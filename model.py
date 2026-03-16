import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rgb_to_ycbcr, rgb_to_lab


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(x_cat)
        return x * self.sigmoid(scale)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, k_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(k_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = CBAM(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ca(out)
        out = out + residual
        return out

def get_dcp_prior(rgb):
    min_c, _ = torch.min(rgb, dim=1, keepdim=True)
    dc = -F.max_pool2d(-min_c, kernel_size=3, stride=1, padding=1)
    return dc


def get_grad_prior(ycbcr):
    y_channel = ycbcr[:, 0:1, :, :]
    dx = y_channel[:, :, :, 1:] - y_channel[:, :, :, :-1]
    dy = y_channel[:, :, 1:, :] - y_channel[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)


def get_illum_prior(lab):
    l_channel = lab[:, 0:1, :, :]
    return 1.0 - torch.clamp(l_channel, 0.0, 1.0)
def get_smoothness_prior(rgb):
    gray = rgb.mean(dim=1, keepdim=True)
    mean_l = F.avg_pool2d(gray, kernel_size=15, stride=1, padding=7)
    sq_l = F.avg_pool2d(gray ** 2, kernel_size=15, stride=1, padding=7)
    var_l = torch.clamp(sq_l - mean_l ** 2, min=1e-8)
    mean_m = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    sq_m = F.avg_pool2d(gray ** 2, kernel_size=7, stride=1, padding=3)
    var_m = torch.clamp(sq_m - mean_m ** 2, min=1e-8)
    var_min = torch.min(var_l, var_m)
    smoothness = torch.exp(-var_min * 80.0)
    return smoothness

class HVStripContextBlock(nn.Module):
    def __init__(self, channels):
        super(HVStripContextBlock, self).__init__()
        mid = max(channels // 4, 16)
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ctx_h = self.h_conv(x.mean(dim=3, keepdim=True)).expand(-1, -1, h, w)
        ctx_v = self.v_conv(x.mean(dim=2, keepdim=True)).expand(-1, -1, h, w)
        attn = self.fuse(torch.cat([ctx_h, ctx_v], dim=1))
        return x * attn


class JointPriorFusion(nn.Module):
    def __init__(self):
        super(JointPriorFusion, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv = nn.Conv2d(16, 16, 7, padding=3, groups=16)
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, ycbcr, lab):
        dcp = get_dcp_prior(rgb)
        grad = get_grad_prior(ycbcr)
        illum = get_illum_prior(lab)
        smooth = get_smoothness_prior(rgb)
        cat_prior = torch.cat([dcp, grad, illum, smooth], dim=1)  # [B, 4, H, W]
        x = self.relu(self.conv1(cat_prior))
        x = self.relu(self.dw_conv(x))
        joint_prior = self.sigmoid(self.conv2(x))
        return joint_prior


class PriorSFTBlock(nn.Module):
    def __init__(self, feat_channels):
        super(PriorSFTBlock, self).__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(1, feat_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels // 2, feat_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma_conv = nn.Conv2d(feat_channels, feat_channels, 1)
        self.beta_conv = nn.Conv2d(feat_channels, feat_channels, 1)

    def forward(self, feat, prior_map):
        cond = self.cond_conv(prior_map)
        gamma = self.gamma_conv(cond)
        beta = self.beta_conv(cond)
        return feat * (1.0 + gamma) + beta


class CrossDomainPriorBridge(nn.Module):
    def __init__(self, out_channels=64):
        super(CrossDomainPriorBridge, self).__init__()
        self.conv_rgb = nn.Sequential(nn.Conv2d(3, out_channels, 3, padding=1), ResBlock(out_channels))
        self.conv_ycbcr = nn.Sequential(nn.Conv2d(3, out_channels, 3, padding=1), ResBlock(out_channels))
        self.conv_lab = nn.Sequential(nn.Conv2d(3, out_channels, 3, padding=1), ResBlock(out_channels))

        self.joint_prior_net = JointPriorFusion()
        self.sft_rgb = PriorSFTBlock(out_channels)
        self.sft_ycbcr = PriorSFTBlock(out_channels)
        self.sft_lab = PriorSFTBlock(out_channels)

        self.fusion_ycbcr = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), ResBlock(out_channels))
        self.fusion_lab = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), ResBlock(out_channels))
        self.final_fusion = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), ResBlock(out_channels))

    def forward(self, rgb, lab, ycbcr):
        f_rgb = self.conv_rgb(rgb)
        f_ycbcr = self.conv_ycbcr(ycbcr)
        f_lab = self.conv_lab(lab)
        joint_prior = self.joint_prior_net(rgb, ycbcr, lab)
        f_rgb = self.sft_rgb(f_rgb, joint_prior)
        f_ycbcr = self.sft_ycbcr(f_ycbcr, joint_prior)
        f_lab = self.sft_lab(f_lab, joint_prior)
        f_stream_a = self.fusion_ycbcr(torch.cat([f_rgb, f_ycbcr], dim=1))
        f_stream_b = self.fusion_lab(torch.cat([f_rgb, f_lab], dim=1))
        f_out = self.final_fusion(torch.cat([f_stream_a, f_stream_b], dim=1))
        return f_out, joint_prior


class PriorGuidedSkipFusion(nn.Module):
    def __init__(self, channels):
        super(PriorGuidedSkipFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Sigmoid()
        )
        self.reduce = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x_decoder, x_encoder, prior):
        cat_feat = torch.cat([x_decoder, x_encoder, prior], dim=1)
        mask = self.attn(cat_feat)
        x_encoder_filtered = x_encoder * mask
        out = torch.cat([x_decoder, x_encoder_filtered], dim=1)
        return self.reduce(out)


class PixelDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelDown, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return self.conv(x)


class PixelUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelUp, self).__init__()
        self.expand_conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.expand_conv(x)
        return self.pixel_shuffle(x)


class SpatiallyAdaptiveCEM(nn.Module):
    def __init__(self, channels):
        super(SpatiallyAdaptiveCEM, self).__init__()
        self.conv_color = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1))
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1), nn.Sigmoid())
        self.spatial_curve = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 3, 1), nn.Tanh())

    def forward(self, x):
        feat_weighted = x * self.spatial_weight(x)
        color_residue = self.conv_color(feat_weighted)
        gamma = torch.exp(self.spatial_curve(x) * 0.5)
        return color_residue, gamma


class MultiScaleGridContext(nn.Module):
    def __init__(self, channels, grid_size):
        super(MultiScaleGridContext, self).__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 3, channels, 1, bias=False)
        )
        self.post_smooth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        avg_pool = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        max_pool = F.adaptive_max_pool2d(x, (self.grid_size, self.grid_size))
        min_pool = -F.adaptive_max_pool2d(-x, (self.grid_size, self.grid_size))

        pool_cat = torch.cat([avg_pool, max_pool, min_pool], dim=1)
        feat = self.conv(pool_cat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        out = self.post_smooth(out)
        return out


class AdaptiveMultiScaleBlock(nn.Module):

    def __init__(self, channels):
        super(AdaptiveMultiScaleBlock, self).__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.branch2 = MultiScaleGridContext(channels, grid_size=32)
        self.branch3 = MultiScaleGridContext(channels, grid_size=16)
        self.branch4 = HVStripContextBlock(channels)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 4, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        self.pw_conv = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = SpatialAttention(kernel_size=7)

    def forward(self, x):
        residual = x
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)

        cat_feat = torch.cat([f1, f2, f3, f4], dim=1)
        gates = self.spatial_gate(cat_feat)
        g1 = gates[:, 0:1, :, :]
        g2 = gates[:, 1:2, :, :]
        g3 = gates[:, 2:3, :, :]
        g4 = gates[:, 3:4, :, :]

        out = (f1 * g1) + (f2 * g2) + (f3 * g3) + (f4 * g4)
        out = self.pw_conv(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ca(out)
        return out + residual


class TripleSpaceDehazeNet(nn.Module):
    def __init__(self):
        super(TripleSpaceDehazeNet, self).__init__()
        base_c = 64
        self.bgb = CrossDomainPriorBridge(out_channels=base_c)

        self.down1 = nn.Sequential(
            PixelDown(base_c, base_c * 2), ResBlock(base_c * 2), ResBlock(base_c * 2))
        self.down2 = nn.Sequential(
            PixelDown(base_c * 2, base_c * 4), ResBlock(base_c * 4), ResBlock(base_c * 4))

        self.bottleneck = nn.Sequential(
            AdaptiveMultiScaleBlock(256),
            AdaptiveMultiScaleBlock(256),
            _HVStripResWrapper(256),
            AdaptiveMultiScaleBlock(256),
            AdaptiveMultiScaleBlock(256),
            AdaptiveMultiScaleBlock(256),
        )

        self.skip_b1 = PriorGuidedSkipFusion(base_c * 4)
        self.up1 = PixelUp(base_c * 4, base_c * 2)

        self.skip1 = PriorGuidedSkipFusion(base_c * 2)
        self.dec1 = nn.Sequential(ResBlock(base_c * 2), ResBlock(base_c * 2))

        self.up2 = PixelUp(base_c * 2, base_c * 1)
        self.skip2 = PriorGuidedSkipFusion(base_c)
        self.dec2 = nn.Sequential(ResBlock(base_c), ResBlock(base_c))

        self.output_conv = nn.Conv2d(base_c, 3, 3, padding=1)
        self.cem = SpatiallyAdaptiveCEM(base_c)

    def forward(self, x):
        x_ycbcr = rgb_to_ycbcr(x)
        x_lab = rgb_to_lab(x)
        f1, prior_s1 = self.bgb(x, x_lab, x_ycbcr)

        f2 = self.down1(f1)
        prior_s2 = F.avg_pool2d(prior_s1, kernel_size=2, stride=2)

        f3 = self.down2(f2)
        prior_s3 = F.avg_pool2d(prior_s2, kernel_size=2, stride=2)

        neck = self.bottleneck(f3)
        neck = self.skip_b1(neck, f3, prior_s3)

        u1 = self.up1(neck)
        u1 = self.skip1(u1, f2, prior_s2)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = self.skip2(u2, f1, prior_s1)
        u2 = self.dec2(u2)

        basic_residual = self.output_conv(u2)
        color_residual, gamma = self.cem(u2)

        J_base = x + basic_residual + color_residual * 0.5
        J_base = torch.clamp(J_base, 1e-6, 1.0)
        J_enhanced = J_base.pow(gamma)

        out = torch.clamp(J_enhanced, 0.0, 1.0)
        return out

class _HVStripResWrapper(nn.Module):
    def __init__(self, channels):
        super(_HVStripResWrapper, self).__init__()
        self.strip = HVStripContextBlock(channels)
    def forward(self, x):
        return x + self.strip(x)