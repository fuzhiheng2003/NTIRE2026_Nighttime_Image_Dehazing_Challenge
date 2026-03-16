import torch
import torch.nn as nn
import torch.fft
import torchvision.models as models
from utils import rgb_to_ycbcr
import torch.nn.functional as F
class CharbonnierLoss(nn.Module):
    # 【优化 1】：将 eps 从 1e-3 降低到 1e-6，使其在误差极小时依然能提供稳定的梯度，逼近纯粹的 L1 Loss
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return loss

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super(FFTLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm='backward')
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm='backward')
        loss_amp = self.criterion(torch.abs(pred_fft), torch.abs(target_fft))
        loss_phase = self.criterion(torch.angle(pred_fft), torch.angle(target_fft))
        return self.loss_weight * (loss_amp + loss_phase)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:12]).eval()
        self.criterion = nn.MSELoss()
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return self.criterion(self.vgg_layers(pred), self.vgg_layers(target))

class TVLoss(nn.Module):

    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class ColorCosineLoss(nn.Module):
    def __init__(self):
        super(ColorCosineLoss, self).__init__()

    def forward(self, pred, target):
        # 沿着通道维度(dim=1)进行 L2 归一化
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        # 计算余弦相似度，并求误差
        cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
        return torch.mean(1.0 - cosine_sim)


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred, target):
        # 计算水平方向的梯度 (边缘)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # 计算垂直方向的梯度 (边缘)
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # 对边缘的差异计算 L1 Loss
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)

        return loss_dx + loss_dy

# 3. 修改主 Loss 函数
class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss, self).__init__()
        self.charbonnier = CharbonnierLoss()
        self.fft_loss = FFTLoss(loss_weight=0.05)
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
        self.tv_loss = TVLoss(weight=1.0)
        self.color_cosine = ColorCosineLoss()  # 实例化新增的色彩角度损失

        # 调整权重分布
        self.w_pixel = 1.0  # 稍微提高像素级损失权重
        self.w_fft = 0.1
        self.w_percep = 0.1
        self.w_color = 0.5  # YCbCr 色彩损失
        self.w_color_cos = 0.2  # RGB 角度相似度损失

        # TV Loss 权重可以适度调高，因为现在它只惩罚残差(噪声)
        self.w_tv = 0.01

    def forward(self, pred, target):
        loss_pixel = self.charbonnier(pred, target)
        loss_fft = self.fft_loss(pred, target)
        loss_percep = self.perceptual_loss(pred, target)
        # loss_edge = self.edge_loss(pred, target)
        pred_ycbcr = rgb_to_ycbcr(pred)
        target_ycbcr = rgb_to_ycbcr(target)
        loss_color = self.charbonnier(pred_ycbcr, target_ycbcr)

        # 计算颜色角度损失
        loss_color_cos = self.color_cosine(pred, target)

        # 核心改动：将 TV Loss 作用于误差图 (pred - target)
        # 这能极大程度抑制去雾产生的伪影和高频噪声，且不损伤真实纹理
        loss_tv = self.tv_loss(pred - target)

        total_loss = (self.w_pixel * loss_pixel) + \
                     (self.w_fft * loss_fft) + \
                     (self.w_percep * loss_percep) + \
                     (self.w_color * loss_color) + \
                     (self.w_color_cos * loss_color_cos) + \
                     (self.w_tv * loss_tv) #+ (loss_edge*0.5)

        return total_loss