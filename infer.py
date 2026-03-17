import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from model import TripleSpaceDehazeNet

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'weight/model_latest.pth'
INPUT_DIR  = 'data/test'
OUTPUT_DIR = './output_submission'

# ── Sliding-window inference hyperparameters ───────────────────────────────
TILE_SIZE = 512    # Should match the training resolution
OVERLAP   = 128    # Overlap in pixels between adjacent tiles (recommended 20%~30% of tile_size)
#   stride = TILE_SIZE - OVERLAP = 384
#   For a 1800×1200 image, ~18 tiles are generated per TTA augmentation, 8×18=144 forwards total
#   If VRAM is insufficient, reduce OVERLAP to 64 (stride=448)
# ───────────────────────────────────────────────────────────────────────────


# ======================================================================
# Core: Overlapping sliding-window inference (Hann window weighted blending)
# ======================================================================
def _make_hann_window(size_h: int, size_w: int, device: torch.device) -> torch.Tensor:
    """Generate a 2D Hann window with shape (1, 1, size_h, size_w)."""
    h1d = torch.hann_window(size_h, device=device)
    w1d = torch.hann_window(size_w, device=device)
    win = h1d.unsqueeze(1) * w1d.unsqueeze(0)
    return win.unsqueeze(0).unsqueeze(0)


def _get_starts(length: int, tile: int, stride: int) -> list[int]:
    """Compute the list of start coordinates for the sliding window along a single axis,
    ensuring full coverage of the right/bottom edges."""
    starts = list(range(0, length - tile + 1, stride))
    if not starts:
        starts = [0]
    elif starts[-1] + tile < length:
        starts.append(max(length - tile, 0))
    return starts


def overlap_tile_forward(
    model:      torch.nn.Module,
    img_tensor: torch.Tensor,
    tile_size:  int = TILE_SIZE,
    overlap:    int = OVERLAP,
) -> torch.Tensor:
    """
    Run overlapping sliding-window inference on a single image already on device.

    Returns a Tensor of shape (1, 3, H, W) in the range [0, 1] on CPU.
    """
    _, C, H, W = img_tensor.shape
    stride = tile_size - overlap

    # If the image is smaller than a tile, run a direct forward pass to avoid padding artifacts
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            return model(img_tensor).clamp(0, 1).cpu()

    output = torch.zeros(1, C, H, W, device=img_tensor.device)
    weight = torch.zeros(1, 1, H, W, device=img_tensor.device)

    y_starts = _get_starts(H, tile_size, stride)
    x_starts = _get_starts(W, tile_size, stride)

    for y in y_starts:
        for x in x_starts:
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            patch = img_tensor[:, :, y:y_end, x:x_end]
            ph, pw = patch.shape[2], patch.shape[3]

            # Reflect-pad edge tiles that are smaller than tile_size
            pad_h = tile_size - ph
            pad_w = tile_size - pw
            if pad_h > 0 or pad_w > 0:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                pred = model(patch).clamp(0, 1)           # (1, 3, T, T)

            # Crop back to actual size and apply the corresponding Hann weights
            pred = pred[:, :, :ph, :pw]
            win  = _make_hann_window(ph, pw, img_tensor.device)

            output[:, :, y:y_end, x:x_end] += pred * win
            weight[:, :, y:y_end, x:x_end] += win

    output = (output / weight.clamp(min=1e-6)).clamp(0, 1)
    return output.cpu()


# ======================================================================
# TTA: 8-way augmentation × overlapping sliding-window → inverse transform → mean
# ======================================================================
def tta_tile_predict(
    model:      torch.nn.Module,
    img_tensor: torch.Tensor,    # (C, H, W), CPU, range [0, 1]
    tile_size:  int = TILE_SIZE,
    overlap:    int = OVERLAP,
) -> torch.Tensor:
    """
    Run TTA (8-way) × overlapping sliding-window inference on a single image.

    Pipeline for each augmentation:
        forward transform → unsqueeze(0) → to(device) → overlap_tile_forward
        → squeeze(0) → inverse transform

    Returns the pixel-wise mean of all 8 results.
    """

    # ------------------------------------------------------------------
    # Define 8 pairs of (forward augmentation, inverse restoration) functions.
    # rot90 operates only on the H/W dimensions — no black borders for non-square images.
    # After rotation the image shape becomes (C, W, H); overlap_tile handles this adaptively.
    # ------------------------------------------------------------------
    transforms = [
        # (forward_fn, inverse_fn)
        (lambda t: t, lambda t: t),                                                                   # 1. Original
        (lambda t: TF.hflip(t), lambda t: TF.hflip(t)),                                               # 2. Horizontal flip
        (lambda t: TF.vflip(t), lambda t: TF.vflip(t)),                                               # 3. Vertical flip
        (lambda t: TF.vflip(TF.hflip(t)), lambda t: TF.hflip(TF.vflip(t))),                           # 4. Horizontal + vertical flip
        (lambda t: torch.rot90(t, 1, [-2, -1]), lambda t: torch.rot90(t, -1, [-2, -1])), # 5. Rotate 90° CCW
        (lambda t: TF.hflip(torch.rot90(t, 1, [-2, -1])),                                    # 6. Rotate 90° CCW + horizontal flip
         lambda t: torch.rot90(TF.hflip(t), -1, [-2, -1])),
        (lambda t: TF.vflip(torch.rot90(t, 1, [-2, -1])),                                    # 7. Rotate 90° CCW + vertical flip
         lambda t: torch.rot90(TF.vflip(t), -1, [-2, -1])),
        (lambda t: TF.vflip(TF.hflip(torch.rot90(t, 1, [-2, -1]))),                          # 8. Rotate 90° CCW + horizontal + vertical flip
         lambda t: torch.rot90(TF.hflip(TF.vflip(t)), -1, [-2, -1])),
    ]

    results = []
    for fwd, inv in transforms:
        aug   = fwd(img_tensor)
        batch = aug.unsqueeze(0).to(DEVICE)               # (1, C, H', W') on device
        pred  = overlap_tile_forward(model, batch,        # (1, C, H', W') on CPU
                                     tile_size, overlap)
        pred  = pred.squeeze(0)                           # (C, H', W')
        results.append(inv(pred))                         # Inverse transform → (C, H, W)

    return torch.stack(results).mean(dim=0)               # (C, H, W)


# ======================================================================
# Model loading
# ======================================================================
def load_model() -> torch.nn.Module:
    model = TripleSpaceDehazeNet().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Strip the "module." prefix that may appear from multi-GPU training
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


# ======================================================================
# Main pipeline
# ======================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()

    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    elapsed_times: list[float] = []   # Record inference time (seconds) for each image

    for fname in files:
        img_path = os.path.join(INPUT_DIR, fname)
        img      = Image.open(img_path).convert('RGB')
        tensor   = TF.to_tensor(img)                   # (3, H, W), CPU

        H, W = tensor.shape[1], tensor.shape[2]
        n_tiles = (len(_get_starts(H, TILE_SIZE, TILE_SIZE - OVERLAP)) *
                   len(_get_starts(W, TILE_SIZE, TILE_SIZE - OVERLAP)))
        print(f"[{fname}]  size={W}×{H}  tiles/aug={n_tiles}  total_fwd={n_tiles*8}")

        # ── Timing: synchronize GPU before start to ensure a clean baseline ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        result = tta_tile_predict(model, tensor, TILE_SIZE, OVERLAP)  # (3, H, W)

        # ── Timing: wait for all GPU kernels to finish before recording end time ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        elapsed_times.append(elapsed)
        print(f"  inference time : {elapsed:.3f} s")

        out_img = TF.to_pil_image(result.clamp(0, 1))
        out_img.save(os.path.join(OUTPUT_DIR, fname))
        print(f"  → saved to {OUTPUT_DIR}/{fname}")

    # ── Summary statistics ───────────────────────────────────────────────
    if elapsed_times:
        n          = len(elapsed_times)
        mean_t     = sum(elapsed_times) / n
        min_t      = min(elapsed_times)
        max_t      = max(elapsed_times)
        total_t    = sum(elapsed_times)
        print("\n" + "=" * 52)
        print(f"  Inference time summary  ({n} image{'s' if n > 1 else ''})")
        print("=" * 52)
        print(f"  Total   : {total_t:>8.3f} s")
        print(f"  Average : {mean_t:>8.3f} s / image")
        print(f"  Min     : {min_t:>8.3f} s")
        print(f"  Max     : {max_t:>8.3f} s")
        print("=" * 52)


if __name__ == "__main__":
    main()