import torch
import numpy as np
import random
import os


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Differentiable RGB to YCbCr conversion."""
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    # Ensure images are in [0, 1]
    image = torch.clamp(image, 0, 1)

    r: torch.Tensor = image[:, 0, :, :]
    g: torch.Tensor = image[:, 1, :, :]
    b: torch.Tensor = image[:, 2, :, :]

    delta = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], dim=1)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:

    image = torch.clamp(image, 0, 1)

    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    x_n = x / 0.95047
    y_n = y / 1.00000
    z_n = z / 1.08883


    threshold = 0.008856

    def f(t):
        return torch.where(t > threshold, torch.pow(t + 1e-6, 1 / 3), 7.787 * t + 16 / 116)

    l_channel = 116 * f(y_n) - 16
    a_channel = 500 * (f(x_n) - f(y_n))
    b_channel = 200 * (f(y_n) - f(z_n))

    l_norm = l_channel / 100.0
    a_norm = (a_channel + 128) / 255.0
    b_norm = (b_channel + 128) / 255.0

    return torch.stack([l_norm, a_norm, b_norm], dim=1)