import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.padding import apply_padding


def apply_convolution(
    image_arr: np.ndarray, kernel: np.ndarray, padding_mode: str = "zero"
) -> np.ndarray:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be a 2D square matrix")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel size must be odd")
    if image_arr.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    is_grayscale = image_arr.ndim == 2
    if is_grayscale:
        image_arr = image_arr[..., None]

    h, w, c = image_arr.shape
    k_h, k_w = kernel.shape

    padded = apply_padding(image_arr, kernel.shape, mode=padding_mode)

    windows = sliding_window_view(padded, (k_h, k_w), axis=(0, 1))
    result = np.einsum("hwcxy,xy->hwc", windows, kernel, optimize="greedy")

    result = np.clip(result, 0, 255).astype(np.uint8)

    if is_grayscale:
        result = result[:, :, 0]

    return result
