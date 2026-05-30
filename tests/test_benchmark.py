import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter

from convolution import apply_convolution
from kernels import get_kernel, KERNELS

KERNEL_NAMES = list(KERNELS.keys())
SIZES = [(256, 256, 3), (256, 256)]


def run_pillow(img, kernel):
    flat = kernel.ravel().tolist()
    k_w, k_h = kernel.shape
    return np.array(
        Image.fromarray(img).filter(ImageFilter.Kernel((k_w, k_h), flat, scale=1)),
        dtype=np.uint8,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize("size", SIZES)
def test_my(benchmark, kernel_name, size):
    kernel = get_kernel(kernel_name)
    img = np.random.randint(0, 255, size, dtype=np.uint8)
    benchmark(apply_convolution, img, kernel, padding_mode="zero")


@pytest.mark.benchmark
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize("size", SIZES)
def test_pillow(benchmark, kernel_name, size):
    kernel = get_kernel(kernel_name)
    img = np.random.randint(0, 255, size, dtype=np.uint8)
    benchmark(run_pillow, img, kernel)


@pytest.mark.benchmark
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize("size", SIZES)
def test_opencv(benchmark, kernel_name, size):
    kernel = get_kernel(kernel_name)
    img = np.random.randint(0, 255, size, dtype=np.uint8)
    benchmark(cv2.filter2D, img, -1, kernel.astype(np.float32))
