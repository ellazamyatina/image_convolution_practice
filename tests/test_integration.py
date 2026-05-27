from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.convolution import apply_convolution
from src.kernels import get_kernel

GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden(name):
    return np.array(Image.open(GOLDEN_DIR / name), dtype=np.uint8)


@pytest.fixture(scope="session")
def input_img():
    return np.array(Image.open(GOLDEN_DIR / "input.png").convert("RGB"), dtype=np.uint8)


SCENARIOS = [
    ("box_blur_3x3", "zero"),
    ("box_blur_5x5", "edge"),
    ("sharpen_3x3", "edge"),
    ("gaussian_3x3", "reflect"),
    ("sharpen_5x5", "zero"),
    ("gaussian_5x5", "wrap"),
    ("identity_kernel_3x3", "zero"),
]


@pytest.mark.parametrize("kernel_name,padding", SCENARIOS)
def test_golden(input_img, kernel_name, padding):
    kernel = get_kernel(kernel_name)
    result = apply_convolution(input_img, kernel, padding_mode=padding)
    expected = load_golden(f"{kernel_name}_{padding}.png")
    assert result.shape == expected.shape
    assert np.array_equal(result, expected), (
        f"{kernel_name} {padding} differs from golden"
    )
