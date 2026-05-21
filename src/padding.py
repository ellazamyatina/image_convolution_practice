import numpy as np

PADDING_MODES = {
    "zero": "constant",
    "edge": "edge",
    "reflect": "reflect",
    "wrap": "wrap",
}


def apply_padding(image: np.ndarray,
                  kernel_shape: tuple[int, int],
                  mode: str = "zero") -> np.ndarray:
    if mode not in PADDING_MODES:
        raise ValueError(
            f"Unknown padding mode: {mode}. "
            f"Available: {list(PADDING_MODES.keys())}"
        )

    m, n = kernel_shape
    padding_height = m // 2
    padding_width = n // 2

    #if the image is grayscale => r=g=b
    pad_width = (
        (padding_height, padding_height),
        (padding_width, padding_width),
        (0, 0),
    )

    return np.pad(
        image,
        pad_width,
        mode=PADDING_MODES[mode],
        constant_values=0,
    )
