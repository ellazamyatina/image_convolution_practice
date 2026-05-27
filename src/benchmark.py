import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from src.convolution import apply_convolution
from src.kernels import get_kernel, KERNELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def bench(fn, n=5):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def apply_pillow(img_arr, kernel):
    k_h, k_w = kernel.shape
    flat = kernel.ravel().tolist()
    img_pil = Image.fromarray(img_arr)
    if img_pil.mode == "L":
        return np.array(
            img_pil.filter(ImageFilter.Kernel((k_w, k_h), flat, scale=1)),
            dtype=np.uint8,
        )
    return np.array(
        img_pil.filter(ImageFilter.Kernel((k_w, k_h), flat, scale=1)), dtype=np.uint8
    )


def run_scenario(kernel_name, image_shape):
    kernel = get_kernel(kernel_name)
    img = np.random.randint(0, 255, image_shape, dtype=np.uint8)

    my_time = bench(lambda: apply_convolution(img, kernel, padding_mode="zero"))
    pil_time = bench(lambda: apply_pillow(img, kernel))
    cv_time = bench(lambda: cv2.filter2D(img, -1, kernel.astype(np.float32)))

    mode_label = "RGB" if len(image_shape) == 3 else "GRAY"
    clean_name = kernel_name.replace("_", " ").replace("kernel ", "")
    label = f"{clean_name} {mode_label}"
    return label, my_time * 1000, pil_time * 1000, cv_time * 1000


def plot_results(results):
    labels = [r[0] for r in results]
    my_ms = [r[1] for r in results]
    sp_ms = [r[2] for r in results]
    cv_ms = [r[3] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width, my_ms, width, label="My implementation", color="#3366cc")
    bars2 = ax.bar(x, sp_ms, width, label="Pillow", color="#ff9933")
    bars3 = ax.bar(x + width, cv_ms, width, label="OpenCV", color="#339933")

    ax.set_ylabel("Time, ms (log scale)")
    ax.set_yscale("log")
    ax.set_title("Convolution performance comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "benchmark.png"
    fig.savefig(path, dpi=150)
    print(f"Chart saved: {path}")
    plt.close(fig)


def main():
    kernel_names = list(KERNELS.keys())
    sizes = [(256, 256, 3), (256, 256)]

    results = []
    for kernel_name in kernel_names:
        for shape in sizes:
            label, my_ms, sp_ms, cv_ms = run_scenario(kernel_name, shape)
            results.append((label, my_ms, sp_ms, cv_ms))
            mode = "RGB" if len(shape) == 3 else "GRAY"
            print(
                f"{kernel_name:25s} {mode:5s} | my:{my_ms:8.2f} | pillow:{sp_ms:8.2f} | cv:{cv_ms:8.2f} ms"  # noqa: E501
            )

    data = [
        {"scenario": r[0], "my_ms": r[1], "pillow_ms": r[2], "cv_ms": r[3]}
        for r in results
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(data, f, indent=2)

    plot_results(results)


if __name__ == "__main__":
    main()
