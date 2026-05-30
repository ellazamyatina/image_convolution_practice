import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_benchmark_data():
    bench_dir = Path(".benchmarks")
    if not bench_dir.exists():
        return None
    files = sorted(bench_dir.rglob("*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def plot_results(data):
    benchmarks = data["benchmarks"]

    impl_map = {
        "my": "My",
        "custom_conv": "My",
        "opencv": "OpenCV",
        "opencv_conv": "OpenCV",
        "pillow": "Pillow",
        "pillow_conv": "Pillow",
    }

    rows = {}
    seen_labels = set()
    for b in benchmarks:
        name = b["name"]
        params = b["params"]
        impl = name.split("[")[0].replace("test_", "")
        impl = impl_map.get(impl, impl)
        kernel = params["kernel_name"].replace("_", " ").replace("kernel ", "")
        size = params.get("size")
        if isinstance(size, list):
            mode = "GRAY" if len(size) == 2 else "RGB"
            label = f"{kernel} {mode}"
        else:
            mode = "GRAY" if params.get("gray") else "RGB"
            label = f"{kernel} {mode} ({size}x{size})"
        seen_labels.add(label)

        mean_ms = b["stats"]["mean"] * 1000
        rows[(label, impl)] = mean_ms

    labels = sorted(seen_labels)

    my_ms = [rows.get((lab, "My"), 0) for lab in labels]
    pil_ms = [rows.get((lab, "Pillow"), 0) for lab in labels]
    cv_ms = [rows.get((lab, "OpenCV"), 0) for lab in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width, my_ms, width, label="My implementation", color="#3366cc")
    bars2 = ax.bar(x, pil_ms, width, label="Pillow", color="#ff9933")
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
    data = load_benchmark_data()
    if data is None:
        print(
            "No benchmark data found."
            "Run: pytest tests/test_benchmark.py --benchmark-autosave"
        )
        return
    plot_results(data)


if __name__ == "__main__":
    main()
