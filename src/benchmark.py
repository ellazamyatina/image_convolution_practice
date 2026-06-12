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
    for b in benchmarks:
        name = b["name"]
        params = b["params"]
        impl = name.split("[")[0].replace("test_", "")
        impl = impl_map.get(impl, impl)
        kernel = params["kernel_name"].replace("_", " ").replace("kernel ", "")
        size = params.get("size")
        if isinstance(size, list):
            mode = "GRAY" if len(size) == 2 else "RGB"
            h, w = size[:2]
            label = f"{kernel} {mode} ({h}x{w})"
        else:
            mode = "GRAY" if params.get("gray") else "RGB"
            label = f"{kernel} {mode} ({size}x{size})"
        mean_ms = b["stats"]["mean"] * 1000
        std_ms = b["stats"]["stddev"] * 1000
        rows[(label, impl)] = (mean_ms, std_ms)

    kernels_3x3 = ["box blur 3x3", "sharpen 3x3", "gaussian 3x3", "identity 3x3"]
    kernels_5x5 = ["box blur 5x5", "sharpen 5x5", "gaussian 5x5", "identity 5x5"]
    size_modes = [(1024, "RGB"), (1024, "GRAY"), (2048, "RGB"), (2048, "GRAY")]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    for row in range(4):
        for col in range(2):
            ax = axes[row][col]
            kernel_name = kernels_3x3[row] if col == 0 else kernels_5x5[row]

            my_data = []
            pil_data = []
            cv_data = []
            for s, m in size_modes:
                label = f"{kernel_name} {m} ({s}x{s})"
                my_data.append(rows.get((label, "My"), (0, 0)))
                pil_data.append(rows.get((label, "Pillow"), (0, 0)))
                cv_data.append(rows.get((label, "OpenCV"), (0, 0)))
            my_vals, my_errs = zip(*my_data)
            pil_vals, pil_errs = zip(*pil_data)
            cv_vals, cv_errs = zip(*cv_data)

            x = np.arange(len(size_modes))
            width = 0.25

            err_kw = {"lw": 1.5, "capsize": 3, "capthick": 1.5}

            ax.bar(
                x - width,
                my_vals,
                width,
                label="My",
                color="#3366cc",
                yerr=my_errs,
                error_kw=err_kw,
            )
            ax.bar(
                x,
                pil_vals,
                width,
                label="Pillow",
                color="#ff9933",
                yerr=pil_errs,
                error_kw=err_kw,
            )
            ax.bar(
                x + width,
                cv_vals,
                width,
                label="OpenCV",
                color="#339933",
                yerr=cv_errs,
                error_kw=err_kw,
            )

            ax.set_yscale("log")
            ax.set_title(kernel_name)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{s} {m}" for s, m in size_modes],
                rotation=20,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel("Time (ms)")

            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "benchmark.pdf"
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
