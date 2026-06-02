import argparse
import sys
import time

import numpy as np
from PIL import Image

from kernels import get_kernel, KERNELS
from convolution import apply_convolution


def main():
    parser = argparse.ArgumentParser(description="Image convolution tool")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to the output image")
    parser.add_argument(
        "--kernel", default="blur", choices=KERNELS.keys(), help="Type of kernel"
    )
    parser.add_argument(
        "--padding",
        default="zero",
        choices=["zero", "edge", "reflect", "wrap"],
        help="Padding",
    )
    parser.add_argument(
        "--gray", action="store_true", help="Converge into grayscale image"
    )
    args = parser.parse_args()

    try:
        img = Image.open(args.input)
    except FileNotFoundError:
        print(f"Error: file {args.input} does not exist")
        sys.exit(1)

    img = img.convert("L") if args.gray else img.convert("RGB")

    arr = np.array(img)
    kernel = get_kernel(args.kernel)

    start = time.time()
    result = apply_convolution(arr, kernel, padding_mode=args.padding)
    elapsed = time.time() - start

    Image.fromarray(result).save(args.output)
    print(f"Succesfull! The image was saved in {args.output}")
    print(f"Time: {elapsed:.4f} sec")


if __name__ == "__main__":
    main()
