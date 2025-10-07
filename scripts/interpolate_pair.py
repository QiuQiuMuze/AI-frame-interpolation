"""Command line helper to interpolate a pair of images using rife-ncnn-vulkan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ai_frame_interpolation import RifeNcnnConfig, RifeNcnnInterpolator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate a middle frame between two input images using the "
            "rife-ncnn-vulkan executable."
        )
    )
    parser.add_argument("first", type=Path, help="Path to the first input image")
    parser.add_argument("second", type=Path, help="Path to the second input image")
    parser.add_argument("output", type=Path, help="Where to write the interpolated frame")
    parser.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="Path to the rife-ncnn-vulkan executable",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional model name to load (for example 'rife-v4.6'). Uses the "
            "binary default when omitted."
        ),
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.5,
        help="Interpolation time value between 0 and 1 (default: 0.5)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Override the internal scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation for potentially higher quality",
    )
    parser.add_argument(
        "--uhd",
        action="store_true",
        help="Enable UHD mode, recommended for 4K and above",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Optional GPU index to run on",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional number of CPU threads",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the rife-ncnn-vulkan binary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RifeNcnnConfig(
        binary=args.binary,
        model=args.model,
        scale=args.scale,
        tta=args.tta,
        uhd=args.uhd,
        gpu_id=args.gpu_id,
        num_threads=args.threads,
    )
    interpolator = RifeNcnnInterpolator(config)

    extra_args: Optional[list[str]] = None
    if args.extra_args:
        # argparse includes the leading '--' separator, remove it when present.
        extra_args = [arg for arg in args.extra_args if arg != "--"]

    interpolator.interpolate(
        first_image=args.first,
        second_image=args.second,
        output=args.output,
        time=args.time,
        extra_args=extra_args,
    )


if __name__ == "__main__":
    main()
