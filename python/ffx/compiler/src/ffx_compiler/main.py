import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import ffx_compiler
from ffx_compiler.compiler import Compiler
from ffx_compiler.error import FfxError
from ffx_compiler.setup_logging import setup_logging

if sys.version_info[:2] < (3, 12):
    sys.exit(
        f"Python {'.'.join(map(str, sys.version_info[:3]))} is not supported. You should run {ffx_compiler.__title__} with Python 3.12 or later"
    )

logger = logging.getLogger("ffx_compiler.main")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(argv[0]),
        description=ffx_compiler.__summary__,
        # epilog=epilog(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        required=True,
        type=Path,
        help="path to exported model file",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default="codegen",
        help="Path to workspace where code is emitted",
    )
    parser.add_argument(
        "-O",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        metavar="LEVEL",
        help=(
            "Optimization level (default: 1):\n"
            "  0  no optimizations (-O0)\n"
            "  1  reserved for future passes (currently same as -O0)\n"
            "  2  reserved for future passes (currently same as -O0)\n"
            "  3  reserved for future passes (currently same as -O0)\n"
        ),
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="ffx_runtime",
        choices=["ffx_runtime"],
        metavar="RUNTIME",
        help="Runtime framework to emit compatible inference code",
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        choices=["serial", "cuda", "hip", "omp2", "tbb"],
        default=["serial", "cuda", "hip", "omp2", "tbb"],
        help="One or more hardware execution targets to synthesise code for, default all supported accelerators",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite content of previous compilation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + ffx_compiler.__version__
    )
    return parser.parse_args()


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    try:
        compiler = Compiler(
            model_path=args.model,
            output_path=args.output,
            level=args.O,
            runtime=args.runtime,
            override=args.force,
        )
        return compiler.compile()
    except FfxError as error:
        logger.error(error)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
