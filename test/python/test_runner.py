import argparse
import os
import subprocess
import sys
from typing import List, Optional


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test",
        metavar="PATH",
        required=True,
        type=str,
        help="path to test to be executed",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        type=str,
        help="output path",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    return parser.parse_args()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv if argv is None else argv)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    test_info = subprocess.run(
        [sys.executable, args.test, "--output", args.output],
        capture_output=True,
        text=True,
    )
    if test_info.returncode:
        print("STDERR:", test_info.stderr)
        return test_info.returncode

    print(test_info.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
