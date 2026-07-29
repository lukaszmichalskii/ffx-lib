import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)


@dataclasses.dataclass
class TestCase:
    name: str
    batch_size: int
    in_channels: int
    in_height: int
    in_width: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]


TEST_CASES: List[TestCase] = [
    # Non-overlapping 2×2 pool — most common use-case
    TestCase("basic_2x2", 1, 1, 4, 4, (2, 2), (2, 2), (0, 0)),
    # Multiple channels — verifies per-channel independence
    TestCase("multi_channel", 1, 3, 6, 6, (2, 2), (2, 2), (0, 0)),
    # Overlapping 3×3 with stride=1, multi-batch — window overlap exercised
    TestCase("overlapping_3x3", 2, 4, 8, 8, (3, 3), (1, 1), (0, 0)),
    # Symmetric padding — output size grows vs. the unpadded case
    TestCase("with_padding", 1, 2, 5, 5, (2, 2), (2, 2), (1, 1)),
    # Asymmetric kernel and stride — height and width treated independently
    TestCase("asymmetric", 1, 1, 4, 6, (2, 3), (2, 3), (0, 0)),
]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        type=str,
        help="output directory path",
    )
    return parser.parse_args()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv if argv is None else argv)
    os.makedirs(args.output, exist_ok=True)


    bin_path = os.path.join(args.output, f"{Path(__file__).stem}.bin")

    with open(bin_path, "wb") as f:
        for tc in TEST_CASES:
            input_tensor = torch.randn(
                tc.batch_size, tc.in_channels, tc.in_height, tc.in_width
            )
            with torch.no_grad():
                output_tensor = F.max_pool2d(
                    input_tensor,
                    kernel_size=tc.kernel_size,
                    stride=tc.stride,
                    padding=tc.padding,
                )

            # Binary layout per test case: input → output
            for tensor in [input_tensor, output_tensor]:
                f.write(tensor.detach().cpu().numpy().astype(np.float32).tobytes())

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
