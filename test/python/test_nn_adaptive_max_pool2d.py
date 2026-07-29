import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Optional

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
    out_height: int
    out_width: int


TEST_CASES: List[TestCase] = [
    # Clean halving — in_h % out_h == 0 for both dims
    TestCase("clean_halving", 1, 1, 4, 4, 2, 2),
    # Multi-channel, non-square input
    TestCase("multi_channel", 1, 3, 6, 4, 2, 2),
    # Global pooling — collapses entire spatial map to a single value per channel
    TestCase("global_pool", 2, 4, 8, 8, 1, 1),
    # Non-divisible spatial dims — exercises the floor/ceiling window formula
    TestCase("non_divisible", 1, 2, 7, 7, 3, 3),
    # Asymmetric output — different reduction factors on H and W
    TestCase("asymmetric_output", 1, 1, 8, 6, 4, 3),
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
                output_tensor = F.adaptive_max_pool2d(
                    input_tensor,
                    output_size=(tc.out_height, tc.out_width),
                )

            # Binary layout per test case: input → output
            for tensor in [input_tensor, output_tensor]:
                f.write(tensor.detach().cpu().numpy().astype(np.float32).tobytes())

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
