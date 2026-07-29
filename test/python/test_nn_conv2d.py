import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)


@dataclasses.dataclass
class TestCase:
    name: str
    batch_size: int
    in_dims: Tuple[int, int]
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]


TEST_CASES = [
    TestCase("test_conv2d_asymmetric", 1, (5, 3), 2, 3, (3, 2), (2, 1), (1, 0)),
    TestCase("test_conv2d_multi_batch", 3, (8, 8), 8, 16, (3, 3), (1, 1), (1, 1)),
    TestCase("test_conv2d_pointwise", 1, (14, 14), 4, 8, (1, 1), (1, 1), (0, 0)),
    TestCase("test_conv2d_valid_stride", 1, (7, 7), 1, 1, (3, 3), (3, 3), (0, 0)),
]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(argv[0]), formatter_class=argparse.RawTextHelpFormatter
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
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    bin_path = os.path.join(args.output, f"{Path(__file__).stem}.bin")

    with open(bin_path, "wb") as f:
        for test_case in TEST_CASES:
            conv2d = nn.Conv2d(
                in_channels=test_case.in_channels,
                out_channels=test_case.out_channels,
                kernel_size=test_case.kernel_size,
                stride=test_case.stride,
                padding=test_case.padding,
                bias=True,
            )

            input_tensor = torch.randn(
                test_case.batch_size, test_case.in_channels, *test_case.in_dims
            )
            with torch.no_grad():
                output_tensor = conv2d(input_tensor)

            # order must strictly match in c++ test case parsing logic
            for tensor in [conv2d.weight, conv2d.bias, input_tensor, output_tensor]:
                bytes_buffer = (
                    tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                )
                f.write(bytes_buffer)

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
