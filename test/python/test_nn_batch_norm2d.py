import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)


@dataclasses.dataclass
class TestCase:
    name: str
    batch_size: int
    channels: int
    height: int
    width: int


TEST_CASES = [
    TestCase("test_bn2d_standard", batch_size=1, channels=4, height=4, width=4),
    TestCase("test_bn2d_multi_batch", batch_size=2, channels=8, height=4, width=4),
    TestCase("test_bn2d_single_pixel", batch_size=1, channels=16, height=1, width=1),
    TestCase("test_bn2d_deep", batch_size=1, channels=32, height=3, width=3),
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
            # Build nn.BatchNorm2d and switch to eval mode so it uses running statistics
            bn = nn.BatchNorm2d(
                num_features=test_case.channels,
                eps=1e-5,  # matches EpsilonNumerator=1, EpsilonDenominator=100000 in C++
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            bn.eval()

            # Randomise all learnable and running parameters
            with torch.no_grad():
                # weight (gamma) and bias (beta) — per-channel affine parameters
                bn.weight.copy_(torch.randn(test_case.channels).abs() + 0.5)
                bn.bias.copy_(torch.randn(test_case.channels))
                # running statistics — variance must be strictly positive
                bn.running_mean.copy_(torch.randn(test_case.channels))
                bn.running_var.copy_(torch.rand(test_case.channels) + 0.1)

            input_tensor = torch.randn(
                test_case.batch_size,
                test_case.channels,
                test_case.height,
                test_case.width,
            )

            with torch.no_grad():
                output_tensor = bn(input_tensor)

            # Binary write order must exactly match the C++ test parsing layout:
            #   gamma (weight), beta (bias), running_mean, running_var, input, output
            for tensor in [
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
                input_tensor,
                output_tensor,
            ]:
                buf = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                f.write(buf)

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
