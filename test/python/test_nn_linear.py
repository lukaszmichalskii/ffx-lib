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
class LinearTestCase:
    name: str
    batch_size: int
    in_features: int
    out_features: int
    dims: Tuple[int, ...] = ()


TEST_CASES = [
    LinearTestCase(
        "test_linear_standard", batch_size=1, in_features=128, out_features=64
    ),
    LinearTestCase(
        "test_linear_wide_multi_batch", batch_size=4, in_features=32, out_features=512
    ),
    LinearTestCase(
        "test_linear_single_unit", batch_size=1, in_features=1, out_features=1
    ),
    LinearTestCase(
        "test_linear_3d_tensor",
        batch_size=2,
        in_features=64,
        out_features=32,
        dims=(5,),
    ),
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
            linear = nn.Linear(
                in_features=test_case.in_features,
                out_features=test_case.out_features,
                bias=True,
            )

            input_shape = (
                (test_case.batch_size,) + test_case.dims + (test_case.in_features,)
            )
            input_tensor = torch.randn(*input_shape)

            with torch.no_grad():
                output_tensor = linear(input_tensor)

            # order must strictly match in c++ test case parsing logic
            for tensor in [linear.weight, linear.bias, input_tensor, output_tensor]:
                bytes_buffer = (
                    tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                )
                f.write(bytes_buffer)

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
