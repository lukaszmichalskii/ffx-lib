import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

torch.manual_seed(42)


@dataclasses.dataclass
class TestCase:
    name: str
    batch_size: int
    m: int
    k: int
    n: int
    # Strides (in number of elements)
    a_batch_stride: int = 0
    a_row_stride: int = 0
    a_col_stride: int = 0
    b_batch_stride: int = 0
    b_row_stride: int = 0
    b_col_stride: int = 0

    def __post_init__(self):
        # Default row-major contiguous strides if unspecified (0)
        if self.a_batch_stride == 0:
            self.a_batch_stride = self.m * self.k
        if self.a_row_stride == 0:
            self.a_row_stride = self.k
        if self.a_col_stride == 0:
            self.a_col_stride = 1

        if self.b_batch_stride == 0:
            self.b_batch_stride = self.k * self.n
        if self.b_row_stride == 0:
            self.b_row_stride = self.n
        if self.b_col_stride == 0:
            self.b_col_stride = 1


TEST_CASES: List[TestCase] = [
    # Basic 2D MatMul (BatchSize=1)
    TestCase("single_batch", batch_size=1, m=4, k=8, n=6),
    # Batched MatMul (e.g. multi-head attention slice)
    TestCase("batched_standard", batch_size=4, m=16, k=32, n=16),
    # Non-square matrix dimensions
    TestCase("asymmetric_dims", batch_size=2, m=7, k=13, n=5),
    # Transposed B matrix layout (b_row_stride=1, b_col_stride=8)
    TestCase(
        "transposed_b",
        batch_size=2,
        m=8,
        k=16,
        n=8,
        b_batch_stride=128,
        b_row_stride=1,
        b_col_stride=8,
    ),
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
            max_a_offset = (tc.batch_size - 1) * tc.a_batch_stride + (tc.m - 1) * tc.a_row_stride + (tc.k - 1) * tc.a_col_stride + 1
            max_b_offset = (tc.batch_size - 1) * tc.b_batch_stride + (tc.k - 1) * tc.b_row_stride + (tc.n - 1) * tc.b_col_stride + 1

            a_flat_raw = torch.randn(max_a_offset, dtype=torch.float32)
            b_flat_raw = torch.randn(max_b_offset, dtype=torch.float32)

            a_tensor = torch.as_strided(
                a_flat_raw,
                size=(tc.batch_size, tc.m, tc.k),
                stride=(tc.a_batch_stride, tc.a_row_stride, tc.a_col_stride),
            )
            b_tensor = torch.as_strided(
                b_flat_raw,
                size=(tc.batch_size, tc.k, tc.n),
                stride=(tc.b_batch_stride, tc.b_row_stride, tc.b_col_stride),
            )

            c_tensor = torch.matmul(a_tensor, b_tensor)

            f.write(a_flat_raw.numpy().astype(np.float32).tobytes())
            f.write(b_flat_raw.numpy().astype(np.float32).tobytes())
            f.write(c_tensor.detach().cpu().numpy().astype(np.float32).tobytes())

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))