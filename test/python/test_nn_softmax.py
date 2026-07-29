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
    shape: Tuple[int, ...]
    num_reductions: int
    reduction_size: int
    stride_within_dim: int = 1
    outer_stride: int = 0  # 0 auto-calculates as reduction_size * stride_within_dim

    def __post_init__(self):
        if self.outer_stride == 0:
            self.outer_stride = self.reduction_size * self.stride_within_dim


TEST_CASES: List[TestCase] = [
    # 1D Softmax: Single vector normalization (Softmax1D)
    TestCase("softmax_1d", shape=(100,), num_reductions=1, reduction_size=100),
    # 2D Softmax: Batch of feature vectors (Softmax2D)
    TestCase("softmax_2d", shape=(8, 64), num_reductions=8, reduction_size=64),
    # 3D Softmax: Sequence embeddings (Softmax3D)
    TestCase(
        "softmax_3d",
        shape=(2, 16, 128),
        num_reductions=2 * 16,
        reduction_size=128,
    ),
    # 4D Softmax: Attention matrix [Batch, Heads, Queries, Keys] (Softmax4D)
    TestCase(
        "softmax_4d",
        shape=(2, 8, 32, 32),
        num_reductions=2 * 8 * 32,
        reduction_size=32,
    ),
    # Strided / Non-contiguous Softmax (StrideWithinDim > 1)
    TestCase(
        "strided_custom",
        shape=(4, 16),
        num_reductions=4,
        reduction_size=16,
        stride_within_dim=2,
        outer_stride=64,
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
            total_elements = tc.num_reductions * tc.outer_stride

            # allocate flat arrays to support custom striding safely
            input_flat = np.zeros(total_elements, dtype=np.float32)
            output_flat = np.zeros(total_elements, dtype=np.float32)

            for r in range(tc.num_reductions):
                base = r * tc.outer_stride
                raw_vec = torch.randn(tc.reduction_size, dtype=torch.float32)
                soft_vec = F.softmax(raw_vec, dim=-1)

                indices = base + np.arange(tc.reduction_size) * tc.stride_within_dim
                input_flat[indices] = raw_vec.numpy()
                output_flat[indices] = soft_vec.numpy()

            f.write(input_flat.tobytes())
            f.write(output_flat.tobytes())

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))