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
    num_tokens: int
    embedding_dim: int
    has_gamma: bool
    has_beta: bool
    eps_numerator: int = 1
    eps_denominator: int = 1000000


TEST_CASES: List[TestCase] = [
    # Standard transformer shape with affine parameters (gamma & beta)
    TestCase("standard_affine", num_tokens=4, embedding_dim=64, has_gamma=True, has_beta=True),
    # Edge case: single token sequence
    TestCase("single_token", num_tokens=1, embedding_dim=128, has_gamma=True, has_beta=True),
    # Pure normalization — gamma=nullptr and beta=nullptr
    TestCase("no_affine", num_tokens=8, embedding_dim=32, has_gamma=False, has_beta=False),
    # Scale without shift — gamma provided, beta=nullptr
    TestCase("gamma_only", num_tokens=2, embedding_dim=256, has_gamma=True, has_beta=False),
    # Custom epsilon value (1e-5 instead of default 1e-6)
    TestCase(
        "custom_eps",
        num_tokens=16,
        embedding_dim=16,
        has_gamma=True,
        has_beta=True,
        eps_numerator=1,
        eps_denominator=100000,
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
            eps = float(tc.eps_numerator) / float(tc.eps_denominator)

            input_tensor = torch.randn(tc.num_tokens, tc.embedding_dim)
            gamma_tensor = torch.randn(tc.embedding_dim) if tc.has_gamma else None
            beta_tensor = torch.randn(tc.embedding_dim) if tc.has_beta else None

            with torch.no_grad():
                output_tensor = F.layer_norm(
                    input_tensor,
                    normalized_shape=(tc.embedding_dim,),
                    weight=gamma_tensor,
                    bias=beta_tensor,
                    eps=eps,
                )

            f.write(input_tensor.detach().cpu().numpy().astype(np.float32).tobytes())

            if gamma_tensor is not None:
                f.write(gamma_tensor.detach().cpu().numpy().astype(np.float32).tobytes())

            if beta_tensor is not None:
                f.write(beta_tensor.detach().cpu().numpy().astype(np.float32).tobytes())

            f.write(output_tensor.detach().cpu().numpy().astype(np.float32).tobytes())

    print(bin_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))