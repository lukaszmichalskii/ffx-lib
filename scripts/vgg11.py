import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-shapes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="models", type=Path, metavar="PATH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model = models.vgg11_bn()
    model.eval()

    in_tensor = torch.ones(1, 3, 224, 224)
    with torch.no_grad():
        out_tensor = model(in_tensor)

    if args.verbose:
        print(out_tensor)
    else:
        print(out_tensor.shape)

    args.output.mkdir(parents=True, exist_ok=True)
    onnx_filename = f"{Path(__file__).stem}.onnx"
    params = {
        "model": model,
        "f": str(args.output.joinpath(onnx_filename)),
        "args": in_tensor,
        "export_params": True,
        "opset_version": 18,
        "input_names": ["input"],
        "output_names": ["output"],
    }

    if args.dynamic_shapes:
        dynamic_shapes = torch.export.Dim("batch_size", min=1, max=1024)
        params["dynamic_shapes"] = ({0: dynamic_shapes},)

    torch.onnx.export(**params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
