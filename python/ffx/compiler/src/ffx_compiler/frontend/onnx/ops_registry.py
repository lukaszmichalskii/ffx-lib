from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import onnx

from ffx_compiler.common import Registry
from ffx_compiler.ir import Op

if TYPE_CHECKING:
    from ffx_compiler.frontend.onnx.graph_context import OnnxGraphContext

OnnxOpParser = Callable[[onnx.NodeProto, "OnnxGraphContext"], Optional[Op]]
OnnxOpsRegistry = Registry[str, OnnxOpParser](name="OnnxOpsRegistry")
