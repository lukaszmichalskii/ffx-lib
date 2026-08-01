from typing import Any, Callable, Type

from ffx_compiler.common import Registry
from ffx_compiler.ir.ops import (
    ActivationOp,
)

KernelFusionFn = Callable[[Any, ActivationOp], Any]
KernelFusionRegistry = Registry[Type, KernelFusionFn]("KernelFusionRegistry")
