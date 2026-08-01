from ffx_compiler.ir import (
    ActivationOp,
    BatchNorm2d,
    Conv2d,
    FusedBatchNorm2dActivation,
    FusedConv2dActivation,
    FusedLinearActivation,
    Linear,
)
from ffx_compiler.middle_end.ffx_runtime.kernel_fusion_registry import (
    KernelFusionRegistry,
)


@KernelFusionRegistry.register(Conv2d)
def fuse_conv2d(op: Conv2d, act: ActivationOp) -> FusedConv2dActivation:
    kwargs = {field: getattr(op, field) for field in op.__dataclass_fields__}
    return FusedConv2dActivation(**kwargs, activation=act)


@KernelFusionRegistry.register(Linear)
def fuse_linear(op: Linear, act: ActivationOp) -> FusedLinearActivation:
    kwargs = {field: getattr(op, field) for field in op.__dataclass_fields__}
    return FusedLinearActivation(**kwargs, activation=act)


@KernelFusionRegistry.register(BatchNorm2d)
def fuse_batch_norm2d(op: BatchNorm2d, act: ActivationOp) -> FusedBatchNorm2dActivation:
    kwargs = {field: getattr(op, field) for field in op.__dataclass_fields__}
    return FusedBatchNorm2dActivation(**kwargs, activation=act)
