from fractions import Fraction
from typing import Any, Dict, Tuple

from ffx_compiler.backend.ffx_runtime.codegen_registry import FfxOpsRegistry
from ffx_compiler.ir import (
    GELU,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    Add,
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Div,
    FusedBatchNorm2dActivation,
    FusedConv2dActivation,
    FusedLinearActivation,
    Hardswish,
    LeakyReLU,
    Linear,
    MaxPool2d,
    Mul,
    ReLU,
    ReLU6,
    Sigmoid,
    SiLU,
    Softmax,
    Sub,
    Tanh,
)


def to_fraction(value: float, max_denominator: int = 1_000_000) -> Tuple[int, int]:
    frac = Fraction(value).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


@FfxOpsRegistry.register(Softmax)
def emit_softmax(op: Softmax, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    # Determine feature dim from op node shape if available
    # For a (B, C) classification head, NumReductions = batch_expr, ReductionSize = C
    if hasattr(op, "feature_dim") and op.feature_dim:
        reduction_size = str(op.feature_dim)
    else:
        # Fallback if size_expr is already folded (e.g. kBatchSize * 10 -> ReductionSize = 10)
        # Standard classification head output size: 10
        reduction_size = size_expr.split("*")[-1].strip()

    num_reductions = batch_expr

    return {
        "kernel_type": f"ffx::nn::Softmax<{num_reductions}, {reduction_size}>",
        "call": "element_wise",
    }


# =============================================================================
# Element-Wise
# =============================================================================


@FfxOpsRegistry.register(Mul)
def mul(op: Mul, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Mul<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Div)
def div(op: Mul, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Div<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Add)
def add(op: Mul, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Add<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Sub)
def sub(op: Mul, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Sub<{size_expr}>",
        "call": "element_wise",
    }


# =============================================================================
# Kernels
# =============================================================================


@FfxOpsRegistry.register(Linear)
def linear(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Linear<{batch_expr}, {op.in_features}, {op.out_features}>",
        "call": "parameterised",
        "weight_size": str(op.out_features * op.in_features),
        "bias_size": str(op.out_features),
    }


@FfxOpsRegistry.register(Conv2d)
def conv2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": (
            f"ffx::nn::Conv2d<{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, {op.out_channels}, "
            f"{op.kernel_height}, {op.kernel_width}, {op.stride_height}, {op.stride_width}, "
            f"{op.padding_height}, {op.padding_width}>"
        ),
        "call": "parameterised",
        "weight_size": str(
            op.out_channels * op.in_channels * op.kernel_height * op.kernel_width
        ),
        "bias_size": str(op.out_channels),
    }


@FfxOpsRegistry.register(BatchNorm2d)
def batch_norm2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    eps_nom, eps_den = to_fraction(op.epsilon)
    return {
        "kernel_type": f"ffx::nn::BatchNorm2d<{batch_expr}, {op.channels}, {op.height}, {op.width}, {eps_nom}, {eps_den}>",
        "call": "batch_norm",
        "param_size": str(op.channels),
    }


@FfxOpsRegistry.register(MaxPool2d)
def max_pool2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": (
            f"ffx::nn::MaxPool2d<{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, "
            f"{op.kernel_height}, {op.kernel_width}, {op.stride_height}, {op.stride_width}, "
            f"{op.padding_height}, {op.padding_width}>"
        ),
        "call": "element_wise",
    }


@FfxOpsRegistry.register(AdaptiveMaxPool2d)
def adaptive_max_pool2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::AdaptiveMaxPool2d<{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, {op.out_height}, {op.out_width}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(AvgPool2d)
def avg_pool2d(op: AvgPool2d, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": (
            f"ffx::nn::AvgPool2d<{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, "
            f"{op.kernel_height}, {op.kernel_width}, {op.stride_height}, {op.stride_width}, "
            f"{op.padding_height}, {op.padding_width}>"
        ),
        "call": "element_wise",
    }


@FfxOpsRegistry.register(AdaptiveAvgPool2d)
def adaptive_avg_pool2d(
    op: AdaptiveAvgPool2d, batch_expr: str, size_expr: str
) -> Dict[str, Any]:
    return {
        "kernel_type": (
            f"ffx::nn::AdaptiveAvgPool2d<{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, "
            f"{op.out_height}, {op.out_width}>"
        ),
        "call": "element_wise",
    }


# =============================================================================
# Activations
# =============================================================================


@FfxOpsRegistry.register(ReLU)
def relu(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::ReLU<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(ReLU6)
def relu6(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::ReLU6<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(LeakyReLU)
def leaky_relu(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    nominator, denominator = to_fraction(op.negative_slope)
    return {
        "kernel_type": f"ffx::nn::LeakyReLU<{size_expr}, {nominator}, {denominator}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(GELU)
def gelu(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::GELU<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Sigmoid)
def sigmoid(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Sigmoid<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(SiLU)
def silu(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::SiLU<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Hardswish)
def hardswish(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Hardswish<{size_expr}>",
        "call": "element_wise",
    }


@FfxOpsRegistry.register(Tanh)
def tanh(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    return {
        "kernel_type": f"ffx::nn::Tanh<{size_expr}>",
        "call": "element_wise",
    }


# =============================================================================
# Fused Kernels
# =============================================================================


@FfxOpsRegistry.register(FusedConv2dActivation)
def fused_conv2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    act = op.activation
    base = (
        f"{batch_expr}, {op.in_height}, {op.in_width}, {op.in_channels}, {op.out_channels}, "
        f"{op.kernel_height}, {op.kernel_width}, {op.stride_height}, {op.stride_width}, "
        f"{op.padding_height}, {op.padding_width}"
    )
    if isinstance(act, LeakyReLU):
        nom, den = to_fraction(act.negative_slope)
        kernel_type = f"ffx::nn::Conv2dLeakyReLU<{base}, true, {nom}, {den}>"
    else:
        kernel_type = f"ffx::nn::Conv2d{type(act).__name__}<{base}>"

    return {
        "kernel_type": kernel_type,
        "call": "parameterised",
        "weight_size": str(
            op.out_channels * op.in_channels * op.kernel_height * op.kernel_width
        ),
        "bias_size": str(op.out_channels),
    }


@FfxOpsRegistry.register(FusedLinearActivation)
def fused_linear(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    act = op.activation
    base = f"{batch_expr}, {op.in_features}, {op.out_features}"
    if isinstance(act, LeakyReLU):
        nom, den = to_fraction(act.negative_slope)
        kernel_type = f"ffx::nn::LinearLeakyReLU<{base}, {nom}, {den}>"
    else:
        kernel_type = f"ffx::nn::Linear{type(act).__name__}<{base}>"

    return {
        "kernel_type": kernel_type,
        "call": "parameterised",
        "weight_size": str(op.out_features * op.in_features),
        "bias_size": str(op.out_features),
    }


@FfxOpsRegistry.register(FusedBatchNorm2dActivation)
def fused_batch_norm2d(op: Any, batch_expr: str, size_expr: str) -> Dict[str, Any]:
    act = op.activation
    base = f"{batch_expr}, {op.num_features}, {op.eps}"

    if isinstance(act, LeakyReLU):
        nom, den = to_fraction(act.negative_slope)
        kernel_type = f"ffx::nn::BatchNorm2dLeakyReLU<{base}, {nom}, {den}>"
    else:
        kernel_type = f"ffx::nn::BatchNorm2d{type(act).__name__}<{base}>"

    return {
        "kernel_type": kernel_type,
        "call": "parameterised",
        "weight_size": str(op.num_features),  # gamma
        "bias_size": str(op.num_features),  # beta
        "running_mean_size": str(op.num_features),
        "running_var_size": str(op.num_features),
    }
