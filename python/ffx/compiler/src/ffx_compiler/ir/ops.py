from abc import ABC
from dataclasses import dataclass, field

import numpy as np

# =============================================================================
# Base Taxonomy
# =============================================================================


@dataclass
class Op(ABC):
    """Abstract root for all IR operations."""

    pass


@dataclass
class IOOp(Op, ABC):
    """Base class for Graph I/O boundary primitives"""

    index: int


@dataclass
class ActivationOp(Op, ABC):
    """Base class for point=wise activation functions."""

    pass


@dataclass
class TransformOp(Op, ABC):
    """Base class for structural/zero=copy shape transformations."""

    pass


@dataclass
class NnOp(Op, ABC):
    """Base class for neural network operations (layers, pools, norms)."""

    pass


@dataclass
class TrainableOp(NnOp, ABC):
    """Base class for operations carrying trainable weights and biases."""

    weight: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


# =============================================================================
# IO & Structural Operations
# =============================================================================


@dataclass
class Input(IOOp):
    pass


@dataclass
class Output(IOOp):
    pass


@dataclass
class Flatten(TransformOp):
    pass


# =============================================================================
# Element-Wise Binary Operations
# =============================================================================


@dataclass
class Mul(Op):
    """Element-wise multiplication."""

    pass


@dataclass
class Div(Op):
    """Element-wise multiplication."""

    pass


@dataclass
class Add(Op):
    """Element-wise multiplication."""

    pass


@dataclass
class Sub(Op):
    """Element-wise multiplication."""

    pass


# =============================================================================
# Activation Operations
# =============================================================================


@dataclass
class Identity(ActivationOp):
    """ffx::nn::Identity"""

    pass


@dataclass
class ReLU(ActivationOp):
    """ffx::nn::ReLU"""

    pass


@dataclass
class ReLU6(ActivationOp):
    """ffx::nn::ReLU6"""

    pass


@dataclass
class GELU(ActivationOp):
    """ffx::nn::GELU"""

    pass


@dataclass
class LeakyReLU(ActivationOp):
    """ffx::nn::LeakyReLU"""

    negative_slope: float = 0.01


@dataclass
class Sigmoid(ActivationOp):
    """ffx::nn::Sigmoid"""

    pass


@dataclass
class SiLU(ActivationOp):
    """ffx::nn::SiLU"""

    pass


@dataclass
class Hardswish(ActivationOp):
    """ffx::nn::Hardswish"""

    pass


@dataclass
class Tanh(ActivationOp):
    """ffx::nn::Tanh"""

    pass


# =============================================================================
# Neural Network & Trainable Operations
# =============================================================================


@dataclass
class Linear(TrainableOp):
    """ffx::nn::Linear"""

    in_features: int = 0
    out_features: int = 0


@dataclass
class Conv2d(TrainableOp):
    """ffx::nn::Conv2d"""

    in_channels: int = 0
    out_channels: int = 0
    in_height: int = 0
    in_width: int = 0
    kernel_height: int = 0
    kernel_width: int = 0
    stride_height: int = 1
    stride_width: int = 1
    padding_height: int = 0
    padding_width: int = 0


@dataclass
class LayerNorm(TrainableOp):
    normalized_shape: int = 0
    epsilon: float = 1e-5


@dataclass
class RMSNorm(TrainableOp):
    normalized_shape: int = 0
    epsilon: float = 1e-5


@dataclass
class BatchNorm2d(TrainableOp):
    """ffx::nn::BatchNorm2d"""

    channels: int = 0
    height: int = 0
    width: int = 0
    epsilon: float = 1e-5
    mean: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    var: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class MaxPool2d(NnOp):
    """ffx::nn::MaxPool2d"""

    in_channels: int = 0
    in_height: int = 0
    in_width: int = 0
    kernel_height: int = 0
    kernel_width: int = 0
    stride_height: int = 1
    stride_width: int = 1
    padding_height: int = 0
    padding_width: int = 0


@dataclass
class AvgPool2d(NnOp):
    """ffx::nn::AvgPool2d"""

    in_channels: int = 0
    in_height: int = 0
    in_width: int = 0
    kernel_height: int = 0
    kernel_width: int = 0
    stride_height: int = 1
    stride_width: int = 1
    padding_height: int = 0
    padding_width: int = 0


@dataclass
class AdaptiveMaxPool2d(NnOp):
    """ffx::nn::AdaptiveMaxPool2d"""

    in_channels: int = 0
    in_height: int = 0
    in_width: int = 0
    out_height: int = 1
    out_width: int = 1


@dataclass
class AdaptiveAvgPool2d(NnOp):
    in_channels: int = 0
    in_height: int = 0
    in_width: int = 0
    out_height: int = 1
    out_width: int = 1


# =============================================================================
# Fused Operations
# =============================================================================


@dataclass
class FusedConv2dActivation(Conv2d):
    activation: ActivationOp = field(default_factory=Identity)


@dataclass
class FusedLinearActivation(Linear):
    activation: ActivationOp = field(default_factory=Identity)


@dataclass
class FusedBatchNorm2dActivation(BatchNorm2d):
    activation: ActivationOp = field(default_factory=Identity)


# =============================================================================
# Non-Linear Activations Operations
# =============================================================================


@dataclass
class Softmax(Op):
    """Softmax activation along an axis."""

    axis: int = -1


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base Taxonomy
    "Op",
    "IOOp",
    "TransformOp",
    "ActivationOp",
    "NnOp",
    "TrainableOp",
    # Primitive / Boundary
    "Input",
    "Output",
    "Flatten",
    # ElementWise
    "Mul",
    "Div",
    "Add",
    "Sub",
    # Non-Linear Activations
    "Softmax",
    # Activations
    "Identity",
    "ReLU",
    "ReLU6",
    "GELU",
    "LeakyReLU",
    "Sigmoid",
    "SiLU",
    "Hardswish",
    "Tanh",
    # NN
    "Linear",
    "Conv2d",
    "BatchNorm2d",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveMaxPool2d",
    "AdaptiveAvgPool2d",
    "LayerNorm",
    "RMSNorm",
    # Fused
    "FusedConv2dActivation",
    "FusedLinearActivation",
    "FusedBatchNorm2dActivation",
]
