"""ONNX to IR Operator Converters."""

import numpy as np
import onnx

from ffx_compiler.frontend.onnx.graph_context import OnnxGraphContext
from ffx_compiler.frontend.onnx.ops_registry import OnnxOpsRegistry
from ffx_compiler.frontend.onnx.utils import extract_attributes
from ffx_compiler.ir import (
    GELU,
    Add,
    Conv2d,
    Div,
    Flatten,
    Hardswish,
    Identity,
    LeakyReLU,
    Linear,
    Mul,
    Op,
    ReLU,
    ReLU6,
    Sigmoid,
    SiLU,
    Softmax,
    Sub,
    Tanh,
)
from ffx_compiler.ir.ops import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    BatchNorm2d,
    LayerNorm,
    MaxPool2d,
)


def get_tensor_shape(node_input: str, context: OnnxGraphContext):
    """helper to safely fetch tensor shape from metadata context."""
    shape, _ = context.tensor_meta.get(node_input, ((-1, -1, -1, -1), "float32"))
    return shape


# =============================================================================
# Non-Linear Activation Operations
# =============================================================================


@OnnxOpsRegistry.register("Softmax")
def parse_softmax(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    params = extract_attributes(node)
    axis = int(params.get("axis", -1))
    return Softmax(axis=axis)


# =============================================================================
# Element-Wise
# =============================================================================


@OnnxOpsRegistry.register("Mul")
def parse_mul(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Mul()


@OnnxOpsRegistry.register("Div")
def parse_div(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Div()


@OnnxOpsRegistry.register("Add")
def parse_add(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Add()


@OnnxOpsRegistry.register("Sub")
def parse_sub(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Sub()


@OnnxOpsRegistry.register("Relu")
def parse_relu(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return ReLU()


@OnnxOpsRegistry.register("Clip")
def parse_clip(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return ReLU6()


@OnnxOpsRegistry.register("LeakyRelu")
def parse_leaky_relu(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    params = extract_attributes(node)
    alpha = float(params.get("alpha", 0.01))
    return LeakyReLU(negative_slope=alpha)


@OnnxOpsRegistry.register("Sigmoid")
def parse_sigmoid(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Sigmoid()


@OnnxOpsRegistry.register("Tanh")
def parse_tanh(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Tanh()


@OnnxOpsRegistry.register("Silu")
def parse_silu(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return SiLU()


@OnnxOpsRegistry.register("Gelu")
def parse_gelu(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return GELU()


@OnnxOpsRegistry.register("HardSwish")
def parse_hardswish(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Hardswish()


@OnnxOpsRegistry.register("Identity")
def parse_identity(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Identity()


# =============================================================================
# Layers
# =============================================================================


@OnnxOpsRegistry.register("Gemm")
def parse_gemm(node: onnx.NodeProto, context: OnnxGraphContext) -> Linear:
    params = extract_attributes(node)
    weight = context.get_initializer_array(
        node.input[1] if len(node.input) > 1 else None
    )
    bias = context.get_initializer_array(node.input[2] if len(node.input) > 2 else None)

    if weight is None:
        raise ValueError(f"Gemm node '{node.name}' missing weight initializer.")

    trans_b = params.get("transB", 0)
    if trans_b:
        out_features, in_features = weight.shape[0], weight.shape[1]
    else:
        in_features, out_features = weight.shape[0], weight.shape[1]

    return Linear(
        in_features=in_features,
        out_features=out_features,
        weight=weight,
        bias=bias if bias is not None else np.empty(0, dtype=np.float32),
    )


@OnnxOpsRegistry.register("Conv")
def parse_conv2d(node: onnx.NodeProto, context: OnnxGraphContext) -> Conv2d:
    params = extract_attributes(node)
    weight = context.get_initializer_array(
        node.input[1] if len(node.input) > 1 else None
    )
    bias = context.get_initializer_array(node.input[2] if len(node.input) > 2 else None)

    if weight is None:
        raise ValueError(f"Conv node '{node.name}' missing weight initializer.")

    out_channels, in_channels, kh, kw = weight.shape
    strides = params.get("strides", [1, 1])
    pads = params.get("pads", [0, 0, 0, 0])

    in_shape = get_tensor_shape(node.input[0], context)

    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_shape[2] if len(in_shape) > 2 else -1,
        in_width=in_shape[3] if len(in_shape) > 3 else -1,
        kernel_height=kh,
        kernel_width=kw,
        stride_height=strides[0],
        stride_width=strides[1] if len(strides) > 1 else strides[0],
        padding_height=pads[0],
        padding_width=pads[1],
        weight=weight,
        bias=bias if bias is not None else np.empty(0, dtype=np.float32),
    )


# =============================================================================
# Normalization Layers
# =============================================================================


@OnnxOpsRegistry.register("BatchNormalization")
def parse_batch_norm(node: onnx.NodeProto, context: OnnxGraphContext) -> BatchNorm2d:
    params = extract_attributes(node)
    scale = context.get_initializer_array(node.input[1])  # Weight / Gamma
    bias = context.get_initializer_array(node.input[2])  # Beta
    mean = context.get_initializer_array(node.input[3])
    var = context.get_initializer_array(node.input[4])

    epsilon = params.get("epsilon", 1e-5)
    in_shape = get_tensor_shape(node.input[0], context)
    spatial_rank = len(in_shape) - 2

    if spatial_rank == 2:
        height, width = in_shape[2], in_shape[3]
    elif spatial_rank == 1:
        height, width = 1, in_shape[2]
    else:
        height, width = 1, 1

    return BatchNorm2d(
        channels=scale.shape[0] if scale is not None else in_shape[1],
        height=height,
        width=width,
        epsilon=epsilon,
        weight=scale if scale is not None else np.empty(0, dtype=np.float32),
        bias=bias if bias is not None else np.empty(0, dtype=np.float32),
        mean=mean if mean is not None else np.empty(0, dtype=np.float32),
        var=var if var is not None else np.empty(0, dtype=np.float32),
    )


@OnnxOpsRegistry.register("LayerNormalization")
def parse_layer_norm(node: onnx.NodeProto, context: OnnxGraphContext) -> LayerNorm:
    params = extract_attributes(node)
    scale = context.get_initializer_array(node.input[1])
    bias = context.get_initializer_array(node.input[2] if len(node.input) > 2 else None)
    epsilon = params.get("epsilon", 1e-5)

    normalized_shape = scale.shape[0] if scale is not None else -1

    return LayerNorm(
        normalized_shape=normalized_shape,
        epsilon=epsilon,
        weight=scale if scale is not None else np.empty(0, dtype=np.float32),
        bias=bias if bias is not None else np.empty(0, dtype=np.float32),
    )


# =============================================================================
# Pooling Layers
# =============================================================================


@OnnxOpsRegistry.register("MaxPool")
def parse_max_pool(node: onnx.NodeProto, context: OnnxGraphContext) -> MaxPool2d:
    params = extract_attributes(node)
    kernel_shape = params.get("kernel_shape", [2, 2])
    strides = params.get("strides", [1, 1])
    pads = params.get("pads", [0, 0, 0, 0])

    if len(kernel_shape) != 2:
        raise NotImplementedError(
            f"MaxPool with spatial rank {len(kernel_shape)} is not supported. Only 2D pooling supported."
        )

    in_shape = get_tensor_shape(node.input[0], context)

    return MaxPool2d(
        in_channels=in_shape[1] if len(in_shape) > 1 else -1,
        in_height=in_shape[2] if len(in_shape) > 2 else -1,
        in_width=in_shape[3] if len(in_shape) > 3 else -1,
        kernel_height=kernel_shape[0],
        kernel_width=kernel_shape[1],
        stride_height=strides[0],
        stride_width=strides[1] if len(strides) > 1 else strides[0],
        padding_height=pads[0],
        padding_width=pads[1] if len(pads) > 1 else pads[0],
    )


@OnnxOpsRegistry.register("AveragePool")
def parse_average_pool(node: onnx.NodeProto, context: OnnxGraphContext) -> AvgPool2d:
    """Parses standard ONNX AveragePool emitted by PyTorch ONNX export for AdaptiveAvgPool2d."""
    params = extract_attributes(node)
    kernel_shape = params.get("kernel_shape", [1, 1])
    strides = params.get("strides", [1, 1])
    pads = params.get("pads", [0, 0, 0, 0])

    in_shape = get_tensor_shape(node.input[0], context)

    return AvgPool2d(
        in_channels=in_shape[1] if len(in_shape) > 1 else -1,
        in_height=in_shape[2] if len(in_shape) > 2 else -1,
        in_width=in_shape[3] if len(in_shape) > 3 else -1,
        kernel_height=kernel_shape[0],
        kernel_width=kernel_shape[1] if len(kernel_shape) > 1 else kernel_shape[0],
        stride_height=strides[0],
        stride_width=strides[1] if len(strides) > 1 else strides[0],
        padding_height=pads[0],
        padding_width=pads[1] if len(pads) > 1 else pads[0],
    )


@OnnxOpsRegistry.register("GlobalAveragePool")
@OnnxOpsRegistry.register("AdaptiveAvgPool")
def parse_global_avg_pool(
    node: onnx.NodeProto, context: OnnxGraphContext
) -> AdaptiveAvgPool2d:
    in_shape = get_tensor_shape(node.input[0], context)
    return AdaptiveAvgPool2d(
        in_channels=in_shape[1] if len(in_shape) > 1 else -1,
        in_height=in_shape[2] if len(in_shape) > 2 else -1,
        in_width=in_shape[3] if len(in_shape) > 3 else -1,
        out_height=1,
        out_width=1,
    )


@OnnxOpsRegistry.register("ReduceMean")
def parse_reduce_mean(
    node: onnx.NodeProto, context: OnnxGraphContext
) -> AdaptiveAvgPool2d:
    """Lowers spatial ReduceMean (e.g. ResNet global pooling) into AdaptiveAvgPool2d."""
    params = extract_attributes(node)
    axes = params.get("axes", None)

    # In newer ONNX opsets, 'axes' can be passed as the second input tensor
    if axes is None and len(node.input) > 1:
        axes_initializer = context.get_initializer_array(node.input[1])
        if axes_initializer is not None:
            axes = axes_initializer

    in_shape = get_tensor_shape(node.input[0], context)

    # Normalize axes to standard Python ints
    if axes is not None:
        # Convert numpy array / numpy ints to standard python ints and sort them
        normalized_axes = sorted([int(a) for a in np.array(axes).flatten()])
    else:
        normalized_axes = []

    # Check for spatial reduction:
    # Spatial H, W axes for 4D NCHW tensors are [2, 3] (or [-2, -1], which normalizes/sorts to [2, 3] or [-2, -1])
    is_spatial_reduction = (
        normalized_axes in ([2, 3], [-2, -1])
        or not normalized_axes  # Default in ONNX when axes is empty reduces all dimensions
    )

    if is_spatial_reduction:
        return AdaptiveAvgPool2d(
            in_channels=in_shape[1] if len(in_shape) > 1 else -1,
            in_height=in_shape[2] if len(in_shape) > 2 else -1,
            in_width=in_shape[3] if len(in_shape) > 3 else -1,
            out_height=1,
            out_width=1,
        )

    raise NotImplementedError(
        f"ReduceMean node '{node.name}' with axes={axes} is not supported. "
        f"Only global spatial pooling reduction ([2, 3] or [-2, -1]) is supported."
    )


# =============================================================================
# Shape/Metadata Operations
# =============================================================================

_PASSTHROUGH_OPS = [
    "Shape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Slice",
    "Cast",
    "ConstantOfShape",
    "Range",
]


@OnnxOpsRegistry.register("Flatten")
def parse_flatten(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Flatten()


@OnnxOpsRegistry.register("Reshape")
def parse_reshape(node: onnx.NodeProto, context: OnnxGraphContext) -> Op:
    return Flatten()


@OnnxOpsRegistry.register(_PASSTHROUGH_OPS)
def parse_passthrough_shape_ops(
    node: onnx.NodeProto, context: OnnxGraphContext
) -> None:
    data_input = next((inp for inp in node.input if inp in context.value_map), None)
    if data_input:
        for out in node.output:
            context.value_map[out] = context.value_map[data_input]
    return None


# =============================================================================
# Custom Operations
# =============================================================================


@OnnxOpsRegistry.register("CustomOperator")
def parse_custom_operator(node: onnx.NodeProto, context: OnnxGraphContext) -> Op | None:
    return None
