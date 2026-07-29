from typing import Dict, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

ONNX_TO_NUMPY_DTYPE: Dict[int, np.dtype] = {
    onnx.TensorProto.FLOAT: np.dtype(np.float32),
    onnx.TensorProto.DOUBLE: np.dtype(np.float64),
    onnx.TensorProto.INT32: np.dtype(np.int32),
    onnx.TensorProto.INT64: np.dtype(np.int64),
    onnx.TensorProto.BOOL: np.dtype(bool),
}


class OnnxGraphContext:
    def __init__(self, model: onnx.ModelProto):
        self.graph = model.graph
        self.initializers: Dict[str, onnx.TensorProto] = {
            init.name: init for init in self.graph.initializer
        }
        # onnx tensor name -> IR value name
        self.value_map: Dict[str, str] = dict()
        # onnx tensor name -> (shape tuple, dtype string)
        self.tensor_meta: Dict[str, Tuple[Tuple[int, ...], str]] = (
            self.extract_metadata()
        )

    def get_initializer_array(self, tensor_name: Optional[str]) -> Optional[np.ndarray]:
        if not tensor_name or tensor_name not in self.initializers:
            return None
        proto = self.initializers[tensor_name]
        array = numpy_helper.to_array(proto)
        dtype = ONNX_TO_NUMPY_DTYPE.get(proto.data_type, array.dtype)
        return array.astype(dtype=dtype, copy=False)

    def resolve_input_name(self, onnx_input_name: str) -> str:
        return self.value_map.get(onnx_input_name, onnx_input_name)

    def extract_metadata(self) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        metadata = dict()
        for value_info in (
            list(self.graph.input)
            + list(self.graph.output)
            + list(self.graph.value_info)
        ):
            if value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                dtype = (
                    onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower()
                    if tensor_type.HasField("elem_type")
                    else "unknown"
                )
                shape = (
                    tuple(
                        d.dim_value if d.HasField("dim_value") else -1
                        for d in tensor_type.shape.dim
                    )
                    if tensor_type.HasField("shape")
                    else ()
                )
                metadata[value_info.name] = (shape, dtype)

        for name, init in self.initializers.items():
            metadata[name] = (
                tuple(init.dims),
                onnx.TensorProto.DataType.Name(init.data_type).lower(),
            )

        return metadata
