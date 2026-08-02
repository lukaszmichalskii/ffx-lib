from typing import Any, Dict, List, Tuple, TypeAlias

import numpy as np

from ffx_compiler.backend.backend_error import BackendError
from ffx_compiler.backend.ffx_runtime.symbol_lookup_table import SymbolLookupTable
from ffx_compiler.ir import BatchNorm2d, Graph, IOOp, TrainableOp

BufferMetadata: TypeAlias = Dict[str, Any]
ALIGNMENT_BYTES = 64  # cache-line alignment boundary


def align_buffer(buffer: bytearray) -> None:
    remainder = len(buffer) % ALIGNMENT_BYTES
    if remainder != 0:
        buffer.extend(b"\x00" * (ALIGNMENT_BYTES - remainder))


def pack_float32(data: np.ndarray) -> bytes:
    return data.astype(np.float32, copy=False).tobytes()


def pack_float64(data: np.ndarray) -> bytes:
    return data.astype(np.float64, copy=False).tobytes()


def pack_data(
    param_name: str, buffer: bytearray, data: np.ndarray, metadata: List[BufferMetadata]
) -> None:
    if data is None or data.size == 0:
        return

    align_buffer(buffer)
    offset = len(buffer)

    if not isinstance(data, np.ndarray):
        raise BackendError(
            f"Unsupported weight format '{type(data)}'. Expected np.ndarray."
        )

    dtype = np.float64 if data.dtype == np.float64 else np.float32
    data_bytes = data.astype(dtype, copy=False).tobytes()
    buffer.extend(data_bytes)

    metadata.append(
        {
            "param_name": param_name,
            "offset": offset,
            "size": data.size,
            "bytes_size": len(data_bytes),
            "dtype": "double" if data.dtype == np.float64 else "float",
        }
    )


def pack_model_data(
    graph: Graph, symbol_lookup_table: SymbolLookupTable
) -> Tuple[bytearray, List[BufferMetadata]]:
    bytes_buffer = bytearray()
    bytes_buffer_metadata = list()

    for node in graph.nodes:
        if isinstance(node.op, IOOp):
            continue
        buffer_metadata_list = list()
        if isinstance(node.op, TrainableOp):
            pack_data("weight", bytes_buffer, node.op.weight, buffer_metadata_list)
            pack_data("bias", bytes_buffer, node.op.bias, buffer_metadata_list)
            if isinstance(node.op, BatchNorm2d):
                pack_data("mean", bytes_buffer, node.op.mean, buffer_metadata_list)
                pack_data("var", bytes_buffer, node.op.var, buffer_metadata_list)

        if buffer_metadata_list:
            metadata = {
                "node": symbol_lookup_table.get_cpp_var(node.name),
                "data": buffer_metadata_list,
            }
            bytes_buffer_metadata.append(metadata)
    align_buffer(bytes_buffer)
    return bytes_buffer, bytes_buffer_metadata
