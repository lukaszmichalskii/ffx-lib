import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# explicitly import codegen_ops to trigger @register decorators
import ffx_compiler.backend.ffx_runtime.codegen_ops  # noqa: F401
from ffx_compiler.backend.codegen import Codegen
from ffx_compiler.backend.ffx_runtime.codegen_registry import FfxOpsRegistry
from ffx_compiler.backend.ffx_runtime.model_data_utilities import pack_model_data
from ffx_compiler.backend.ffx_runtime.symbol_lookup_table import SymbolLookupTable
from ffx_compiler.ir import Graph, Input, IOOp, Node, Output, TransformOp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IONodeContext:
    index: int
    var: str
    size_expr: str
    dtype: str = "float"
    internal_parent_var: str = ""


@dataclass(frozen=True)
class ComputeNodeContext:
    name: str
    var: str
    input_bufs: List[str]
    output_n: str
    kernel_type: str
    call_type: str
    dtype: str = "float"
    weight_size: str = ""
    bias_size: str = ""


@dataclass(frozen=True)
class RenderingContext:
    model_name: str
    batch_size: int
    dynamic_batch: bool
    inputs: List[IONodeContext]
    outputs: List[IONodeContext]
    nodes: List[ComputeNodeContext]
    initializers: List[str]
    infer_params: List[str]
    infer_args: List[str]
    output_copiers: List[str]
    bytes_buffer_metadata: List[Dict[str, Any]] = field(default_factory=list)
    has_bytes_buffer_metadata: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "dynamic_batch": self.dynamic_batch,
            "inputs": [inp.__dict__ for inp in self.inputs],
            "outputs": [out.__dict__ for out in self.outputs],
            "nodes": [node.__dict__ for node in self.nodes],
            "initializers": self.initializers,
            "infer_params": self.infer_params,
            "infer_args": self.infer_args,
            "output_copiers": self.output_copiers,
            "bytes_buffer_metadata": self.bytes_buffer_metadata,
            "has_bytes_buffer_metadata": self.has_bytes_buffer_metadata,
        }


def format_infer_params(
    inputs: List[IONodeContext], outputs: List[IONodeContext]
) -> List[str]:
    params: List[str] = ["Queue &queue"]
    for inp in inputs:
        params.append(
            f"const ffx::device_buffer<Device, {inp.dtype}[]>& device_{inp.var}"
        )
    for out in outputs:
        params.append(f"ffx::device_buffer<Device, {out.dtype}[]>& device_{out.var}")
    return params


def format_infer_args(
    inputs: List[IONodeContext], outputs: List[IONodeContext]
) -> List[str]:
    args: List[str] = ["queue"]
    for inp in inputs:
        args.append(f"device_{inp.var}")
    for out in outputs:
        args.append(f"device_{out.var}")
    return args


def format_output_copiers(outputs: List[IONodeContext]) -> List[str]:
    return [
        f"alpaka::memcpy(queue, device_{out.var}, device_{out.internal_parent_var}_output_);"
        for out in outputs
    ]


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB"]:
        if abs(size) < 1024:
            return f"{size:.2f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.2f} GB"


class FfxRuntimeCodegen(Codegen):
    _TEMPLATE_MODEL_HEADER = "model.h.j2"
    _TEMPLATE_MAIN_SOURCE = "main.cc.j2"

    def __init__(self, template_dir: Path | None = None) -> None:
        super().__init__()
        self.template_dir = template_dir or Path(__file__).parent.joinpath("templates")

        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    def intermediate_representation_to_source_code(
        self, graph: Graph
    ) -> Tuple[Dict[str, str], bytes]:
        logger.info("Starting C++ source code generation for graph '%s'", graph.name)

        symbols = SymbolLookupTable.from_graph(graph=graph)
        rendering_ctx = self.__build_rendering_context(graph, symbols)
        bytes_buffer, bytes_buffer_metadata = pack_model_data(graph, symbols)

        context_dict = rendering_ctx.to_dict()
        context_dict["bytes_buffer_metadata"] = bytes_buffer_metadata
        context_dict["has_bytes_buffer_metadata"] = bool(bytes_buffer_metadata)

        model_h = self.__render_template(self._TEMPLATE_MODEL_HEADER, context_dict)
        main_cc = self.__render_template(self._TEMPLATE_MAIN_SOURCE, context_dict)

        sources = {
            "model.h": model_h,
            "main.cc": main_cc,
        }

        logger.info(
            "Successfully generated C++ source code for '%s' (%s bytes weights)",
            graph.name,
            format_bytes(len(bytes_buffer)),
        )

        return sources, bytes_buffer

    def __build_rendering_context(
        self, graph: Graph, symbols: SymbolLookupTable
    ) -> RenderingContext:
        input_nodes = sorted(
            [n for n in graph.nodes if isinstance(n.op, Input)],
            key=lambda n: n.op.index,
        )
        output_nodes = sorted(
            [n for n in graph.nodes if isinstance(n.op, Output)],
            key=lambda n: n.op.index,
        )
        compute_nodes = [n for n in graph.nodes if not isinstance(n.op, IOOp)]

        # IO boundaries
        inputs_ctx = [
            IONodeContext(
                index=node.op.index,
                var=f"input_{node.op.index}" if len(input_nodes) > 1 else "input",
                size_expr=symbols.get_size_expression(node),
                dtype=node.dtype,
            )
            for node in input_nodes
        ]

        outputs_ctx = [
            IONodeContext(
                index=node.op.index,
                var=f"output_{node.op.index}" if len(input_nodes) > 1 else "output",
                size_expr=symbols.get_size_expression(node),
                dtype=node.dtype,
                internal_parent_var=symbols.get_cpp_var(
                    node.inputs[0] if node.inputs else node.name
                ),
            )
            for node in output_nodes
        ]

        # map compute nodes & buffer initializers
        nodes_ctx: List[ComputeNodeContext] = []
        initializers: List[str] = []

        for node in compute_nodes:
            # zero-copy structural ops
            if isinstance(node.op, TransformOp):
                continue

            input_bufs = symbols.resolve_input_buffers(node, input_nodes)
            node_ctx = self.__create_node_context(node, input_bufs, symbols)
            nodes_ctx.append(node_ctx)

            initializers.append(
                f"device_{node_ctx.var}_output_(ffx::make_device_buffer<{node_ctx.dtype}[]>(queue, {node_ctx.output_n}))"
            )
            if node_ctx.call_type in ("parameterised", "batch_norm"):
                initializers.append(
                    f"device_{node_ctx.var}_weight_(ffx::make_device_buffer<{node_ctx.dtype}[]>(queue, {node_ctx.weight_size}))"
                )
                initializers.append(
                    f"device_{node_ctx.var}_bias_(ffx::make_device_buffer<{node_ctx.dtype}[]>(queue, {node_ctx.bias_size}))"
                )

        infer_params = format_infer_params(inputs_ctx, outputs_ctx)
        infer_args = format_infer_args(inputs_ctx, outputs_ctx)
        output_copiers = format_output_copiers(outputs_ctx)

        return RenderingContext(
            model_name=graph.name,
            batch_size=graph.batch_size,
            dynamic_batch=(graph.batch_size == -1),
            inputs=inputs_ctx,
            outputs=outputs_ctx,
            nodes=nodes_ctx,
            initializers=initializers,
            infer_params=infer_params,
            infer_args=infer_args,
            output_copiers=output_copiers,
        )

    def __create_node_context(
        self, node: Node, input_buffers: List[str], symbols: SymbolLookupTable
    ) -> ComputeNodeContext:
        op_type = type(node.op)
        emitter_fn = FfxOpsRegistry.get(op_type)

        if not emitter_fn:
            logger.error("No code generator registered for operator %s", op_type)
            raise NotImplementedError(
                f"Unsupported operator '{op_type.__name__}' at node '{node.name}'."
            )

        size_expr = symbols.get_size_expression(node)
        batch_expr = symbols.get_batch_expr()

        # execute registered operator emitter
        op_ctx = emitter_fn(node.op, batch_expr, size_expr)

        return ComputeNodeContext(
            name=node.name,
            var=symbols.get_cpp_var(node.name),
            input_bufs=input_buffers,
            output_n=size_expr,
            kernel_type=op_ctx["kernel_type"],
            call_type=op_ctx.get("call", "element_wise"),
            dtype=node.dtype,
            weight_size=op_ctx.get("weight_size", ""),
            bias_size=op_ctx.get("bias_size", ""),
        )

    def __render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)
