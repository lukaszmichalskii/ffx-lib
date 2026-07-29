from __future__ import annotations

from pathlib import Path
from typing import List

import onnx

# explicitly import ops to trigger @register decorators
import ffx_compiler.frontend.onnx.ops  # noqa: F401
from ffx_compiler.frontend.onnx.graph_context import OnnxGraphContext
from ffx_compiler.frontend.onnx.ops_registry import OnnxOpsRegistry
from ffx_compiler.frontend.parser import Parser
from ffx_compiler.ir import Graph, Input, Node, Output


class OnnxParser(Parser):
    @staticmethod
    def map_inputs(context: OnnxGraphContext) -> List[Node]:
        nodes_ir = list()
        for index, inp in enumerate(context.graph.input):
            if inp.name in context.initializers:
                continue
            shape, dtype = context.tensor_meta.get(inp.name, ((), "unknown"))
            node_ir = Node(name=inp.name, op=Input(index), shape=shape, dtype=dtype)
            context.value_map[inp.name] = inp.name
            nodes_ir.append(node_ir)
        return nodes_ir

    @staticmethod
    def map_outputs(context: OnnxGraphContext) -> List[Node]:
        nodes_ir = list()
        for index, out in enumerate(context.graph.output):
            shape, dtype = context.tensor_meta.get(out.name, ((), "unknown"))
            parent = context.resolve_input_name(out.name)
            node_ir = Node(
                name=out.name,
                op=Output(index),
                inputs=[parent],
                shape=shape,
                dtype=dtype,
            )
            nodes_ir.append(node_ir)
        return nodes_ir

    def topological_graph_traverse(self, context: OnnxGraphContext) -> List[Node]:
        nodes_ir = list()
        for node in context.graph.node:
            output_name = node.output[0] if node.output else f"{node.name}_out"
            node_name = node.name or output_name

            operator_create_fn = OnnxOpsRegistry.get(node.op_type)
            if not operator_create_fn:
                raise NotImplementedError(
                    f"Unsupported ONNX operator '{node.op_type}' at node '{node_name}'."
                )

            operator_instance = operator_create_fn(node, context)
            if operator_instance is None:
                continue

            for out in node.output:
                context.value_map[out] = node_name

            inputs = [
                context.resolve_input_name(inp)
                for inp in node.input
                if inp in context.value_map
            ]
            shape, dtype = context.tensor_meta.get(output_name, ((), "unknown"))

            node_ir = Node(name=node_name, op=operator_instance)
            node_ir.inputs = inputs
            node_ir.shape = shape
            node_ir.dtype = dtype

            nodes_ir.append(node_ir)

        return nodes_ir

    def parse_to_intermediate_representation(self, model_path: Path) -> Graph:
        model = onnx.load(model_path)
        onnx.checker.check_model(model, full_check=True)
        model = onnx.shape_inference.infer_shapes(model)

        context = OnnxGraphContext(model)
        nodes_ir: List[Node] = list()

        nodes_ir.extend(OnnxParser.map_inputs(context))
        nodes_ir.extend(self.topological_graph_traverse(context))
        nodes_ir.extend(OnnxParser.map_outputs(context))

        batch_size = (
            -1
            if any(
                n.shape and n.shape[0] == -1
                for n in nodes_ir
                if isinstance(n.op, Input)
            )
            else 1
        )

        graph_ir = Graph(name=model_path.stem, nodes=nodes_ir)
        graph_ir.batch_size = batch_size
        return graph_ir
