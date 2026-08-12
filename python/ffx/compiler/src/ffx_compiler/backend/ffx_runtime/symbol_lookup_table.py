import re
from typing import Dict, List, Set, Tuple

from typing_extensions import Self

from ffx_compiler.ir import Graph, Input, IOOp, Node, TransformOp


def sanitize_variable_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"var_{sanitized}"
    return sanitized


def get_shape_expression(shape: Tuple[int, ...]) -> str:
    if not shape or len(shape) <= 1:
        return "1"
    return " * ".join(str(d) for d in shape[1:])


class SymbolLookupTable:
    def __init__(self):
        self.ir_to_cpp_variable: Dict[str, str] = dict()
        self.aliases: Dict[str, str] = (
            dict()
        )  # Maps transform_node_id -> parent_node_id
        self.graph_input_cc_variables: Set[str] = set()
        self.graph_input_names: Set[str] = set()
        self.batch_size = 1

    @classmethod
    def from_graph(cls, graph: Graph) -> Self:
        lookup_table = cls()
        lookup_table.batch_size = graph.batch_size

        input_nodes = [n for n in graph.nodes if isinstance(n.op, Input)]
        input_nodes.sort(key=lambda n: n.op.index)

        for node in input_nodes:
            cc_var = f"input_{node.op.index}" if len(input_nodes) > 1 else "input"
            lookup_table.ir_to_cpp_variable[node.name] = cc_var
            lookup_table.graph_input_cc_variables.add(cc_var)
            lookup_table.graph_input_names.add(node.name)

        for node in graph.nodes:
            if isinstance(node.op, IOOp):
                continue

            if isinstance(node.op, TransformOp):
                parent_input = node.inputs[0] if node.inputs else ""
                lookup_table.aliases[node.name] = parent_input
                continue

            cpp_var = sanitize_variable_name(node.name)
            lookup_table.ir_to_cpp_variable[node.name] = cpp_var

        return lookup_table

    def get_physical_node_id(self, node_id: str) -> str:
        """Recursively follows TransformOp aliases to find the physical buffer owner."""
        visited = set()
        curr = node_id
        while curr in self.aliases:
            if curr in visited:
                break
            visited.add(curr)
            curr = self.aliases[curr]
        return curr

    def get_cpp_var(self, node_id: str) -> str:
        physical_id = self.get_physical_node_id(node_id)
        return self.ir_to_cpp_variable.get(
            physical_id, sanitize_variable_name(physical_id)
        )

    def get_buffer_accessor(self, node_id: str) -> str:
        physical_id = self.get_physical_node_id(node_id)
        cpp_var = self.get_cpp_var(physical_id)

        if (
            physical_id in self.graph_input_names
            or cpp_var in self.graph_input_cc_variables
        ):
            return f"device_{cpp_var}.data()"
        return f"device_{cpp_var}_output_.data()"

    def get_size_expression(self, node: Node) -> str:
        shape = node.folded_shape if node.folded_shape else node.shape

        # scalar
        if not shape:
            return "1"

        dims = list()
        for dim in shape:
            if dim == -1:  # dynamic
                dims.append("kBatchSize")
            else:  # static
                dims.append(str(dim))

        return " * ".join(dims)

    def get_batch_expr(self) -> str:
        return "kBatchSize" if self.batch_size <= 0 else str(self.batch_size)

    def resolve_input_buffers(self, node: Node, fallback: List[Node]) -> List[str]:
        inputs = (
            node.inputs if node.inputs else ([fallback[0].name] if fallback else [])
        )
        return [self.get_buffer_accessor(inp) for inp in inputs]
