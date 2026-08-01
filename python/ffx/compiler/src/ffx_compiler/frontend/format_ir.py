from typing import List

import numpy as np

from ffx_compiler.ir import Graph, Node, TrainableOp


def number_of_parameters(node: Node) -> int:
    if not isinstance(node.op, TrainableOp):
        return 0

    total = 0
    for attr in ("weight", "bias", "mean", "var"):
        val = getattr(node.op, attr, None)
        if isinstance(val, np.ndarray) and val.size > 0:
            total += val.size
    return total


def model_summary(graph: Graph) -> str:
    """Generates a Keras/PyTorch style summary table with an explicit Type column."""
    try:
        from tabulate import tabulate
    except ImportError:
        return f"Graph IR: {graph.name} (nodes: {len(graph.nodes)})"

    table_rows: List[List[str]] = []
    total_params = 0

    for i, node in enumerate(graph.nodes):
        if node.shape:
            shape_list = list(node.shape)
            shape_list[0] = "None"
            output_shape_str = str(tuple(shape_list)).replace("'None'", "None")
        else:
            output_shape_str = "(None)"

        node_params = number_of_parameters(node)
        total_params += node_params
        params_str = f"{node_params:,}" if node_params > 0 else "0"

        connected_to_str = ", ".join(node.inputs) if node.inputs else ""

        table_rows.append(
            [node.name, node.type, output_shape_str, params_str, connected_to_str]
        )

        if i < len(graph.nodes) - 1:
            table_rows.append(["---BREAK---", "", "", "", ""])

    headers = ["Layer Name", "Type", "Output Shape", "Param #", "Connected to"]
    raw_table = tabulate(table_rows, headers=headers, tablefmt="plain", stralign="left")
    table_lines = raw_table.split("\n")

    header_line = table_lines[0]
    body_lines = table_lines[1:]

    line_length = max(len(line) for line in table_lines)
    thick_sep = "=" * line_length
    thin_sep = "_" * line_length

    cleaned_body = [thin_sep if "---BREAK---" in line else line for line in body_lines]

    return "\n".join(
        [
            thin_sep,
            header_line,
            thick_sep,
            "\n".join(cleaned_body),
            thick_sep,
            f"Total params: {total_params:,}",
        ]
    )
