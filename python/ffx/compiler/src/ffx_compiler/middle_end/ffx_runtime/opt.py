import logging
import math
from typing import List, Set, Tuple

from ffx_compiler.ir import ActivationOp, Graph, Node
from ffx_compiler.middle_end.ffx_runtime.kernel_fusion_registry import (
    KernelFusionRegistry,
)
from ffx_compiler.middle_end.ffx_runtime.opt_registry import register_pass

logger = logging.getLogger(__name__)


def consteval_dim_expression(dim: int | str | Tuple | List) -> int | str:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, (list, tuple)):
        if all(isinstance(x, int) for x in dim):
            return math.prod(dim)
        return str(dim)
    if isinstance(dim, str):
        try:
            val = eval(dim, {"__builtins__": {}}, {})
            if isinstance(val, (int, float)):
                return int(val)
        except Exception:
            # symbolic expression
            pass
    return dim


# =============================================================================
# -O1
# =============================================================================


@register_pass("constant_folding", min_level=1, order=10)
def constant_folding(graph: Graph) -> Graph:
    return graph


@register_pass("kernel_fusion", min_level=1, order=20)
def kernel_fusion(graph: Graph) -> Graph:
    def get_node_consumers(graph: Graph, producer: Node) -> List[Node]:
        producer_targets = set(getattr(producer, "outputs", []))
        producer_targets.add(producer.name)
        return [
            n for n in graph.nodes if any(inp in producer_targets for inp in n.inputs)
        ]

    logger.debug("Running activation kernel fusion pass")

    nodes_to_remove: Set[str] = set()
    fused_count = 0

    for producer in graph.nodes:
        if producer.name in nodes_to_remove:
            continue

        consumers = get_node_consumers(graph, producer)
        if len(consumers) != 1:
            continue

        consumer = consumers[0]
        if not isinstance(consumer.op, ActivationOp):
            continue

        producer_op_type = type(producer.op)
        fusion_fn = KernelFusionRegistry.get(producer_op_type)

        if fusion_fn is None:
            logger.debug(
                "No fusion fn registered for producer op '%s'", producer_op_type
            )
            continue

        fused_op = fusion_fn(producer.op, consumer.op)
        logger.debug(
            "Fused '%s' (%s) into '%s' (%s) -> %s",
            consumer.name,
            type(consumer.op).__name__,
            producer.name,
            producer.type,
            f"{producer.type}{type(consumer.op).__name__}",
        )

        producer.op = fused_op
        producer.type = type(fused_op).__name__

        consumer_outputs = set(getattr(consumer, "outputs", []))
        consumer_outputs.add(consumer.name)
        for ds_consumer in graph.nodes:
            if ds_consumer.name == consumer.name:
                continue
            ds_consumer.inputs = [
                producer.name if inp in consumer_outputs else inp
                for inp in ds_consumer.inputs
            ]

        nodes_to_remove.add(consumer.name)
        fused_count += 1

    if nodes_to_remove:
        graph.nodes = [n for n in graph.nodes if n.name not in nodes_to_remove]
        logger.info(
            "Kernel fusion complete: fused %d activation nodes",
            fused_count,
        )

    return graph


@register_pass("shape_folding", min_level=1, order=15)
def shape_folding(graph: Graph) -> Graph:
    logger.debug("Folding dimension expressions across IR graph nodes")
    modified_count = 0

    for node in graph.nodes:
        shape = getattr(node, "shape", None)
        if not shape:
            continue

        if len(shape) <= 1:
            node.folded_shape = tuple(shape)
            continue

        batch_dim = shape[0]
        trailing_dims = shape[1:]

        if all(isinstance(d, int) and d > 0 for d in trailing_dims):
            static_volume = math.prod(trailing_dims)
            node.folded_shape = (batch_dim, static_volume)
            modified_count += 1
        else:
            node.folded_shape = tuple(shape)

    logger.debug("Shape folding complete: folded %d node shapes", modified_count)
    return graph


# =============================================================================
# -O2
# =============================================================================


@register_pass("buffer_liveness_analysis", min_level=2, order=5)
def buffer_liveness_analysis(graph: Graph) -> Graph:
    return graph


@register_pass("malloc_optimization", min_level=2, order=10)
def malloc_optimization(graph: Graph) -> Graph:
    return graph


# =============================================================================
# -O3
# =============================================================================


@register_pass("fork_join_stream", min_level=3, order=10)
def parallel_streams(graph: Graph) -> Graph:
    return graph
