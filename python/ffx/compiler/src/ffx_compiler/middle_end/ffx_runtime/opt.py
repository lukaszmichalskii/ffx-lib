import logging
import math
from typing import List, Set, Tuple

# explicitly import kernel_fusion to trigger @register decorators
import ffx_compiler.middle_end.ffx_runtime.kernel_fusion  # noqa: F401
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

        fusion_fn = KernelFusionRegistry.get(type(producer.op))
        if fusion_fn is None:
            logger.debug(
                "No fusion fn registered for producer op '%s'", producer.type
            )
            continue

        fused_op = fusion_fn(producer.op, consumer.op)
        logger.debug(
            "Fused '%s' (%s) into '%s' (%s) -> %s",
            consumer.name,
            consumer.type,
            producer.name,
            producer.type,
            f"{producer.type}{consumer.type}",
        )

        old_producer_name = producer.name
        consumer_suffix = consumer.op.name.lower()
        new_producer_name = f"{old_producer_name}_{consumer_suffix}"

        logger.debug(
            "Fused '%s' (%s) into '%s' (%s) -> %s (renamed to '%s')",
            consumer.name,
            consumer.type,
            old_producer_name,
            producer.type,
            f"{producer.type}{consumer.type}",
            new_producer_name,
        )

        producer.op = fused_op
        producer.type = fused_op.name
        producer.name = new_producer_name

        legacy_names = {old_producer_name, consumer.name}
        consumer_outputs = set(getattr(consumer, "outputs", []))
        legacy_names.update(consumer_outputs)

        for ds_node in graph.nodes:
            if ds_node.name == consumer.name or ds_node.name == old_producer_name:
                continue
            ds_node.inputs = [
                new_producer_name if inp in legacy_names else inp
                for inp in ds_node.inputs
            ]

        if hasattr(graph, "outputs") and graph.outputs:
            graph.outputs = [
                new_producer_name if out in legacy_names else out
                for out in graph.outputs
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

    logger.info("Shape folding complete: folded %d node shapes", modified_count)
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
