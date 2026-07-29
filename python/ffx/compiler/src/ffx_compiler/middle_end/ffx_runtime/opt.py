import logging

from ffx_compiler.ir import Graph
from ffx_compiler.middle_end.ffx_runtime.opt_registry import register_pass

logger = logging.getLogger(__name__)


# =============================================================================
# -O1
# =============================================================================


@register_pass("constant_folding", min_level=1, order=10)
def constant_folding(graph: Graph) -> Graph:
    return graph


@register_pass("kernel_fusion", min_level=1, order=20)
def kernel_fusion(graph: Graph) -> Graph:
    return graph


@register_pass("shape_folding", min_level=1, order=30)
def shape_folding(graph: Graph) -> Graph:
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
