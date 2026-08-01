import logging

import ffx_compiler.middle_end.ffx_runtime.opt  # noqa: F401
from ffx_compiler.ir import Graph
from ffx_compiler.middle_end.ffx_runtime.opt_registry import get_sorted_passes
from ffx_compiler.middle_end.optimizer import Optimizer

logger = logging.getLogger(__name__)


class FfxRuntimeOptimizer(Optimizer):
    def __init__(self, level: int):
        super().__init__(level)

    def optimize_intermediate_representation(self, graph: Graph) -> Graph:
        if self.level == 0:
            logger.debug("Optimization level -O0. Skipping all middle-end passes.")
            return graph

        logger.info(f"Running optimization pass pipeline at -O{self.level}.")
        pipeline = get_sorted_passes(self.level)
        graph_opt = graph
        for pass_name, pass_fn in pipeline:
            graph_opt = pass_fn(graph_opt)
        return graph_opt
