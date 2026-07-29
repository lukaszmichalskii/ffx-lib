from abc import ABC, abstractmethod

from ffx_compiler.ir import Graph


class Optimizer(ABC):
    def __init__(self, level: int):
        self.level = level

    @abstractmethod
    def optimize_intermediate_representation(self, graph: Graph) -> Graph:
        pass
