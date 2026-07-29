from abc import ABC, abstractmethod
from pathlib import Path

from ffx_compiler.ir import Graph


class Parser(ABC):
    @abstractmethod
    def parse_to_intermediate_representation(self, model_path: Path) -> Graph:
        pass
