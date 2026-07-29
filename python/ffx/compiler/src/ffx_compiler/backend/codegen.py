from abc import ABC, abstractmethod
from typing import Dict, Tuple

from ffx_compiler.ir import Graph


class Codegen(ABC):
    @abstractmethod
    def intermediate_representation_to_source_code(
        self, graph: Graph
    ) -> Tuple[Dict[str, str], bytes]:
        pass
