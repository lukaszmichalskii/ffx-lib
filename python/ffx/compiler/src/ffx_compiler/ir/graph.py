from dataclasses import dataclass, field
from typing import List

from ffx_compiler.ir.node import Node


@dataclass
class Graph:
    name: str
    nodes: List[Node] = field(default_factory=list)
    batch_size: int = 1
