from dataclasses import dataclass, field
from typing import List, Tuple

from ffx_compiler.ir.ops import Op


@dataclass
class Node:
    name: str
    op: Op
    inputs: List[str] = field(default_factory=list)
    shape: Tuple[int, ...] = None
    type: str = "unknown"
    dtype: str = "float32"
    # O1
    folded_shape: Tuple[int, ...] = None

    def __post_init__(self):
        self.type = type(self.op).__name__
