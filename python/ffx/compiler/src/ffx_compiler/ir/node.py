from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ffx_compiler.ir.ops import Op


@dataclass
class Node:
    name: str
    op: Op
    inputs: List[str] = field(default_factory=list)
    shape: Tuple[int, ...] = None
    type: str = "unknown"
    dtype: str = "float32"
    folded_size: Optional[int] = None

    def __post_init__(self):
        self.type = type(self.op).__name__
