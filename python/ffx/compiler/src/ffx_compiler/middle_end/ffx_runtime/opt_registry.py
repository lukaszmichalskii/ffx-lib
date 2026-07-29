import logging
from typing import Callable, List, Tuple

from ffx_compiler.common import Registry
from ffx_compiler.ir import Graph

logger = logging.getLogger(__name__)

OptPassFn = Callable[[Graph], Graph]
OptPassMetadata = Tuple[int, int, OptPassFn]
FfxOptRegistry = Registry[str, OptPassMetadata](name="FfxOptRegistry")


def register_pass(name: str, min_level: int = 1, order: int = 100):
    """register an optimization pass function with its required -O level."""

    def decorator(fn: OptPassFn) -> OptPassFn:
        FfxOptRegistry.register(name)((min_level, order, fn))
        return fn

    return decorator


def get_sorted_passes(level: int) -> List[Tuple[str, OptPassFn]]:
    """Retrieves all passes enabled for `level`, sorted FIRST by min_level, THEN by order."""
    enabled_passes = []
    for name, (min_lvl, order, fn) in FfxOptRegistry.items():
        if level >= min_lvl:
            enabled_passes.append((min_lvl, order, name, fn))
    enabled_passes.sort(key=lambda item: (item[0], item[1]))
    return [(name, fn) for _, _, name, fn in enabled_passes]
