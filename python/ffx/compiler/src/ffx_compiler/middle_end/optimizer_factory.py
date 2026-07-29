from ffx_compiler.middle_end.ffx_runtime.optimizer import FfxRuntimeOptimizer
from ffx_compiler.middle_end.middle_end_error import MiddleEndError
from ffx_compiler.middle_end.optimizer import Optimizer


def optimizer_factory(runtime: str, level: int = 0) -> Optimizer:
    if runtime == "ffx_runtime":
        return FfxRuntimeOptimizer(level=level)
    raise MiddleEndError(f"Unsupported runtime: {runtime}")
