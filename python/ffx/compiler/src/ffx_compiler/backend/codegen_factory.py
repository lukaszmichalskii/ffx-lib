from ffx_compiler.backend.backend_error import BackendError
from ffx_compiler.backend.codegen import Codegen
from ffx_compiler.backend.ffx_runtime.codegen import FfxRuntimeCodegen


def codegen_factory(runtime: str) -> Codegen:
    if runtime == "ffx_runtime":
        return FfxRuntimeCodegen()
    raise BackendError(f"Unsupported runtime: {runtime}")
