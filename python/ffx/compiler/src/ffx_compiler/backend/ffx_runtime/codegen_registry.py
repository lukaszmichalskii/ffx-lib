from typing import Any, Callable, Dict, Type

from ffx_compiler.common import Registry
from ffx_compiler.ir import Op

OpCodegenFn = Callable[[Any, str, str], Dict[str, Any]]
FfxOpsRegistry = Registry[Type[Op], OpCodegenFn](name="FfxOpsRegistry")
