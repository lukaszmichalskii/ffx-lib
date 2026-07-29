import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")  # lookup key type
V = TypeVar("V")  # registered value type


def import_submodules(package: ModuleType) -> None:
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = f"{package.__name__}.{module_name}"
        importlib.import_module(full_name)


class Registry(Generic[K, V]):
    def __init__(self, name: str):
        self.name = name
        self.__registry: Dict[K, V] = dict()

    def register(self, key: K | List[K]) -> Callable[[V], V]:
        def decorator(fn: V) -> V:
            keys = key if isinstance(key, list) else [key]
            for k in keys:
                if k in self.__registry:
                    logger.warning(
                        f"[{self.name}] Overwriting existing handler for key '{k}'"
                    )
                self.__registry[k] = fn
            return fn

        return decorator

    def get(self, key: K) -> Optional[V]:
        return self.__registry.get(key)

    def is_supported(self, key: K) -> bool:
        return key in self.__registry

    def keys(self):
        return self.__registry.keys()

    def values(self):
        return self.__registry.values()

    def items(self):
        return self.__registry.items()
