import inspect
from collections import OrderedDict
from typing import Dict, Any, Callable, Set

# =========================================================================

# =========================================================================
class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    
    def inspect(self, func: Callable, pop_first_n: int = 0) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        for _ in range(pop_first_n):
            params.popitem(last=False)
        self.params[func.__name__] = params
        return params  

    def keys(self, func_names: list) -> Set[str]:
        keys = []
        for func_name in func_names:
            keys += list(self.params[func_name].keys())
        return set(keys)

    def implements(self, func_name: str) -> bool:
        return self.__implements__(self.base_class.__class__, func_name)

    def __implements__(self, cls, func_name: str) -> bool:
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def distribute(self, func_name, kwargs: Dict[str, Any]):
        out = {}
        for key, param in self.params[func_name].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

# =========================================================================

# =========================================================================
class CellularInspector(Inspector):
    """Wrapper of the PyTorch Geometric Inspector so to adapt it to our use cases."""

    def __implements__(self, cls, func_name: str) -> bool:
        
        if cls.__name__ == 'CochainMessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)
    
    