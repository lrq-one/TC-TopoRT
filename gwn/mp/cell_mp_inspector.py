import inspect
from collections import OrderedDict
from typing import Dict, Any, Callable, Set

# =========================================================================
# 1. 手动补充被 PyG 移除的 Inspector 基类
# =========================================================================
class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    # [修复]：增加 return params 以匹配 -> Dict[str, Any] 的类型提示
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
# 2. 原始的 CellularInspector 逻辑 (继承上面手写的 Inspector)
# =========================================================================
class CellularInspector(Inspector):
    """Wrapper of the PyTorch Geometric Inspector so to adapt it to our use cases."""

    def __implements__(self, cls, func_name: str) -> bool:
        # 阻断特定类的向上传递检查
        if cls.__name__ == 'CochainMessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)
    
    # [修复]：删除了完全冗余、与父类一模一样的 inspect 方法，直接让它自然继承