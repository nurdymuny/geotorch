"""
DavisTensor NN - Container Modules
===================================

Sequential and other container modules.
"""

from __future__ import annotations
from typing import List

from .module import Module


class Sequential(Module):
    """
    Sequential container for layers.
    """
    
    def __init__(self, *layers: Module):
        super().__init__()
        # Store in ordered dict directly, don't use setattr to avoid double registration
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x
    
    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __repr__(self) -> str:
        lines = [f"Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    """
    Holds submodules in a list.
    """
    
    def __init__(self, modules: List[Module] = None):
        super().__init__()
        if modules is not None:
            for i, module in enumerate(modules):
                self._modules[str(i)] = module
    
    def append(self, module: Module):
        idx = len(self._modules)
        self._modules[str(idx)] = module
    
    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self):
        return iter(self._modules.values())
    
    def forward(self, x):
        raise NotImplementedError("ModuleList does not implement forward()")


class ModuleDict(Module):
    """
    Holds submodules in a dictionary.
    """
    
    def __init__(self, modules: dict = None):
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self._modules[name] = module
    
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]
    
    def __setitem__(self, key: str, module: Module):
        self._modules[key] = module
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self):
        return iter(self._modules)
    
    def keys(self):
        return self._modules.keys()
    
    def values(self):
        return self._modules.values()
    
    def items(self):
        return self._modules.items()
    
    def forward(self, x):
        raise NotImplementedError("ModuleDict does not implement forward()")
