"""Backward-compatible import namespace retained for pre-0.3 clients.

The implementation lives in :mod:`urdwell`; these aliases keep existing Python
clients working during the 0.x transition.
"""

from importlib import import_module
import sys

from urdwell import __version__

_ALIASED_MODULES = (
    "embeddings",
    "integrations",
    "models",
    "pipeline",
    "ranking",
    "storage",
)

for _module_name in _ALIASED_MODULES:
    _module = import_module(f"urdwell.{_module_name}")
    sys.modules[f"{__name__}.{_module_name}"] = _module
    globals()[_module_name] = _module

del _module, _module_name
