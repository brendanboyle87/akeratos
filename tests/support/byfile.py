"""Load a script outside the packages by its path, as the speedrun's tests must.

The tests never import `speedrun/*.py` as a module: each script is loaded from its file.
No `__pycache__` is written beside it.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str, name: str) -> ModuleType:
    """The module at `ROOT / rel`, registered in `sys.modules` as `name`."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod
