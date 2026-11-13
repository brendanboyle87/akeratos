r"""Run a frozen script that imports the package by the project's first name.

`analysis.py` and `overlap.py` never change: their SHA-256s were recorded before the result.
They were written under the project's first name, MetalForge, and `overlap.py` imports the
package by that name. `alias` points that name at the package. Run as a script, this file runs
the script it is given, with that script's arguments:

    uv run --frozen --no-sync python scripts/analysis/firstname.py scripts/analysis/overlap.py \
        --store STORE --out OUT
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import runpy
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType

import akeratos

FIRST = "metalforge"


class _FirstName(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Gives each module asked for under the first name as the package's own module."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        """A spec for `FIRST` and its modules; None for every other name."""
        del path, target
        if fullname != FIRST and not fullname.startswith(FIRST + "."):
            return None
        module = importlib.import_module(akeratos.__name__ + fullname[len(FIRST) :])
        return importlib.util.spec_from_loader(
            fullname, self, is_package=hasattr(module, "__path__")
        )

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """The package's module that the name stands for, already imported."""
        return sys.modules[akeratos.__name__ + spec.name[len(FIRST) :]]

    def exec_module(self, module: ModuleType) -> None:
        """Nothing to run: the module ran when it was imported under its own name."""


def alias() -> None:
    """Make an import of the first name find the package, where that is not its name."""
    if akeratos.__name__ != FIRST and not any(isinstance(f, _FirstName) for f in sys.meta_path):
        sys.meta_path.insert(0, _FirstName())


def main(argv: Sequence[str]) -> None:
    """Run the script `argv[0]` as `__main__` with the rest of `argv` as its arguments."""
    alias()
    sys.argv = list(argv)
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
