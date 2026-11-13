"""The exception tree.

Every error Akeratos raises on purpose derives from `AkeratosError`. The CLI
(`akeratos/cli.py`) turns a `ConfigError` and a `DataMismatchError` into its exit codes.
"""


class AkeratosError(Exception):
    """The root of every error Akeratos raises on purpose."""


class ConfigError(AkeratosError):
    """A run configuration or environment that Akeratos refuses."""


class DataError(AkeratosError):
    """A failure in the data a run reads."""


class DataMismatchError(DataError):
    """Data that breaks its format: the CLI maps this error, and not its parent, to an exit code."""
