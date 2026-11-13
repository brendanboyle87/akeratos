"""The table's configuration: its size, row format, shards and update-rule switches.

Each field is a `table.*` key of the run's configuration. The size and width have no default:
they come from the run's configuration (the width is the model's d), so nothing here assumes a
core size. The shard cap keeps shard bytes at most half the measured forward-and-backward
headroom, floored to a power of two of elements and never above 2³¹ - 2²⁰, because MLX
reductions over more than 2³¹ elements go wrong.

This module imports no MLX, so a configuration can be built and validated first.
"""

import math
from dataclasses import dataclass

from akeratos.errors import ConfigError

SHARD_LIMIT = 2**31 - 2**20  # elements: MLX 0.32.3 reduces bigger arrays to the identity
SHARD_POC = 2**29  # elements: the cap from the M4 Max's measured headroom
FORMATS = ("F2", "F1", "F32")


@dataclass(frozen=True)
class TableConfig:
    """One lookup table: its rows, their format, its shards and its update rule.

    `seed` is the run's master seed; the hash multipliers and the rounding keys derive from it
    by `derive` in `akeratos/rng.py`, on the streams `table.mult` and `table.round`.
    """

    params: float  # table.params: numbers stored
    width: int  # table.width: the model's d
    orders: tuple[int, ...] = (2, 3)  # table.orders; k = len(orders) rows per token per head
    fmt: str = "F2"  # table.format: F2 | F1 | F32 (the test oracle)
    rounding: str = "stochastic"  # table.rounding: stochastic | nearest
    beta2: float = 0.95  # per update event, pinned
    eps: float = 1e-10
    lr_mult: float = 50.0  # M: η_table = M·η_emb
    floor: float = 0.3  # after warm-up the rate stays at or above floor·M·η_emb,peak
    d_peak: float = 0.00784  # gated decay at the peak rate
    decay: str = "follow"  # table.decay: follow | coupled | zero
    lr_schedule: str = "floor"  # table.lr_schedule: floor | to0 | constant
    heads: int = 1  # table.heads: hash heads per order, rows w/heads wide
    bias: str = "global"  # table.bias: global (√(1 - β₂^e)) | per_row (√(1 - β₂^n_j))
    seed: int = 0
    max_shard_elements: int = SHARD_POC

    @property
    def k(self) -> int:
        """Orders per token."""
        return len(self.orders)

    def row_width(self) -> int:
        """The stored row's width, w/heads."""
        return self.width // self.heads

    def lookups(self) -> int:
        """Rows read per token, k·heads."""
        return self.k * self.heads

    def rows(self) -> int:
        """Rows in the global row space, a multiple of `lookups`."""
        r = int(self.params // self.row_width())
        return r - r % self.lookups()

    def rows_per_slot(self) -> int:
        """Rows of one (order, head) slot; slots are contiguous and order-major."""
        return self.rows() // self.lookups()

    def rows_per_shard(self) -> int:
        """Rows in every shard but the last."""
        return self.max_shard_elements // self.row_width()

    def n_shards(self) -> int:
        return math.ceil(self.rows() / self.rows_per_shard())

    def bytes_per_param(self) -> float:
        """Bytes per stored number: the row plus its share of v, the stamp and any count."""
        w = self.row_width()
        count = 4 / w if self.bias == "per_row" else 0.0
        return {"F2": 2 + 8 / w, "F1": 1 + 12 / w, "F32": 4 + 8 / w}[self.fmt] + count

    def bytes(self) -> float:
        """Resident bytes of the whole table."""
        return self.rows() * self.row_width() * self.bytes_per_param()

    def validate(self) -> "TableConfig":
        """Return self, or raise ConfigError naming every invalid field."""
        checks = [
            (self.fmt in FORMATS, f"table.format {self.fmt!r}"),
            (self.rounding in ("stochastic", "nearest"), f"table.rounding {self.rounding!r}"),
            (self.decay in ("follow", "coupled", "zero"), f"table.decay {self.decay!r}"),
            (
                self.lr_schedule in ("floor", "to0", "constant"),
                f"table.lr_schedule {self.lr_schedule!r}",
            ),
            (self.bias in ("global", "per_row"), f"table.bias {self.bias!r}"),
            (bool(self.orders) and min(self.orders) >= 1, f"table.orders {self.orders!r}"),
            (
                0 < self.max_shard_elements <= SHARD_LIMIT,
                f"shard of {self.max_shard_elements:,} elements: the cap is {SHARD_LIMIT:,}"
                " (MLX 0.32.3 reduces bigger arrays to the identity)",
            ),
            (
                self.heads >= 1 and self.width % self.heads == 0,
                f"table.heads {self.heads} must divide table.width {self.width}",
            ),
            (0 < self.beta2 < 1, f"table.beta2 {self.beta2}"),
        ]
        problems = [message for ok, message in checks if not ok]
        if not problems and (self.rows_per_shard() < 1 or self.rows_per_slot() < 1):
            problems.append("table too small for its width")
        if problems:
            raise ConfigError("; ".join(problems))
        return self
