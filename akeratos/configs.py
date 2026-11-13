"""Named configurations: the two released runs as their records hold them, a starter, a 10M model.

`BILLET` is the 150M model with the n-gram table. `BILLET_TWIN` is the same run with the table
off. Each equals the `config` object of the `config.json` beside its weights: what the training
run recorded as its own configuration. Both trained on the banded attention path. `STARTER` is
Billet's configuration at the 10M model's size, for a first run of about 40 minutes. `LAUNCH` and
`STARTER_LAUNCH` say how each is launched.

`ModelConfig`'s defaults are the 10M model's shape. In `from_keys`, a key that is absent takes
its field's default. This module imports no MLX, so a configuration can be built before MLX
loads.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, cast

VOCAB = 49_152  # SmolLM2's vocabulary
SEP_ID = 0  # <|endoftext|>: separates documents and opens a segment
MASKED_IDS = tuple(range(1, 17))  # ids 1-16: reserved, never emitted, masked out
MUP_BASE_D = 256  # μP base shape: the proof of concept's width
ID_LIMIT = 65_536  # ids are stored as uint16


@dataclass(frozen=True)
class ModelConfig:
    """One model: its shape and the state of every model switch.

    `window` None turns windows off, so every layer is global. `xl_init` is the x0 scalar at
    layer 0 and at the last layer, linear between. `table_sites` () enters the table term at
    every layer. The table's own configuration (rows, format, rates) is in `akeratos.table`.
    """

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    head_dim: int = 64
    n_kv_heads: int = 2
    ffn: int = 1152  # ReLU² width, 4.5·d
    vocab_size: int = VOCAB
    window: int | None = 512  # attn.window
    global_every: int = 5  # global at every 5th layer (0-based 4, 9, ...) and the last
    rope_theta_local: float = 1e4
    rope_theta_global: float = 1e6
    rope_global_fraction: float = 1.0  # rope.global_fraction p
    rms_norm_eps: float = 1e-5
    norm_placement: str = "peri"  # norm.placement: peri | pre | post
    norm_out_gain_init: float = 1.0  # norm.out_gain_init
    ffn_kind: str = "relu2"  # ffn.kind: relu2 | glu (GELU-tanh, width 3·d)
    x0_inject: bool = True  # residual.x0_inject
    xl_init: tuple[float, float] = (0.20, 0.05)
    init_out_proj: str = "fan_in"  # init.out_proj: fan_in | zero (pre-norm only)
    # attn.path: banded | masked (the full-mask oracle) | fused (kernels); a fresh entry with
    # no switch resolves it to fused where the kernels run (Checkpointer.attn_path)
    attn_path: str = "banded"
    doc_mask: bool = True  # mask.packing: document (True) | causal (False)
    embed_std: float = 0.02
    input_scale: str = "norm"  # embed.input_scale: norm | sqrt_d | none
    value_embed: str = "odd"  # value_embed.layers: odd | off | every
    ve_gate_width: int = 12
    logit_cap: float | None = 15.0  # loss.final_softcap; None is off
    masked_ids: tuple[int, ...] = MASKED_IDS  # head.masked_ids
    z_loss: float = 0.0  # loss.z
    loss_path: str = "unchunked"  # loss.path: unchunked | chunked
    loss_chunk: int = 256  # loss.chunk_tokens
    table: bool = True  # the table term's entry scalars tl
    tl_init: float = 0.1
    table_site_scale: bool = True  # table.site_scale: tl starts at tl_init·8/L
    table_sites: tuple[int, ...] = ()  # table.sites
    param_kind: str = "mup"  # param.kind: mup (logit multiplier 256/d) | sp (1)

    def ffn_width(self) -> int:
        """The feed-forward width: `ffn`, or 3·d under `ffn.kind` glu."""
        return 3 * self.d_model if self.ffn_kind == "glu" else self.ffn

    def is_global(self, i: int) -> bool:
        """Whether layer i attends globally: every 5th layer and the last, or all without windows."""
        if self.window is None:
            return True
        return (i + 1) % self.global_every == 0 or i == self.n_layers - 1

    def ve_layers(self) -> list[int]:
        """The layers with a value embedding: the odd layers by default."""
        if self.value_embed == "off":
            return []
        step, first = (1, 0) if self.value_embed == "every" else (2, 1)
        return list(range(first, self.n_layers, step))

    def m_logit(self) -> float:
        """μP's logit multiplier: 256/d, or 1 under `param.kind` sp."""
        return MUP_BASE_D / self.d_model if self.param_kind == "mup" else 1.0

    def sites(self) -> list[int]:
        """The layers whose entry adds tl·t; one tl per site."""
        return list(self.table_sites) if self.table_sites else list(range(self.n_layers))

    def xl_values(self) -> list[float]:
        """The initial x0 scalars, from `xl_init[0]` at layer 0 to `xl_init[1]` at the last."""
        a, b = self.xl_init
        last = max(self.n_layers - 1, 1)
        return [a + (b - a) * i / last for i in range(self.n_layers)]

    def tl_value(self) -> float:
        """The table's initial entry strength: `tl_init`·8/L under `table.site_scale`."""
        return self.tl_init * 8 / self.n_layers if self.table_site_scale else self.tl_init

    def validate(self) -> "ModelConfig":
        """Return self, or raise ValueError naming every invalid field."""
        p = self.rope_global_fraction
        pairs = p * self.head_dim / 2
        sites = list(self.table_sites)
        checks = [
            (
                self.norm_placement in ("peri", "pre", "post"),
                f"norm.placement {self.norm_placement!r}",
            ),
            (self.ffn_kind in ("relu2", "glu"), f"ffn.kind {self.ffn_kind!r}"),
            (self.init_out_proj in ("fan_in", "zero"), f"init.out_proj {self.init_out_proj!r}"),
            (
                self.init_out_proj != "zero" or self.norm_placement == "pre",
                "init.out_proj zero is a configuration error unless norm.placement is pre",
            ),
            (
                self.attn_path in ("banded", "masked", "fused"),
                f"attn.path {self.attn_path!r}",
            ),
            (
                self.input_scale in ("norm", "sqrt_d", "none"),
                f"embed.input_scale {self.input_scale!r}",
            ),
            (
                self.value_embed in ("odd", "off", "every"),
                f"value_embed.layers {self.value_embed!r}",
            ),
            (self.loss_path in ("unchunked", "chunked"), f"loss.path {self.loss_path!r}"),
            (self.param_kind in ("mup", "sp"), f"param.kind {self.param_kind!r}"),
            (self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"),
            (self.head_dim % 2 == 0, "head_dim must be even (half-split RoPE)"),
            (
                0 <= p <= 1 and pairs == int(pairs),
                f"rope.global_fraction {p}: a whole number of pairs",
            ),
            (self.ve_gate_width <= self.d_model, "ve_gate_width exceeds d_model"),
            (self.vocab_size <= ID_LIMIT, "ids are stored as uint16"),
            (self.logit_cap is None or self.logit_cap > 0, "loss.final_softcap must be positive"),
            (
                all(0 < i < self.vocab_size for i in self.masked_ids),
                "head.masked_ids outside 1..V-1",
            ),
            (self.loss_chunk > 0, "loss.chunk_tokens must be positive"),
            (
                sites == sorted(set(sites)) and all(0 <= i < self.n_layers for i in sites),
                f"table.sites {self.table_sites!r}: sorted, unique layers in [0, {self.n_layers})",
            ),
        ]
        problems = [message for ok, message in checks if not ok]
        if problems:
            raise ValueError("; ".join(problems))
        return self


# Switch key -> ModelConfig field. Absent keys take the field's default, the "off" state.
SWITCHES = {
    "norm.placement": "norm_placement",
    "norm.out_gain_init": "norm_out_gain_init",
    "ffn.kind": "ffn_kind",
    "residual.x0_inject": "x0_inject",
    "init.out_proj": "init_out_proj",
    "attn.window": "window",
    "attn.path": "attn_path",
    "rope.global_fraction": "rope_global_fraction",
    "mask.packing": "doc_mask",
    "embed.input_scale": "input_scale",
    "value_embed.layers": "value_embed",
    "loss.final_softcap": "logit_cap",
    "head.masked_ids": "masked_ids",
    "loss.z": "z_loss",
    "loss.path": "loss_path",
    "loss.chunk_tokens": "loss_chunk",
    "table": "table",
    "table.sites": "table_sites",
    "table.site_scale": "table_site_scale",
    "param.kind": "param_kind",
}

_PACKING = {"document": True, "causal": False}
_SITES: dict[str, tuple[int, ...]] = {"all": (), "every": (), "input": (0,)}


def _value(name: str, value: object) -> object:
    """Normalise a switch's spelling to its field's type."""
    if name == "doc_mask" and isinstance(value, str):
        return _PACKING[value]
    if name == "table_sites" and isinstance(value, str):
        return _SITES[value]
    if name == "table_sites" and value is None:
        return ()
    if name in ("table_sites", "masked_ids", "xl_init") and isinstance(value, (list, tuple)):
        return tuple(cast("list[object]", value))
    return value


def from_keys(keys: Mapping[str, Any]) -> ModelConfig:
    """Build and validate a ModelConfig from switch keys and/or field names."""
    names = {f.name for f in fields(ModelConfig)}
    kw: dict[str, Any] = {}
    for key, value in keys.items():
        name = SWITCHES.get(key, key)
        if name not in names:
            raise KeyError(f"unknown model configuration key {key!r}")
        kw[name] = _value(name, value)
    return ModelConfig(**kw).validate()


def describe(cfg: ModelConfig) -> list[str]:
    """One log line per switch, so a run's log names every switch's state."""
    return [f"switch {key} = {getattr(cfg, name)!r}" for key, name in SWITCHES.items()]


# The 10M model that bake bench and the speedrun measure.
MODEL_10M: dict[str, Any] = {
    "d_model": 256,
    "n_layers": 8,
    "n_heads": 4,
    "head_dim": 64,
    "n_kv_heads": 2,
    "ffn": 1152,
    "ffn_kind": "relu2",
    "vocab_size": 49152,
    "window": 512,
    "global_every": 5,
    "rope_theta_local": 10000.0,
    "rope_theta_global": 1000000.0,
    "rms_norm_eps": 1e-05,
    "value_embed": "odd",
    "ve_gate_width": 12,
}

BILLET: dict[str, Any] = {
    "model": {
        "attn_path": "banded",
        "d_model": 768,
        "doc_mask": True,
        "embed_std": 0.02,
        "ffn": 3456,
        "ffn_kind": "relu2",
        "global_every": 5,
        "head_dim": 128,
        "init_out_proj": "fan_in",
        "input_scale": "norm",
        "logit_cap": 15.0,
        "loss_chunk": 256,
        "loss_path": "unchunked",
        "masked_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "n_heads": 6,
        "n_kv_heads": 2,
        "n_layers": 18,
        "norm_out_gain_init": 1.0,
        "norm_placement": "peri",
        "param_kind": "mup",
        "rms_norm_eps": 1e-05,
        "rope_global_fraction": 1.0,
        "rope_theta_global": 1000000.0,
        "rope_theta_local": 10000.0,
        "table": True,
        "table_site_scale": True,
        "table_sites": [],
        "tl_init": 0.1,
        "value_embed": "odd",
        "ve_gate_width": 12,
        "vocab_size": 49152,
        "window": 512,
        "x0_inject": True,
        "xl_init": [0.2, 0.05],
        "z_loss": 0.0,
    },
    "optim": {
        "beta1": 0.9,
        "beta2_fixed": 0.95,
        "beta2_rule": "halflife",
        "clip": 1.0,
        "decay_ind": None,
        "decay_tokens": None,
        "eps": 1e-08,
        "eta_emb": None,
        "eta_ref": 0.001,
        "lr_peak": 0.01,
        "tokens": 3200000000.0,
        "tokens_per_update": 131072,
        "weight_decay_mode": "independent",
    },
    "recipe": {
        "decay_frac": 0.2,
        "depth_scaling": "off",
        "ema": True,
        "ema_half_life": 0.02,
        "final_frac": 0.0,
        "mup_base_d": 256,
        "param_kind": "mup",
        "schedule": "wsd",
        "shape": "1-sqrt",
        "warmup": None,
    },
    "seed": "0x0000000000000000",
    "init_seed": "0x7883aaf03a901492",
    "data_seed": "0x84d17ad2e2342315",
    "compute_dtype": "bfloat16",
    "mtp": False,
    "accum": 32,
    "table": {
        "beta2": 0.95,
        "bias": "global",
        "d_peak": 0.00784,
        "decay": "follow",
        "eps": 1e-10,
        "floor": 0.3,
        "fmt": "F2",
        "heads": 1,
        "lr_mult": 50.0,
        "lr_schedule": "floor",
        "max_shard_elements": 536870912,
        "orders": [2, 3],
        "params": 6000000000.0,
        "rounding": "stochastic",
        "seed": 0,
        "width": 768,
    },
}

BILLET_TWIN: dict[str, Any] = BILLET | {"model": BILLET["model"] | {"table": False}, "table": None}

# How both runs were launched. A run record's `config` does not hold these.
LAUNCH: dict[str, Any] = {
    "rows": 2,
    "seq_len": 2048,
    "ckpt_every": 240,
    "keep": 3,
    "pin_decay": True,
    "cache_limit_gb": 3.81,
}

# A first run of about 40 minutes that goes through the whole trainer, on the speedrun's
# training slice. It is Billet's configuration at the 10M model's size. Its optimiser values
# are the trainer's defaults but for the budget, and its recipe and table values are the
# trainer's defaults at that size. 75M tokens is 9,156 updates of 8,192 tokens, under one pass
# of the slice: packed into these rows, the slice feeds 9,720 to 9,769 updates over seeds 0 to
# 199, so 80M tokens would not fit. It holds no init or data seed: the trainer derives both
# from the seed.
_STARTER_SHAPE = ("d_model", "n_layers", "n_heads", "head_dim", "ffn")
STARTER: dict[str, Any] = {
    "model": BILLET["model"] | {key: MODEL_10M[key] for key in _STARTER_SHAPE},
    "optim": BILLET["optim"] | {"lr_peak": 0.003, "tokens": 7.5e7, "tokens_per_update": 8192},
    "recipe": dict(BILLET["recipe"]),
    "seed": BILLET["seed"],
    "compute_dtype": BILLET["compute_dtype"],
    "mtp": BILLET["mtp"],
    "accum": 2,
    "table": BILLET["table"] | {"params": 1e9, "width": 256},
}

# How the starter is launched: no checkpoint pinned at the decay's start, and MLX's cache capped
# at 2.0 GiB.
STARTER_LAUNCH: dict[str, Any] = {
    "rows": 2,
    "seq_len": 2048,
    "ckpt_every": 2000,
    "keep": 2,
    "pin_decay": False,
    "cache_limit_gb": 2.0,
}
