"""The hardware-class reader in `akeratos/macops.py`.

`parse_hardware` reads `ioreg -rc AGXAccelerator` (an excerpt) and `pmset -g batt` as captured on
an M4 Max MacBook Pro on 2026-11-02 (`tests/fixtures/bench/`), with that Mac's
`device_info` values as its probe records give them, and variants of all three. No test runs a
command, reads this machine's hardware or touches Metal: `read_hardware` runs off a Mac, and on
one with a fake `mlx.core` and the captured output; `read_conditions` gets captured pmset text
and a fixed thermal level.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from akeratos import macops
from akeratos.errors import ConfigError
from akeratos.macops import Hardware

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bench"
IOREG = (FIXTURES / "ioreg-m4-max.txt").read_text(encoding="utf-8")
PMSET = (FIXTURES / "pmset-m4-max.txt").read_text(encoding="utf-8")  # a laptop, on battery
NO_BATTERY = "Now drawing from 'AC Power'\n"  # what pmset prints on a Mac without one
# A UPS's power source line in pmset's format, written for this test: a charge, no InternalBattery.
UPS = (
    "Now drawing from 'AC Power'\n"
    " -UPS-0 (id=4325376)\t100%; charged; 0:00 remaining present: true\n"
)
# MLX 0.32.3's `device_info` for the M4 Max's GPU, as that Mac's probe environment records give it.
INFO: dict[str, object] = {
    "device_name": "Apple M4 Max",
    "architecture": "applegpu_g16s",
    "memory_size": 51_539_607_552,
    "max_buffer_length": 30_150_672_384,
    "max_recommended_working_set_size": 40_200_896_512,
}
MODEL = "Mac16,5\n"  # `sysctl -n hw.model` on the same Mac
M4_MAX = Hardware("Apple M4 Max", 40, "applegpu_g16s", 48, "26.6.2", "laptop", "Mac16,5")


def parse(info=INFO, ioreg=IOREG, batt=PMSET, macos="26.6.2", model=MODEL) -> Hardware:
    """`parse_hardware` on the M4 Max's reads, with any of them changed."""
    return macops.parse_hardware(info, ioreg, batt, macos, model)


def test_a_laptop() -> None:
    assert parse() == M4_MAX


@pytest.mark.parametrize("batt", [NO_BATTERY, UPS], ids=["no-battery", "ups"])
def test_a_desktop_is_one_on_a_ups_too(batt: str) -> None:
    assert parse(batt=batt) == M4_MAX._replace(form="desktop")


def test_ioreg_without_a_core_count_is_refused() -> None:
    lines = IOREG.splitlines(keepends=True)
    without = "".join(line for line in lines if "gpu-core-count" not in line)
    assert without != IOREG
    with pytest.raises(ConfigError, match=r"^cannot read this Mac's gpu_cores$"):
        parse(ioreg=without)


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"info": {**INFO, "device_name": ""}}, "chip"),
        ({"ioreg": None}, "gpu_cores"),
        ({"info": {**INFO, "architecture": None}}, "gpu_arch"),
        # The CPU device's `device_info` has no memory_size.
        ({"info": {"device_name": "Apple M4 Max", "architecture": "applegpu_g16s"}}, "memory_gb"),
        ({"macos": ""}, "macos"),  # `platform.mac_ver()` off a Mac
        ({"batt": None}, "form"),
        ({"batt": "pmset: garbled\n"}, "form"),
        ({"model": None}, "model"),
    ],
    ids=["chip", "no-ioreg", "arch", "memory", "macos", "no-pmset", "garbled-pmset", "model"],
)
def test_a_field_that_cannot_be_read_is_refused(change, field: str) -> None:
    with pytest.raises(ConfigError, match=rf"^cannot read this Mac's {field}$"):
        parse(**change)


def test_memory_is_rounded_to_gib() -> None:
    assert parse(info={**INFO, "memory_size": 16 * 2**30}).memory_gb == 16
    assert parse(info={**INFO, "memory_size": 24 * 2**30 - 2**20}).memory_gb == 24


def test_the_model_stays_out_of_the_key_and_the_form_stays_in() -> None:
    assert M4_MAX._replace(model="Mac16,6").key() == M4_MAX.key()
    assert M4_MAX._replace(form="desktop").key() != M4_MAX.key()


# Classes made up from M4_MAX for the key alone, which reads neither gpu_arch nor model.
@pytest.mark.parametrize(
    ("hw", "key"),
    [
        (M4_MAX, "m4-max-40c-48gb-macos26-laptop"),
        (
            M4_MAX._replace(chip="Apple M2 Ultra", gpu_cores=76, memory_gb=192, form="desktop"),
            "m2-ultra-76c-192gb-macos26-desktop",
        ),
        (
            M4_MAX._replace(chip="Apple M1", gpu_cores=8, memory_gb=16, macos="15"),
            "m1-8c-16gb-macos15-laptop",
        ),
        (M4_MAX._replace(chip="Apple M4 Max (Virtual)"), "m4-max-virtual-40c-48gb-macos26-laptop"),
    ],
    ids=["m4-max", "m2-ultra", "m1", "punctuation"],
)
def test_the_keys_format(hw: Hardware, key: str) -> None:
    assert hw.key() == key


def test_describe_holds_every_field_and_the_class() -> None:
    assert M4_MAX.describe() == {
        "chip": "Apple M4 Max",
        "gpu_cores": 40,
        "gpu_arch": "applegpu_g16s",
        "memory_gb": 48,
        "macos": "26.6.2",
        "form": "laptop",
        "model": "Mac16,5",
        "class": "m4-max-40c-48gb-macos26-laptop",
    }


def test_off_a_mac_read_hardware_refuses_and_runs_nothing() -> None:
    asked: list[object] = []
    with pytest.raises(ConfigError, match=r"^the hardware class is read only on a Mac$"):
        macops.read_hardware(asked.append, mac=False)
    assert asked == []


def test_on_a_mac_read_hardware_reads_the_gpu_and_each_command(monkeypatch) -> None:
    gpu = object()
    devices: list[object] = []

    def device_info(device=None):
        devices.append(device)
        return INFO

    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(gpu=gpu, device_info=device_info))
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.6.2", ("", "", ""), "arm64"))
    output = {macops.IOREG_GPU: IOREG, macops.PMSET_BATT: PMSET, macops.HW_MODEL: MODEL}
    asked: list[tuple[str, ...]] = []

    def run(argv):
        asked.append(tuple(argv))
        return output[tuple(argv)]

    assert macops.read_hardware(run, mac=True) == M4_MAX
    assert devices == [gpu]  # the GPU's, whatever the default device
    assert sorted(asked) == sorted(output)


def test_read_conditions_gives_the_power_source_and_the_thermal_level(monkeypatch) -> None:
    monkeypatch.setattr(macops, "thermal_level", lambda: 3)
    asked: list[tuple[str, ...]] = []

    def run(argv):
        asked.append(tuple(argv))
        return PMSET

    assert macops.read_conditions(run) == {"power": macops.BATTERY, "thermal_level": 3}
    assert asked == [macops.PMSET_BATT]
    assert macops.read_conditions(lambda argv: NO_BATTERY)["power"] == macops.AC
    assert macops.read_conditions(lambda argv: None) == {"power": None, "thermal_level": 3}
