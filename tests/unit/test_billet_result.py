"""The Billet result in results/billet/: its pages quote the report, and the report is clean.

Each page quotes the report's primary readout: the held-out difference in bits per byte and
its 95% interval. The report must match what `SCRIPT` writes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.docs  # it reads docs/billet.md

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "billet" / "report.json"
SCRIPT = ROOT / "scripts" / "analysis" / "analysis.py"
SCRIPT_SHA256 = "7876fbb31c28c6b5a082284a3a2693f57acc30476a70b4f5eedc8372617ab85b"
PAGES = ("README.md", "results/billet/README.md", "docs/billet.md")


def page_problems(text: str, report: dict[str, Any]) -> list[str]:
    """Return what is wrong with a page's quote of the report's primary readout."""
    text = text.replace(chr(0x2212), "-")  # a typeset minus sign reads as "-"
    primary = report["primary"]
    problems = []
    delta = format(primary["delta"], "+.4f") + " bits per byte"
    if delta not in text:
        problems.append(f"the page does not say {delta!r} (primary.delta)")
    low, high = (format(bound, "+.4f") for bound in primary["ci95"])
    if low not in text or high not in text:
        problems.append(f"the page does not say {low!r} and {high!r} (primary.ci95)")
    if primary["band"] not in text:
        problems.append(f"the page does not say {primary['band']!r} (primary.band)")
    return problems


def _report() -> dict[str, Any]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", PAGES)
def test_page_quotes_the_report(page: str) -> None:
    assert page_problems((ROOT / page).read_text(encoding="utf-8"), _report()) == []


@pytest.mark.parametrize("page", PAGES)
def test_a_changed_report_is_caught(page: str) -> None:
    changed = copy.deepcopy(_report())
    changed["primary"]["delta"] += 0.5
    changed["primary"]["ci95"] = [bound + 0.5 for bound in changed["primary"]["ci95"]]
    changed["primary"]["band"] += " (changed)"
    problems = page_problems((ROOT / page).read_text(encoding="utf-8"), changed)
    assert len(problems) == len(("delta", "ci95", "band"))


def test_report_names_the_script_it_came_from() -> None:
    script = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert _report()["analysis sha256"] == script == SCRIPT_SHA256


def test_report_has_no_local_paths() -> None:
    assert not [name for name in _report()["inputs"] if name.startswith("/")]
    assert "/Users/" not in REPORT.read_text(encoding="utf-8")
