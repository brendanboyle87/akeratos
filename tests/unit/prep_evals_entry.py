"""The eval prep script's test-only entry point, with a fake Hub.

Its first two arguments are SCRIPT and a WORLD JSON file. It loads SCRIPT
(`scripts/data/prepare_evals.py`) in a fresh process, as the script is started on the Mac,
replaces its Hub client with `FakeHub` over WORLD, and calls its `main` with the remaining
arguments. WORLD maps each repository to `info` (revision -> the dict `HubClient.info` in
`scripts/data/prepare_store.py` returns), `tree` (the recursive listing) and `files`
(repository path -> the local file served for it). Every call is appended to WORLD's `log` file.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class FakeHub:
    """Answers `info`, `tree` and `download` from WORLD, and never opens a socket."""

    def __init__(self, world: dict[str, Any], prep: ModuleType) -> None:
        self.w, self.prep = world, prep

    def _note(self, *call: object) -> None:
        with Path(self.w["log"]).open("a") as f:
            f.write(json.dumps(call) + "\n")

    def _repo(self, repo_id: str) -> dict[str, Any]:
        if repo_id not in self.w["repos"]:
            raise self.prep.base.Missing(f"{repo_id} not found (fake)")
        return self.w["repos"][repo_id]

    def info(self, repo_id: str, repo_type: str, revision: str) -> dict[str, Any]:
        self._note("info", repo_id, revision)
        r = self._repo(repo_id)
        if revision not in r["info"]:
            raise self.prep.base.Missing(f"{repo_id}@{revision} not found (fake)")
        return dict(r["info"][revision])

    def tree(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        path: str | None = None,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        self._note("tree", repo_id, revision)
        return [dict(e) for e in self._repo(repo_id)["tree"]]

    def download(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        filename: str,
        local_dir: str,
        cache_dir: str,
    ) -> str:
        self._note("download", repo_id, revision, filename)
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.chmod(0o644)
            dst.unlink()
        shutil.copyfile(self._repo(repo_id)["files"][filename], dst)
        return str(dst)


def main() -> int:
    """Load the script, give it the fake Hub and run its `main`."""
    world = json.loads(Path(sys.argv[2]).read_text())
    spec = importlib.util.spec_from_file_location("prep_evals", sys.argv[1])
    assert spec is not None
    assert spec.loader is not None
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.__dict__["make_hub"] = lambda: FakeHub(world, prep)
    return int(prep.main(sys.argv[3:]))


if __name__ == "__main__":
    sys.exit(main())
