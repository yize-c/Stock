"""Holdout guard (spec §14.4 step 4).

The holdout phase may run only when ``preregistration.md``

* is committed and unmodified in the working tree,
* says ``Status: FINAL`` in the committed version,
* states ``Config SHA-256: <hash>`` matching the committed, unmodified config file,
* names the selected development attempt (``選定的調參嘗試編號：<n>``; the holdout result is
  compared with it for the overfitting warning, spec v2 P1-9),

and only once: the first holdout run writes ``results/holdout_lock.json`` before computing
anything, and any later run refuses while that file exists (removing it shows in git).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FILE = "preregistration.md"
LOCK = Path("results") / "holdout_lock.json"
_FINAL = re.compile(r"^Status:\s*FINAL\s*$", re.MULTILINE)
_CONFIG_HASH = re.compile(r"^Config SHA-256:\s*`?([0-9a-f]{64})`?\s*$", re.MULTILINE)
_SELECTED = re.compile(r"^-?\s*選定的調參嘗試編號：\s*(\d+)\s*$", re.MULTILINE)


class HoldoutBlocked(Exception):
    """The holdout phase must not run (yet, or again)."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(root: Path = Path()) -> tuple[str | None, bool]:
    """(HEAD commit, working tree dirty?); (None, True) outside a git repository."""
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        return None, True
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    return head.stdout.strip(), dirty


def _committed_and_clean(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root)
    if _git(root, "ls-files", "--error-unmatch", str(relative)).returncode != 0:
        raise HoldoutBlocked(f"{relative} is not committed")
    if _git(root, "status", "--porcelain", "--", str(relative)).stdout.strip():
        raise HoldoutBlocked(f"{relative} has uncommitted changes")
    shown = _git(root, "show", f"HEAD:{relative.as_posix()}")
    if shown.returncode != 0:
        raise HoldoutBlocked(f"{relative} is not in HEAD")
    return shown.stdout


@dataclass(frozen=True)
class Registration:
    commit: str
    config_sha256: str
    selected_attempt: int  # the development attempt the final parameters come from


def check_holdout_allowed(directory: Path, config_path: Path) -> Registration:
    top = _git(directory if directory.exists() else Path(), "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise HoldoutBlocked("not inside a git repository: preregistration cannot be checked")
    root = Path(top.stdout.strip()).resolve()
    if (directory / LOCK).exists():
        raise HoldoutBlocked(
            f"the holdout already ran ({directory / LOCK}); it may run only once (spec §14.4)"
        )
    prereg = directory / FILE
    if not prereg.exists():
        raise HoldoutBlocked(f"{prereg} does not exist")
    text = _committed_and_clean(root, prereg)
    if not _FINAL.search(text):
        raise HoldoutBlocked(f"{prereg} is not marked 'Status: FINAL' in HEAD")
    stated = _CONFIG_HASH.search(text)
    if stated is None:
        raise HoldoutBlocked(f"{prereg} does not state 'Config SHA-256: <hash>'")
    _committed_and_clean(root, config_path)
    actual = file_sha256(config_path)
    if actual != stated.group(1):
        raise HoldoutBlocked(
            f"{config_path} (sha256 {actual[:12]}…) differs from the preregistered config "
            f"({stated.group(1)[:12]}…)"
        )
    selected = _SELECTED.search(text)
    if selected is None:
        raise HoldoutBlocked(f"{prereg} does not state '選定的調參嘗試編號：<number>'")
    commit = _git(root, "log", "-1", "--format=%H", "--", str(prereg.resolve().relative_to(root)))
    return Registration(commit.stdout.strip(), actual, int(selected.group(1)))


def acquire_holdout_lock(directory: Path, registration: Registration, now: datetime) -> Path:
    path = directory / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:  # never overwrite
        json.dump(
            {
                "started_at": now.isoformat(),
                "preregistration_commit": registration.commit,
                "config_sha256": registration.config_sha256,
                "selected_attempt": registration.selected_attempt,
            },
            handle,
            indent=2,
        )
    return path
