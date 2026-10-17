"""PreToolUse hook: block `git commit` when the commit could carry a secret.

Runs before the command, so `git add -A && git commit` has not staged anything yet: the hook
scans everything a commit could pick up (staged, unstaged and untracked, non-ignored files) and
blocks on a `.env` file, the value of any token from the environment or `.env`, or a string
shaped like an Anthropic key or a JWT. Exit code 2 blocks the tool call.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_NAMES = ("FINMIND_TOKEN", "ANTHROPIC_API_KEY")
PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)
MIN_SECRET_LENGTH = 8
COMMIT = re.compile(r"\bgit\b[^|;&]*\bcommit\b")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, errors="replace", check=False
    )
    return result.stdout


def _secret_values(root: Path) -> list[str]:
    values = {os.environ.get(name, "") for name in SECRET_NAMES}
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in SECRET_NAMES:
                values.add(value.strip().strip("'\""))
    return [v for v in values if len(v) >= MIN_SECRET_LENGTH]


def _candidate_text(root: Path) -> tuple[list[str], str]:
    """Files a commit could include, and the lines it would add."""
    names = set(_git(root, "diff", "--cached", "--name-only").split())
    names |= set(_git(root, "diff", "--name-only").split())
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").split()
    names |= set(untracked)
    added = [
        line[1:]
        for line in (_git(root, "diff", "--cached", "-U0") + _git(root, "diff", "-U0")).splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    for name in untracked:
        path = root / name
        try:
            added.append(path.read_text())
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a text secret
    return sorted(names), "\n".join(added)


def problems(root: Path) -> list[str]:
    names, text = _candidate_text(root)
    found = [f"{n}: .env files must never be committed" for n in names if Path(n).name == ".env"]
    found += [
        f"the value of a secret ({len(v)} chars) appears in the changes"
        for v in _secret_values(root)
        if v in text
    ]
    found += [
        f"a string shaped like a key or token matches /{p.pattern}/"
        for p in PATTERNS
        if p.search(text)
    ]
    return found


def main() -> int:
    event = json.load(sys.stdin)
    command = event.get("tool_input", {}).get("command", "")
    if not COMMIT.search(command):
        return 0
    root = Path(event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    found = problems(root)
    if not found:
        return 0
    print("Commit blocked: possible secret in the changes.", file=sys.stderr)
    for item in found:
        print(f"- {item}", file=sys.stderr)
    print("Keep tokens in .env only (see CLAUDE.md).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
