"""PostToolUse hook: after a Python file is edited, run `ruff format` on it and the tests that
import it (or the test file itself). Failing tests exit 2 so the output reaches Claude.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TAIL_LINES = 40


def module_name(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to((root / "src").resolve())
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def related_tests(path: Path, root: Path) -> list[Path]:
    tests = root / "tests"
    resolved = path.resolve()
    if tests.resolve() in resolved.parents:
        return [resolved] if resolved.name.startswith("test_") else []
    module = module_name(path, root)
    if module is None or not tests.is_dir():
        return []
    return sorted(t for t in tests.rglob("test_*.py") if module in t.read_text(errors="replace"))


def main() -> int:
    event = json.load(sys.stdin)
    raw = event.get("tool_input", {}).get("file_path", "")
    root = Path(event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    path = Path(raw) if Path(raw).is_absolute() else root / raw
    if path.suffix != ".py" or not path.is_file():
        return 0
    subprocess.run(["uv", "run", "-q", "ruff", "format", "-q", str(path)], cwd=root, check=False)
    tests = related_tests(path, root)
    if not tests:
        return 0
    result = subprocess.run(
        [
            "uv",
            "run",
            "-q",
            "pytest",
            "-q",
            "-x",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            *map(str, tests),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return 0
    tail = (result.stdout + result.stderr).splitlines()[-TAIL_LINES:]
    print(f"Related tests failed after editing {raw}:", file=sys.stderr)
    print("\n".join(tail), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
