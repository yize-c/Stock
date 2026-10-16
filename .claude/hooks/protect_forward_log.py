"""PreToolUse hook: files already in validation/forward_log/ must never be modified.

The forward log is the prospective-tracking record (spec §14.5); changing an entry after the
fact would defeat it. New files may be created with the Write tool. Shell commands that write,
move or delete with a forward_log path as a target are blocked (conservatively, even for new
files); heredoc bodies are ignored because they are data, not targets. Exit code 2 blocks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

FOLDER = Path("validation") / "forward_log"
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.DOTALL)
SEGMENT = re.compile(r"&&|\|\||[;|\n]")
# The folder itself: "forward_log/..." or a bare "forward_log", not "forward_log.py".
DIRECTORY = re.compile(r"forward_log(?=/|\s|$|['\"])")
REDIRECT = re.compile(r">>?\s*['\"]?\S*forward_log(?=/|\s|$|['\"])")
WRITERS = re.compile(r"^\s*(sudo\s+)?(rm|mv|cp|tee|truncate|sed\s+-i|perl\s+-i|git\s+(rm|mv))\b")


def _inside(path: Path, root: Path) -> bool:
    folder = (root / FOLDER).resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    return resolved == folder or folder in resolved.parents


def shell_writes_forward_log(command: str) -> bool:
    for segment in SEGMENT.split(HEREDOC.sub("", command)):
        if not DIRECTORY.search(segment):
            continue
        if REDIRECT.search(segment) or WRITERS.search(segment):
            return True
    return False


def reason(event: dict, root: Path) -> str | None:
    tool, data = event.get("tool_name", ""), event.get("tool_input", {})
    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        raw = data.get("file_path") or data.get("notebook_path") or ""
        path = Path(raw) if Path(raw).is_absolute() else root / raw
        if raw and _inside(path, root) and path.exists():
            return f"{raw} already exists in {FOLDER}/ and must not be modified"
        return None
    if tool == "Bash" and shell_writes_forward_log(data.get("command", "")):
        return f"shell command writes, moves or deletes files in {FOLDER}/"
    return None


def main() -> int:
    event = json.load(sys.stdin)
    root = Path(event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ".")
    found = reason(event, root)
    if found is None:
        return 0
    print(f"Blocked: {found}. Add a new dated file with the Write tool instead.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
