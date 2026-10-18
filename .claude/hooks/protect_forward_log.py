"""PreToolUse hook: append-only validation records must never be modified.

The folders: the forward log (prospective tracking, spec §14.5), the expert records
(成長 × 風險 §3) and the monthly reviews (§3.1). Changing an entry after the fact would defeat
them. New files may be created with the Write tool. Shell commands that write, move or delete
with a path in one of them as a target are blocked (conservatively, even for new files);
heredoc bodies are ignored because they are data, not targets. Exit code 2 blocks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

FOLDERS = tuple(Path("validation") / name for name in ("forward_log", "expert_forms", "reviews"))
NAMES = "|".join(f.name for f in FOLDERS)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.DOTALL)
SEGMENT = re.compile(r"&&|\|\||[;|\n]")
# The folder itself: "forward_log/..." or a bare "forward_log", not "forward_log.py".
DIRECTORY = re.compile(rf"(?:{NAMES})(?=/|\s|$|['\"])")
REDIRECT = re.compile(rf">>?\s*['\"]?\S*(?:{NAMES})(?=/|\s|$|['\"])")
WRITERS = re.compile(r"^\s*(sudo\s+)?(rm|mv|cp|tee|truncate|sed\s+-i|perl\s+-i|git\s+(rm|mv))\b")


def _inside(path: Path, root: Path) -> Path | None:
    resolved = (path if path.is_absolute() else root / path).resolve()
    for folder in FOLDERS:
        absolute = (root / folder).resolve()
        if resolved == absolute or absolute in resolved.parents:
            return folder
    return None


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
        folder = _inside(path, root) if raw else None
        if folder is not None and path.exists():
            return f"{raw} already exists in {folder}/ and must not be modified"
        return None
    if tool == "Bash" and shell_writes_forward_log(data.get("command", "")):
        names = ", ".join(f"{f}/" for f in FOLDERS)
        return f"shell command writes, moves or deletes files in an append-only folder ({names})"
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
