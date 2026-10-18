"""Tests for the Claude Code hooks in .claude/hooks (run as subprocesses, like Claude Code)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".claude" / "hooks"
FAKE_TOKEN = "fake-finmind-token-value-1234567890"
SECRETS = ("FINMIND_TOKEN", "ANTHROPIC_API_KEY")


def _run(hook: str, event: dict, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOKS / hook)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _load(hook: str):
    spec = importlib.util.spec_from_file_location(hook.removesuffix(".py"), HOOKS / hook)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(**extra: str) -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SECRETS} | extra


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return tmp_path


def _commit(root: Path, command: str = "git add -A && git commit -m x") -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(root)}


def test_settings_reference_existing_hooks():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    commands = [
        hook["command"]
        for groups in settings["hooks"].values()
        for group in groups
        for hook in group["hooks"]
    ]
    assert commands
    for command in commands:
        name = command.split("/.claude/hooks/")[1].rstrip('"')
        assert (HOOKS / name).is_file()
    assert "Bash(git push:*)" in settings["permissions"]["ask"]


def test_secret_guard_ignores_other_commands(repo):
    (repo / "a.py").write_text(f"x = '{FAKE_TOKEN}'\n")
    result = _run("guard_secrets.py", _commit(repo, "git status"), _env(FINMIND_TOKEN=FAKE_TOKEN))
    assert result.returncode == 0


def test_secret_guard_passes_clean_changes(repo):
    (repo / "a.py").write_text("x = 2\n")
    (repo / "b.py").write_text("y = 3\n")
    result = _run("guard_secrets.py", _commit(repo), _env(FINMIND_TOKEN=FAKE_TOKEN))
    assert result.returncode == 0


def test_secret_guard_blocks_token_value_from_environment(repo):
    (repo / "a.py").write_text(f"x = '{FAKE_TOKEN}'\n")
    result = _run("guard_secrets.py", _commit(repo), _env(FINMIND_TOKEN=FAKE_TOKEN))
    assert result.returncode == 2
    assert "secret" in result.stderr


def test_secret_guard_blocks_token_value_from_dotenv_in_untracked_file(repo):
    (repo / ".env").write_text(f'ANTHROPIC_API_KEY="{FAKE_TOKEN}"\n')
    (repo / "notes.txt").write_text(f"key {FAKE_TOKEN}\n")
    assert _run("guard_secrets.py", _commit(repo), _env()).returncode == 2


def test_secret_guard_blocks_key_shaped_strings(repo):
    (repo / "a.py").write_text("k = 'sk-ant-api03-" + "A" * 60 + "'\n")
    assert _run("guard_secrets.py", _commit(repo), _env()).returncode == 2


def test_secret_guard_blocks_force_added_dotenv(repo):
    (repo / ".env").write_text("FINMIND_TOKEN=\n")
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, check=True)
    result = _run("guard_secrets.py", _commit(repo, "git commit -m x"), _env())
    assert result.returncode == 2
    assert ".env" in result.stderr


def _edit(root: Path, path: str, tool: str = "Write") -> dict:
    return {"tool_name": tool, "tool_input": {"file_path": path}, "cwd": str(root)}


def test_forward_log_allows_new_files_and_blocks_existing(tmp_path):
    log = tmp_path / "validation" / "forward_log"
    log.mkdir(parents=True)
    (log / "2026-01-31.md").write_text("entry\n")
    new = _edit(tmp_path, "validation/forward_log/2026-02-28.md")
    assert _run("protect_forward_log.py", new).returncode == 0
    existing = _edit(tmp_path, str(log / "2026-01-31.md"), "Edit")
    result = _run("protect_forward_log.py", existing)
    assert result.returncode == 2
    assert "must not be modified" in result.stderr
    other = _edit(tmp_path, "validation/preregistration.md", "Edit")
    assert _run("protect_forward_log.py", other).returncode == 0


@pytest.mark.parametrize("folder", ["expert_forms", "reviews"])
def test_expert_records_and_reviews_are_append_only_too(tmp_path, folder):
    """成長 × 風險 §3: expert records and monthly reviews follow the forward log's rule."""
    records = tmp_path / "validation" / folder
    records.mkdir(parents=True)
    (records / "2026-10-17_record.yaml").write_text("x: 1\n")
    new = _edit(tmp_path, f"validation/{folder}/2026-10-18_follow_up.md")
    assert _run("protect_forward_log.py", new).returncode == 0
    existing = _edit(tmp_path, f"validation/{folder}/2026-10-17_record.yaml", "Edit")
    assert _run("protect_forward_log.py", existing).returncode == 2
    shell = {
        "tool_name": "Bash",
        "tool_input": {"command": f"rm validation/{folder}/2026-10-17_record.yaml"},
        "cwd": str(tmp_path),
    }
    assert _run("protect_forward_log.py", shell).returncode == 2


@pytest.mark.parametrize(
    ("command", "code"),
    [
        ("ls validation/forward_log", 0),
        ("cat validation/forward_log/a.md | wc -l", 0),
        ("git log --follow validation/forward_log/a.md", 0),
        ("echo x >> validation/forward_log/a.md", 2),
        ("echo x > other.md && rm validation/forward_log/a.md", 2),
        ("sed -i s/a/b/ validation/forward_log/a.md", 2),
        ("mv validation/forward_log/a.md /tmp/", 2),
        ("echo x > other.md", 0),
        ("cat > src/pkg/forward_log.py <<'EOF'\nx = 1\nEOF", 0),
        ("rm validation/forward_log", 2),
        ("cat > t.py <<'EOF'\nrun('echo x >> validation/forward_log/a.md')\nEOF\npytest", 0),
    ],
)
def test_forward_log_shell_commands(tmp_path, command, code):
    event = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(tmp_path)}
    assert _run("protect_forward_log.py", event).returncode == code


def test_format_hook_finds_related_tests():
    hook = _load("format_and_test.py")
    source = ROOT / "src" / "stock_research" / "analysis" / "weighting.py"
    assert hook.module_name(source, ROOT) == "stock_research.analysis.weighting"
    assert ROOT / "tests" / "test_weighting.py" in hook.related_tests(source, ROOT)
    test_file = ROOT / "tests" / "test_risk.py"
    assert hook.related_tests(test_file, ROOT) == [test_file]
    assert hook.module_name(ROOT / "README.md", ROOT) is None
    init = ROOT / "src" / "stock_research" / "__init__.py"
    assert hook.module_name(init, ROOT) == "stock_research"


def test_format_hook_skips_non_python(tmp_path):
    (tmp_path / "notes.md").write_text("x\n")
    assert _run("format_and_test.py", _edit(tmp_path, "notes.md")).returncode == 0
