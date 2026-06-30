"""Wire the UrdWell MCP server into coding agents, and remove it again.

`urdwell install` detects which agents are present on the machine and adds a
stdio server entry to each one's configuration; `urdwell uninstall`
removes exactly those entries. Both commands are idempotent and never overwrite
unrelated configuration.

Agents are described declaratively in ``REGISTRY``. Adding support for a new
agent is one new ``Agent`` entry plus, if its file format is new, one writer
pair — existing agents are never modified (open/closed principle). Three config
formats are covered:

- standard JSON with a top-level ``mcpServers`` map (Claude Desktop, Cursor,
  Windsurf, Gemini CLI, Kiro);
- the opencode JSON schema (``mcp.<name>`` with ``type: "local"``);
- the Codex CLI TOML file (``[mcp_servers.<name>]``).

Claude Code is configured through its own ``claude mcp`` CLI when available.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The installed console command and the subcommand that starts the stdio server.
SERVER_NAME = "urdwell"
LEGACY_SERVER_NAMES = ("contextmemory",)
_COMMAND = "urdwell"
_ARGS = ["serve"]


def _args_for(agent_key: str | None) -> list[str]:
    """Server args for one agent, tagging its writes with their origin.

    ``urdwell install`` already knows each agent's key, so the identity is
    injected here at wiring time rather than trusted from the model at run time.
    """
    if agent_key is None:
        return list(_ARGS)
    return [*_ARGS, "--agent", agent_key]


class _SkipAgent(Exception):
    """Raised when an agent is present but cannot be modified safely."""


# ---------- Filesystem location helpers ----------


def _home() -> Path:
    return Path.home()


def _config_home() -> Path:
    """XDG config root (``~/.config`` unless ``XDG_CONFIG_HOME`` overrides it)."""
    override = os.environ.get("XDG_CONFIG_HOME")
    return Path(override) if override else _home() / ".config"


def _claude_desktop_dir() -> Path | None:
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "Claude" if appdata else None
    if system == "Darwin":
        return _home() / "Library" / "Application Support" / "Claude"
    return _config_home() / "Claude"


def _claude_desktop_config() -> Path | None:
    directory = _claude_desktop_dir()
    return directory / "claude_desktop_config.json" if directory else None


# ---------- JSON config I/O ----------


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise _SkipAgent(
            f"{path} is not plain JSON (comments?); left untouched"
        )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------- Writer 1: standard ``mcpServers`` JSON ----------


def _configure_mcpservers(path: Path, agent_key: str | None = None) -> None:
    data = _load_json(path)
    servers = data.setdefault("mcpServers", {})
    for legacy_name in LEGACY_SERVER_NAMES:
        servers.pop(legacy_name, None)
    servers[SERVER_NAME] = {"command": _COMMAND, "args": _args_for(agent_key)}
    _write_json(path, data)


def _unconfigure_mcpservers(path: Path) -> bool:
    if not path.exists():
        return False
    data = _load_json(path)
    servers = data.get("mcpServers")
    removed = False
    if isinstance(servers, dict):
        for name in (SERVER_NAME, *LEGACY_SERVER_NAMES):
            removed = servers.pop(name, None) is not None or removed
    if removed:
        _write_json(path, data)
        return True
    return False


# ---------- Writer 2: opencode JSON schema ----------


def _configure_opencode(path: Path, agent_key: str | None = None) -> None:
    data = _load_json(path)
    data.setdefault("$schema", "https://opencode.ai/config.json")
    servers = data.setdefault("mcp", {})
    for legacy_name in LEGACY_SERVER_NAMES:
        servers.pop(legacy_name, None)
    servers[SERVER_NAME] = {
        "type": "local",
        "command": [_COMMAND, *_args_for(agent_key)],
        "enabled": True,
    }
    _write_json(path, data)


def _unconfigure_opencode(path: Path) -> bool:
    if not path.exists():
        return False
    data = _load_json(path)
    servers = data.get("mcp")
    removed = False
    if isinstance(servers, dict):
        for name in (SERVER_NAME, *LEGACY_SERVER_NAMES):
            removed = servers.pop(name, None) is not None or removed
    if removed:
        _write_json(path, data)
        return True
    return False


# ---------- Writer 3: Codex CLI TOML ----------
# Edited as text so the user's hand-written config.toml keeps its comments and
# formatting; the standard library can read TOML but cannot write it.

_CODEX_SECTION = f"[mcp_servers.{SERVER_NAME}]"
_LEGACY_CODEX_SECTIONS = {
    f"[mcp_servers.{name}]" for name in LEGACY_SERVER_NAMES
}


def _codex_block(agent_key: str | None) -> str:
    """Render the Codex TOML table. A JSON string array is also valid TOML."""
    args = json.dumps(_args_for(agent_key))
    return f'\n{_CODEX_SECTION}\ncommand = "{_COMMAND}"\nargs = {args}\n'


def _remove_codex_sections(text: str, sections: set[str]) -> tuple[str, bool]:
    """Remove selected TOML tables without reformatting unrelated content."""
    kept: list[str] = []
    in_section = False
    removed = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_section and stripped in sections:
            in_section = True
            removed = True
            continue
        if in_section:
            if stripped.startswith("[") and stripped.endswith("]"):
                if stripped in sections:
                    removed = True
                    continue
                in_section = False
                kept.append(line)
            continue
        kept.append(line)

    cleaned = "".join(kept).rstrip("\n")
    return (cleaned + "\n" if cleaned else "", removed)


def _configure_codex(path: Path, agent_key: str | None = None) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    text, migrated = _remove_codex_sections(text, _LEGACY_CODEX_SECTIONS)
    if _CODEX_SECTION in text:
        if migrated:
            path.write_text(text, encoding="utf-8")
        return  # already present; keep the file as the user left it
    if text and not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + _codex_block(agent_key), encoding="utf-8")


def _unconfigure_codex(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    sections = {_CODEX_SECTION, *_LEGACY_CODEX_SECTIONS}
    cleaned, removed = _remove_codex_sections(text, sections)
    if not removed:
        return False
    path.write_text(cleaned, encoding="utf-8")
    return True


# ---------- Writer 4: Claude Code via its own CLI ----------


def _claude_code_present() -> bool:
    return shutil.which("claude") is not None


def _configure_claude_code() -> str:
    claude = shutil.which("claude")
    if claude is None:
        raise _SkipAgent("claude CLI not found on PATH")
    # Remove new and legacy names so re-running also performs the 0.3 migration.
    for name in (SERVER_NAME, *LEGACY_SERVER_NAMES):
        subprocess.run(
            [claude, "mcp", "remove", "--scope", "user", name],
            check=False,
            capture_output=True,
            text=True,
        )
    result = subprocess.run(
        [
            claude, "mcp", "add", "--scope", "user", SERVER_NAME,
            "--", _COMMAND, *_args_for("claude-code"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise _SkipAgent((result.stderr or result.stdout).strip() or "claude mcp add failed")
    return "user scope (claude CLI)"


def _unconfigure_claude_code() -> bool:
    claude = shutil.which("claude")
    if claude is None:
        return False
    removed = False
    for name in (SERVER_NAME, *LEGACY_SERVER_NAMES):
        result = subprocess.run(
            [claude, "mcp", "remove", "--scope", "user", name],
            check=False,
            capture_output=True,
            text=True,
        )
        removed = result.returncode == 0 or removed
    return removed


# ---------- Agent registry ----------


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    detect: Callable[[], bool]
    configure: Callable[[], str]
    unconfigure: Callable[[], bool]


def _file_agent(
    key: str,
    name: str,
    marker: Callable[[], Path | None],
    config_path: Callable[[], Path | None],
    configure_writer: Callable[[Path, str], None],
    unconfigure_writer: Callable[[Path], bool],
) -> Agent:
    """Build an Agent backed by a config file, detected by a marker directory."""

    def detect() -> bool:
        location = marker()
        return location is not None and location.exists()

    def configure() -> str:
        path = config_path()
        if path is None:
            raise _SkipAgent("no config location on this platform")
        configure_writer(path, key)
        return str(path)

    def unconfigure() -> bool:
        path = config_path()
        return unconfigure_writer(path) if path is not None else False

    return Agent(key, name, detect, configure, unconfigure)


REGISTRY: list[Agent] = [
    Agent(
        "claude-code",
        "Claude Code",
        _claude_code_present,
        _configure_claude_code,
        _unconfigure_claude_code,
    ),
    _file_agent(
        "claude-desktop",
        "Claude Desktop",
        _claude_desktop_dir,
        _claude_desktop_config,
        _configure_mcpservers,
        _unconfigure_mcpservers,
    ),
    _file_agent(
        "cursor",
        "Cursor",
        lambda: _home() / ".cursor",
        lambda: _home() / ".cursor" / "mcp.json",
        _configure_mcpservers,
        _unconfigure_mcpservers,
    ),
    _file_agent(
        "windsurf",
        "Windsurf",
        lambda: _home() / ".codeium" / "windsurf",
        lambda: _home() / ".codeium" / "windsurf" / "mcp_config.json",
        _configure_mcpservers,
        _unconfigure_mcpservers,
    ),
    _file_agent(
        "gemini",
        "Gemini CLI",
        lambda: _home() / ".gemini",
        lambda: _home() / ".gemini" / "settings.json",
        _configure_mcpservers,
        _unconfigure_mcpservers,
    ),
    _file_agent(
        "kiro",
        "Kiro",
        lambda: _home() / ".kiro",
        lambda: _home() / ".kiro" / "settings" / "mcp.json",
        _configure_mcpservers,
        _unconfigure_mcpservers,
    ),
    _file_agent(
        "codex",
        "Codex CLI",
        lambda: _home() / ".codex",
        lambda: _home() / ".codex" / "config.toml",
        _configure_codex,
        _unconfigure_codex,
    ),
    _file_agent(
        "opencode",
        "opencode",
        lambda: _config_home() / "opencode",
        lambda: _config_home() / "opencode" / "opencode.json",
        _configure_opencode,
        _unconfigure_opencode,
    ),
]


# ---------- Orchestration ----------


def _report(title: str, items: list[str]) -> None:
    if items:
        print(f"\n{title}:")
        for item in items:
            print(f"  - {item}")


def install() -> None:
    """Detect installed agents and add the UrdWell server to each."""
    configured: list[str] = []
    skipped: list[str] = []
    for agent in REGISTRY:
        if not agent.detect():
            continue
        # One misconfigured agent must not abort the others, so failures are
        # isolated and reported rather than raised.
        try:
            target = agent.configure()
            configured.append(f"{agent.name} -> {target}")
        except _SkipAgent as exc:
            skipped.append(f"{agent.name}: {exc}")
        except OSError as exc:
            skipped.append(f"{agent.name}: {exc}")

    if not configured and not skipped:
        print("No supported agents detected; nothing to configure.")
        return
    _report("Configured", configured)
    _report("Skipped", skipped)
    print(
        "\nOpen a new terminal or restart the agent for the change to take effect."
    )


def uninstall() -> None:
    """Remove UrdWell and its legacy name from every configured agent."""
    removed: list[str] = []
    errors: list[str] = []
    for agent in REGISTRY:
        try:
            if agent.unconfigure():
                removed.append(agent.name)
        except (_SkipAgent, OSError) as exc:
            errors.append(f"{agent.name}: {exc}")

    if not removed and not errors:
        print("UrdWell was not configured in any detected agent.")
        return
    _report("Removed", removed)
    _report("Errors", errors)


if __name__ == "__main__":
    sys.exit(install())
