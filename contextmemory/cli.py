"""Command-line interface for ContextMemory.

``contextmemory`` with no command (or ``contextmemory serve``) runs the MCP
server over stdio, which is what agents invoke. The other commands manage the
installation: wiring the server into coding agents and upgrading the tool.

Heavy dependencies (the MCP server, the embedding backend) are imported lazily
inside each command so that management commands such as ``install`` stay fast
and do not load the embedding model.
"""

import argparse
import shutil
import subprocess
import sys

from contextmemory import __version__


def _run_serve() -> None:
    from contextmemory.server import serve

    serve()


def _run_install() -> None:
    from contextmemory import integrations

    integrations.install()


def _run_uninstall() -> None:
    from contextmemory import integrations

    integrations.uninstall()


def _run_upgrade() -> None:
    """Update the installed tool in place using uv, which manages the runtime."""
    uv = shutil.which("uv")
    if uv is None:
        print(
            "uv was not found on PATH. Reinstall with the same method you used, "
            "for example: uv tool upgrade contextmemory",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise SystemExit(subprocess.call([uv, "tool", "upgrade", "contextmemory"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextmemory",
        description="ContextMemory: a long-term memory MCP server for LLM agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"contextmemory {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser(
        "serve",
        help="Run the MCP server over stdio (default when no command is given).",
    )
    subcommands.add_parser(
        "install",
        help="Detect installed coding agents and wire ContextMemory into each.",
    )
    subcommands.add_parser(
        "uninstall",
        help="Remove ContextMemory from every agent it configured.",
    )
    subcommands.add_parser(
        "upgrade",
        help="Update the installed ContextMemory tool in place.",
    )
    subcommands.add_parser("version", help="Print the installed version.")
    return parser


_COMMANDS = {
    "serve": _run_serve,
    "install": _run_install,
    "uninstall": _run_uninstall,
    "upgrade": _run_upgrade,
    "version": lambda: print(__version__),
}


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    command = args.command or "serve"
    _COMMANDS[command]()


if __name__ == "__main__":
    main()
