"""Command-line interface for UrdWell.

``urdwell`` with no command (or ``urdwell serve``) runs the MCP
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

from urdwell import __version__


def _run_serve(agent: str | None = None) -> None:
    from urdwell.server import serve

    serve(agent=agent)


def _run_install() -> None:
    from urdwell import integrations

    integrations.install()


def _run_uninstall() -> None:
    from urdwell import integrations

    integrations.uninstall()


def _run_upgrade() -> None:
    """Update the installed tool in place using uv, which manages the runtime."""
    uv = shutil.which("uv")
    if uv is None:
        print(
            "uv was not found on PATH. Reinstall with the same method you used, "
            "for example: uv tool upgrade urdwell",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise SystemExit(subprocess.call([uv, "tool", "upgrade", "urdwell"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urdwell",
        description="UrdWell: durable, evolving memory for AI agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"urdwell {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command")
    serve_parser = subcommands.add_parser(
        "serve",
        help="Run the MCP server over stdio (default when no command is given).",
    )
    serve_parser.add_argument(
        "--agent",
        default=None,
        help="Identifier of the agent this server is wired into; "
        "recorded as the provenance of every memory it writes. "
        "Set automatically by `urdwell install`.",
    )
    subcommands.add_parser(
        "install",
        help="Detect installed coding agents and wire UrdWell into each.",
    )
    subcommands.add_parser(
        "uninstall",
        help="Remove UrdWell from every agent it configured.",
    )
    subcommands.add_parser(
        "upgrade",
        aliases=["update"],
        help="Update the installed UrdWell tool in place.",
    )
    subcommands.add_parser("version", help="Print the installed version.")
    return parser


_COMMANDS = {
    "serve": _run_serve,
    "install": _run_install,
    "uninstall": _run_uninstall,
    "upgrade": _run_upgrade,
    "update": _run_upgrade,
    "version": lambda: print(__version__),
}


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    command = args.command or "serve"
    if command == "serve":
        _run_serve(agent=getattr(args, "agent", None))
        return
    _COMMANDS[command]()


if __name__ == "__main__":
    main()
