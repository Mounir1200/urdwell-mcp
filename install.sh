#!/bin/sh
# UrdWell installer for macOS / Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Mounir1200/urdwell-mcp/main/install.sh | sh
#
# Installs uv (which manages its own Python runtime) if needed, then installs
# the `urdwell` command as an isolated tool. Override the source with
# URDWELL_PACKAGE (for example "urdwell" once on PyPI).

set -e

PACKAGE="${URDWELL_PACKAGE:-${CONTEXTMEMORY_PACKAGE:-git+https://github.com/Mounir1200/urdwell-mcp}}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Python toolchain manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing urdwell from $PACKAGE ..."
uv tool install --force "$PACKAGE"
uv tool update-shell 2>/dev/null || true

echo
echo "Done. Next steps:"
echo "  1. Open a NEW terminal so 'urdwell' resolves on PATH."
echo "  2. Run: urdwell install    (wires it into your agents)"
