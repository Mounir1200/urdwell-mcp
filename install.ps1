# ContextMemory installer for Windows PowerShell.
#
#   irm https://raw.githubusercontent.com/Mounir1200/contextmemory-mcp/main/install.ps1 | iex
#
# Installs uv (which manages its own Python runtime) if needed, then installs
# the `contextmemory` command as an isolated tool. Override the source with
# $env:CONTEXTMEMORY_PACKAGE (for example "contextmemory" once on PyPI).

$ErrorActionPreference = "Stop"

$Package = if ($env:CONTEXTMEMORY_PACKAGE) {
    $env:CONTEXTMEMORY_PACKAGE
} else {
    "git+https://github.com/Mounir1200/contextmemory-mcp"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python toolchain manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing contextmemory from $Package ..."
uv tool install --force $Package
uv tool update-shell

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Open a NEW terminal so 'contextmemory' resolves on PATH."
Write-Host "  2. Run: contextmemory install    (wires it into your agents)"
