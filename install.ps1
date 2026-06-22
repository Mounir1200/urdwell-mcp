# UrdWell installer for Windows PowerShell.
#
#   irm https://raw.githubusercontent.com/Mounir1200/urdwell-mcp/main/install.ps1 | iex
#
# Installs uv (which manages its own Python runtime) if needed, then installs
# the `urdwell` command as an isolated tool. Override the source with
# $env:URDWELL_PACKAGE (for example "urdwell" once on PyPI).

$ErrorActionPreference = "Stop"

$Package = if ($env:URDWELL_PACKAGE) {
    $env:URDWELL_PACKAGE
} elseif ($env:CONTEXTMEMORY_PACKAGE) {
    $env:CONTEXTMEMORY_PACKAGE
} else {
    "git+https://github.com/Mounir1200/urdwell-mcp"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python toolchain manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing urdwell from $Package ..."
uv tool install --force $Package
uv tool update-shell

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Open a NEW terminal so 'urdwell' resolves on PATH."
Write-Host "  2. Run: urdwell install    (wires it into your agents)"
