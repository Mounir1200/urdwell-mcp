param(
    [ValidateSet("oracle", "s")]
    [string]$Dataset = "s",

    [ValidateSet("context-memory", "raw-history", "no-memory")]
    [string]$System = "context-memory",

    [ValidateSet("llm", "verbatim")]
    [string]$IngestionMode = "verbatim",

    [string]$RunName = "local-smoke-3",

    [int]$Limit = 3,

    [string]$Model = "",

    [ValidateSet("cosine", "hybrid")]
    [string]$Ranking = "cosine",

    [string]$ReuseStoresFrom = "",

    [int]$TopK = 5,

    [double]$Threshold = 0,

    [switch]$Resume,

    [switch]$VerboseOutput
)

$projectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_end_to_end.py"
$datasetPath = if ($Dataset -eq "s") {
    Join-Path $PSScriptRoot "longmemeval_s_cleaned.json"
} else {
    Join-Path $PSScriptRoot "longmemeval_oracle.json"
}
$selectedModel = if ($Model) {
    $Model
} elseif ($System -eq "raw-history") {
    "gemma4-8b-128k"
} else {
    "gemma4:26b"
}

if (-not (Test-Path $python)) {
    throw "Virtual environment not found: $python"
}
if (-not (Test-Path $datasetPath)) {
    throw "Dataset not found: $datasetPath"
}

$arguments = @(
    $runner,
    "--dataset", $datasetPath,
    "--system", $System,
    "--ingestion-mode", $IngestionMode,
    "--run-name", $RunName,
    "--reader-model", $selectedModel,
    "--judge-model", $selectedModel,
    "--base-url", "http://127.0.0.1:11434/v1",
    "--reasoning-effort", "none",
    "--ranking", $Ranking,
    "--top-k", $TopK,
    "--threshold", $Threshold
)

if ($ReuseStoresFrom) {
    $arguments += @("--reuse-stores-from", $ReuseStoresFrom)
}
if ($Limit -gt 0) {
    $arguments += @("--limit", $Limit)
}
if ($Resume) {
    $arguments += "--resume"
}
if ($VerboseOutput) {
    $arguments += "--verbose"
}

& $python @arguments
exit $LASTEXITCODE
