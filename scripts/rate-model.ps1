param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string[]]$TrainingData = @(),
    [string]$BenchmarkRoot = 'runs/cardplay-rating-v1',
    [int]$Port = 18765
)

$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repoDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Python environment missing: $pythonExecutable"
}

Push-Location -LiteralPath $repoDirectory
try {
    $registerArguments = @('-m', 'bridgemind.rating', '--root', $BenchmarkRoot, '--port', "$Port",
        'register', '--id', $Id, '--kind', 'neural', '--checkpoint', $Checkpoint)
    foreach ($trainingDirectory in $TrainingData) {
        $registerArguments += @('--training-data', $trainingDirectory)
    }
    & $pythonExecutable @registerArguments
    if ($LASTEXITCODE -ne 0) { throw 'Model registration failed' }
    & $pythonExecutable -m bridgemind.rating --root $BenchmarkRoot --port $Port evaluate --id $Id
    if ($LASTEXITCODE -ne 0) { throw 'Model evaluation failed; completed matches remain resumable' }
    Get-Content -LiteralPath (Join-Path $BenchmarkRoot 'leaderboard.md')
}
finally {
    Pop-Location
}
