param(
    [string]$BindAddress,
    [int]$Port = 8000,
    [string]$Checkpoint = 'runs/play-sl-v2-20261026-2h/best.pt',
    [string]$BiddingCheckpoint = 'runs/sl-pilot/last.pt',
    [switch]$SkipBuild,
    [switch]$RestartGateway,
    [switch]$Restart
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $repo '.venv\Scripts\python.exe'
$logDir = Join-Path $repo '.build\lan'
$stateFile = Join-Path $logDir 'state.json'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create the repository Python environment first.' }
if (-not $BindAddress) {
    $addresses = @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.|198\.18\.)' -and $_.AddressState -eq 'Preferred' })
    $BindAddress = ($addresses | Where-Object { $_.InterfaceAlias -match 'WLAN|Wi-Fi|无线' } | Select-Object -First 1).IPAddress
    if (-not $BindAddress) { $BindAddress = ($addresses | Select-Object -First 1).IPAddress }
}
if (-not $BindAddress -or -not (Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -eq $BindAddress)) { throw 'Choose a local IPv4 address using -BindAddress.' }
$checkpointPath = if ([IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint } else { Join-Path $repo $Checkpoint }
if (-not (Test-Path -LiteralPath $checkpointPath)) { throw "Model checkpoint not found: $checkpointPath" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$baseUrl = "http://${BindAddress}:$Port"

function Read-Health {
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/api/v1/health" -TimeoutSec 3
        if ($health.service -eq 'BridgeMind LAN') { return $health }
    } catch {}
    return $null
}
function Read-Dds {
    try {
        $raw = & $python -c "import json; from bridgemind.dds import SolverClient; c=SolverClient(connect_timeout=1); print(json.dumps(c.get_info())); c.close()" 2>$null
        if ($LASTEXITCODE -eq 0) { return ($raw | ConvertFrom-Json) }
    } catch {}
    return $null
}

Push-Location -LiteralPath $repo
try {
    if (($Restart -or $RestartGateway) -and (Test-Path -LiteralPath $stateFile)) {
        $prior = Get-Content -Raw -LiteralPath $stateFile | ConvertFrom-Json
        foreach ($entry in @(@{id=$prior.gatewayPid; module='bridgemind.web.server'; owned=$true}, @{id=$prior.ddsPid; module='bridgemind.dds.server'; owned=($prior.ddsOwned -and $Restart)})) {
            if ($entry.id -and $entry.owned) {
                $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.id)" -ErrorAction SilentlyContinue
                if ($process -and $process.CommandLine -like "*-m $($entry.module)*") { Stop-Process -Id $entry.id -Force }
            }
        }
    }
    $envFile = Join-Path $repo 'frontend\.env.local'
    # .env.local is checkout-local and ignored by Git; retain other user variables.
    $lines = if (Test-Path -LiteralPath $envFile) { @(Get-Content -LiteralPath $envFile | Where-Object { $_ -notmatch '^VITE_ANALYSIS_SERVER_URL=' }) } else { @() }
    @($lines; "VITE_ANALYSIS_SERVER_URL=$baseUrl") | Set-Content -LiteralPath $envFile -Encoding utf8
    if (-not $SkipBuild) {
        Push-Location -LiteralPath (Join-Path $repo 'frontend')
        try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' } } finally { Pop-Location }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repo 'frontend\dist\index.html'))) { throw 'Build the frontend before using -SkipBuild.' }
    $dds = Read-Dds
    $ddsOwned = $false
    if (-not $dds) {
        Start-Process -FilePath $python -ArgumentList @('-m','bridgemind.dds.server','--workers','2','--port','18765') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'dds.stdout.log') -RedirectStandardError (Join-Path $logDir 'dds.stderr.log') | Out-Null
        $ddsOwned = $true
        for ($attempt=0; $attempt -lt 40; $attempt++) { Start-Sleep -Milliseconds 300; $dds=Read-Dds; if ($dds) { break } }
    } elseif (Test-Path -LiteralPath $stateFile) {
        $prior = Get-Content -Raw -LiteralPath $stateFile | ConvertFrom-Json
        $ddsOwned = $prior.ddsOwned -and $prior.ddsPid -eq $dds.process_id
    }
    if (-not $dds) { throw "DDS startup failed; see $logDir" }
    $health = Read-Health
    if (-not $health) {
        if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is occupied by another service." }
        $biddingPath = if ([IO.Path]::IsPathRooted($BiddingCheckpoint)) { $BiddingCheckpoint } else { Join-Path $repo $BiddingCheckpoint }
        if (-not (Test-Path -LiteralPath $biddingPath)) { throw "Bidding model checkpoint not found: $biddingPath" }
        $arguments = @('-m','bridgemind.web.server','--host',$BindAddress,'--port',"$Port",'--checkpoint',('"' + $checkpointPath + '"'),'--bidding-checkpoint',('"' + $biddingPath + '"'),'--web-root',('"' + (Join-Path $repo 'frontend\dist') + '"'))
        Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'gateway.stdout.log') -RedirectStandardError (Join-Path $logDir 'gateway.stderr.log') | Out-Null
        for ($attempt=0; $attempt -lt 60; $attempt++) { Start-Sleep -Milliseconds 300; $health=Read-Health; if ($health) { break } }
    }
    if (-not $health) { throw "Gateway startup failed; see $logDir" }
    $state = @{ url=$baseUrl; gatewayPid=$health.processId; ddsPid=$dds.process_id; ddsOwned=$ddsOwned; checkpoint=$checkpointPath }
    $state | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding utf8
    Write-Output "BridgeMind: $baseUrl"
    Write-Output "Health: $($health.status); AI: $($health.services.ai.ready); DDS: $($health.services.dds.ready)"
    Write-Output "Logs: $logDir"
} finally { Pop-Location }
