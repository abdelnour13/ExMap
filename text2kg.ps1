<#
.SYNOPSIS
  Text2KG backend on Windows: install it, start it, stop it, check it. (Linux/macOS: text2kg.sh.)
.DESCRIPTION
  .\text2kg.cmd install [-Rebuild] [-ForceCpu]
                                     build the images, start the containers, download the resources
                                     (-ForceCpu: do not look for a GPU, run on the CPU)
  .\text2kg.cmd start                start the containers and wait until the API answers
  .\text2kg.cmd stop                 stop the containers (nothing is deleted)
  .\text2kg.cmd status               show the containers and whether the API answers
  .\text2kg.cmd logs [service]       follow the logs

  `install` is safe to run again at any time: it only redoes what is missing. -Rebuild rebuilds every image
  from scratch (slow). -ForceCpu skips the GPU test and saves the CPU mode; run install without it to detect
  the GPU again. Needs Docker Desktop (WSL 2 backend). Keep this folder inside the WSL file system
  (or at least on a local disk): bind mounts from network drives are very slow.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('install', 'start', 'stop', 'status', 'logs', 'help')]
  [string]$Command = 'help',
  [switch]$Rebuild,
  [switch]$ForceCpu,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$ApiUrl      = if ($env:DATA_SERVICE_URL) { $env:DATA_SERVICE_URL.TrimEnd('/') } else { 'http://127.0.0.1:8000' }
$ReadyUrl    = "$ApiUrl/openapi.json"
$WaitSeconds = if ($env:TEXT2KG_WAIT_SECONDS) { [int]$env:TEXT2KG_WAIT_SECONDS } else { 180 }
$ModeFile    = '.text2kg-mode'   # "cpu" or "gpu", written by install and read by the other commands
$script:Current = ''

function Step([string]$Number, [string]$Text) {
  $script:Current = "$Number $Text"
  Write-Host ""
  Write-Host "[$Number] $Text"
}

# ---------- Docker ----------

# Runs docker and stops the script if it fails. (Not named "Docker": PowerShell names are case-insensitive, so
# a function with that name would hide docker.exe and call itself forever.)
function Invoke-Docker {
  & docker.exe @args
  if ($LASTEXITCODE -ne 0) { throw "'docker $($args -join ' ')' failed (exit code $LASTEXITCODE)." }
}

# Runs docker and only says whether it worked. (Windows PowerShell 5.1 turns anything a program writes to
# stderr into an error when $ErrorActionPreference is 'Stop', so it is relaxed here.)
function Test-Docker {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & docker.exe @args *> $null; return ($LASTEXITCODE -eq 0) }
  catch { return $false }
  finally { $ErrorActionPreference = $previous }
}

function Assert-Docker {
  if (-not (Get-Command docker.exe -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed. Install Docker Desktop (https://docs.docker.com/desktop/setup/install/windows-install/) and try again.'
  }
  # `docker --version` works without the engine; asking the server for its version only works when the engine runs.
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $said = (& docker.exe version --format '{{.Server.Version}}' 2>&1 | Out-String).Trim(); $running = ($LASTEXITCODE -eq 0) }
  catch { $said = $_.Exception.Message; $running = $false }
  finally { $ErrorActionPreference = $previous }
  if (-not $running) {
    throw "Docker is not running, or this user is not allowed to use it. Start Docker Desktop and wait until it says it is running; if it already is, add your Windows user to the 'docker-users' group, then sign out and in. Docker said:`n$said"
  }
  if (-not (Test-Docker compose version)) {
    throw "'docker compose' (Compose v2) is missing. Update Docker Desktop and try again."
  }
}

# GPU: the only reliable test is to start a container with one. $env:TEXT2KG_MODE = 'cpu' or 'gpu' skips the test.
function Get-Mode {
  if ($env:TEXT2KG_MODE -in @('cpu', 'gpu')) { return $env:TEXT2KG_MODE }
  if (Test-Docker run --rm --gpus all alpine true) { return 'gpu' }
  return 'cpu'
}

function Get-SavedMode {
  if (Test-Path -LiteralPath $ModeFile) { return (Get-Content -LiteralPath $ModeFile -Raw).Trim() }
  return 'cpu'
}

function Compose {
  if ((Get-SavedMode) -eq 'gpu') { Invoke-Docker compose -f docker-compose.yml -f docker-compose.gpu.yml @args }
  else { Invoke-Docker compose @args }
}

function Assert-Installed {
  if (-not (Test-Path -LiteralPath $ModeFile)) { throw 'The backend is not installed yet. Run: .\text2kg.cmd install' }
}

# ---------- Is the API answering? ----------

# True when the service answers with any HTTP response (even an error page).
function Test-Api {
  try {
    Invoke-WebRequest -Uri $ReadyUrl -UseBasicParsing -TimeoutSec 3 | Out-Null
    return $true
  } catch {
    return [bool]$_.Exception.Response   # an error status still means the server is up
  }
}

function Wait-Ready {
  Write-Host -NoNewline "Waiting for the API at $ApiUrl "
  $waited = 0
  while (-not (Test-Api)) {
    if ($waited -ge $WaitSeconds) {
      Write-Host ''
      Write-Host "The API did not answer within $WaitSeconds seconds. The last log lines of the main container:"
      try { Compose logs --tail 30 text2kg } catch { }
      throw 'The API is not ready. See the logs above, or run: .\text2kg.cmd logs text2kg'
    }
    Write-Host -NoNewline '.'
    Start-Sleep -Seconds 2
    $waited += 2
  }
  Write-Host ''
  Write-Host 'The API is ready.'
}

# ---------- Commands ----------

function Install-Backend {
  Step '1/5' 'Checking Docker'
  Assert-Docker

  Step '2/5' 'Looking for a GPU'
  if ($ForceCpu) {
    $mode = 'cpu'
    Write-Host '-ForceCpu: not looking for a GPU, starting in CPU mode.'
  } else {
    $mode = Get-Mode
    if ($mode -eq 'gpu') { Write-Host 'GPU found: starting in GPU mode.' } else { Write-Host 'No GPU available: starting in CPU mode.' }
  }
  Set-Content -LiteralPath $ModeFile -Value $mode -NoNewline

  Step '3/5' 'Building the images (the first time this takes several minutes)'
  try { Compose pull neo4j } catch { Write-Host '(Could not refresh the Neo4j image; using the one that is already here.)' }
  if ($Rebuild) { Compose build --pull --no-cache } else { Compose build --pull }

  Step '4/5' 'Starting the containers'
  if ($Rebuild) { Compose up -d --force-recreate } else { Compose up -d }

  Step '5/5' 'Downloading the resources from the manifest'
  Compose exec -T text2kg python -m src.manifest download
  Write-Host 'Restarting Neo4j so that it loads the downloaded plugins...'
  Compose restart neo4j
  Wait-Ready

  $script:Current = ''
  Write-Host ''
  Write-Host 'Installation complete. Open the desktop app.'
}

function Start-Backend {
  Assert-Docker
  Assert-Installed
  Write-Host "Starting the Text2KG backend ($(Get-SavedMode) mode)..."
  $script:Current = 'start'
  Compose up -d
  Wait-Ready
  $script:Current = ''
}

function Stop-Backend {
  Assert-Docker
  Assert-Installed
  Compose stop
  Write-Host 'The Text2KG backend is stopped. Its data is kept.'
}

function Show-Status {
  Assert-Docker
  Assert-Installed
  Compose ps
  if (Test-Api) { Write-Host "The API answers at $ApiUrl." }
  else { Write-Host "The API does not answer at $ApiUrl."; exit 1 }
}

try {
  switch ($Command) {
    'install' { Install-Backend }
    'start'   { Start-Backend }
    'stop'    { Stop-Backend }
    'status'  { Show-Status }
    'logs'    { Assert-Docker; Assert-Installed; Compose logs -f --tail 100 @Rest }
    default   { Get-Help $PSCommandPath -Full | Out-String | Write-Host }
  }
} catch {
  Write-Host ''
  Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
  if ($script:Current) { Write-Host "Stopped at step $($script:Current). Fix the problem above and run the same command again: it is safe to repeat." -ForegroundColor Red }
  exit 1
}
