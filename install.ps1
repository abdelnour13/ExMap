<#
.SYNOPSIS
  Text2KG backend on Windows: install it, start it, stop it, check it. (Linux/macOS: text2kg.sh.)
.DESCRIPTION
  .\text2kg.cmd install [-Version NAME | -v NAME] [-NoUI] [-Rebuild] [-ForceCpu]
                                     install a version: switch the backend files to the commit it names,
                                     build the images, start the containers, download the resources and
                                     the desktop app
                                     -Version, -v   which version to install (default: "default")
                                     -NoUI          do not download the desktop app
                                     -Rebuild       rebuild every image from scratch (slow)
                                     -ForceCpu      do not look for a GPU, run on the CPU
  .\text2kg.cmd list-versions        list the available versions (the files in versions\)
  .\text2kg.cmd start                start the containers and wait until the API answers
  .\text2kg.cmd stop                 stop the containers (nothing is deleted)
  .\text2kg.cmd status               show the version, the containers and whether the API answers
  .\text2kg.cmd logs [service]       follow the logs

  A version is a file versions\<name>.env with these lines (default.env always exists):
    DATA_SERVICE_URL=                address of the API (an environment variable DATA_SERVICE_URL wins over it)
    BACKEND_GITHUB_HASH=             commit of the backend to install (empty: keep the files as they are)
    DESKTOP_APP_BASE_CURL_URL=       where the desktop app files are downloaded from
    DESKTOP_APP_NAME_LINUX=          file name of the Linux build
    DESKTOP_APP_NAME_WINDOWS=        file name of the Windows build

  `install` is safe to run again at any time: it only redoes what is missing. -ForceCpu saves the CPU mode;
  run install without it to detect the GPU again. Needs Docker Desktop (WSL 2 backend). Keep this folder
  inside the WSL file system (or at least on a local disk): bind mounts from network drives are very slow.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('install', 'list-versions', 'start', 'stop', 'status', 'logs', 'help')]
  [string]$Command = 'help',
  [Alias('v')]
  [string]$Version = 'default',
  [switch]$NoUI,
  [switch]$Rebuild,
  [switch]$ForceCpu,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$VersionsDir = 'versions'
$Snapshot    = '.text2kg-version.env'   # what install used, read by the other commands (versions\ can change with the checkout)
$ModeFile    = '.text2kg-mode'          # "cpu" or "gpu", written by install and read by the other commands
$WaitSeconds = if ($env:TEXT2KG_WAIT_SECONDS) { [int]$env:TEXT2KG_WAIT_SECONDS } else { 180 }
$EnvApiUrl   = $env:DATA_SERVICE_URL    # an environment variable wins over the version file
$script:Current = ''

function Step([string]$Number, [string]$Text) {
  $script:Current = "$Number $Text"
  Write-Host ""
  Write-Host "[$Number] $Text"
}

# ---------- Versions ----------

function New-VersionTable {
  return @{
    DATA_SERVICE_URL          = ''
    BACKEND_GITHUB_HASH       = ''
    DESKTOP_APP_BASE_CURL_URL = ''
    DESKTOP_APP_NAME_LINUX    = ''
    DESKTOP_APP_NAME_WINDOWS  = ''
  }
}

# Reads KEY=VALUE lines. The file is only read, never run as a script. Blank lines and lines starting with #
# are ignored, as are unknown keys; the values may be in quotes.
function Read-VersionFile([string]$Path) {
  $table = New-VersionTable
  foreach ($raw in @(Get-Content -LiteralPath $Path)) {
    $line = $raw.Trim()
    if (-not $line -or $line.StartsWith('#')) { continue }
    $i = $line.IndexOf('=')
    if ($i -lt 1) { continue }
    $key = $line.Substring(0, $i).Trim()
    $value = $line.Substring($i + 1).Trim()
    if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[$value.Length - 1] -eq '"') -or
                                  ($value[0] -eq "'" -and $value[$value.Length - 1] -eq "'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }
    if ($table.ContainsKey($key)) { $table[$key] = $value }
  }
  return $table
}

function Test-VersionName([string]$Name) { return ($Name -match '^[A-Za-z0-9_-][A-Za-z0-9._-]*$') }

function Get-VersionPath([string]$Name) { return (Join-Path $VersionsDir "$Name.env") }

# The names of the available versions, "default" first.
function Get-VersionNames {
  if (-not (Test-Path -LiteralPath $VersionsDir -PathType Container)) { return @() }
  $all = @(Get-ChildItem -LiteralPath $VersionsDir -Filter '*.env' -File |
           ForEach-Object { $_.BaseName } | Where-Object { Test-VersionName $_ } | Sort-Object)
  $names = @()
  if ($all -contains 'default') { $names += 'default' }
  $names += @($all | Where-Object { $_ -ne 'default' })
  return $names
}

function Import-Version([string]$Name) {
  if (-not (Test-VersionName $Name)) {
    throw "'$Name' is not a valid version name (letters, digits, dot, dash and underscore only)."
  }
  $path = Get-VersionPath $Name
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $names = @(Get-VersionNames)
    if ($names.Count -gt 0) { throw "There is no version called '$Name' ($path does not exist). Available versions: $($names -join ' ')" }
    throw "There is no version called '$Name': the folder '$VersionsDir' has no <name>.env file."
  }
  $script:V = Read-VersionFile $path
}

# What install used is kept in a file of its own, because versions\ changes when the backend is switched to
# another commit.
function Save-Snapshot([string]$Name) {
  $lines = @("# version: $Name") + @($script:V.Keys | Sort-Object | ForEach-Object { "$_=$($script:V[$_])" })
  Set-Content -LiteralPath $Snapshot -Value $lines -Encoding UTF8
}

function Get-InstalledVersion {
  if (-not (Test-Path -LiteralPath $Snapshot)) { return '' }
  $first = @(Get-Content -LiteralPath $Snapshot -TotalCount 1)[0]
  if ($first -match '^# version: (.+)$') { return $Matches[1] }
  return ''
}

function Set-ApiUrl {
  $url = if ($EnvApiUrl) { $EnvApiUrl } elseif ($script:V.DATA_SERVICE_URL) { $script:V.DATA_SERVICE_URL } else { 'http://127.0.0.1:8000' }
  $script:ApiUrl = $url.TrimEnd('/')
  $script:ReadyUrl = "$($script:ApiUrl)/openapi.json"
}

$script:V = New-VersionTable
if (Test-Path -LiteralPath $Snapshot) { $script:V = Read-VersionFile $Snapshot }
Set-ApiUrl

# ---------- Backend files (git) ----------

# Runs git and returns its exit code and output (Windows PowerShell 5.1 turns anything a program writes to
# stderr into an error when $ErrorActionPreference is 'Stop', so it is relaxed here).
function Invoke-Git {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = (& git.exe @args 2>&1 | Out-String).Trim(); $code = $LASTEXITCODE }
  catch { $out = $_.Exception.Message; $code = 1 }
  finally { $ErrorActionPreference = $previous }
  return [pscustomobject]@{ Code = $code; Out = $out }
}

# Puts the backend files at the commit the version names. Without a git clone (a ZIP download) it cannot, and
# says so instead of failing.
function Switch-Backend([string]$Ref) {
  if (-not $Ref) {
    Write-Host 'This version does not name a backend commit: keeping the files in this folder.'
    return
  }
  if (-not (Test-Path -LiteralPath '.git') -or -not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Write-Host 'Warning: this folder is not a git clone (a ZIP download?) or git is not installed, so the backend'
    Write-Host "         cannot be switched to $Ref. Continuing with the files in this folder."
    return
  }
  $spec = "$($Ref)^{commit}"
  $found = Invoke-Git rev-parse --verify --quiet $spec
  if ($found.Code -ne 0) {
    Write-Host "Commit $Ref is not in this clone yet: fetching it..."
    $fetch = Invoke-Git fetch --quiet --tags origin
    if ($fetch.Code -ne 0) { throw "Could not fetch from GitHub. Check the internet connection and try again. Git said: $($fetch.Out)" }
    $found = Invoke-Git rev-parse --verify --quiet $spec
    if ($found.Code -ne 0) { throw "The commit $Ref does not exist in the repository." }
  }
  $target = $found.Out
  $short = $target.Substring(0, [Math]::Min(7, $target.Length))
  if ($target -eq (Invoke-Git rev-parse HEAD).Out) {
    Write-Host "The backend files are already at commit $short."
    return
  }
  $checkout = Invoke-Git checkout --quiet $target
  if ($checkout.Code -ne 0) {
    throw "Could not switch the backend files to commit $short. If you changed files in this folder, commit or stash those changes first. Git said: $($checkout.Out)"
  }
  Write-Host "The backend files are now at commit $short (git is on a detached HEAD; 'git checkout <branch>' leaves it)."
}

# ---------- Docker ----------

# Runs docker and stops the script if it fails. (Not named "Docker": PowerShell names are case-insensitive, so
# a function with that name would hide docker.exe and call itself forever.)
function Invoke-Docker {
  & docker.exe @args
  if ($LASTEXITCODE -ne 0) { throw "'docker $($args -join ' ')' failed (exit code $LASTEXITCODE)." }
}

# Runs docker and only says whether it worked. (Same stderr remark as for git.)
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
    Invoke-WebRequest -Uri $script:ReadyUrl -UseBasicParsing -TimeoutSec 3 | Out-Null
    return $true
  } catch {
    return [bool]$_.Exception.Response   # an error status still means the server is up
  }
}

function Wait-Ready {
  Write-Host -NoNewline "Waiting for the API at $($script:ApiUrl) "
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

# ---------- Desktop app ----------

function Get-DesktopApp {
  $name = $script:V.DESKTOP_APP_NAME_WINDOWS
  $base = $script:V.DESKTOP_APP_BASE_CURL_URL
  if (-not $name -or -not $base) {
    Write-Host 'This version lists no Windows desktop app (DESKTOP_APP_NAME_WINDOWS / DESKTOP_APP_BASE_CURL_URL): skipped.'
    return
  }
  if ($name -match '[\\/]') { throw "The desktop app file name '$name' must not contain a slash." }

  if (Test-Path -LiteralPath $name) {
    Write-Host "$name is already in this folder: not downloading it again (delete it to download it again)."
  } else {
    $url = $base.TrimEnd('/') + '/' + [uri]::EscapeDataString($name)   # spaces become %20
    $part = "$name.part"
    Write-Host "Downloading $name"
    Write-Host "  from $url"
    $ok = $false
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
      & curl.exe -fL --retry 3 -o $part $url
      $ok = ($LASTEXITCODE -eq 0)
    } else {
      try {
        $ProgressPreference = 'SilentlyContinue'   # the progress bar makes Windows PowerShell 5.1 very slow
        Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing
        $ok = $true
      } catch { $ok = $false }
    }
    if (-not $ok) {
      Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
      throw "Could not download the desktop app from $url. The backend is installed: run the install again, or use -NoUI and download the app yourself."
    }
    Move-Item -LiteralPath $part -Destination $name -Force
  }
  Write-Host "Run `"$name`" to install the desktop app."
}

# ---------- Commands ----------

function Install-Backend {
  $total = if ($NoUI) { 6 } else { 7 }

  Import-Version $Version   # before the first step: a wrong name is reported without a "stopped at step" line
  Step "1/$total" "Selecting version '$Version'"
  Switch-Backend $script:V.BACKEND_GITHUB_HASH
  Save-Snapshot $Version
  Set-ApiUrl

  Step "2/$total" 'Checking Docker'
  Assert-Docker

  Step "3/$total" 'Looking for a GPU'
  if ($ForceCpu) {
    $mode = 'cpu'
    Write-Host '-ForceCpu: not looking for a GPU, starting in CPU mode.'
  } else {
    $mode = Get-Mode
    if ($mode -eq 'gpu') { Write-Host 'GPU found: starting in GPU mode.' } else { Write-Host 'No GPU available: starting in CPU mode.' }
  }
  Set-Content -LiteralPath $ModeFile -Value $mode -NoNewline

  Step "4/$total" 'Building the images (the first time this takes several minutes)'
  try { Compose pull neo4j } catch { Write-Host '(Could not refresh the Neo4j image; using the one that is already here.)' }
  $progress = if ($env:TEXT2KG_BUILD_PROGRESS) { $env:TEXT2KG_BUILD_PROGRESS } else { 'auto' }
  try {
    if ($Rebuild) { Compose build --pull --no-cache --progress $progress } else { Compose build --pull --progress $progress }
  } catch {
    Write-Host ''
    Write-Host 'The build failed. To see the full output of every build step, run: $env:TEXT2KG_BUILD_PROGRESS = ''plain''; .\text2kg.cmd install'
    throw
  }

  Step "5/$total" 'Starting the containers'
  if ($Rebuild) { Compose up -d --force-recreate } else { Compose up -d }

  Step "6/$total" 'Downloading the resources from the manifest'
  Compose exec -T text2kg python -m src.manifest download
  Write-Host 'Restarting Neo4j so that it loads the downloaded plugins...'
  Compose restart neo4j
  Wait-Ready

  if (-not $NoUI) {
    Step "7/$total" 'Downloading the desktop app'
    Get-DesktopApp
  }

  $script:Current = ''
  Write-Host ''
  Write-Host "Installation of version $Version complete."
  if ($NoUI) { Write-Host 'The desktop app was not downloaded (-NoUI).' }
}

function Show-Versions {
  if (-not (Test-Path -LiteralPath $VersionsDir -PathType Container)) { throw "There is no '$VersionsDir' folder next to this script." }
  $names = @(Get-VersionNames)
  if ($names.Count -eq 0) { throw "No version found: the folder '$VersionsDir' has no <name>.env file." }
  $installed = Get-InstalledVersion
  Write-Host "Available versions ('*' marks the installed one):"
  Write-Host ('  {0,-2} {1,-18} {2,-9} {3}' -f '', 'VERSION', 'BACKEND', 'DESKTOP APP (Windows)')
  foreach ($n in $names) {
    $v = Read-VersionFile (Get-VersionPath $n)
    $mark = if ($n -eq $installed) { '*' } else { '' }
    $hash = if ($v.BACKEND_GITHUB_HASH) { $v.BACKEND_GITHUB_HASH.Substring(0, [Math]::Min(7, $v.BACKEND_GITHUB_HASH.Length)) } else { '-' }
    $app = if ($v.DESKTOP_APP_NAME_WINDOWS) { $v.DESKTOP_APP_NAME_WINDOWS } else { '-' }
    Write-Host ('  {0,-2} {1,-18} {2,-9} {3}' -f $mark, $n, $hash, $app)
  }
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
  $installed = Get-InstalledVersion
  if ($installed) { Write-Host "Version: $installed" }
  Compose ps
  if (Test-Api) { Write-Host "The API answers at $($script:ApiUrl)." }
  else { Write-Host "The API does not answer at $($script:ApiUrl)."; exit 1 }
}

try {
  switch ($Command) {
    'install'       { Install-Backend }
    'list-versions' { Show-Versions }
    'start'         { Start-Backend }
    'stop'          { Stop-Backend }
    'status'        { Show-Status }
    'logs'          { Assert-Docker; Assert-Installed; Compose logs -f --tail 100 @Rest }
    default         { Get-Help $PSCommandPath -Full | Out-String | Write-Host }
  }
} catch {
  Write-Host ''
  Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
  if ($script:Current) { Write-Host "Stopped at step $($script:Current). Fix the problem above and run the same command again: it is safe to repeat." -ForegroundColor Red }
  exit 1
}