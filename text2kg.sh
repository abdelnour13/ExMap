#!/bin/sh
# Text2KG backend: install it, start it, stop it, check it. Linux and macOS (Windows: text2kg.ps1).
#
#   ./text2kg.sh install [--version NAME | -v NAME] [--no-ui] [--rebuild] [--force-cpu]
#                                      install a version: switch the backend files to the commit it names,
#                                      build the images, start the containers, download the resources and
#                                      the desktop app
#                                      --version, -v   which version to install (default: "default")
#                                      --no-ui         do not download the desktop app
#                                      --rebuild       rebuild every image from scratch (slow)
#                                      --force-cpu     do not look for a GPU, run on the CPU
#   ./text2kg.sh list-versions         list the available versions (the files in versions/)
#   ./text2kg.sh start                 start the containers and wait until the API answers
#   ./text2kg.sh stop                  stop the containers (nothing is deleted)
#   ./text2kg.sh status                show the version, the containers and whether the API answers
#   ./text2kg.sh logs [service]        follow the logs
#
# A version is a file versions/<name>.env with these lines (default.env always exists):
#   DATA_SERVICE_URL=                  address of the API (an exported DATA_SERVICE_URL wins over it)
#   BACKEND_GITHUB_HASH=               commit of the backend to install (empty: keep the files as they are)
#   DESKTOP_APP_BASE_CURL_URL=         where the desktop app files are downloaded from
#   DESKTOP_APP_NAME_LINUX=            file name of the Linux build
#   DESKTOP_APP_NAME_WINDOWS=          file name of the Windows build
#
# `install` is safe to run again at any time: it only redoes what is missing. TEXT2KG_BUILD_PROGRESS=plain shows
# the full output of every build step (default: auto). `--force-cpu` saves the CPU mode; run install without it
# to detect the GPU again.
set -eu
cd "$(dirname "$0")"

VERSIONS_DIR="versions"
SNAPSHOT=".text2kg-version.env"   # what install used, read by the other commands (versions/ can change with the checkout)
MODE_FILE=".text2kg-mode"         # "cpu" or "gpu", written by install and read by the other commands
WAIT_SECONDS="${TEXT2KG_WAIT_SECONDS:-180}"
ENV_API_URL="${DATA_SERVICE_URL:-}"   # an exported variable wins over the version file

say()  { printf '%s\n' "$*"; }
step() { CURRENT="$1 $2"; printf '\n[%s] %s\n' "$1" "$2"; }
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }

CURRENT=""
trap 'rc=$?; if [ "$rc" -ne 0 ] && [ -n "$CURRENT" ]; then printf "\nStopped at step %s. Fix the problem above and run the same command again: it is safe to repeat.\n" "$CURRENT" >&2; fi' EXIT

# ---------- Versions ----------

trim() {
  s=$1
  s=${s#"${s%%[![:space:]]*}"}
  s=${s%"${s##*[![:space:]]}"}
  printf '%s' "$s"
}

reset_version() {
  V_DATA_SERVICE_URL=""
  V_BACKEND_GITHUB_HASH=""
  V_DESKTOP_APP_BASE_CURL_URL=""
  V_DESKTOP_APP_NAME_LINUX=""
  V_DESKTOP_APP_NAME_WINDOWS=""
}

# Reads KEY=VALUE lines into the V_* variables. The file is only read, never run as a script. Blank lines and
# lines starting with # are ignored, as are unknown keys; the values may be in quotes; CRLF files are fine.
read_version_file() {
  reset_version
  bom="$(printf '\357\273\277')"
  cr="$(printf '\r')"
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line#"$bom"}
    line=${line%"$cr"}
    line="$(trim "$line")"
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    case "$value" in
      \"*\") value=${value#\"}; value=${value%\"} ;;
      \'*\') value=${value#\'}; value=${value%\'} ;;
    esac
    case "$key" in
      DATA_SERVICE_URL)           V_DATA_SERVICE_URL="$value" ;;
      BACKEND_GITHUB_HASH)        V_BACKEND_GITHUB_HASH="$value" ;;
      DESKTOP_APP_BASE_CURL_URL)  V_DESKTOP_APP_BASE_CURL_URL="$value" ;;
      DESKTOP_APP_NAME_LINUX)     V_DESKTOP_APP_NAME_LINUX="$value" ;;
      DESKTOP_APP_NAME_WINDOWS)   V_DESKTOP_APP_NAME_WINDOWS="$value" ;;
    esac
  done < "$1"
}

valid_name() {
  case "$1" in ''|.*|*[!A-Za-z0-9._-]*) return 1 ;; esac
  return 0
}

version_file() { printf '%s/%s.env' "$VERSIONS_DIR" "$1"; }

# The names of the available versions, "default" first.
version_names() {
  [ -f "$(version_file default)" ] && printf 'default\n'
  for f in "$VERSIONS_DIR"/*.env; do
    [ -f "$f" ] || continue
    b=${f##*/}; b=${b%.env}
    [ "$b" = default ] && continue
    valid_name "$b" && printf '%s\n' "$b"
  done
  return 0
}

load_version() {
  valid_name "$1" || die "'$1' is not a valid version name (letters, digits, dot, dash and underscore only)."
  if [ ! -f "$(version_file "$1")" ]; then
    names="$(version_names | tr '\n' ' ')"
    if [ -n "$names" ]; then
      die "There is no version called '$1' ($(version_file "$1") does not exist). Available versions: $names"
    fi
    die "There is no version called '$1': the folder '$VERSIONS_DIR' has no <name>.env file."
  fi
  read_version_file "$(version_file "$1")"
}

# What install used is kept in a file of its own, because versions/ changes when the backend is switched to
# another commit.
write_snapshot() {
  {
    printf '# version: %s\n' "$1"
    printf 'DATA_SERVICE_URL=%s\n' "$V_DATA_SERVICE_URL"
    printf 'BACKEND_GITHUB_HASH=%s\n' "$V_BACKEND_GITHUB_HASH"
    printf 'DESKTOP_APP_BASE_CURL_URL=%s\n' "$V_DESKTOP_APP_BASE_CURL_URL"
    printf 'DESKTOP_APP_NAME_LINUX=%s\n' "$V_DESKTOP_APP_NAME_LINUX"
    printf 'DESKTOP_APP_NAME_WINDOWS=%s\n' "$V_DESKTOP_APP_NAME_WINDOWS"
  } > "$SNAPSHOT"
}

installed_version() {
  [ -f "$SNAPSHOT" ] || return 0
  first="$(head -n 1 "$SNAPSHOT")"
  case "$first" in '# version: '*) printf '%s' "${first#'# version: '}" ;; esac
  return 0
}

set_api_url() {
  API_URL="${ENV_API_URL:-${V_DATA_SERVICE_URL:-http://127.0.0.1:8000}}"
  READY_URL="${API_URL%/}/openapi.json"
}

reset_version
if [ -f "$SNAPSHOT" ]; then read_version_file "$SNAPSHOT"; fi
set_api_url

# ---------- Backend files (git) ----------

short() { printf '%.7s' "$1"; }

# Puts the backend files at the commit the version names. Without a git clone (a ZIP download) it cannot, and
# says so instead of failing.
switch_backend() {
  ref=$1
  if [ -z "$ref" ]; then
    say "This version does not name a backend commit: keeping the files in this folder."
    return 0
  fi
  if [ ! -e .git ] || ! command -v git >/dev/null 2>&1; then
    say "Warning: this folder is not a git clone (a ZIP download?) or git is not installed, so the backend"
    say "         cannot be switched to $ref. Continuing with the files in this folder."
    return 0
  fi
  if ! target="$(git rev-parse --verify --quiet "${ref}^{commit}")"; then
    say "Commit $ref is not in this clone yet: fetching it..."
    git fetch --quiet --tags origin || die "Could not fetch from GitHub. Check the internet connection and try again."
    target="$(git rev-parse --verify --quiet "${ref}^{commit}")" || die "The commit $ref does not exist in the repository."
  fi
  if [ "$target" = "$(git rev-parse HEAD)" ]; then
    say "The backend files are already at commit $(short "$target")."
    return 0
  fi
  if ! said="$(git checkout --quiet "$target" 2>&1)"; then
    die "Could not switch the backend files to commit $(short "$target"). If you changed files in this folder, commit or stash those changes first. Git said: $said"
  fi
  say "The backend files are now at commit $(short "$target") (git is on a detached HEAD; 'git checkout <branch>' leaves it)."
}

# ---------- Checks ----------

check_docker() {
  command -v docker >/dev/null 2>&1 ||
    die "Docker is not installed. Install Docker Engine (https://docs.docker.com/engine/install/) and try again."
  if ! said="$(docker info 2>&1 >/dev/null)"; then
    die "Docker is not running, or this user is not allowed to use it. Start Docker (and, on Linux, add your user to the 'docker' group), then try again. Docker said: $said"
  fi
  docker compose version >/dev/null 2>&1 ||
    die "'docker compose' (Compose v2) is missing. Install the Docker Compose plugin and try again."
}

# GPU: the only reliable test is to start a container with one. TEXT2KG_MODE=cpu|gpu skips the test.
detect_mode() {
  case "${TEXT2KG_MODE:-}" in
    cpu|gpu) printf '%s' "$TEXT2KG_MODE"; return ;;
  esac
  if docker run --rm --gpus all alpine true >/dev/null 2>&1; then printf gpu; else printf cpu; fi
}

mode() { cat "$MODE_FILE" 2>/dev/null || printf cpu; }

compose() {
  if [ "$(mode)" = gpu ]; then
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml "$@"
  else
    docker compose "$@"
  fi
}

require_installed() {
  [ -f "$MODE_FILE" ] || die "The backend is not installed yet. Run: ./text2kg.sh install"
}

# ---------- Is the API answering? ----------

# Succeeds when the service answers with any HTTP response (even an error page). Returns 2 when this machine
# has neither curl nor wget, so the caller can say it could not check.
api_answers() {
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null --max-time 3 "$READY_URL" 2>/dev/null
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O /dev/null -T 3 "$READY_URL" 2>/dev/null && return 0
    [ "$?" -eq 8 ]   # 8: the server answered with an error status, so it is up
  else
    return 2
  fi
}

wait_ready() {
  printf 'Waiting for the API at %s ' "$API_URL"
  waited=0
  while :; do
    rc=0; api_answers || rc=$?
    if [ "$rc" -eq 0 ]; then printf '\nThe API is ready.\n'; return 0; fi
    if [ "$rc" -eq 2 ]; then printf '\n(Neither curl nor wget is installed: cannot check. Give it a minute.)\n'; return 0; fi
    if [ "$waited" -ge "$WAIT_SECONDS" ]; then
      printf '\n'
      say "The API did not answer within ${WAIT_SECONDS} seconds. The last log lines of the main container:"
      compose logs --tail 30 text2kg || true
      die "The API is not ready. See the logs above, or run: ./text2kg.sh logs text2kg"
    fi
    printf '.'; sleep 2; waited=$((waited + 2))
  done
}

# ---------- Desktop app ----------

# Spaces and the few characters that mean something in a URL, so that "Text2EM Setup 1.0.0.exe" works.
urlencode() {
  printf '%s' "$1" | sed -e 's/%/%25/g' -e 's/ /%20/g' -e 's/#/%23/g' -e 's/?/%3F/g'
}

download_app() {
  case "$(uname -s)" in
    Linux) ;;
    *) say "There is no desktop app build for this system: skipped."; return 0 ;;
  esac
  name="$V_DESKTOP_APP_NAME_LINUX"
  base="$V_DESKTOP_APP_BASE_CURL_URL"
  if [ -z "$name" ] || [ -z "$base" ]; then
    say "This version lists no Linux desktop app (DESKTOP_APP_NAME_LINUX / DESKTOP_APP_BASE_CURL_URL): skipped."
    return 0
  fi
  case "$name" in */*|*\\*) die "The desktop app file name '$name' must not contain a slash." ;; esac

  if [ -f "$name" ]; then
    say "$name is already in this folder: not downloading it again (delete it to download it again)."
  else
    command -v curl >/dev/null 2>&1 ||
      die "curl is needed to download the desktop app. Install it, or run the install again with --no-ui."
    url="${base%/}/$(urlencode "$name")"
    say "Downloading $name"
    say "  from $url"
    if ! curl -fL --retry 3 -o "$name.part" "$url"; then
      rm -f "$name.part"
      die "Could not download the desktop app from $url. The backend is installed: run the install again, or use --no-ui and download the app yourself."
    fi
    mv "$name.part" "$name"
  fi

  case "$name" in
    *.AppImage|*.appimage)
      chmod +x "$name"
      say "Run it with: ./$name   (if it complains about FUSE: sudo apt install libfuse2)" ;;
    *.deb)
      say "Install it with: sudo apt install ./$name" ;;
    *)
      say "Saved as $name." ;;
  esac
}

# ---------- Commands ----------

cmd_install() {
  rebuild=0
  force_cpu=0
  no_ui=0
  version=default
  while [ $# -gt 0 ]; do
    case "$1" in
      --rebuild) rebuild=1 ;;
      --force-cpu) force_cpu=1 ;;
      --no-ui) no_ui=1 ;;
      --version|-v)
        [ $# -ge 2 ] || die "$1 needs a version name. Run ./text2kg.sh list-versions to see them."
        version=$2; shift ;;
      --version=*) version=${1#--version=} ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done
  total=7
  if [ "$no_ui" -eq 1 ]; then total=6; fi

  step "1/$total" "Selecting version '$version'"
  load_version "$version"
  switch_backend "$V_BACKEND_GITHUB_HASH"
  write_snapshot "$version"
  set_api_url

  step "2/$total" "Checking Docker"
  check_docker

  step "3/$total" "Looking for a GPU"
  if [ "$force_cpu" -eq 1 ]; then
    detected=cpu
    say "--force-cpu: not looking for a GPU, starting in CPU mode."
  else
    detected="$(detect_mode)"
    if [ "$detected" = gpu ]; then say "GPU found: starting in GPU mode."; else say "No GPU available: starting in CPU mode."; fi
  fi
  printf '%s' "$detected" > "$MODE_FILE"

  step "4/$total" "Building the images (the first time this takes several minutes)"
  compose pull neo4j || say "(Could not refresh the Neo4j image; using the one that is already here.)"
  progress="${TEXT2KG_BUILD_PROGRESS:-auto}"
  if [ "$rebuild" -eq 1 ]; then flags="--pull --no-cache"; else flags="--pull"; fi
  # shellcheck disable=SC2086
  compose build $flags --progress "$progress" || {
    say ""
    say "The build failed. To see the full output of every build step, run: TEXT2KG_BUILD_PROGRESS=plain ./text2kg.sh install"
    exit 1
  }

  step "5/$total" "Starting the containers"
  if [ "$rebuild" -eq 1 ]; then compose up -d --force-recreate; else compose up -d; fi

  step "6/$total" "Downloading the resources from the manifest"
  compose exec -T text2kg python -m src.manifest download
  say "Restarting Neo4j so that it loads the downloaded plugins..."
  compose restart neo4j
  wait_ready

  if [ "$no_ui" -eq 0 ]; then
    step "7/$total" "Downloading the desktop app"
    download_app
  fi

  CURRENT=""
  printf '\nInstallation of version %s complete.\n' "$version"
  if [ "$no_ui" -eq 1 ]; then say "The desktop app was not downloaded (--no-ui)."; fi
}

cmd_list_versions() {
  [ -d "$VERSIONS_DIR" ] || die "There is no '$VERSIONS_DIR' folder next to this script."
  names="$(version_names)"
  [ -n "$names" ] || die "No version found: the folder '$VERSIONS_DIR' has no <name>.env file."
  installed="$(installed_version)"
  say "Available versions ('*' marks the installed one):"
  printf '  %-2s %-18s %-9s %s\n' "" VERSION BACKEND "DESKTOP APP (Linux)"
  for n in $names; do
    read_version_file "$(version_file "$n")"
    mark=""
    if [ "$n" = "$installed" ]; then mark="*"; fi
    h="-"
    if [ -n "$V_BACKEND_GITHUB_HASH" ]; then h="$(short "$V_BACKEND_GITHUB_HASH")"; fi
    app="${V_DESKTOP_APP_NAME_LINUX:--}"
    printf '  %-2s %-18s %-9s %s\n' "$mark" "$n" "$h" "$app"
  done
}

cmd_start() {
  check_docker
  require_installed
  say "Starting the Text2KG backend ($(mode) mode)..."
  CURRENT="start"
  compose up -d
  wait_ready
  CURRENT=""
}

cmd_stop() {
  check_docker
  require_installed
  compose stop
  say "The Text2KG backend is stopped. Its data is kept."
}

cmd_status() {
  check_docker
  require_installed
  v="$(installed_version)"
  if [ -n "$v" ]; then say "Version: $v"; fi
  compose ps
  rc=0; api_answers || rc=$?
  case "$rc" in
    0) say "The API answers at $API_URL." ;;
    2) say "Cannot check the API: neither curl nor wget is installed." ;;
    *) say "The API does not answer at $API_URL."; CURRENT=""; exit 1 ;;
  esac
}

cmd_logs() {
  check_docker
  require_installed
  compose logs -f --tail 100 "$@"
}

# Prints the comment block at the top of this file.
usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

# The "; exit" is on the same line on purpose: install can switch the repository to another commit, which
# replaces this file while it is running, and the shell must not read anything after this line.
case "${1:-help}" in
  install)       shift; cmd_install "$@" ;;
  list-versions) cmd_list_versions ;;
  start)         cmd_start ;;
  stop)          cmd_stop ;;
  status)        cmd_status ;;
  logs)          shift; cmd_logs "$@" ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac; exit $?