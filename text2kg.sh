#!/bin/sh
# Text2KG backend: install it, start it, stop it, check it. Linux and macOS (Windows: text2kg.ps1).
#
#   ./text2kg.sh install [--rebuild]   build the images, start the containers, download the resources
#   ./text2kg.sh start                 start the containers and wait until the API answers
#   ./text2kg.sh stop                  stop the containers (nothing is deleted)
#   ./text2kg.sh status                show the containers and whether the API answers
#   ./text2kg.sh logs [service]        follow the logs
#
# `install` is safe to run again at any time: it only redoes what is missing. `--rebuild` rebuilds every
# image from scratch (slow); use it after a change of the Dockerfiles or to refresh the base images.
set -eu
cd "$(dirname "$0")"

API_URL="${DATA_SERVICE_URL:-http://127.0.0.1:8000}"
READY_URL="${API_URL%/}/openapi.json"
WAIT_SECONDS="${TEXT2KG_WAIT_SECONDS:-180}"
MODE_FILE=".text2kg-mode"   # "cpu" or "gpu", written by install and read by the other commands

say()  { printf '%s\n' "$*"; }
step() { CURRENT="$1 $2"; printf '\n[%s] %s\n' "$1" "$2"; }
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }

CURRENT=""
trap 'rc=$?; if [ "$rc" -ne 0 ] && [ -n "$CURRENT" ]; then printf "\nStopped at step %s. Fix the problem above and run the same command again: it is safe to repeat.\n" "$CURRENT" >&2; fi' EXIT

# ---------- Checks ----------

check_docker() {
  command -v docker >/dev/null 2>&1 ||
    die "Docker is not installed. Install Docker Engine (https://docs.docker.com/engine/install/) and try again."
  docker info >/dev/null 2>&1 ||
    die "Docker is not running, or this user is not allowed to use it. Start Docker (and, on Linux, add your user to the 'docker' group), then try again."
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

# ---------- Commands ----------

cmd_install() {
  rebuild=0
  for arg in "$@"; do
    case "$arg" in
      --rebuild) rebuild=1 ;;
      *) die "Unknown option: $arg" ;;
    esac
  done

  step 1/5 "Checking Docker"
  check_docker

  step 2/5 "Looking for a GPU"
  detected="$(detect_mode)"
  printf '%s' "$detected" > "$MODE_FILE"
  if [ "$detected" = gpu ]; then say "GPU found: starting in GPU mode."; else say "No GPU available: starting in CPU mode."; fi

  step 3/5 "Building the images (the first time this takes several minutes)"
  compose pull neo4j || say "(Could not refresh the Neo4j image; using the one that is already here.)"
  if [ "$rebuild" -eq 1 ]; then compose build --pull --no-cache; else compose build --pull; fi

  step 4/5 "Starting the containers"
  if [ "$rebuild" -eq 1 ]; then compose up -d --force-recreate; else compose up -d; fi

  step 5/5 "Downloading the resources from the manifest"
  compose exec -T text2kg python -m src.manifest download
  say "Restarting Neo4j so that it loads the downloaded plugins..."
  compose restart neo4j
  wait_ready

  CURRENT=""
  printf '\nInstallation complete. Open the desktop app.\n'
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

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-help}" in
  install) shift; cmd_install "$@" ;;
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  logs)    shift; cmd_logs "$@" ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac