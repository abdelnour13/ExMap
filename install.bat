@echo off

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker is not running. Please start Docker Desktop and try again.
  pause
  exit /b 1
)

docker run --rm --gpus all alpine true >nul 2>&1
if errorlevel 1 (
  echo No GPU available, starting in CPU mode
  docker compose up -d
) else (
  echo GPU detected, starting in GPU mode
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
)

pause