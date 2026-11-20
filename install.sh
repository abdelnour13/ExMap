#!/bin/sh
set -e

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

if docker run --rm --gpus all alpine true >/dev/null 2>&1; then
  echo "GPU detected, starting in GPU mode"
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
else
  echo "No GPU available, starting in CPU mode"
  docker compose up -d --build
fi