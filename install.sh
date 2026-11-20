#!/bin/sh
set -e

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

if docker run --rm --gpus all alpine true >/dev/null 2>&1; then
  echo "GPU detected, starting in GPU mode"
  COMPOSE="docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
else
  echo "No GPU available, starting in CPU mode"
  COMPOSE="docker compose"
fi

$COMPOSE up -d --build

echo "Downloading resources from the manifest..."
$COMPOSE exec text2kg python -m manifest download

# Neo4j only loads plugins at startup
$COMPOSE restart neo4j