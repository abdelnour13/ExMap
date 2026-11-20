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

# --------------------------------------------------
# 1. Build images from scratch and start containers
# --------------------------------------------------
$COMPOSE build --no-cache --pull
$COMPOSE up -d --force-recreate

# --------------------------------------------------
# 2. Download resources from the manifest
#    (runs inside the container, where the project is mounted)
# --------------------------------------------------
echo "Downloading resources from the manifest..."
$COMPOSE exec text2kg python -m manifest download

# --------------------------------------------------
# 3. Restart Neo4j so it loads the downloaded plugins
# --------------------------------------------------
$COMPOSE restart neo4j

echo "Installation complete."