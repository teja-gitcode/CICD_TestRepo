#!/bin/bash
# Build Docker images

set -e

echo "========================================="
echo "Building Docker Images"
echo "========================================="
echo ""

# Change to docker directory
cd "$(dirname "$0")/../docker"

# Load environment variables
if [ -f ../config/.env ]; then
    echo "Loading environment variables from config/.env"
    export $(cat ../config/.env | grep -v '^#' | xargs)
fi

# Build OpenCV GPU image
echo "Building OpenCV GPU image..."
docker compose build opencv

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""

# Clean up dangling images (old untagged images after new build)
echo "Cleaning up dangling images..."
docker image prune -f
echo ""

# Display built images
echo "Built images:"
docker images | grep -E "REPOSITORY|jetson-cuda-opencv" || true

