#!/bin/bash
# Deploy application to production

set -e

echo "========================================="
echo "Deploying to Production"
echo "========================================="
echo ""

# Change to docker directory
cd "$(dirname "$0")/../docker"

# Load environment variables
if [ -f ../config/.env ]; then
    echo "Loading environment variables from config/.env"
    export $(cat ../config/.env | grep -v '^#' | xargs)
fi

# Stop existing containers
echo "Stopping existing containers..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

echo ""

# Start containers
echo "Starting containers..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo ""
echo "========================================="
echo "Deployment complete!"
echo "========================================="
echo ""

# Display running containers
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

