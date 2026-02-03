#!/bin/bash
# Cleanup script to remove unused Docker images and cache

set -e

echo "========================================="
echo "Docker Cleanup Script"
echo "========================================="
echo ""

# Function to display size
display_disk_usage() {
    echo "Current disk usage:"
    df -h | grep -E "Filesystem|/$"
    echo ""
}

# Display initial disk usage
echo "BEFORE CLEANUP:"
display_disk_usage

# Clean Docker build cache
echo "Cleaning Docker build cache..."
docker builder prune -af
echo ""

# Clean unused images
echo "Cleaning unused Docker images..."
docker image prune -a -f
echo ""

# Clean unused volumes
echo "Cleaning unused Docker volumes..."
docker volume prune -f
echo ""

# Clean unused networks
echo "Cleaning unused Docker networks..."
docker network prune -f
echo ""

# Clean stopped containers
echo "Removing stopped containers..."
docker container prune -f
echo ""

# Display final disk usage
echo "AFTER CLEANUP:"
display_disk_usage

echo "========================================="
echo "Cleanup complete!"
echo "========================================="

