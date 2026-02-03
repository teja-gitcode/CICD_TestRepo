#!/bin/bash
# Health check script to verify deployment

set -e

echo "========================================="
echo "Health Check"
echo "========================================="
echo ""

# Load environment variables
if [ -f "$(dirname "$0")/../config/.env" ]; then
    export $(cat "$(dirname "$0")/../config/.env" | grep -v '^#' | xargs)
fi

API_PORT=${API_PORT:-5001}
API_URL="http://localhost:${API_PORT}"

# Check if API is responding
echo "Checking API health at ${API_URL}/health..."
response=$(curl -s -o /dev/null -w "%{http_code}" ${API_URL}/health || echo "000")

if [ "$response" -eq 200 ]; then
    echo "✅ API is healthy (HTTP $response)"
    
    # Get detailed health info
    echo ""
    echo "Health details:"
    curl -s ${API_URL}/health | python3 -m json.tool || echo "Could not parse JSON response"
    
    exit 0
else
    echo "❌ API health check failed (HTTP $response)"
    echo ""
    echo "Container status:"
    docker ps --filter "name=opencv-gpu" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "Recent logs:"
    docker logs --tail 20 opencv-gpu
    
    exit 1
fi

