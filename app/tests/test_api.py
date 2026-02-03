#!/usr/bin/env python3
"""
Unit tests for API endpoints
"""

import pytest
import sys
import os

# Add app directory to Python path
sys.path.insert(0, '/workspace')

from app.api.server import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test /health endpoint returns 200 and correct status"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data is not None
    assert data['status'] == 'healthy'
    assert data['service'] == 'opencv-gpu-api'
    assert 'timestamp' in data


def test_root_endpoint(client):
    """Test / endpoint returns API documentation"""
    response = client.get('/')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data is not None
    assert data['service'] == 'OpenCV GPU API Server'
    assert 'endpoints' in data
    assert '/health' in data['endpoints']


def test_container_stats_endpoint(client):
    """Test /api/container/stats endpoint"""
    response = client.get('/api/container/stats')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data is not None
    assert data['status'] == 'success'
    assert 'uptime_seconds' in data
    assert 'uptime_human' in data
    assert data['container_name'] == 'opencv-gpu'


def test_pidtest_status_endpoint(client):
    """Test /api/pidtest/status endpoint when service is not running"""
    response = client.get('/api/pidtest/status')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data is not None
    # Status should indicate not running or provide status info
    assert 'status' in data or 'running' in data


def test_invalid_endpoint(client):
    """Test that invalid endpoints return 404"""
    response = client.get('/api/invalid/endpoint')
    assert response.status_code == 404


def test_health_endpoint_method(client):
    """Test that /health only accepts GET requests"""
    response = client.post('/health')
    assert response.status_code == 405  # Method Not Allowed

