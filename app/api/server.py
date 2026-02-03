#!/usr/bin/env python3
"""
Flask REST API Server for OpenCV GPU Operations
Runs inside opencv-gpu container
Exposes endpoints for Node-RED to call

Lightweight API server - GPU operations are executed as subprocesses
This ensures GPU crashes don't affect the API server
"""

from flask import Flask, jsonify, request
from datetime import datetime
import logging
import subprocess
import json
import os
import time
import sys

# Add app directory to Python path
sys.path.insert(0, '/workspace')

from app.config import config

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track container startup time
CONTAINER_START_TIME = datetime.now()
CONTAINER_START_TIMESTAMP = time.time()


def run_gpu_operation(operation, timeout=30):
    """
    Run GPU operation as subprocess

    Args:
        operation: 'info' or 'test'
        timeout: Maximum execution time in seconds

    Returns:
        dict: Result from GPU operation
    """
    try:
        # Build command
        cmd = ['python3', str(config.GPU_OPERATIONS_SCRIPT), f'--{operation}']

        logger.info(f"Executing GPU operation: {operation}")

        # Run subprocess with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(config.APP_ROOT / 'services')
        )

        # Parse JSON output from stdout
        if result.stdout:
            output = json.loads(result.stdout)
            logger.info(f"GPU operation {operation} completed with status: {output.get('status', 'unknown')}")
            return output
        else:
            # No output - return error
            error_msg = result.stderr if result.stderr else "No output from GPU operation"
            logger.error(f"GPU operation {operation} failed: {error_msg}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }

    except subprocess.TimeoutExpired:
        logger.error(f"GPU operation {operation} timed out after {timeout} seconds")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Operation timed out after {timeout} seconds"
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPU operation output: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Failed to parse output: {str(e)}",
            "raw_output": result.stdout if 'result' in locals() else None
        }

    except Exception as e:
        logger.error(f"Unexpected error running GPU operation {operation}: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def run_pid_test_operation(operation, timeout=10):
    """
    Run PID test service operation as subprocess

    Args:
        operation: 'start', 'stop', or 'status'
        timeout: Maximum execution time in seconds

    Returns:
        dict: Result from PID test operation
    """
    try:
        # Build command
        cmd = ['python3', str(config.PID_TEST_SCRIPT), f'--{operation}']

        logger.info(f"Executing PID test operation: {operation}")

        # For 'start' operation, use Popen to run in background
        if operation == 'start':
            # Start process in background
            # CRITICAL: Do NOT capture stderr/stdout to avoid pipe buffer deadlock
            # Let output go to Docker logs instead
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                cwd=str(config.APP_ROOT / 'services')
            )

            # Return immediately with process info
            logger.info(f"PID test service started in background (PID: {process.pid})")
            return {
                "status": "started",
                "timestamp": datetime.now().isoformat(),
                "message": "PID test service started in background",
                "process_pid": process.pid
            }

        else:
            # For 'stop' and 'status', wait for completion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(config.APP_ROOT / 'services')
            )

            # Parse JSON output from stdout
            if result.stdout:
                output = json.loads(result.stdout)
                logger.info(f"PID test operation {operation} completed")
                return output
            else:
                error_msg = result.stderr if result.stderr else "No output"
                logger.error(f"PID test operation {operation} failed: {error_msg}")
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg
                }

    except Exception as e:
        logger.error(f"Error running PID test operation {operation}: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "opencv-gpu-api",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/container/stats', methods=['GET'])
def container_stats():
    """Get container statistics without using Docker socket"""
    current_time = time.time()
    uptime_seconds = current_time - CONTAINER_START_TIMESTAMP

    # Calculate human-readable uptime
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    uptime_human = f"{days}d {hours}h {minutes}m {seconds}s"

    return jsonify({
        "status": "success",
        "container_name": "opencv-gpu",
        "api_server_started_at": CONTAINER_START_TIME.isoformat(),
        "current_time": datetime.now().isoformat(),
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_human": uptime_human,
        "note": "Uptime resets when container or API server restarts"
    }), 200


@app.route('/api/gpu/test', methods=['GET', 'POST'])
def gpu_test():
    """Run GPU test via subprocess"""
    logger.info("GPU test endpoint called")
    result = run_gpu_operation('test', timeout=30)
    status_code = 200 if result.get("status") == "success" else 500
    return jsonify(result), status_code


@app.route('/api/gpu/info', methods=['GET'])
def gpu_info():
    """Get GPU and OpenCV information via subprocess"""
    logger.info("GPU info endpoint called")
    result = run_gpu_operation('info', timeout=10)
    status_code = 200 if result.get("status") == "success" else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/start', methods=['POST'])
def pidtest_start():
    """Start PID test service in background"""
    logger.info("PID test start endpoint called")
    result = run_pid_test_operation('start', timeout=5)
    status_code = 200 if result.get("status") == "started" else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/stop', methods=['POST'])
def pidtest_stop():
    """Stop PID test service"""
    logger.info("PID test stop endpoint called")
    result = run_pid_test_operation('stop', timeout=5)
    status_code = 200 if result.get("status") in ["stopped", "warning"] else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/status', methods=['GET'])
def pidtest_status():
    """Check PID test service status"""
    logger.info("PID test status endpoint called")
    result = run_pid_test_operation('status', timeout=5)
    return jsonify(result), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        "service": "OpenCV GPU API Server",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/api/container/stats": "Get container statistics",
            "/api/gpu/test": "Run GPU test (GET or POST)",
            "/api/gpu/info": "Get GPU and OpenCV information",
            "/api/pidtest/start": "Start PID test service (POST)",
            "/api/pidtest/stop": "Stop PID test service (POST)",
            "/api/pidtest/status": "Check PID test service status (GET)"
        },
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    logger.info("Starting OpenCV GPU API Server (Lightweight Mode)...")
    logger.info(f"Environment: {config.APP_ENV}")
    logger.info("GPU operations will be executed as subprocesses")

    # Check if GPU operations script exists
    if config.GPU_OPERATIONS_SCRIPT.exists():
        logger.info(f"GPU operations script found: {config.GPU_OPERATIONS_SCRIPT}")
    else:
        logger.warning(f"GPU operations script not found: {config.GPU_OPERATIONS_SCRIPT}")

    # Run Flask server
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False, threaded=True)

