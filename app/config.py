#!/usr/bin/env python3
"""
Centralized configuration management
Loads configuration from environment variables
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Application
    APP_NAME = os.getenv('APP_NAME', 'jetson-cv-pipeline')
    APP_ENV = os.getenv('APP_ENV', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # MQTT Configuration
    MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
    MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
    MQTT_TOPIC_PREFIX = os.getenv('MQTT_TOPIC_PREFIX', '/docker')
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5001'))
    STREAM_PORT = int(os.getenv('STREAM_PORT', '5002'))
    
    # Paths
    WORKSPACE_ROOT = Path('/workspace')
    APP_ROOT = WORKSPACE_ROOT / 'app'
    CONFIG_ROOT = WORKSPACE_ROOT / 'config'
    
    # Scripts paths
    GPU_OPERATIONS_SCRIPT = APP_ROOT / 'services' / 'gpu_operations.py'
    PID_TEST_SCRIPT = APP_ROOT / 'services' / 'pid_test.py'
    
    # Timezone
    TZ = os.getenv('TZ', 'America/New_York')
    
    @classmethod
    def is_production(cls):
        """Check if running in production environment"""
        return cls.APP_ENV.lower() == 'production'
    
    @classmethod
    def is_development(cls):
        """Check if running in development environment"""
        return cls.APP_ENV.lower() == 'development'
    
    @classmethod
    def get_mqtt_topic(cls, subtopic):
        """Get full MQTT topic path"""
        return f"{cls.MQTT_TOPIC_PREFIX}/{subtopic}"


# Create singleton instance
config = Config()

