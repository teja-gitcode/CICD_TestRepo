# Jetson CV Pipeline

Production-grade CI/CD pipeline for NVIDIA Jetson AGX Orin computer vision applications.

## Features

- GPU-accelerated OpenCV operations
- Flask REST API server
- MQTT integration
- Docker containerization with NVIDIA runtime
- Azure DevOps CI/CD pipeline
- Automated testing
- Resource management and monitoring

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp config/.env.example config/.env

# Edit config/.env with your settings
nano config/.env
```

### 2. Build and Deploy

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Build Docker images
./scripts/build.sh

# Deploy containers
cd docker
docker compose up -d

# Check health
../scripts/health_check.sh
```

## Directory Structure

```
jetson-cv-pipeline/
├── app/                    # Application code
│   ├── api/               # Flask API server
│   ├── services/          # Business logic
│   └── tests/             # Unit tests
├── config/                # Configuration files
├── docker/                # Docker configuration
├── scripts/               # Automation scripts
├── myagent/              # Azure Pipelines agent
└── wheels/               # Custom Python wheels
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/container/stats` - Container statistics
- `GET /api/gpu/info` - GPU information
- `GET|POST /api/gpu/test` - Run GPU test
- `POST /api/pidtest/start` - Start PID test service
- `POST /api/pidtest/stop` - Stop PID test service
- `GET /api/pidtest/status` - PID test status

## Scripts

- `scripts/build.sh` - Build Docker images
- `scripts/deploy.sh` - Deploy to production
- `scripts/cleanup.sh` - Clean up Docker resources
- `scripts/health_check.sh` - Verify deployment

## Testing

```bash
# Run unit tests
docker exec opencv-gpu pytest /workspace/app/tests/ -v
```

## Monitoring

Access Portainer at http://localhost:9001 for container monitoring.

## CI/CD Pipeline

The project uses Azure DevOps with a self-hosted agent running on the Jetson device.

See `myagent/README.md` for agent setup instructions.

## Resource Limits

- OpenCV GPU Container:
  - Memory: 24GB limit, 8GB reservation
  - CPU: 6 cores limit, 4 cores reservation
  - GPU: All NVIDIA GPUs with full capabilities

## License

Proprietary - Ashley Furniture Industries

