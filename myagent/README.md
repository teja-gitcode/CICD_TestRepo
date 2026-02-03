# Azure Pipelines Agent Setup

This directory contains the Azure Pipelines self-hosted agent for the Jetson device.

## Installation

### 1. Download Agent

```bash
cd myagent

# Download the latest agent (ARM64 version for Jetson)
wget https://vstsagentpackage.azureedge.net/agent/3.236.1/vsts-agent-linux-arm64-3.236.1.tar.gz

# Extract
tar zxvf vsts-agent-linux-arm64-3.236.1.tar.gz
```

### 2. Configure Agent

```bash
# Run configuration
./config.sh

# You will be prompted for:
# - Azure DevOps URL: https://dev.azure.com/your-organization
# - Personal Access Token (PAT)
# - Agent pool name: Jetson-Self-Hosted
# - Agent name: jetson-agx-orin-01
# - Work folder: _work (default)
```

### 3. Install as Service

```bash
# Install as systemd service
sudo ./svc.sh install

# Start service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

## Verify Installation

```bash
# Check agent status
./run.sh --once

# View logs
journalctl -u vsts.agent.* -f
```

## Uninstall

```bash
# Stop service
sudo ./svc.sh stop

# Uninstall service
sudo ./svc.sh uninstall

# Remove agent configuration
./config.sh remove
```

## Troubleshooting

### Agent Not Connecting

1. Check network connectivity
2. Verify PAT token is valid
3. Check firewall settings
4. Review agent logs: `journalctl -u vsts.agent.* -n 100`

### Permission Issues

```bash
# Ensure agent user has Docker permissions
sudo usermod -aG docker $(whoami)

# Restart Docker service
sudo systemctl restart docker
```

## Notes

- The agent runs as a systemd service
- Logs are available via journalctl
- The `_work` directory contains build artifacts (gitignored)
- Agent credentials are stored in `.credentials` (gitignored)

