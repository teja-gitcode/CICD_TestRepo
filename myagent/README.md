# Azure Pipelines Agent

This directory contains the Azure Pipelines self-hosted agent for the Jetson CV Pipeline.

## Important Security Notice

**This entire folder is gitignored except for this README.**

The following files contain sensitive information and are NOT committed to Git:
- `.credentials` - PAT token
- `.credentials_rsaparams` - RSA credentials
- `.agent` - Agent configuration
- `config.sh`, `env.sh`, `svc.sh` - Agent scripts
- `bin/`, `externals/` - Agent binaries
- `_work/`, `_diag/` - Runtime files

## Setup Instructions

### 1. Download the Agent

Download the ARM64 version of the Azure Pipelines agent:

```bash
cd /home/afi/Documents/Teja/jetson-cv-pipeline/myagent

# Download ARM64 agent (NOT x64!)
wget https://vstsagentpackage.azureedge.net/agent/4.268.0/vsts-agent-linux-arm64-4.268.0.tar.gz

# Extract
tar zxvf vsts-agent-linux-arm64-4.268.0.tar.gz
```

**Important:** Make sure to download the **ARM64** version, not x64!

### 2. Configure the Agent

```bash
./config.sh
```

You will be prompted for:
- **Azure DevOps URL**: `https://dev.azure.com/your-organization`
- **PAT Token**: Your Personal Access Token
- **Agent Pool**: `MFG-EdgeAIML_arm64`
- **Agent Name**: `jetson-agx-orin-01` (or your preferred name)
- **Work folder**: `_work` (default)

### 3. Install as Service (Optional)

To run the agent as a systemd service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

Check status:
```bash
sudo ./svc.sh status
```

### 4. Run Interactively (Alternative)

To run the agent in the foreground:

```bash
./run.sh
```

## Verification

After configuration, verify the agent appears in Azure DevOps:
1. Go to Azure DevOps → Project Settings → Agent pools
2. Select `MFG-EdgeAIML_arm64`
3. You should see your agent listed and online

## Troubleshooting

### Architecture Error
If you get "cannot execute binary file: Exec format error":
- You downloaded the wrong architecture (x64 instead of ARM64)
- Delete everything and download the ARM64 version

### Agent Not Appearing
- Check PAT token has correct permissions (Agent Pools: Read & manage)
- Verify network connectivity to Azure DevOps
- Check agent logs in `_diag/` folder

## References

- [Azure Pipelines Agent Documentation](https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/agents)
- [Self-hosted Linux agents](https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/linux-agent)

