# Multi-Agent Council System - Deployment Guide

## Overview

This guide covers deploying the distributed multi-agent council system across T4 and A100 GPU nodes.

## Prerequisites

### Hardware Requirements

**T4 Node:**
- 4x NVIDIA T4 GPUs (16GB each)
- 64GB+ RAM
- Fast inter-GPU connectivity (PCIe Gen3+)

**A100 Node:**
- 4x NVIDIA A100 GPUs (40GB or 80GB)
- 128GB+ RAM
- NVLink connectivity between GPUs

**Orchestrator Node:**
- CPU-only or 1 GPU (optional)
- Network connectivity to both GPU nodes

### Software Requirements

- Python 3.9+
- CUDA 11.8+ or 12.0+
- PyTorch 2.1.0+
- See `requirements.txt` for complete dependencies

## Installation

### 1. Clone and Setup

```bash
git clone <repository>
cd multi-agent-test

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/model/cache

# Models will be downloaded automatically on first run
# or pre-download:
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
"
```

### 3. Configure Endpoints

Edit `config/endpoints.yaml` to match your network setup:

```yaml
members:
  - id: "member_1"
    url: "http://t4-node-hostname:8001"  # Update hostname
    temperature: 0.3
  # ... etc

chairman:
  url: "http://a100-node-hostname:8020"  # Update hostname
```

## Deployment

### Option 1: Single-Node Testing (All on one machine)

**Requirements:** 1 machine with 8 GPUs

```bash
# Terminal 1: T4 Cluster (uses GPUs 0-3)
cd serving/t4_cluster
python launcher.py

# Terminal 2: A100 Large Model Twin (uses GPUs 4-5)
cd serving/a100_cluster
CUDA_VISIBLE_DEVICES=4,5 python large_model_twin.py

# Terminal 3: Chairman (uses GPUs 6-7)
cd serving/a100_cluster
CUDA_VISIBLE_DEVICES=6,7 python chairman_tp.py

# Terminal 4: Orchestrator
cd orchestrator
python main.py --mode example
```

### Option 2: Multi-Node Production (Recommended)

#### T4 Node Setup

```bash
# SSH to T4 node
ssh user@t4-node

cd multi-agent-test
source venv/bin/activate

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0

# Launch T4 cluster
cd serving/t4_cluster
python launcher.py
```

#### A100 Node Setup - Large Model Twin

```bash
# SSH to A100 node - Terminal 1
ssh user@a100-node

cd multi-agent-test
source venv/bin/activate

# Use GPUs 0-1 for large model
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export MASTER_PORT=29501

cd serving/a100_cluster
python large_model_twin.py
```

#### A100 Node Setup - Chairman

```bash
# SSH to A100 node - Terminal 2
# Use GPUs 2-3 for chairman
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_DEBUG=INFO
export MASTER_PORT=29502

cd serving/a100_cluster
python chairman_tp.py
```

#### Orchestrator Setup

```bash
# SSH to orchestrator node (can be any machine)
ssh user@orchestrator-node

cd multi-agent-test
source venv/bin/activate

# Update config/endpoints.yaml with actual hostnames/IPs

cd orchestrator
python main.py --mode interactive
```

## Configuration

### Profiling

Enable/disable profiling:

```bash
# Enable profiling (generates traces)
export ENABLE_PROFILING=true

# Disable profiling (production)
export ENABLE_PROFILING=false
```

Configure per-GPU profiling in `config/profiling.yaml`.

### Model Selection

Edit `config/models.yaml` to use different models:

```yaml
small_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"  # Or any compatible model
  
large_model:
  name: "meta-llama/Llama-2-70b-chat-hf"  # Or any compatible model
```

### Network Tuning

For best performance:

```bash
# NCCL optimizations
export NCCL_SOCKET_IFNAME=eth0  # Use your network interface
export NCCL_IB_DISABLE=0        # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=3     # GPU Direct RDMA

# PyTorch optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
```

## Verification

### Health Checks

```bash
# Check T4 members
curl http://t4-node:8001/health
curl http://t4-node:8002/health
curl http://t4-node:8003/health

# Check A100 members
curl http://a100-node:8010/health
curl http://a100-node:8011/health

# Check Chairman
curl http://a100-node:8020/health
```

### Run Tests

```bash
# Unit tests
cd tests
bash run_tests.sh

# Integration test with orchestrator
cd orchestrator
python main.py --mode example
```

## Monitoring

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Per-process GPU usage
nvidia-smi pmon -i 0,1,2,3
```

### Log Monitoring

```bash
# Orchestrator logs
tail -f orchestrator.log

# Worker logs (check stdout/stderr)
```

### Profiling Analysis

```bash
# Analyze profiling traces
python tools/analyze_profiling.py

# Compare GPU performance
python tools/analyze_profiling.py --trace-dir ./profiling_traces
```

## Troubleshooting

### Issue: Services not starting

**Check:**
1. GPU availability: `nvidia-smi`
2. Port conflicts: `netstat -tuln | grep 800[0-9]`
3. CUDA/PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: NCCL initialization failures

**Solutions:**
```bash
# Check NCCL debug output
export NCCL_DEBUG=INFO

# Try disabling P2P if hardware doesn't support it
export NCCL_P2P_DISABLE=1

# Specify network interface
export NCCL_SOCKET_IFNAME=eth0
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in code
2. Use smaller models
3. Enable gradient checkpointing
4. Reduce max sequence length
5. Use model quantization (int8)

### Issue: Slow KV transfers

**Check:**
1. GPU topology: `nvidia-smi topo -m`
2. NVLink connectivity
3. PCIe bandwidth

**Solutions:**
1. Enable P2P: `NCCL_P2P_DISABLE=0`
2. Use faster GPUs with better interconnect
3. Consider doing full prefill on each worker

## Performance Tuning

See [tools/optimization_guide.md](tools/optimization_guide.md) for detailed optimization strategies.

## Docker Deployment (Optional)

### Build Images

```bash
# T4 worker image
docker build -t council-t4-worker -f docker/Dockerfile.t4 .

# A100 worker image
docker build -t council-a100-worker -f docker/Dockerfile.a100 .

# Orchestrator image
docker build -t council-orchestrator -f docker/Dockerfile.orchestrator .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

## Security Considerations

1. **Network Security:**
   - Use VPN or private network for inter-node communication
   - Consider TLS for HTTP endpoints
   - Firewall rules to restrict access

2. **Model Security:**
   - Verify model checksums after download
   - Use private model registry for proprietary models
   - Monitor for prompt injection attacks

3. **Resource Limits:**
   - Set request rate limits
   - Implement timeout policies
   - Monitor for resource exhaustion

## Scaling

### Horizontal Scaling

To add more council members:

1. Add GPU nodes
2. Update `config/endpoints.yaml`
3. Modify `orchestrator/council_workflow.py` to include new members

### Vertical Scaling

To handle more concurrent requests:

1. Implement request batching
2. Use continuous batching (vLLM-style)
3. Add load balancer in front of members

## Maintenance

### Model Updates

```bash
# Download new model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('new-model')"

# Update config/models.yaml

# Restart services
```

### Log Rotation

```bash
# Setup logrotate for orchestrator.log
sudo logrotate -f /etc/logrotate.d/council
```

### Profiling Cleanup

```bash
# Clean old profiling traces
find profiling_traces/ -name "*.json" -mtime +7 -delete
```

## Support

For issues and questions:
- Check logs in `orchestrator.log`
- Review profiling traces in `profiling_traces/`
- See optimization guide in `tools/optimization_guide.md`

