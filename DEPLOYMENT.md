# Multi-Agent Council System - Deployment Guide

## Overview

This guide covers deploying the two-phase council system with synthesis across T4 and A100 GPU nodes.

**Architecture:**
- **T4 Node**: 4 GPUs running council members with shared-prefill
- **A100 Node**: 4 GPUs running single model with 4-way Tensor Parallelism for two-phase synthesis
- **Orchestrator**: Coordinates the two-phase workflow with parallel execution

## Prerequisites

### Hardware Requirements

**T4 Node (Council Members):**
- 4x NVIDIA T4 GPUs (16GB each)
- 64GB+ RAM
- Fast inter-GPU connectivity (PCIe Gen3+)
- Runs: 3 council members + 1 prefill server

**A100 Node (Synthesis with 4-way TP):**
- 4x NVIDIA A100-80GB GPUs (**80GB required for Llama-2-70B**)
- 128GB+ RAM
- **NVLink connectivity required** (for efficient Tensor Parallelism)
- All 4 GPUs dedicated to single model with TP=4

**Orchestrator Node:**
- CPU-only machine (no GPU required)
- Network connectivity to both GPU nodes
- Coordinates two-phase workflow

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
# Set HuggingFace cache directory (optional)
export HF_HOME=/path/to/model/cache

# Pre-download models (recommended to avoid first-run delays)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

# T4 model (smaller, ~5GB)
print('Downloading Phi-2 for T4 council...')
AutoModelForCausalLM.from_pretrained('microsoft/phi-2')

# A100 model (large, ~140GB)
print('Downloading Llama-2-70B for A100 synthesis...')
# Note: Requires HuggingFace authentication for Llama-2
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
"
```

**For Llama-2-70B Access:**
1. Visit: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
2. Accept the license agreement
3. Create a token at: https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login`

**Storage Requirements:**
- Phi-2: ~5GB
- Llama-2-70B: ~140GB (stored once, used with TP across 4 GPUs)
- Total: ~145GB disk space

### 3. Configure Endpoints

Edit `config/endpoints.yaml` to match your network setup:

```yaml
# Council members on T4 node
members:
  - id: "member_1"
    url: "http://t4-node-hostname:8001"
    temperature: 0.3
    node: "t4"
    gpu: 1
  
  - id: "member_2"
    url: "http://t4-node-hostname:8002"
    temperature: 0.7
    node: "t4"
    gpu: 2
  
  - id: "member_3"
    url: "http://t4-node-hostname:8003"
    temperature: 1.0
    node: "t4"
    gpu: 3

# Synthesis server on A100 node (4-way TP)
synthesis:
  url: "http://a100-node-hostname:8020"
  node: "a100"
  gpus: [0, 1, 2, 3]

# Timeout settings (seconds)
timeouts:
  generation: 120
  synthesis: 180
  cache_timeout: 300  # 5 minutes for KV cache
```

## Deployment

### Option 1: Single-Node Testing (All on one machine)

**Requirements:** 1 machine with 8 GPUs (4 T4 + 4 A100)

```bash
# Terminal 1: T4 Council (uses GPUs 0-3)
cd serving/t4_cluster
export CUDA_VISIBLE_DEVICES=0,1,2,3
python launcher.py

# Terminal 2: A100 Synthesis Server with 4-way TP (uses GPUs 4-7)
cd serving/a100_cluster
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29500
python synthesis_server.py

# Terminal 3: Orchestrator
cd orchestrator
python main.py --mode two_phase
```

### Option 2: Multi-Node Production (Recommended)

#### T4 Node Setup (Council Members)

```bash
# SSH to T4 node
ssh user@t4-node

cd multi-agent-test
source venv/bin/activate

# Set environment for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0

# Launch T4 council cluster (3 members + prefill server)
cd serving/t4_cluster
python launcher.py
```

**Expected Output:**
```
INFO: GPU 0 - Prefill Server started on port 8000
INFO: GPU 1 - Member 1 started on port 8001 (temp=0.3)
INFO: GPU 2 - Member 2 started on port 8002 (temp=0.7)
INFO: GPU 3 - Member 3 started on port 8003 (temp=1.0)
INFO: T4 Council ready for requests
```

#### A100 Node Setup (4-way TP Synthesis Server)

```bash
# SSH to A100 node
ssh user@a100-node

cd multi-agent-test
source venv/bin/activate

# Set environment for 4-way Tensor Parallelism
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=5  # Enable GPU Direct RDMA for NVLink
export MASTER_PORT=29500

# Launch synthesis server with 4-way TP
cd serving/a100_cluster
python synthesis_server.py
```

**Expected Output:**
```
INFO: [Rank 0] Initializing synthesis worker on GPU 0
INFO: [Rank 1] Initializing synthesis worker on GPU 1
INFO: [Rank 2] Initializing synthesis worker on GPU 2
INFO: [Rank 3] Initializing synthesis worker on GPU 3
INFO: Loading Llama-2-70b-chat-hf with 4-way Tensor Parallelism...
INFO: [Rank 0] Model shard loaded (35GB)
INFO: [Rank 1] Model shard loaded (35GB)
INFO: [Rank 2] Model shard loaded (35GB)
INFO: [Rank 3] Model shard loaded (35GB)
INFO: NCCL process group initialized
INFO: Synthesis server ready on port 8020
```

#### Orchestrator Setup

```bash
# SSH to orchestrator node (can be any machine with network access)
ssh user@orchestrator-node

cd multi-agent-test
source venv/bin/activate

# Update config/endpoints.yaml with actual hostnames/IPs:
# - members: http://t4-node:800X
# - synthesis: http://a100-node:8020

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

### Model Configuration

Edit `config/models.yaml`:

```yaml
# T4 Council Members
t4_model:
  name: "microsoft/phi-2"
  node: "t4"
  gpus: [0, 1, 2, 3]  # GPU 0 = prefill, GPUs 1-3 = members
  precision: "bf16"

# A100 Synthesis (4-way TP)
synthesis_model:
  name: "meta-llama/Llama-2-70b-chat-hf"
  node: "a100"
  gpus: [0, 1, 2, 3]  # All 4 GPUs for TP
  tensor_parallel: 4
  precision: "bf16"
  
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

