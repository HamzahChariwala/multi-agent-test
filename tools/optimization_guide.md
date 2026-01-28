# Optimization Guide

## Overview

This guide provides strategies for optimizing the multi-agent council system based on profiling data.

## Key Metrics to Monitor

### 1. KV Transfer Latency

**What to measure:**
- Time to broadcast KV cache from prefill worker to decode workers (T4 cluster)
- Time to fork KV cache between large model GPUs (A100 cluster)

**Expected values:**
- Within-node (NVLink): 1-10 ms for typical KV caches
- PCIe: 10-50 ms depending on size

**Optimization strategies:**
- Reduce KV cache size by using fewer layers or smaller hidden dimensions
- Use compression for KV caches (quantization to int8)
- Batch multiple requests to amortize transfer overhead

### 2. Prefill Throughput

**What to measure:**
- Tokens processed per second during prefill phase
- Time from input to KV cache generation

**Expected values:**
- T4: 100-500 tokens/second for 7B models
- A100: 500-2000 tokens/second for 70B models

**Optimization strategies:**
- Use FlashAttention for faster attention computation
- Batch multiple prompts together
- Use tensor cores (ensure bf16/fp16 precision)
- Enable CUDA graphs for repeated operations

### 3. Memory Usage

**What to measure:**
- Peak GPU memory during prefill
- KV cache memory per request
- Model weights memory

**Expected values:**
- 7B model (bf16): ~14 GB for weights
- 70B model (bf16): ~140 GB for weights
- KV cache: ~1-10 GB depending on sequence length

**Optimization strategies:**
- Use PagedAttention to reduce KV cache fragmentation
- Implement KV cache eviction for long sequences
- Use model quantization (int8, int4)
- Enable activation checkpointing

## Bottleneck Analysis

### Scenario 1: High KV Transfer Time

**Symptoms:**
- KV transfer takes >50ms
- Decode workers idle waiting for KV cache

**Solutions:**
1. Check GPU topology: `nvidia-smi topo -m`
2. Ensure NVLink is enabled: `NCCL_P2P_DISABLE=0`
3. Reduce KV cache size via quantization
4. Consider doing full prefill on each worker if transfer is too slow

### Scenario 2: Low Prefill Throughput

**Symptoms:**
- Prefill takes >1s for typical prompts
- High latency before generation starts

**Solutions:**
1. Enable FlashAttention or FlashAttention-2
2. Use continuous batching
3. Check if model is using tensor cores (bf16/fp16)
4. Profile attention kernels specifically

### Scenario 3: OOM Errors

**Symptoms:**
- CUDA out of memory errors
- System crashes during generation

**Solutions:**
1. Reduce batch size
2. Reduce maximum sequence length
3. Enable KV cache eviction
4. Use model sharding or quantization
5. Monitor with `nvidia-smi`

## Profiling Workflow

### 1. Capture Traces

```bash
# Enable profiling
export ENABLE_PROFILING=true

# Run workload
python orchestrator/main.py --mode example

# Traces saved to profiling_traces/
```

### 2. Analyze Traces

```bash
# Analyze specific GPU
python tools/analyze_profiling.py --gpu-id t4_gpu0 --request-id <request_id>

# Compare all GPUs
python tools/analyze_profiling.py
```

### 3. Identify Bottlenecks

Look for:
- Operations taking >100ms
- GPU idle time between operations
- Memory allocation spikes
- NCCL collective slowness

### 4. Apply Optimizations

Based on findings:
- Adjust batch sizes
- Tune NCCL parameters
- Enable kernel optimizations
- Modify KV cache strategy

### 5. Verify Improvements

Re-run with profiling and compare metrics.

## Advanced Optimizations

### NCCL Tuning

```bash
# Enable debug output
export NCCL_DEBUG=INFO

# Tune buffer sizes
export NCCL_BUFFSIZE=8388608

# Select best network interface
export NCCL_SOCKET_IFNAME=eth0

# Enable IB/RoCE if available
export NCCL_IB_DISABLE=0
```

### CUDA Optimization

```bash
# Enable TF32 for Ampere+
export NVIDIA_TF32_OVERRIDE=1

# Disable CUDA memory caching for profiling
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Enable CUDA graphs
export TORCH_CUDNN_V8_API_ENABLED=1
```

### Model-Specific

**For Llama-2:**
- Use RoPE caching
- Enable grouped-query attention (GQA)
- Use FP8 quantization on newer GPUs

**For Mistral:**
- Enable sliding window attention
- Use sparse attention patterns

## Performance Targets

### Generation Phase

| GPU Type | Model Size | Target Latency | Target Throughput |
|----------|------------|----------------|-------------------|
| T4       | 7B         | <2s            | 100+ tokens/s     |
| A100     | 70B        | <5s            | 200+ tokens/s     |

### KV Transfer

| Setup           | Target Latency |
|-----------------|----------------|
| T4 (PCIe)       | <20ms          |
| A100 (NVLink)   | <5ms           |

### End-to-End Council

| Phase      | Target Time |
|------------|-------------|
| Generate   | <10s        |
| Judge      | <5s         |
| Synthesize | <10s        |
| **Total**  | **<30s**    |

## Monitoring in Production

### Metrics to Track

1. **Latency percentiles** (p50, p95, p99)
2. **Throughput** (requests/second)
3. **GPU utilization** (target >80%)
4. **Memory usage** (stay <90% capacity)
5. **Error rates** (target <1%)

### Alerting Thresholds

- P95 latency >2x target
- GPU utilization <50% (underutilized)
- Memory usage >95% (risk of OOM)
- Error rate >5%

## Tools

- `nvidia-smi`: Monitor GPU usage
- `nsys`: NVIDIA Nsight Systems profiler
- `ncu`: NVIDIA Nsight Compute profiler
- `tensorboard`: Visualize profiling traces
- Chrome tracing: View detailed execution timeline

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [vLLM Documentation](https://docs.vllm.ai/)

