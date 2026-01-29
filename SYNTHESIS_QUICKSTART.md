# A100 Synthesis Server - Quick Start Guide

## Overview

This guide will help you quickly get the A100 Synthesis Server running and test it.

**What it does:**
- Runs a single large model (Llama-2-70B) with 4-way Tensor Parallelism
- Two-phase workflow with KV cache reuse for efficient synthesis
- Processes council results and generates final synthesis

---

## Prerequisites

- 4x NVIDIA A100-80GB GPUs with NVLink
- CUDA 11.8+ or 12.0+
- Python environment with dependencies installed

---

## Quick Start (5 Steps)

### Step 1: Set Environment Variables

```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
```

### Step 2: Start Synthesis Server

```bash
python serving/a100_cluster/synthesis_server.py
```

**Expected output:**
```
======================================================================
A100 Synthesis Server - 4-way Tensor Parallelism
======================================================================
Model: meta-llama/Llama-2-70b-chat-hf
Precision: bf16
Port: 8020
Cache timeout: 300s
Profiling: False
======================================================================
[Rank 0] Initializing SynthesisWorker
[Rank 1] Initializing SynthesisWorker
[Rank 2] Initializing SynthesisWorker
[Rank 3] Initializing SynthesisWorker
...
[Rank 0] Model loaded successfully (XXX params sharded)
[Rank 0] Starting HTTP server on port 8020
```

**Wait for:** "Uvicorn running on http://0.0.0.0:8020"

### Step 3: Test Phase 1 (In New Terminal)

```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate

python test_synthesis_phase1.py
```

**This tests:**
- Initial context prefill
- KV cache storage
- Server responsiveness

**Expected:** ~600ms prefill time

### Step 4: Test Full Workflow

```bash
python test_synthesis_full.py
```

**This tests:**
- Phase 1: Prefill initial context
- Phase 2: Append council results and synthesize
- KV cache reuse working properly

**Expected:** 
- Phase 1: ~600ms
- Phase 2: ~2-3s (prefill + generation)

### Step 5: Run All Tests

```bash
./run_synthesis_tests.sh all
```

---

## Command Reference

### Start Server

```bash
# Standard server (5-minute cache timeout)
python serving/a100_cluster/synthesis_server.py

# With profiling enabled
python serving/a100_cluster/synthesis_server.py --profiling-enabled

# Custom cache timeout (for testing)
python serving/a100_cluster/synthesis_server.py --cache-timeout 30

# Different model
python serving/a100_cluster/synthesis_server.py --model-name mistralai/Mixtral-8x7B-Instruct-v0.1

# Custom port
python serving/a100_cluster/synthesis_server.py --port 8021
```

### Run Tests

```bash
# Individual tests
python test_synthesis_phase1.py        # Phase 1 only
python test_synthesis_full.py          # Full two-phase workflow
python test_synthesis_timeout.py       # Cache expiration (needs test server)

# Test runner
./run_synthesis_tests.sh phase1        # Phase 1 test
./run_synthesis_tests.sh full          # Full workflow test
./run_synthesis_tests.sh all           # All tests

# Manual API testing
curl http://localhost:8020/health
curl http://localhost:8020/debug/cache_status
curl http://localhost:8020/debug/tp_status
```

---

## API Usage Examples

### Phase 1: Prefill Initial Context

```bash
curl -X POST http://localhost:8020/prefill_initial \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "context": "Your initial context here...",
    "metadata": {"source": "test"}
  }'
```

**Response:**
```json
{
  "status": "success",
  "request_id": "test_001",
  "cache_stored": true,
  "context_tokens": 1534,
  "prefill_time_ms": 450.2,
  "cache_size_mb": 2.5
}
```

### Phase 2: Synthesize Final

```bash
curl -X POST http://localhost:8020/synthesize_final \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "appended_text": "Council results here...",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "status": "success",
  "request_id": "test_001",
  "synthesis": "Generated synthesis text...",
  "num_tokens": 342,
  "prefill_time_ms": 120.5,
  "decode_time_ms": 2100.3,
  "total_time_ms": 2220.8
}
```

---

## Monitoring

### Check Server Health

```bash
curl http://localhost:8020/health
```

### Check Active Caches

```bash
curl http://localhost:8020/debug/cache_status
```

**Response:**
```json
{
  "active_caches": [
    {
      "request_id": "test_001",
      "age_seconds": 45.2,
      "seq_len": 1534,
      "context_preview": "Your initial context...",
      "expires_in": 254.8
    }
  ],
  "active_session_id": "test_001",
  "cache_timeout": 300
}
```

### Check Tensor Parallel Status

```bash
curl http://localhost:8020/debug/tp_status
```

**Response:**
```json
{
  "rank": 0,
  "world_size": 4,
  "device": "cuda:0",
  "model": "meta-llama/Llama-2-70b-chat-hf",
  "tp_info": {
    "initialized": true,
    "rank": 0,
    "world_size": 4,
    "backend": "nccl"
  }
}
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check memory per GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

**Expected memory per GPU:** ~40GB / 80GB (50% utilization)

---

## Troubleshooting

### Server won't start

**Problem:** "Address already in use"
```bash
# Find and kill process on port 8020
lsof -ti:8020 | xargs kill -9

# Or use different port
python serving/a100_cluster/synthesis_server.py --port 8021
```

**Problem:** "NCCL initialization failed"
```bash
# Check CUDA_VISIBLE_DEVICES shows 4 GPUs
echo $CUDA_VISIBLE_DEVICES  # Should be: 0,1,2,3

# Check GPUs are available
nvidia-smi

# Try different MASTER_PORT
export MASTER_PORT=29501
```

### Phase 2 fails with "cache not found"

**Causes:**
1. Phase 1 was never called
2. Cache expired (>5 minutes elapsed)
3. Wrong request_id

**Solution:**
- Always call Phase 1 before Phase 2
- Use same request_id for both phases
- Complete workflow within 5 minutes

### Slow performance

**Check:**
1. GPU utilization: Should be ~90-100% during generation
2. NCCL communication: Check logs for NCCL warnings
3. Model loaded correctly: All 4 ranks should show "Model loaded successfully"

**If slow:**
```bash
# Enable NVLink optimizations
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_DISABLE=1

# Restart server with profiling to identify bottleneck
python serving/a100_cluster/synthesis_server.py --profiling-enabled
```

### Out of memory

**Symptoms:** "CUDA out of memory" during model load

**Solutions:**
1. Check model size: Llama-2-70B requires 80GB A100s (not 40GB)
2. Ensure 4-way TP is working: Each GPU should use ~35GB
3. Close other processes using GPU memory

```bash
# Check GPU memory before starting
nvidia-smi

# Kill any lingering processes
pkill -f synthesis_server
```

---

## Performance Expectations

### Phase 1 (Prefill Initial Context)
- **Input:** 1000-2000 tokens
- **Time:** 400-600ms
- **Memory:** ~2.5GB KV cache per GPU

### Phase 2 (Synthesis)
- **Incremental prefill:** 300-500 new tokens → 120-200ms
- **Generation:** 400 tokens @ 5-6 tok/s → 2000-2500ms
- **Total:** 2.1-2.7 seconds

### Memory Usage (per GPU)
- Model weights: 35GB
- KV cache: 2.5GB
- Activations: 2GB
- **Total: ~40GB / 80GB**

---

## Next Steps

1. **Test with T4 Council:** Integrate with T4 node for full workflow
2. **Performance Tuning:** Profile and optimize NCCL communication
3. **Production Deployment:** Update orchestrator to use synthesis server

See `markdowns/A100_SYNTHESIS_ARCHITECTURE.md` for detailed architecture documentation.

---

## Support

**Logs:** Check terminal where server is running

**Profiling:** Traces saved to `profiling_traces/a100_gpu{0-3}/`

**Documentation:**
- Architecture: `markdowns/A100_SYNTHESIS_ARCHITECTURE.md`
- Deployment: `DEPLOYMENT.md`
- Implementation: `markdowns/IMPLEMENTATION_ROADMAP.md`

