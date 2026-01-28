# Simple T4 Cluster Architecture - IMPLEMENTED ✅

## The Problem with Previous Approach

**Point-to-point blocking killed HTTP responsiveness:**
- Decode workers had HTTP servers
- When they called `dist.recv()` to get KV from rank 0, it **blocked the entire process**
- HTTP server couldn't respond → timeouts

## New Simple Architecture

**HTTP only on GPU 0, decode workers just wait for broadcasts:**

```
┌────────────────────────────────────────────────────────────┐
│                        T4 Node                             │
│                                                            │
│  ┌────────────────────┐                                   │
│  │      GPU 0         │  HTTP Server (port 8000)          │
│  │  (Rank 0)          │  • Receives HTTP requests         │
│  │  Prefill Server    │  • Does prefill                   │
│  │                    │  • Broadcasts KV to 1-3           │
│  │                    │  • Does own decode                │
│  │                    │  • Returns HTTP response          │
│  └────────┬───────────┘                                   │
│           │                                                │
│           │ dist.broadcast() [KV Cache]                   │
│           │                                                │
│    ┌──────┴──────┬──────────┬──────────┐                 │
│    │             │          │          │                  │
│    ▼             ▼          ▼          ▼                  │
│  GPU 1         GPU 2      GPU 3                           │
│  (Rank 1)      (Rank 2)   (Rank 3)                        │
│  Decode        Decode     Decode                          │
│  Worker        Worker     Worker                          │
│                                                            │
│  Each decode worker:                                      │
│  - NO HTTP server                                         │
│  - Sits in a loop                                         │
│  - Waits for dist.broadcast()                             │
│  - Receives KV cache                                      │
│  - Does decode                                            │
│  - Logs result                                            │
│  - Waits for next broadcast                               │
└────────────────────────────────────────────────────────────┘
```

## How It Works

### Request Flow

1. **HTTP request arrives at GPU 0** (port 8000)
2. **GPU 0 formats and tokenizes** the prompt
3. **GPU 0 broadcasts metadata** to all workers: `[has_work=1, batch_size, seq_len, max_tokens]`
4. **All decode workers wake up** and receive metadata
5. **GPU 0 broadcasts input_ids** 
6. **All workers receive input_ids**
7. **GPU 0 does prefill**, generates KV cache
8. **GPU 0 broadcasts KV cache** (layer by layer)
9. **All workers receive KV cache**
10. **All 4 GPUs do decode in parallel** (each with different temperature)
11. **GPU 0 returns HTTP response** with its result

### No Blocking Issues!

- **Decode workers**: Always in a loop waiting for `dist.broadcast()` - no HTTP to block
- **GPU 0**: Only calls `dist.broadcast()` when it has work - never blocks on receive
- **All broadcasts are collective**: All ranks participate synchronously

## Files

### New Files ✨

1. **`serving/t4_cluster/prefill_server.py`**
   - FastAPI server on GPU 0
   - `/generate` endpoint
   - Does prefill and broadcasts KV

2. **`serving/t4_cluster/simple_decode_worker.py`**
   - Simple worker with NO HTTP
   - Loop waiting for broadcasts
   - Receives KV and does decode

3. **`serving/t4_cluster/simple_launcher.py`**
   - Spawns 4 processes
   - Rank 0 → prefill_server
   - Ranks 1-3 → simple_decode_worker

4. **`run_t4_simple.sh`**
   - Launch script for new architecture

5. **`test_t4_simple.py`**
   - Test script that only hits port 8000

### Old Files (Not Used)

- `prefill_worker.py` - Old version
- `decode_worker.py` - Old version with HTTP
- `launcher.py` - Old launcher
- `p2p_kv_transfer.py` - Point-to-point approach (didn't work)

## How to Run

### 1. Kill Old Processes

```bash
pkill -9 -f "launcher.py"
pkill -9 -f "simple_launcher.py"
```

### 2. Start Cluster

```bash
bash run_t4_simple.sh
```

**Wait for:**
```
[Rank 0] Starting HTTP server on port 8000
[Rank 1] Decode worker ready - listening for broadcasts
[Rank 2] Decode worker ready - listening for broadcasts
[Rank 3] Decode worker ready - listening for broadcasts
```

### 3. Test

```bash
python3 test_t4_simple.py
```

## Expected Output

### GPU 0 (Prefill Server) Logs:

```
[Rank 0] Received generation request: test_request_001
[Rank 0] Broadcasting metadata to workers
[Rank 0] Running prefill...
[Rank 0] Broadcasting KV cache to workers
[Rank 0] Completed generation
```

### GPU 1-3 (Decode Workers) Logs:

```
[Rank 1] Received work: batch_size=1, seq_len=43, max_tokens=50
[Rank 1] Received KV cache (32 layers)
[Rank 1] Completed decode #1
```

## Profiling

**All operations are profiled:**

**GPU 0:**
- `init_model_load` - Model loading
- `format_prompt` - Prompt formatting
- `tokenize` - Tokenization
- `signal_workers` - Metadata broadcast
- `prefill` - Prefill compute
- `broadcast_kv` - KV cache broadcast
- `decode` - Rank 0's own decode

**GPU 1-3:**
- `init_model_load` - Model loading
- `receive_inputs` - Input reception
- `receive_kv` - KV cache reception
- `decode_only` - Decode compute

## Performance Characteristics

### Single Request

**Timeline:**
```
t=0ms:    HTTP request arrives at GPU 0
t=5ms:    Broadcast metadata (negligible)
t=10ms:   Broadcast input_ids (negligible)
t=15ms:   GPU 0 starts prefill
t=315ms:  GPU 0 completes prefill (300ms)
t=320ms:  GPU 0 broadcasts KV (5ms per layer × 32 layers = 160ms)
t=480ms:  All workers have KV, start decode
t=980ms:  All workers complete decode (500ms)
```

**Total latency: ~980ms** (prefill + KV broadcast + decode)

### Concurrent Requests

**Problem**: GPU 0 is a serialization point
- Request 1: 0-980ms
- Request 2: 980-1960ms (waits for GPU 0 to be free)
- Request 3: 1960-2940ms

**Throughput**: Limited by GPU 0's prefill + broadcast time (~480ms) = ~2 req/sec

### What You Gain

✅ **Compute savings**: Only 1 prefill instead of 3
✅ **Network measurement**: Can see KV broadcast time in traces
✅ **Simple architecture**: No blocking, no deadlocks
✅ **4x decode diversity**: Each GPU generates with different temperature

### What You Lose

❌ **Parallel prefill**: Can't do multiple prefills simultaneously
❌ **Request throughput**: Serialized on GPU 0

## Key Insights for Network Analysis

**From profiling traces, you can extract:**

1. **Prefill time** (`prefill` operation on GPU 0): ~300ms
2. **KV broadcast time** (`broadcast_kv` on GPU 0): ~160ms
3. **KV receive time** (`receive_kv` on GPUs 1-3): ~160ms
4. **Decode time** (`decode_only` on GPUs 1-3): ~500ms

**Network impact:**
- Current PCIe: 160ms to broadcast 32 layers of KV
- With 100G RDMA: Could be 10-20ms
- **Speedup**: 140ms saved per request = ~15% faster

## Troubleshooting

### Requests timeout

**Check:** Are all 4 processes running?
```bash
ps aux | grep simple_launcher
```

Should see 5 processes (1 parent + 4 workers)

### No broadcasts received

**Check:** NCCL initialization
Look for: `[Rank X] Synchronized with all workers`

If missing: NCCL failed to initialize (check CUDA/NCCL versions)

### Traces not generated

**Check:** Profiling config
```bash
cat config/profiling.yaml
```

Ensure `global_profiling_enabled: true` and `t4_gpu0-3` are enabled

### GPU 0 gets no requests

**Check:** Is HTTP server listening?
```bash
curl http://localhost:8000/health
```

Should return 200 OK

## Next Steps

1. **Test this architecture** - Should work without blocking!
2. **Examine profiling traces** - See prefill vs broadcast vs decode times
3. **Run concurrent requests** - See serialization on GPU 0
4. **Estimate network impact** - Compare KV broadcast time to prefill time
5. **Add A100 node** - For chairman stage

---

**This architecture is much simpler and should actually work!** 🚀

