# Profiling Issues Fixed

## Issue 1: Can't See JSON Files in Editor

**Problem:** The trace JSON files are visible with `ls` but don't show up in your editor.

**Cause:** The files are **massive** (up to 336 MB). Most editors refuse to open or index files > 100MB for performance reasons.

**Solutions:**

### 1. View in Chrome Trace Viewer (Recommended)
```bash
# Open Chrome/Chromium
google-chrome chrome://tracing

# Or if on remote machine, download the file
scp user@remote:~/multi-agent-test/profiling_traces/t4_gpu0/test_request_001_trace.json ./

# Then drag-and-drop the JSON file into chrome://tracing
```

### 2. Use Command-Line Tools
```bash
# Install jq for JSON parsing
sudo apt-get install jq

# Find NCCL broadcast operations
jq '.traceEvents[] | select(.name == "nccl:broadcast")' \
  profiling_traces/t4_gpu0/test_request_001_trace.json | head -20

# Find operations by name
jq '.traceEvents[] | select(.cat == "user_annotation")' \
  profiling_traces/t4_gpu0/test_request_001_trace.json | less

# Get operation timing summary
jq '.traceEvents[] | select(.name == "prefill") | {name, dur_ms: (.dur/1000)}' \
  profiling_traces/t4_gpu0/test_request_001_trace.json
```

### 3. Use Python to Analyze
```python
import json
import pandas as pd

# Load trace
with open('profiling_traces/t4_gpu0/test_request_001_trace.json') as f:
    trace = json.load(f)

# Extract events
events = trace['traceEvents']

# Filter to user annotations (our custom operations)
user_ops = [e for e in events if e.get('cat') == 'user_annotation']
print(f"Found {len(user_ops)} user operations")

# Find broadcast operations
broadcasts = [e for e in events if 'broadcast' in e.get('name', '').lower()]
print(f"Found {len(broadcasts)} broadcast operations")

# Timing analysis
df = pd.DataFrame(user_ops)
print(df[['name', 'dur']].groupby('name').sum())
```

### 4. Configure Editor for Large Files

**VSCode:**
```json
// settings.json
{
  "files.maxMemoryForLargeFilesMB": 512
}
```

**Vim:**
```bash
# Add to ~/.vimrc
set maxmempattern=2000000
```

---

## Issue 2: ExecutionTraceObserver Not Generating Files

**Problem:** `execution_trace.enabled: true` in config, but no `*_et.json` files were being created.

**Root Cause:** The code was completely wrong. It tried to call `profiler.export_execution_trace()`, which **doesn't exist**.

### What Was Wrong

```python
# OLD CODE (BROKEN)
if self.et_file:
    profiler_kwargs["experimental_config"] = torch._C._profiler._ExperimentalConfig(
        verbose=True
    )

# ...later in __exit__...
if hasattr(self.profiler, 'export_execution_trace'):  # This method doesn't exist!
    self.profiler.export_execution_trace(self.et_file)
```

### The Fix

ExecutionTraceObserver is a **separate observer** that needs to be registered as a callback:

```python
# NEW CODE (WORKING)
if et_config.get("enabled", False):
    et_format = et_config.get("export_format", "json")
    self.et_file = str(self.output_dir / f"{self.request_id}_et.{et_format}")
    
    # Create and register ExecutionTraceObserver
    self.et_observer = torch.profiler.ExecutionTraceObserver()
    self.et_observer.register_callback(self.et_file)  # Register output file
    self.et_observer.start()  # Start observing

# ...later in __exit__...
if self.et_observer is not None:
    self.et_observer.stop()  # Stop observing
    self.et_observer.unregister_callback()  # Clean up
```

**Key insight:** ExecutionTraceObserver runs **independently** from the main profiler and captures the CPU-side execution graph.

---

## What You Get Now (Per GPU)

### 1. Chrome Trace (`*_trace.json`)
**Size:** 8-189 MB per GPU  
**Contains:**
- CUDA kernel launches and timing
- CPU operations
- Memory allocations
- Your custom annotations (`signal_workers`, `prefill`, `broadcast_kv`, `decode_only`)

**Use for:**
- Visualizing timeline in Chrome
- Finding GPU bottlenecks
- Measuring NCCL communication time
- Seeing overlaps between compute and communication

**View in:** `chrome://tracing`

### 2. ExecutionTrace (`*_et.json`)
**Size:** 14-336 MB per GPU  
**Format:** Chakra Execution Trace schema  
**Contains:**
- CPU-side execution graph (operator DAG)
- Operator dependencies
- Tensor shapes, types, and strides
- Process group info (NCCL)
- Control flow dependencies

**Use for:**
- Understanding operator-level dependencies
- Analyzing execution graph structure
- Identifying unnecessary serialization
- Optimizing operator fusion opportunities

**Example content:**
```json
{
  "id": 3, 
  "name": "## process_group:init ##",
  "inputs": {
    "values": ["{\"pg_name\": \"0\", \"backend_config\": \"cuda:nccl\", \"ranks\": [], \"group_size\": 4}"]
  }
},
{
  "id": 4, 
  "name": "format_prompt",
  "ctrl_deps": 2
},
{
  "id": 6,
  "name": "aten::empty",
  "inputs": {"values": [[1,7],4,0,"cpu",false,"<None>"]},
  "outputs": {"values": [[7,8,0,7,8,"cpu"]]},
  "attrs": [
    {"name": "op_schema", "value": "aten::empty.memory_format(...)"}
  ]
}
```

---

## Current Test Results

```
GPU 0 (Prefill Worker):
  - test_request_001_trace.json: 94.62 MB
  - test_request_001_et.json: 167.90 MB

GPU 1 (Decode Worker):
  - rank1_req1_trace.json: 189.70 MB
  - rank1_req1_et.json: 336.65 MB

GPU 2 (Decode Worker):
  - rank2_req1_trace.json: 8.33 MB
  - rank2_req1_et.json: 14.40 MB

GPU 3 (Decode Worker):
  - rank3_req1_trace.json: 57.19 MB
  - rank3_req1_et.json: 101.29 MB
```

All 4 GPUs successfully:
- ✅ Did their assigned work (prefill or decode)
- ✅ Generated Chrome traces
- ✅ Generated ExecutionTraces

---

## Analyzing KV Broadcast Performance

### Quick Analysis Script

```python
import json

def analyze_kv_broadcast(gpu0_trace_path):
    """Extract KV broadcast timing from GPU 0 trace."""
    with open(gpu0_trace_path) as f:
        trace = json.load(f)
    
    events = trace['traceEvents']
    
    # Find our custom operations
    user_ops = {e['name']: e for e in events if e.get('cat') == 'user_annotation'}
    
    # Extract timings (dur is in microseconds)
    prefill_time = user_ops['prefill']['dur'] / 1000  # ms
    broadcast_time = user_ops['broadcast_kv']['dur'] / 1000  # ms
    decode_time = user_ops['decode']['dur'] / 1000  # ms
    
    print(f"Prefill time: {prefill_time:.2f} ms")
    print(f"KV broadcast time: {broadcast_time:.2f} ms")
    print(f"Decode time: {decode_time:.2f} ms")
    print(f"\nBroadcast overhead: {broadcast_time/prefill_time*100:.1f}% of prefill")
    
    # Find NCCL broadcast kernels
    nccl_ops = [e for e in events if 'nccl:broadcast' in e.get('name', '')]
    print(f"\nTotal NCCL broadcasts: {len(nccl_ops)}")
    total_nccl_time = sum(e['dur'] for e in nccl_ops) / 1000
    print(f"Total NCCL time: {total_nccl_time:.2f} ms")

# Run it
analyze_kv_broadcast('profiling_traces/t4_gpu0/test_request_001_trace.json')
```

---

## Why ExecutionTrace Matters

The Chrome trace shows you **when** things happened. The ExecutionTrace shows you **why** they happened in that order.

For example:
- **Chrome trace:** "NCCL broadcast took 5ms at timestamp X"
- **ExecutionTrace:** "NCCL broadcast has control dependency on prefill completion, which depends on aten::matmul chain"

This helps you answer:
- Can I overlap communication with compute?
- Are there unnecessary dependencies blocking parallelism?
- Why is GPU X waiting when GPU Y is busy?

---

## Files Modified

- `serving/common/profiling.py`:
  - Added `self.et_observer = None` to `__init__`
  - In `__enter__`: Create `ExecutionTraceObserver()`, register callback, start
  - In `__exit__`: Stop and unregister observer before profiler cleanup
  - Removed broken `_trace_handler` method

---

## Verification

```bash
# Check all GPUs generated both trace types
ls -lh profiling_traces/t4_gpu*/

# Should see 2 files per GPU:
# - *_trace.json (Chrome trace)
# - *_et.json (ExecutionTrace)

# Verify ExecutionTrace format
head -50 profiling_traces/t4_gpu0/test_request_001_et.json

# Should see JSON with "schema", "nodes", operator names like "aten::empty"
```

---

## Next Steps

1. **Analyze Chrome traces visually** in `chrome://tracing` to see timeline
2. **Compare prefill vs broadcast timing** across GPUs
3. **Use ExecutionTrace** to identify optimization opportunities
4. **Consider overlap**: Can we start decode on GPU 1 before GPU 3 finishes receiving KV?

The traces now contain everything you need to estimate performance with faster networking (IB/NVLink).

