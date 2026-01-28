# Running T4 Cluster in Two Terminals

This guide shows how to run the T4 cluster without background processes, using two separate terminals for better visibility and control.

## Terminal 1: Cluster Server

This terminal runs the 4-GPU cluster (1 prefill worker + 3 decode workers).

### Commands

```bash
# Navigate to project
cd /home/azureuser/multi-agent-test

# Activate virtual environment
source venv/bin/activate

# Run cluster (stays in foreground)
python3 serving/t4_cluster/simple_launcher.py
```

### What You'll See

```
INFO:serving.t4_cluster.prefill_server:[Rank 0] Loading model microsoft/phi-2...
INFO:serving.t4_cluster.simple_decode_worker:[Rank 1] Loading model microsoft/phi-2...
INFO:serving.t4_cluster.simple_decode_worker:[Rank 2] Loading model microsoft/phi-2...
INFO:serving.t4_cluster.simple_decode_worker:[Rank 3] Loading model microsoft/phi-2...

[... model loading progress ...]

INFO:serving.t4_cluster.prefill_server:[Rank 0] All workers synchronized
INFO:serving.t4_cluster.simple_decode_worker:[Rank 1] Decode worker ready - listening for broadcasts
INFO:serving.t4_cluster.simple_decode_worker:[Rank 2] Decode worker ready - listening for broadcasts
INFO:serving.t4_cluster.simple_decode_worker:[Rank 3] Decode worker ready - listening for broadcasts
INFO:serving.t4_cluster.prefill_server:[Rank 0] Starting HTTP server on port 8000
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**When you see "Uvicorn running", the cluster is ready for requests.**

### During Request Handling

You'll see real-time logs of the KV broadcast:

```
INFO:serving.t4_cluster.prefill_server:[Rank 0] Broadcasted metadata to workers
INFO:serving.t4_cluster.prefill_server:[Rank 0] Broadcasted input_ids
INFO:serving.t4_cluster.prefill_server:[Rank 0] Prefill complete, got 32 layers of KV cache
INFO:serving.t4_cluster.prefill_server:[Rank 0] Broadcasted KV cache (32 layers)
INFO:serving.t4_cluster.simple_decode_worker:[Rank 1] Received KV cache (32 layers)
INFO:serving.t4_cluster.simple_decode_worker:[Rank 2] Received KV cache (32 layers)
INFO:serving.t4_cluster.simple_decode_worker:[Rank 3] Received KV cache (32 layers)
INFO:serving.t4_cluster.prefill_server:[Rank 0] Completed full request
INFO:serving.t4_cluster.simple_decode_worker:[Rank 1] Completed decode #1
INFO:serving.t4_cluster.simple_decode_worker:[Rank 2] Completed decode #1
INFO:serving.t4_cluster.simple_decode_worker:[Rank 3] Completed decode #1
```

### Stopping the Cluster

Press `Ctrl+C` in Terminal 1 to gracefully shut down all 4 GPU processes.

---

## Terminal 2: Test Client

This terminal sends requests and runs tests against the cluster.

### Wait for Cluster Startup

**Important**: Wait ~90 seconds after starting Terminal 1 for model loading to complete. Look for "Uvicorn running" message.

### Commands

```bash
# Navigate to project
cd /home/azureuser/multi-agent-test

# Activate virtual environment
source venv/bin/activate

# Run test script
python3 test_t4_simple.py
```

### What You'll See

```
======================================================================
T4 SIMPLE CLUSTER TEST
======================================================================

Step 1: Checking service health...
✓ Service is ready

Step 2: Testing generation...
Sending request: What is the capital of France?
✓ SUCCESS (took 26.68s)
Generated text: What is the capital of France?
A) Paris
B) Berlin
C) Rome
D) Madrid

Answer: A) Paris
...

======================================================================
Checking Profiling Traces
======================================================================
✓ GPU 0: Found 1 trace file(s)
  - test_request_001_trace.json (97.69 MB)
✓ GPU 1: Found 1 trace file(s)
  - rank1_req1_trace.json (188.06 MB)
✓ GPU 2: Found 1 trace file(s)
  - rank2_req1_trace.json (83.02 MB)
✓ GPU 3: Found 1 trace file(s)
  - rank3_req1_trace.json (187.99 MB)

======================================================================
TEST SUMMARY
======================================================================
✓ Generation: PASS
✓ Profiling traces: FOUND
======================================================================

✓ TEST PASSED!
```

### Interactive Testing

You can also send custom requests using `curl` or Python:

#### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Generation request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_prompt": "What is 2+2?",
    "context": [],
    "max_tokens": 50,
    "request_id": "manual_test_001"
  }'
```

#### Using Python

```python
import httpx
import json

# Send request
response = httpx.post(
    "http://localhost:8000/generate",
    json={
        "task_prompt": "Explain quantum computing",
        "context": [],
        "max_tokens": 100,
        "request_id": "test_002"
    },
    timeout=120.0
)

print(json.dumps(response.json(), indent=2))
```

---

## Terminal Layout Tips

### Option 1: Side-by-Side (tmux/screen)

```bash
# Install tmux if needed
sudo apt-get install tmux

# Start tmux session
tmux new -s t4_cluster

# Split window vertically
Ctrl+b %

# Navigate between panes
Ctrl+b <arrow keys>

# Terminal 1 (left): Run cluster
cd /home/azureuser/multi-agent-test && source venv/bin/activate
python3 serving/t4_cluster/simple_launcher.py

# Terminal 2 (right): Run tests
cd /home/azureuser/multi-agent-test && source venv/bin/activate
python3 test_t4_simple.py
```

### Option 2: Separate Terminal Windows

Just open two SSH sessions or terminal windows and follow the commands above.

---

## Troubleshooting

### "Connection refused" in Terminal 2

**Cause**: Cluster not fully started yet.

**Solution**: Wait for "Uvicorn running" message in Terminal 1 before running tests.

### GPU Out of Memory

**Cause**: Previous processes not fully cleaned up.

**Solution**: In Terminal 1, press `Ctrl+C` to stop, then:

```bash
# Kill any lingering processes
pkill -9 python3
sleep 3

# Check GPU memory is clear
nvidia-smi

# Restart cluster
python3 serving/t4_cluster/simple_launcher.py
```

### Port Already in Use

**Cause**: Previous cluster process still running.

**Solution**:

```bash
# Find process on port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)

# Or kill all python3
pkill -9 python3
```

---

## Advantages of Two-Terminal Setup

1. **Real-time visibility**: See KV broadcast logs as they happen
2. **Easy debugging**: Errors appear immediately in Terminal 1
3. **Quick iteration**: `Ctrl+C` to stop, edit code, restart
4. **No log file hunting**: All output directly visible
5. **Better profiling**: See exact timing of operations across GPUs

## Workflow Example

```bash
# Terminal 1
cd /home/azureuser/multi-agent-test && source venv/bin/activate
python3 serving/t4_cluster/simple_launcher.py

# Wait for "Uvicorn running"...

# Terminal 2
cd /home/azureuser/multi-agent-test && source venv/bin/activate
python3 test_t4_simple.py

# Watch Terminal 1 for real-time KV broadcast logs
# See all 4 GPUs participate in the request

# When done, Ctrl+C in Terminal 1
```

This setup is ideal for development, debugging, and understanding the distributed coordination.

