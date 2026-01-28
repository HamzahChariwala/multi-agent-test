# Quick Test Commands

## Step 1: Kill Any Existing Processes

```bash
pkill -9 python3
sleep 3
```

## Step 2: Start the T4 Cluster

```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate
python3 serving/t4_cluster/simple_launcher.py
```

**Wait for this message:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

This takes ~90 seconds for model loading.

## Step 3: Run Test (In Another Terminal)

```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate
python3 test_t4_simple.py
```

## Step 4: Check Results

### Expected Output:
```
✓ TEST PASSED!

Generation: PASS
Profiling traces: FOUND
- GPU 0: Found trace files
- GPU 1: Found trace files
- GPU 2: Found trace files
- GPU 3: Found trace files
```

### View Generated Text from All GPUs:
```bash
grep "Generated:" /tmp/t4_cluster.log
```

### Check Trace Files:
```bash
ls -lh profiling_traces/t4_gpu*/
```

You should see 2 files per GPU:
- `*_trace.json` (Chrome trace)
- `*_et.json` (ExecutionTrace)

### Analyze Generation:
```bash
python3 analyze_generation.py
```

## What Gets Tested

✅ **GPU 0:** Prefill + KV broadcast + decode  
✅ **GPUs 1-3:** Receive KV cache + decode  
✅ **All 4 GPUs:** Generate diverse outputs with different temperatures  
✅ **Profiling:** Chrome traces + ExecutionTraces for all GPUs  
✅ **NCCL:** 130 broadcasts per GPU (KV cache transfer)  

## Stop the Cluster

In Terminal 1 where the cluster is running, press `Ctrl+C`.

