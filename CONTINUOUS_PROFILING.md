# Continuous Profiling Implementation

## What Changed

Implemented **continuous profiling with idle timeout** - exactly what you requested!

### Features:

✅ **One trace per GPU** for entire workflow (not split by phase)  
✅ **Profiler runs continuously** across all HTTP requests  
✅ **30-second idle timeout** - auto-exports only after inactivity  
✅ **Timer resets** on each new request  
✅ **Background monitoring** task checks every 5 seconds  

## Architecture

```
Request 1 (Phase 1) arrives
  ↓
Profiler STARTS (if not already running)
  ↓
Request 1 completes
  ↓
Timer: 0s idle... 5s... 10s... 15s...
  ↓
Request 2 (Phase 2) arrives
  ↓
Timer RESETS to 0s
  ↓
Request 2 completes
  ↓
Timer: 0s... 5s... 10s... 15s... 20s... 25s... 30s
  ↓
EXPORT TRACES (idle timeout reached)
```

## Output Files

Each GPU now produces **exactly 2 files** per session:

- `two_phase_session_trace.json` - Chrome trace (all phases)
- `two_phase_session_et.json` - ExecutionTrace (all phases)

## Implementation Details

### New File: `serving/common/continuous_profiler.py`

**ContinuousProfiler class:**
- Manages one profiler per GPU
- Tracks `last_activity` timestamp
- Exports only when idle timeout expires

**ContinuousProfilerManager class:**
- Manages all GPU profilers
- Runs background monitoring task
- Checks for idle profilers every 5 seconds

### Integration Points

**In `prefill_server.py` and `simple_decode_worker.py`:**

1. **Initialize monitoring** (once at startup):
```python
manager = get_manager(idle_timeout=30.0)
manager.start_monitoring()
```

2. **Mark activity** (on each request):
```python
record_request_activity(gpu_id, "two_phase_session")
```

That's it! The profiler handles everything else automatically.

## Testing

### When to Restart:

**Terminal 1 (Server): MUST RESTART**
- Modified: `serving/t4_cluster/prefill_server.py`
- Modified: `serving/t4_cluster/simple_decode_worker.py`
- Added: `serving/common/continuous_profiler.py`

**Terminal 2 (Client): Just rerun**
- Modified: `test_t4_two_phase.py` (only error handling)

### Steps:

```bash
# Terminal 1: Restart server
# Press Ctrl+C to stop
cd /home/azureuser/multi-agent-test
bash run_t4_simple.sh

# Terminal 2: Wait for server ready, then test
python test_t4_two_phase.py
```

### What to Expect:

1. **Phase 1 completes** - profiler starts, timer at 0s
2. **Phase 2 completes** - timer resets to 0s
3. **Wait 30 seconds** - profiler auto-exports
4. **Check traces** - ONE file per GPU:
   ```
   profiling_traces/t4_gpu0/two_phase_session_trace.json
   profiling_traces/t4_gpu0/two_phase_session_et.json
   profiling_traces/t4_gpu1/two_phase_session_trace.json
   profiling_traces/t4_gpu1/two_phase_session_et.json
   ... (and so on for GPU 2 and 3)
   ```

## Benefits

✅ **Clean traces** - No phase splits, one continuous timeline  
✅ **Automatic management** - No manual export calls needed  
✅ **Session-aware** - Multiple requests = one trace  
✅ **Resource efficient** - Profiler only runs when needed  
✅ **Easy to use** - Just call `record_request_activity()`  

## Configuration

Change the idle timeout:

```python
# In initialize() method:
manager = get_manager(idle_timeout=60.0)  # 60 seconds
```

Or disable completely by setting `enabled: false` in `config/profiling.yaml`.

## Troubleshooting

**Traces not exporting?**
- Wait the full 30 seconds after last request
- Check logs for "Idle timeout reached, exporting traces"

**Multiple trace files per GPU?**
- Old files from previous runs
- Delete `profiling_traces/t4_gpu*/` before testing

**Profiler not starting?**
- Check `config/profiling.yaml` has `enabled: true`
- Check env variable: `export ENABLE_PROFILING=true`

