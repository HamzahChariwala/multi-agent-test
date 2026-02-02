# Repository Cleanup and Current Status

**Date:** 2026-01-30

## Cleanup Summary

### Files Deleted (27 total)

#### Test Scripts (13 files)
- `demo_proof.py` - Old demonstration script
- `test_input_small.txt` - Test input file
- `test_quick.py` - Quick test script
- `test_accelerate_tp.sh` - Accelerate launcher
- `test_interactive.py` - Interactive test
- `extract_gpu_kernels.py` - Kernel extraction utility
- `test_synthesis_full.py` - Old synthesis test
- `test_synthesis_phase1.py` - Old phase 1 test
- `test_synthesis_timeout.py` - Old timeout test
- `run_synthesis_tests.sh` - Old test runner
- `test_simple_synthesis.py` - Simple test
- `test_full_pipeline_profiled.py` - Pipeline test
- `hf_login.py` - HuggingFace login helper

#### Documentation (10 files)
- `SYNTHESIS_QUICKSTART.md` - Redundant quickstart
- `IMPLEMENTATION_STATUS.md` - Redundant status doc
- `NCCL_BROADCAST_SUCCESS.md` - Success documentation
- `ACCELERATE_FIX_SUMMARY.md` - Accelerate summary
- `BEFORE_AFTER_COMPARISON.md` - Comparison doc
- `TP_TESTING_GUIDE.md` - Testing guide
- `TRACE_SPLITTING_EXPLAINED.md` - Trace splitting doc
- `PER_GPU_TRACES_SUMMARY.md` - Per-GPU traces doc
- `DEEPSPEED_QUICKSTART.md` - DeepSpeed quickstart
- (Replaced by: `TENSOR_PARALLELISM_APPROACH.md`)

#### Server Files (4 files)
- `serving/a100_cluster/chairman_tp.py` - Old unused server
- `serving/a100_cluster/deepspeed_server.py` - DeepSpeed implementation
- `serving/a100_cluster/synthesis_server.py` - Accelerate-based server
- `serving/common/trace_splitter.py` - Trace splitting utility
- `run_deepspeed_server.sh` - DeepSpeed launcher

## Current Repository Structure

### Core Directories

```
multi-agent-test/
├── config/                      # Configuration files
│   ├── endpoints.yaml
│   ├── models.yaml
│   └── profiling.yaml
│
├── orchestrator/                # T4 orchestrator (main entry point)
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   ├── council_workflow.py
│   └── main.py
│
├── serving/                     # Server implementations
│   ├── a100_cluster/
│   │   ├── __init__.py
│   │   └── manual_tp_server.py  # ← CURRENT: 4-way TP server
│   ├── common/
│   │   ├── __init__.py
│   │   ├── http_server.py       # HTTP utilities
│   │   ├── inference.py         # Inference logic
│   │   ├── model_loader.py      # Model loading
│   │   ├── profiling.py         # ← ProfilerContext, ET observer
│   │   └── tp_utils.py          # TP utilities
│   └── t4_cluster/
│       ├── __init__.py
│       ├── kv_transfer.py
│       ├── prefill_server.py
│       ├── simple_decode_worker.py
│       └── simple_launcher.py
│
├── schemas/                     # Data models
│   ├── __init__.py
│   ├── chairman.py
│   ├── generation.py
│   └── judging.py
│
├── tests/                       # Integration tests
│   ├── __init__.py
│   ├── run_tests.sh
│   ├── test_integration.py
│   ├── test_kv_transfer.py
│   └── test_profiling.py
│
├── tools/                       # Analysis tools
│   ├── analyze_profiling.py
│   └── optimization_guide.md
│
├── plots/                       # Visualization
│   ├── timeline_plot.py
│   └── timeline_gibberish.png
│
├── markdowns/                   # Technical documentation
│   └── [various .md files]
│
└── profiling_traces/            # Output traces
    ├── a100_gpu0/
    ├── a100_gpu1/
    ├── a100_gpu2/
    └── a100_gpu3/
```

### Key Files

- **`run_manual_tp.sh`** - Launcher for 4-way TP server
- **`run_t4_simple.sh`** - Launcher for T4 cluster
- **`sample.txt`** - Sample input for testing
- **`test_t4_simple.py`** - T4 cluster test script
- **`README.md`** - Main documentation
- **`DEPLOYMENT.md`** - Deployment guide
- **`TENSOR_PARALLELISM_APPROACH.md`** - TP implementation details
- **`requirements.txt`** - Python dependencies

## Current Implementation: Tensor Parallelism

### File: `serving/a100_cluster/manual_tp_server.py`

**Approach:** Using `tensor_parallel` library with `torchrun`

**Architecture:**
- 4 separate processes (one per A100 GPU)
- Each process loads model on CPU, then wraps with `tp.tensor_parallel()`
- Distributed coordination via NCCL (torch.distributed)
- Rank 0 runs FastAPI server, broadcasts work to other ranks
- Each rank generates its own profiling traces

**Status:** In development - last attempt had issues with `tensor_parallel` API

### Next Steps for TP Implementation

1. **Verify tensor_parallel usage** - Ensure API calls are correct
2. **Test model loading** - Confirm 4-way sharding works
3. **Validate generation** - Check output quality (not gibberish)
4. **Inspect traces** - Look for NCCL collectives and synchronization
5. **Integrate with orchestrator** - Connect T4 node to A100 cluster

## Dependencies

- **torch** - PyTorch with CUDA support
- **transformers** - HuggingFace models
- **tensor_parallel** - TP library
- **fastapi** - HTTP server
- **httpx** - HTTP client
- Other: pydantic, pyyaml, etc.

## Running the System

### A100 TP Server
```bash
./run_manual_tp.sh
# Launches 4 processes on ports via torchrun
# Profiling enabled by default
```

### T4 Cluster
```bash
./run_t4_simple.sh
# Launches T4 decode workers
```

### Orchestrator
```bash
python orchestrator/main.py
# Coordinates between T4 and A100
```

## Profiling Output

Traces are written to: `profiling_traces/a100_gpu{0-3}/`

Each request generates:
- `{request_id}_chrome.json` - Chrome trace (visualize in chrome://tracing)
- `{request_id}_et.json` - ExecutionTrace (PyTorch graph format)

## Documentation

- **`TENSOR_PARALLELISM_APPROACH.md`** - Detailed TP approach and goals
- **`README.md`** - Project overview and architecture
- **`DEPLOYMENT.md`** - Hardware requirements and deployment
- **`markdowns/`** - Technical deep-dives and architecture docs


