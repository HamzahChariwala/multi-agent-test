# Multi-Agent Council System

A distributed two-phase council workflow with custom HuggingFace + NCCL implementation, spanning 8 GPUs across 2 nodes (T4 and A100) with advanced KV cache optimization.

## Architecture Overview

### T4 Node (4 GPUs) - Council Deliberation
- **GPU 0**: Shared prefill server for all council members
- **GPUs 1-3**: 3 council members with diverse temperatures
- **Purpose**: Generate diverse answers and perform peer rankings
- **Model**: Phi-2 (2.7B parameters)
- **Technique**: Shared-prefill with KV cache broadcasting

### A100 Node (4 GPUs) - Synthesis with 4-Way Tensor Parallelism
- **All 4 GPUs**: Single large model (Llama-2-70B) split via Tensor Parallelism
- **Purpose**: Two-phase workflow with KV cache persistence
  - **Phase 1**: Prefill context (parallel with T4 generation)
  - **Phase 2**: Append council results and generate final synthesis
- **Technique**: Manual 4-way TP with incremental prefill and KV cache reuse

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Question Received                                               │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─────────────────────┬────────────────────────────────┐
             ▼                     ▼                                │
    ┌────────────────┐    ┌──────────────────┐                    │
    │ T4 Node        │    │ A100 Node        │                    │
    │ (GPU 0-3)      │    │ (GPU 0-3)        │                    │
    │                │    │                  │                    │
    │ Phase 1a:      │    │ Phase 1b:        │                    │
    │ • Prefill      │    │ • Prefill same   │                    │
    │ • Broadcast    │    │   context        │                    │
    │ • Generate     │    │ • Store KV cache │                    │
    │   (3 members)  │    │ • Wait           │                    │
    │                │    │                  │                    │
    │ Phase 2:       │    │                  │                    │
    │ • Peer ranking │    │                  │                    │
    └────────┬───────┘    └──────────────────┘                    │
             │                     ▲                                │
             │  Council Results    │                                │
             └─────────────────────┤                                │
                                   │                                │
                          ┌────────┴─────────┐                     │
                          │ A100 Node        │                     │
                          │ Phase 2:         │                     │
                          │ • Retrieve KV    │                     │
                          │ • Append text    │                     │
                          │ • Prefill new    │                     │
                          │ • Synthesize     │                     │
                          └──────────────────┘                     │
                                   │                                │
                                   ▼                                │
                          Final Synthesis ◄────────────────────────┘
```

**Key Innovation:** A100 prefills context in parallel with T4 generation, stores KV cache, then reuses it for synthesis - saving 1.5-2 seconds per deliberation.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit configuration files in `config/`:
- `models.yaml`: Model specifications and GPU assignments
- `endpoints.yaml`: HTTP endpoint definitions for each member
- `profiling.yaml`: PyTorch profiling settings per GPU

## Running the System

### T4 Node (Council Members)
```bash
cd serving/t4_cluster
export CUDA_VISIBLE_DEVICES=0,1,2,3
python launcher.py
```

### A100 Node (Synthesis Server with 4-way TP)
```bash
cd serving/a100_cluster
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29500
python synthesis_server.py
```

### Orchestrator
```bash
cd orchestrator
python main.py
```

## Profiling

Profiling traces are saved to `profiling_traces/{gpu_id}/` directory.
- Enable/disable via `ENABLE_PROFILING` environment variable
- View traces using TensorBoard or Chrome tracing viewer

## Project Structure

```
multi-agent-test/
├── orchestrator/          # Council workflow orchestrator (two-phase)
├── serving/              # Model serving infrastructure
│   ├── t4_cluster/       # T4 GPU cluster (3 council members)
│   │   ├── prefill_server.py     # Shared prefill (GPU 0)
│   │   ├── member_server.py      # Member workers (GPUs 1-3)
│   │   └── launcher.py            # Launch all T4 services
│   ├── a100_cluster/     # A100 GPU cluster (4-way TP synthesis)
│   │   └── synthesis_server.py   # Two-phase synthesis with KV cache reuse
│   └── common/           # Shared utilities (inference, TP, profiling)
│       └── tp_utils.py   # Tensor parallelism utilities
├── schemas/              # Data contracts (Pydantic models)
├── config/               # Configuration files
│   ├── models.yaml       # Model specs and TP settings
│   ├── endpoints.yaml    # HTTP endpoints
│   └── profiling.yaml    # Profiling configuration
├── markdowns/            # Architecture documentation
│   └── A100_SYNTHESIS_ARCHITECTURE.md  # Detailed A100 implementation guide
└── profiling_traces/     # Profiling output directory
```

## Key Documentation

- **[A100_SYNTHESIS_ARCHITECTURE.md](markdowns/A100_SYNTHESIS_ARCHITECTURE.md)**: Deep dive on two-phase synthesis with 4-way TP
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Step-by-step setup and deployment guide
- **[KV_CACHE_INSIGHT.md](markdowns/KV_CACHE_INSIGHT.md)**: KV cache broadcasting implementation details

