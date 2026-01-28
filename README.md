# Multi-Agent Council System

A distributed 3-stage council workflow (Generate → Judge → Chairman) with custom HuggingFace + NCCL implementation, spanning 8 GPUs across 2 nodes (T4 and A100) with KV cache optimization.

## Architecture

- **T4 Node (4 GPUs)**: 3 council members using shared-prefill architecture
- **A100 Node (4 GPUs)**: 2 large model members + 1 chairman (TP=2)
- **Orchestrator**: CPU-based coordinator managing the 3-stage workflow

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

### T4 Node
```bash
cd serving/t4_cluster
python launcher.py
```

### A100 Node (Large Model Twin)
```bash
cd serving/a100_cluster
python large_model_twin.py
```

### A100 Node (Chairman)
```bash
cd serving/a100_cluster
python chairman_tp.py
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
├── orchestrator/          # Council workflow orchestrator
├── serving/              # Model serving infrastructure
│   ├── t4_cluster/       # T4 GPU cluster (3 members)
│   ├── a100_cluster/     # A100 GPU cluster (2 members + chairman)
│   └── common/           # Shared utilities
├── schemas/              # Data contracts (Pydantic models)
├── config/               # Configuration files
└── profiling_traces/     # Profiling output directory
```

