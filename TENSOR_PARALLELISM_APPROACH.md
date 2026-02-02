# Tensor Parallelism Implementation for Llama-2-70B

## High-Level Goal

Implement **true 4-way tensor parallelism** across 4 NVIDIA A100 GPUs (80GB each) for serving Llama-2-70B-chat-hf, with the following requirements:

### Core Requirements

1. **Separate Process Per GPU**
   - 4 independent processes, one per GPU (rank 0-3)
   - Each process manages its own shard of the model weights
   - Launched via `torchrun --nproc_per_node=4`

2. **Synchronous Execution**
   - All 4 GPUs must participate in every forward pass
   - NCCL collectives (all-reduce, broadcast, all-gather) ensure synchronization
   - No GPU proceeds independently - they work in lockstep

3. **Correct Tensor Parallelism**
   - Model weights are sharded across GPUs
   - Column-parallel layers (q_proj, k_proj, v_proj, gate_proj, up_proj): output dimension sharded, followed by all-reduce
   - Row-parallel layers (o_proj, down_proj): input dimension sharded, followed by all-reduce
   - All ranks produce identical final outputs (after collectives)

4. **Per-GPU Profiling**
   - Each GPU process generates its own Chrome trace (`profiling_traces/a100_gpu{N}/{request_id}_chrome.json`)
   - Each GPU process generates its own ExecutionTrace (`profiling_traces/a100_gpu{N}/{request_id}_et.json`)
   - Traces capture GPU kernels, CUDA operations, and NCCL collectives
   - No combined traces - each rank writes independently

5. **Distributed Coordination**
   - Rank 0 runs FastAPI server and receives HTTP requests
   - Rank 0 broadcasts work to ranks 1-3 via NCCL broadcast
   - All ranks execute generation simultaneously
   - Only rank 0 returns HTTP response

## Current Implementation Status

### Challenges Encountered

1. **Manual TP with Hooks (Failed)**
   - Initial attempt: manually shard weights and add forward hooks for all-reduce
   - Problem: Hooks execute after layer computation, not during
   - Result: Gibberish output - each GPU sampled independently

2. **Accelerate Library (Partial Success)**
   - Used HuggingFace `accelerate` with `device_map="auto"`
   - Problem: Pipeline parallelism, not tensor parallelism (sequential layer execution)
   - Problem: Single combined trace, not per-GPU traces
   - Result: Correct output but wrong parallelism strategy

3. **tensor_parallel Library (In Progress)**
   - Using `tensor_parallel.tensor_parallel()` to wrap model
   - Properly handles TP weight sharding and collectives
   - Each process gets `device_ids=[cuda:rank]` with `distributed=True`
   - Current status: Attempting to load and verify

### Expected Trace Characteristics

When working correctly, traces should show:

1. **NCCL Collectives**
   - `ncclAllReduce` operations after column-parallel and row-parallel layers
   - `ncclBroadcast` for work distribution from rank 0
   - Synchronized timing across GPUs (operations align temporally)

2. **GPU Kernels**
   - Matrix multiplications on sharded dimensions
   - Attention computations distributed across heads
   - Each GPU processes 1/4 of the work (for column-parallel layers)

3. **Dense Prefill, Iterative Decode**
   - Prefill: Dense operations, all tokens processed in parallel
   - Decode: Iterative, one token at a time, much lighter load
   - Traces should clearly show these two phases

## File Structure

```
serving/
├── a100_cluster/
│   └── manual_tp_server.py          # Current TP server implementation
├── common/
│   ├── profiling.py                 # ProfilerContext and ExecutionTraceObserver
│   └── tp_utils.py                  # TP utility functions (shard_*, all_reduce_*)
profiling_traces/
├── a100_gpu0/
│   ├── {request_id}_chrome.json
│   └── {request_id}_et.json
├── a100_gpu1/
│   └── ...
├── a100_gpu2/
│   └── ...
└── a100_gpu3/
    └── ...
```

## Success Criteria

1. ✅ Model loads successfully across 4 GPUs (~32GB per GPU)
2. ❓ Generated text is coherent and correct (not gibberish)
3. ❓ 4 separate profiling traces generated per request
4. ❓ Traces show NCCL collectives at expected points
5. ❓ Traces show synchronized execution across GPUs
6. ❓ Performance is reasonable (prefill + decode completes in <2 minutes for 50 tokens)

## Next Steps

1. Verify `tensor_parallel` library correctly implements TP
2. Test generation and inspect output quality
3. Examine traces for NCCL operations and synchronization
4. Validate per-GPU trace separation
5. Once working: integrate with T4 orchestrator node

