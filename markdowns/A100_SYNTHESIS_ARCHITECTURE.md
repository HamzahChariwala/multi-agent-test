# A100 Synthesis Server Architecture

## Overview

The A100 node runs a single large language model (Llama-2-70B) using **4-way Tensor Parallelism** across all 4 GPUs. It performs a two-phase workflow with KV cache persistence between phases.

## Design Goals

1. **Parallel Prefill**: A100 prefills context simultaneously with T4 member generation
2. **KV Cache Reuse**: Store KV cache from Phase 1, append new content in Phase 2
3. **Latency Optimization**: Avoid re-processing the initial large context
4. **Single Session**: One deliberation at a time (simplified implementation)

---

## Workflow

### Phase 1: Initial Context Prefill (Parallel with T4 Generation)

**Trigger:** Orchestrator receives question

**Simultaneous Actions:**
- T4 GPU0: Prefills context → broadcasts to T4 members (GPUs 1-3)
- A100 (4-way TP): Receives same context → prefills → **stores KV cache**

**A100 Phase 1 Steps:**
1. Receive request with `request_id` and `context`
2. Tokenize context (typically 1000-2000 tokens)
3. Perform prefill across 4 GPUs with TP
4. **Store resulting KV cache** (indexed by `request_id`)
5. Return success acknowledgment
6. **Wait for Phase 2** (up to 5 minutes)

**Important:** No text generation in Phase 1 - pure prefill only.

```python
# Conceptual Phase 1 flow:
input_ids = tokenizer(context)  # ~1500 tokens
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True
    )
    past_kv = outputs.past_key_values  # Store this!

kv_cache_store[request_id] = {
    'kv': past_kv,
    'timestamp': time.time(),
    'seq_len': input_ids.shape[1]
}
```

### Phase 2: Synthesis with KV Cache Reuse

**Trigger:** T4 council completes generation + judgement stages

**T4 → A100:**
- Sends: `request_id` + formatted results (member responses + rankings)
- Typical size: 300-800 additional tokens

**A100 Phase 2 Steps:**
1. Retrieve stored KV cache using `request_id`
2. Tokenize new appended text (T4 results + synthesis instruction)
3. Perform **incremental prefill** on new tokens only
4. Generate synthesis response (~200-500 tokens)
5. Delete KV cache for this session
6. Return synthesis to orchestrator

```python
# Conceptual Phase 2 flow:
past_kv = kv_cache_store[request_id]['kv']
new_input_ids = tokenizer(appended_text)  # ~500 tokens

# Incremental forward pass
with torch.no_grad():
    outputs = model(
        input_ids=new_input_ids,
        past_key_values=past_kv,  # Reuse Phase 1 cache!
        use_cache=True
    )
    extended_kv = outputs.past_key_values

# Now generate synthesis
synthesis_ids = model.generate(
    input_ids=new_input_ids[:, -1:],  # Last token
    past_key_values=extended_kv,
    max_new_tokens=500
)

# Cleanup
del kv_cache_store[request_id]
```

---

## Technical Implementation Details

### 1. Tensor Parallelism (4-Way)

**Model Sharding Strategy:**

```
Llama-2-70B: ~140GB in bf16
Per GPU: 35GB (leaves 45GB free for KV cache, activations, etc.)

Sharding Layout:
- GPU 0: Layers 0-79, parameters [:, 0:shard_size]
- GPU 1: Layers 0-79, parameters [:, shard_size:2*shard_size]
- GPU 2: Layers 0-79, parameters [:, 2*shard_size:3*shard_size]
- GPU 3: Layers 0-79, parameters [:, 3*shard_size:]
```

**Layer Types:**
- **Column-parallel**: q_proj, k_proj, v_proj, gate_proj, up_proj
  - Split along output dimension
  - Each GPU computes 1/4 of outputs
- **Row-parallel**: o_proj, down_proj
  - Split along input dimension
  - Requires all-reduce after computation

**Forward Pass:**
```python
def tp_forward(model, input_ids, past_key_values=None):
    # Each rank computes on its shard
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True
    )
    
    # All-reduce logits across 4 ranks
    if dist.is_initialized():
        dist.all_reduce(outputs.logits, op=dist.ReduceOp.SUM)
        outputs.logits /= 4
    
    return outputs
```

### 2. KV Cache Management

**Structure Per Session:**
```python
{
    'request_id': str,
    'kv': Tuple[Tuple[Tensor, Tensor], ...],  # (num_layers, 2, ...)
    'timestamp': float,
    'seq_len': int,
    'metadata': {
        'phase': 1 or 2,
        'context_preview': str  # First 100 chars for debugging
    }
}
```

**Memory Profile:**
- **KV cache size**: `2 × num_layers × batch × num_heads × seq_len × head_dim × sizeof(bf16)`
- For Llama-2-70B at 2048 tokens:
  - `2 × 80 × 1 × 8 × 2048 × 128 × 2 bytes ≈ 10GB total`
  - **Per GPU with 4-way TP**: `10GB / 4 ≈ 2.5GB`
- Single session: Total A100 memory usage ~38GB per GPU

**Distributed Storage:**
Each TP rank stores its partition of the KV cache:
- GPU 0: heads 0-1
- GPU 1: heads 2-3  
- GPU 2: heads 4-5
- GPU 3: heads 6-7

### 3. Session Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ Session State Machine                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [IDLE] ──(Phase 1)──> [PREFILLED] ──(Phase 2)──> [COMPLETE]
│                              │                              │
│                              │                              │
│                         (timeout 5min)                      │
│                              │                              │
│                              └──────────> [EXPIRED]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**State Transitions:**
1. **IDLE → PREFILLED**: `/prefill_initial` endpoint called
2. **PREFILLED → COMPLETE**: `/synthesize_final` endpoint called
3. **PREFILLED → EXPIRED**: 5-minute timeout, cache cleaned up
4. **COMPLETE → IDLE**: Cache deleted, ready for new session

**Timeout Implementation:**
```python
async def cache_cleanup_task():
    """Background task to clean up expired caches."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        current_time = time.time()
        expired_ids = []
        
        async with cache_lock:
            for req_id, cache_data in kv_cache_store.items():
                age = current_time - cache_data['timestamp']
                if age > 300:  # 5 minutes
                    expired_ids.append(req_id)
            
            for req_id in expired_ids:
                logger.warning(f"Cleaning up expired cache: {req_id}")
                del kv_cache_store[req_id]
```

### 4. NCCL Configuration

**Process Group Initialization:**
```python
def init_tp_group(rank: int, world_size: int = 4):
    """Initialize NCCL process group for 4-way TP."""
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    # Optional optimizations for NVLink
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
    
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://localhost:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
```

**Communication Patterns:**
- **All-Reduce**: After row-parallel layers (logits)
- **All-Gather**: If needed for attention (usually not with proper sharding)
- **Broadcast**: Initial metadata sync

---

## API Specification

### Endpoint 1: `/prefill_initial`

**Purpose:** Phase 1 - Prefill initial context and store KV cache

**Request:**
```json
{
    "request_id": "unique-session-id",
    "context": "Full context string (1000-2000 tokens)",
    "metadata": {
        "source": "orchestrator",
        "timestamp": 1234567890
    }
}
```

**Response:**
```json
{
    "status": "success",
    "request_id": "unique-session-id",
    "cache_stored": true,
    "context_tokens": 1534,
    "prefill_time_ms": 450,
    "cache_size_mb": 2.5
}
```

**Implementation Notes:**
- No text generation
- Pure prefill operation
- Stores KV cache internally
- Returns immediately after prefill

### Endpoint 2: `/synthesize_final`

**Purpose:** Phase 2 - Append new text and generate synthesis

**Request:**
```json
{
    "request_id": "unique-session-id",
    "appended_text": "T4 member responses + rankings + synthesis instruction",
    "max_tokens": 500,
    "temperature": 0.7,
    "metadata": {
        "t4_completion_time": 1234567899
    }
}
```

**Response:**
```json
{
    "status": "success",
    "request_id": "unique-session-id",
    "synthesis": "Final synthesized answer...",
    "num_tokens": 342,
    "prefill_time_ms": 120,
    "decode_time_ms": 2100,
    "total_time_ms": 2220
}
```

**Implementation Notes:**
- Retrieves KV cache by `request_id`
- Performs incremental prefill on appended text only
- Generates synthesis response
- Deletes KV cache after completion

### Endpoint 3: `/health`

**Purpose:** Health check and capacity status

**Response:**
```json
{
    "status": "healthy",
    "model": "Llama-2-70b-chat-hf",
    "tp_world_size": 4,
    "active_sessions": 0,
    "max_concurrent_sessions": 1,
    "available": true,
    "gpu_memory_per_device_gb": {
        "0": {"used": 36.2, "total": 80.0},
        "1": {"used": 36.1, "total": 80.0},
        "2": {"used": 36.3, "total": 80.0},
        "3": {"used": 36.2, "total": 80.0}
    }
}
```

---

## Implementation Files

### Core Server: `serving/a100_cluster/synthesis_server.py`

**Structure:**
```python
class SynthesisWorker:
    """Worker process for one TP rank."""
    
    def __init__(self, rank: int, world_size: int = 4):
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}"
        self.model = None
        self.tokenizer = None
        self.kv_cache_store = {}  # Only rank 0 stores metadata
        
    async def initialize(self):
        """Load model shard and init TP group."""
        
    async def prefill_initial(self, request_id: str, context: str):
        """Phase 1: Prefill and store KV cache."""
        
    async def synthesize_final(self, request_id: str, appended_text: str, max_tokens: int):
        """Phase 2: Retrieve KV, append, generate."""
        
    def _shard_model(self, model):
        """Shard model weights across 4 GPUs."""
        
    def _tp_forward(self, input_ids, past_key_values=None):
        """TP forward pass with all-reduce."""

def main():
    """Launch 4 worker processes."""
    mp.spawn(worker_process, args=(4,), nprocs=4, join=True)
```

### Utilities: `serving/common/tp_utils.py`

**Functions:**
- `shard_column(weight, rank, world_size)`: Column-parallel sharding
- `shard_row(weight, rank, world_size)`: Row-parallel sharding
- `all_reduce_logits(logits, world_size)`: All-reduce and average
- `validate_kv_cache(kv_cache)`: Validate KV cache structure
- `estimate_kv_cache_size(kv_cache)`: Calculate memory usage

---

## Error Handling

### Common Error Cases

**1. Request ID Not Found (Phase 2)**
```json
{
    "error": "cache_not_found",
    "message": "No KV cache found for request_id: xyz",
    "request_id": "xyz"
}
```
**Cause:** Phase 2 called before Phase 1, or cache expired
**Recovery:** Client should restart with Phase 1

**2. Cache Expired**
```json
{
    "error": "cache_expired",
    "message": "KV cache expired after 5 minutes",
    "request_id": "xyz",
    "cache_age_seconds": 305
}
```
**Cause:** T4 took too long (>5 minutes)
**Recovery:** Restart deliberation

**3. Session Conflict**
```json
{
    "error": "session_conflict",
    "message": "Another session is active. Max concurrent sessions: 1",
    "active_request_id": "abc"
}
```
**Cause:** Attempted to start Phase 1 while another session active
**Recovery:** Wait for active session to complete

**4. NCCL Communication Failure**
```json
{
    "error": "nccl_failure",
    "message": "NCCL all-reduce failed on rank 2",
    "rank": 2
}
```
**Cause:** GPU communication issue, possibly GPU fault
**Recovery:** Restart server

---

## Performance Characteristics

### Expected Latencies

**Phase 1 (Prefill Initial):**
- Context size: 1500 tokens
- 4-way TP overhead: ~50ms (all-reduce per layer)
- Expected time: **400-600ms**

**Phase 2 (Synthesis Final):**
- Incremental prefill: 500 new tokens → **120-180ms**
- Decode generation: 400 tokens → **2000-2500ms** (5-6 tokens/sec)
- Total Phase 2: **2.1-2.7 seconds**

**Overall Deliberation:**
- T4 generation: 10-15 seconds (parallel with Phase 1)
- T4 judgement: 5-8 seconds
- A100 Phase 1: 0.5 seconds (parallel with T4 generation)
- A100 Phase 2: 2.5 seconds
- **Total: ~18-26 seconds**

**Latency Savings:**
Without KV cache reuse, Phase 2 would need to prefill full context (2000 tokens):
- Avoided prefill time: **1.5-2.0 seconds saved per deliberation**

### Memory Usage

**Per GPU (4-way TP):**
- Model weights: 35GB
- KV cache (2048 seq): 2.5GB
- Activations: 1-2GB
- OS/Other: 1GB
- **Total: ~40GB / 80GB (50% utilization)**

### Throughput

**Single Session Mode:**
- Concurrent requests: 1
- Throughput: Limited by T4 council speed (~20-30 seconds per deliberation)
- Effective: **2-3 deliberations per minute**

---

## Monitoring & Debugging

### Key Metrics to Track

1. **Cache Hit Rate**: Should be 100% (single session)
2. **Cache Expiration Rate**: Should be 0% (T4 completes within 5 min)
3. **Phase 1 Latency**: Target <600ms
4. **Phase 2 Latency**: Target <3s
5. **NCCL Communication Time**: Target <100ms per forward pass

### Logging Strategy

```python
logger.info(f"[Phase1] request_id={req_id}, tokens={n_tokens}, time={t:.2f}ms")
logger.info(f"[Phase2] request_id={req_id}, append_tokens={n}, prefill={t1:.2f}ms, decode={t2:.2f}ms")
logger.warning(f"[Cleanup] Expired cache: {req_id}, age={age:.1f}s")
logger.error(f"[Error] NCCL failure on rank {rank}")
```

### Debug Endpoints

**`/debug/cache_status`** - View active caches:
```json
{
    "active_caches": [
        {
            "request_id": "xyz",
            "age_seconds": 45,
            "seq_len": 1534,
            "size_mb": 2.5
        }
    ]
}
```

**`/debug/tp_status`** - View TP health:
```json
{
    "ranks": [
        {"rank": 0, "healthy": true, "gpu_id": 0},
        {"rank": 1, "healthy": true, "gpu_id": 1},
        {"rank": 2, "healthy": true, "gpu_id": 2},
        {"rank": 3, "healthy": true, "gpu_id": 3}
    ],
    "nccl_initialized": true
}
```

---

## Testing Strategy

### Unit Tests

1. **Model Sharding**: Verify weight partitioning correctness
2. **KV Cache Storage**: Test store/retrieve/delete operations
3. **Timeout Logic**: Verify 5-minute expiration
4. **Session Conflict**: Test rejection of concurrent sessions

### Integration Tests

1. **Phase 1 Only**: Verify prefill and cache storage
2. **Phase 1 + 2 Sequential**: Full flow with cache reuse
3. **Cache Expiration**: Wait >5 min, verify cleanup
4. **Incremental Prefill**: Verify new tokens appended correctly

### Load Tests

1. **Sequential Deliberations**: Run 10 deliberations back-to-back
2. **Memory Leak Check**: Monitor memory over 100 deliberations
3. **NCCL Stability**: Verify no communication failures over time

---

## Future Enhancements

### Multi-Session Support (If Needed)

Current: 1 concurrent session
Possible: 4-6 concurrent sessions

**Changes required:**
- Session queue with priority
- Per-session KV cache partitioning
- Resource allocation policy

### Model Swapping

Support multiple models (e.g., 70B vs 13B):
- Different models for different deliberation types
- Hot-swap without server restart

### Quantization

If memory becomes constraint:
- 8-bit quantization: 70GB → 35GB (17.5GB per GPU)
- Allows larger batch sizes or longer contexts

### Speculative Decoding

For faster Phase 2 generation:
- Use small draft model for speculation
- Verify with large model
- Potential 2-3x speedup

---

## Conclusion

This architecture provides:
- ✅ Efficient KV cache reuse between phases
- ✅ Parallel processing (T4 and A100 work simultaneously)
- ✅ Simple single-session model (easy to implement)
- ✅ Robust error handling and timeout management
- ✅ Scalable to larger models or more concurrent sessions

The two-phase design with KV cache persistence is the key innovation, saving 1.5-2 seconds per deliberation while keeping the implementation tractable.




