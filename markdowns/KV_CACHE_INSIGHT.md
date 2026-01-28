# KV Cache Management Implementation Insights

## Overview

This document explains the manual KV cache management, movement, and decode coordination implemented for the T4 cluster's shared-prefill architecture.

## Architecture Pattern

**GPU 0 (Prefill + Broadcast)** → **GPUs 1-3 (Receive + Decode)**

- GPU 0 receives HTTP requests, does prefill, broadcasts KV cache
- GPUs 1-3 wait for broadcasts, receive KV cache, do decode-only generation
- All 4 GPUs generate text in parallel using the shared KV cache

## Key Implementation Details

### 1. KV Cache Structure (Model-Specific)

**Critical Discovery: Phi-2 Returns 3-Element Tuples**

When calling `model()` with `use_cache=True`, the returned `past_key_values` structure varies by model:

```python
# Standard models (Llama, Mistral):
past_key_values = (
    (key_layer0, value_layer0),
    (key_layer1, value_layer1),
    ...
)

# Phi-2 quirk:
past_key_values = (
    (key_layer0, value_layer0, extra_data),
    (key_layer1, value_layer1, extra_data),
    ...
)
```

**Solution**: Extract only the first 2 elements per layer:

```python
for layer_idx, kv_pair in enumerate(past_key_values):
    key_cache = kv_pair[0]  # Always key
    value_cache = kv_pair[1]  # Always value
    # Ignore kv_pair[2:] if present
```

### 2. Broadcasting Strategy (GPU 0 → All Others)

#### Step 1: Signal Workers
```python
# Pack metadata into a single tensor
metadata = torch.tensor(
    [has_work, batch_size, seq_len, max_tokens],
    dtype=torch.long,
    device='cuda'
)
dist.broadcast(metadata, src=0)
```

#### Step 2: Broadcast Input IDs
```python
# All workers need the original input for attention mask
dist.broadcast(input_ids.contiguous(), src=0)
```

#### Step 3: Do Prefill (GPU 0 Only)
```python
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,  # Generate KV cache
    )
    past_key_values = outputs.past_key_values
```

#### Step 4: Broadcast KV Cache Layer-by-Layer
```python
# For Phi-2: 32 layers × 2 tensors = 64 broadcasts
for layer_idx, kv_pair in enumerate(past_key_values):
    key_cache = kv_pair[0]
    value_cache = kv_pair[1]
    
    # CRITICAL: Must be contiguous for NCCL
    dist.broadcast(key_cache.contiguous(), src=0)
    dist.broadcast(value_cache.contiguous(), src=0)
```

**Why layer-by-layer?** NCCL broadcasts operate on single tensors. Each key/value is a separate 4D tensor: `(batch, num_heads, seq_len, head_dim)`.

### 3. Receiving KV Cache (GPUs 1-3)

#### Step 1: Allocate Receive Buffers
```python
# Get model architecture info
num_layers = model.config.num_hidden_layers  # 32 for Phi-2
num_heads = model.config.num_attention_heads  # 32 for Phi-2
head_dim = model.config.hidden_size // num_heads  # 80 for Phi-2

# Shape for each KV tensor
key_shape = (batch_size, num_heads, seq_len, head_dim)
value_shape = (batch_size, num_heads, seq_len, head_dim)

# Use model's dtype (bfloat16 or float16)
dtype = next(model.parameters()).dtype
```

#### Step 2: Receive Broadcasts into Buffers
```python
cache = DynamicCache()  # Transformers' cache object

for layer_idx in range(num_layers):
    # Allocate
    key = torch.zeros(key_shape, dtype=dtype, device='cuda')
    value = torch.zeros(value_shape, dtype=dtype, device='cuda')
    
    # Receive (blocking call - waits for GPU 0)
    dist.broadcast(key, src=0)
    dist.broadcast(value, src=0)
    
    # Add to cache
    cache.update(key, value, layer_idx)

return cache
```

### 4. DynamicCache Requirement

**Critical Error Fixed**: Model's forward pass expects a `DynamicCache` object, not plain tuples.

```python
# ❌ WRONG - Causes AttributeError: 'tuple' object has no attribute 'get_seq_length'
past_key_values = [(key0, val0), (key1, val1), ...]

# ✅ CORRECT - Model can query cache length
from transformers import DynamicCache
cache = DynamicCache()
cache.update(key, value, layer_idx)
```

**Why?** During decode, the model calls:
```python
past_seen_tokens = past_key_values.get_seq_length()  # Only works on DynamicCache
```

### 5. Decode-Only Generation

Once KV cache is received, all GPUs perform decode:

```python
generated_ids, _ = generate(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,  # Original prompt
    past_key_values=cache,  # Shared KV cache from prefill
    max_tokens=max_tokens,
    temperature=0.7,
)
```

**Decode Loop Behavior**:
- First call: `input_ids.shape[1] == seq_len` (full prompt), but prefill is skipped because `past_key_values` is provided
- Subsequent calls: `input_ids.shape[1] == 1` (single token), appended to growing KV cache
- Each GPU maintains its own growing cache independently after the initial broadcast

### 6. Memory Layout Requirements

**NCCL requires contiguous tensors:**

```python
# May fail if tensor is a view or sliced
dist.broadcast(key_cache, src=0)  # ❌

# Always works
dist.broadcast(key_cache.contiguous(), src=0)  # ✅
```

**Why?** NCCL operates on raw memory pointers. Non-contiguous tensors have strided memory layouts that NCCL can't handle directly.

## Performance Insights

### Expected Trace Sizes
- **GPU 0**: 98 MB (prefill + 64 broadcasts + decode)
- **GPU 1-3**: 83-188 MB (64 receives + decode)

### Critical Operations to Profile
1. **Prefill** (`cuda_time`): Time to generate KV cache on GPU 0
2. **broadcast_kv** (`cuda_time` + `cpu_time`): NCCL communication overhead
3. **receive_kv** (`cuda_time` + `cpu_time`): NCCL receive + memcpy overhead
4. **decode_only** (`cuda_time`): Pure generation time per GPU

### Bottleneck Analysis

**Questions to answer from traces:**
- Is prefill time >> decode time? (Validates shared-prefill approach)
- What's the KV broadcast latency? (Network/PCIe bound)
- Are all decode workers balanced? (Load distribution)
- What's the overhead of NCCL vs compute?

## Synchronization Points

### Initialization
```python
dist.init_process_group(backend='nccl')
dist.barrier()  # All workers must reach here before proceeding
```

### Per-Request
1. GPU 0 broadcasts metadata → GPUs 1-3 block on `dist.broadcast(metadata, src=0)`
2. GPU 0 broadcasts input_ids → GPUs 1-3 block
3. GPU 0 does prefill (others wait)
4. GPU 0 broadcasts 64 KV tensors → GPUs 1-3 receive sequentially
5. All GPUs do decode in parallel (no synchronization)

**No locks, no queues, no async** - pure blocking NCCL collectives for simplicity and determinism.

## Edge Cases Handled

### 1. Multiple Requests
```python
while True:
    metadata = torch.zeros(4, dtype=torch.long, device='cuda')
    dist.broadcast(metadata, src=0)
    
    if metadata[0] == 0:
        continue  # No work, keep-alive ping
```

### 2. Batch Size = 1
All tensors have `batch_size` dimension, even for single requests. No special casing needed.

### 3. Variable Sequence Length
`seq_len` is communicated via metadata. Receive buffers are allocated dynamically per request.

## Why This Works

1. **Deterministic ordering**: All workers call `dist.broadcast()` in the same order
2. **Blocking semantics**: No race conditions, no need for explicit sync
3. **Single coordinator**: GPU 0 is the source of truth
4. **Stateless workers**: GPUs 1-3 maintain no state between requests
5. **No network assumptions**: Works on PCIe, NVLink, or InfiniBand

## Comparison to Production Systems

### vLLM / TGI Approach
- PagedAttention with dynamic KV cache allocation
- Continuous batching with token-level scheduling
- Complex memory management, but better throughput

### Our Approach
- Simple broadcast-based sharing
- Request-level synchronization
- Easy to reason about, profile, and debug
- Good for multi-agent scenarios where we want N diverse outputs per prompt

## Next Steps for Optimization

1. **Pipeline broadcasts**: Start decode on GPU 1 after receiving first few layers
2. **Reduce broadcasts**: Pack multiple layers into single broadcast
3. **Async coordination**: Use non-blocking broadcasts with manual sync
4. **Compare with NCCL_NET**: Test IB vs PCIe vs NVLink performance

## Files Modified

- `serving/t4_cluster/prefill_server.py` - Implements broadcast logic
- `serving/t4_cluster/simple_decode_worker.py` - Implements receive logic
- Both files use `torch.distributed` (NCCL backend) directly

## Testing

```bash
# Start cluster
cd /home/azureuser/multi-agent-test
source venv/bin/activate
python3 serving/t4_cluster/simple_launcher.py

# In another terminal, test
python3 test_t4_simple.py

# Verify all 4 GPUs participated
tail -100 /tmp/t4_cluster.log | grep "Completed"
```

## Key Takeaway

**Manual KV cache management is straightforward when you:**
1. Understand the model's exact KV structure (use debugging to inspect)
2. Use `DynamicCache` API for compatibility with `transformers`
3. Keep coordination simple (blocking broadcasts, single coordinator)
4. Profile everything to find bottlenecks

The "magic" is just understanding NCCL semantics and the transformers library's expectations.

