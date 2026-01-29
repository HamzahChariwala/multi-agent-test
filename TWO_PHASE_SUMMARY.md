# Two-Phase Implementation Summary

## What Changed

### 1. Infrastructure Updates

#### `serving/t4_cluster/prefill_server.py`
- ✅ Added session state storage: `current_kv_cache`, `current_input_ids`, `member_responses`
- ✅ Added `/judge` endpoint for Phase 2
- ✅ Modified `/generate` to collect all member responses via `all_gather`
- ✅ Stores KV cache after Phase 1 for reuse in Phase 2
- ✅ Formats responses as A, B, C, D with ranking instruction
- ✅ Appends ranking instruction to existing KV cache
- ✅ Broadcasts extended KV cache to workers
- ✅ Collects rankings from all members

#### `serving/t4_cluster/simple_decode_worker.py`
- ✅ Added phase detection (1=generate, 2=judge)
- ✅ Stores KV cache and input_ids after Phase 1
- ✅ Sends generated text back via `all_gather` in Phase 1
- ✅ Receives ranking instruction in Phase 2
- ✅ Receives extended KV cache in Phase 2
- ✅ Generates rankings with lower temperature (0.3)
- ✅ Sends rankings back via `all_gather` in Phase 2

### 2. New Files

#### `test_t4_two_phase.py`
- New test script that exercises both phases
- Calls `/generate` to get 4 diverse answers
- Calls `/judge` to get 4 rankings
- Validates responses and checks profiling traces

#### `TWO_PHASE_GUIDE.md`
- Complete guide explaining the two-phase workflow
- API documentation
- Architecture diagrams
- Running instructions
- Troubleshooting tips

## How It Works

### Phase 1: Generate

```
1. HTTP request arrives at GPU 0
2. GPU 0 tokenizes and prefills
3. GPU 0 broadcasts [phase=1, metadata, input_ids, KV cache]
4. All GPUs (0-3) generate with different temperatures
5. All GPUs participate in all_gather to share responses
6. GPU 0 returns all 4 responses to client
7. All GPUs store KV cache for Phase 2
```

### Phase 2: Judge

```
1. HTTP request arrives at GPU 0 (/judge endpoint)
2. GPU 0 formats responses as A, B, C, D + ranking instruction
3. GPU 0 tokenizes ranking instruction
4. GPU 0 appends to existing KV cache (prefill new tokens only)
5. GPU 0 broadcasts [phase=2, metadata, ranking_ids, extended_KV]
6. All GPUs generate rankings with T=0.3
7. All GPUs participate in all_gather to share rankings
8. GPU 0 returns all 4 rankings to client
```

## Key Benefits

1. **KV Cache Reuse**: Phase 2 doesn't re-prefill the original prompt
2. **Temperature Diversity**: Different temperatures create diverse perspectives
3. **Efficient Communication**: `all_gather` collects results from all ranks simultaneously
4. **Profiling Support**: Both phases generate complete profiling traces
5. **No HTTP Overhead**: Only GPU 0 has HTTP server, workers use fast NCCL

## Communication Primitives

- **`broadcast(tensor, src=0)`**: One rank sends to all others
  - Used for: metadata, input_ids, KV cache
  
- **`all_gather(list, tensor)`**: All ranks send to all others
  - Used for: collecting responses, collecting rankings

## Temperature Settings

- **Generation (Phase 1)**:
  - GPU 0: 0.7 (moderate)
  - GPU 1: 0.8 (more creative)
  - GPU 2: 0.9 (very creative)
  - GPU 3: 1.0 (maximum creativity)

- **Ranking (Phase 2)**:
  - All GPUs: 0.3 (consistent, deterministic)

## Prompt Format (Phase 2)

```
[Original prompt from Phase 1...]

Here are the responses from all council members:

[A] [First member's response]

[B] [Second member's response]

[C] [Third member's response]

[D] [Fourth member's response]

Rank these responses from best to worst by listing only the letters in order (e.g., 'B A D C'). Provide ONLY the ranking with no explanation:
```

## Testing

```bash
# Terminal 1: Start cluster
bash run_t4_simple.sh

# Terminal 2: Run two-phase test
python test_t4_two_phase.py
```

## Next Steps

1. Integrate A100 node for final synthesis (Phase 3)
2. Add more sophisticated ranking metrics
3. Implement consensus algorithm from rankings
4. Add trace analysis tools for Phase 1 vs Phase 2 comparison

