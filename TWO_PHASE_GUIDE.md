# Two-Phase T4 Council Workflow

## Overview

The T4 cluster now implements a full two-phase council workflow:

1. **Phase 1 (Generate)**: All 4 GPUs generate diverse answers to the prompt
2. **Phase 2 (Judge)**: All 4 GPUs rank the answers from Phase 1

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: GENERATE                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HTTP → GPU 0 (T=0.7)  ──┬── Prefill + KV Broadcast        │
│                          │                                  │
│                          ├── GPU 1 (T=0.8) → Answer A       │
│                          ├── GPU 2 (T=0.9) → Answer B       │
│                          ├── GPU 3 (T=1.0) → Answer C       │
│                          └── GPU 0 (T=0.7) → Answer D       │
│                                                             │
│  Collect all 4 answers via torch.distributed.all_gather     │
│  Store KV cache for Phase 2                                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: JUDGE                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Format: "[A] answer1\n[B] answer2\n[C] answer3\n[D]..."    │
│  Append: "Rank these from best to worst (e.g., 'B A D C')" │
│                                                             │
│  GPU 0 ──┬── Tokenize ranking instruction                   │
│          ├── Append to existing KV cache (prefill new)      │
│          └── Broadcast extended KV cache                    │
│                                                             │
│  All GPUs generate rankings:                                │
│    GPU 0 → "B D A C"                                        │
│    GPU 1 → "D B C A"                                        │
│    GPU 2 → "B A D C"                                        │
│    GPU 3 → "A D B C"                                        │
│                                                             │
│  Collect all rankings via torch.distributed.all_gather      │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. KV Cache Reuse
- Phase 1 KV cache is stored on all GPUs
- Phase 2 appends new tokens to existing cache
- Saves ~2-3 seconds of redundant prefill computation

### 2. Temperature Diversity
- GPU 0: T=0.7 (moderate creativity)
- GPU 1: T=0.8 (more creative)
- GPU 2: T=0.9 (very creative)
- GPU 3: T=1.0 (maximum creativity)
- Phase 2: T=0.3 (consistent rankings)

### 3. Communication Pattern
- **Broadcast**: GPU 0 → All (metadata, inputs, KV cache)
- **All-Gather**: All GPUs → All (collect responses/rankings)

## API Endpoints

### POST /generate (Phase 1)

**Request:**
```json
{
  "task_prompt": "Your question here",
  "max_tokens": 100,
  "request_id": "unique_id"
}
```

**Response:**
```json
{
  "answer": "GPU 0's answer",
  "member_id": "member_0",
  "confidence": 0.8,
  "request_id": "unique_id",
  "member_responses": [
    {"member_id": "member_0", "answer": "...", "confidence": 0.8},
    {"member_id": "member_1", "answer": "...", "confidence": 0.8},
    {"member_id": "member_2", "answer": "...", "confidence": 0.8},
    {"member_id": "member_3", "answer": "...", "confidence": 0.8}
  ]
}
```

### POST /judge (Phase 2)

**Request:**
```json
{
  "request_id": "unique_id_judge",
  "max_tokens": 50
}
```

**Response:**
```json
{
  "rankings": [
    {"judge_id": "member_0", "ranking": "B A D C"},
    {"judge_id": "member_1", "ranking": "D B C A"},
    {"judge_id": "member_2", "ranking": "B A D C"},
    {"judge_id": "member_3", "ranking": "A D B C"}
  ],
  "request_id": "unique_id_judge"
}
```

## Running the Test

### Terminal 1: Start T4 Cluster
```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate
bash run_t4_simple.sh
```

### Terminal 2: Run Two-Phase Test
```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate
python test_t4_two_phase.py
```

## Expected Output

```
==================================================
T4 TWO-PHASE COUNCIL TEST
==================================================
Phase 1: Generate diverse answers (4 members)
Phase 2: Peer ranking of all answers
==================================================

==================================================
PHASE 1: GENERATE
==================================================

Collected 4 member responses:
[A] member_0: ...
[B] member_1: ...
[C] member_2: ...
[D] member_3: ...

==================================================
PHASE 2: PEER RANKING
==================================================

Collected 4 rankings:
member_0: B A D C
member_1: D B C A
member_2: B D A C
member_3: A B D C
```

## Profiling

Both phases generate separate profiling traces:

- Phase 1: `profiling_traces/t4_gpu*/test_two_phase_001_*.json`
- Phase 2: `profiling_traces/t4_gpu*/test_two_phase_001_judge_*.json`

Look for:
- `prefill` operations in Phase 1
- `broadcast_kv` operations in Phase 1
- `prefill_ranking` operations in Phase 2 (should be much faster)
- `broadcast_extended_kv` operations in Phase 2

## Troubleshooting

**Error: "No member responses from Phase 1"**
- Run `/generate` endpoint before `/judge`

**Error: "No KV cache from Phase 1"**
- Ensure Phase 1 completed successfully
- Check that `self.current_kv_cache` is stored

**Ranks don't synchronize:**
- Check NCCL initialization logs
- Verify all 4 GPUs are visible
- Ensure no hanging broadcast/gather operations

