# Retest with Better Error Handling

## Changes Made:

1. **Added full traceback to Phase 2 exception handling**
   - Now we'll see exactly what error occurred

2. **Increased Phase 2 timeout to 300 seconds** (5 minutes)
   - Phase 2 has more work: append to KV cache, generate rankings from all 4 GPUs
   - Your large prompt (88,914 characters) makes this take longer

## What We Know So Far:

✅ **Phase 1 is PERFECT**
- All 4 members generated responses
- All responses collected successfully

✅ **Phase 2 is EXECUTING**
- Trace files confirm all GPUs are working:
  - GPU 0: `test_two_phase_001_judge_et.json` (6.79 MB)
  - GPU 1: `rank1_req2_judge_et.json` (336.95 MB)
  - GPU 2: `rank2_req2_judge_et.json` (336.89 MB)
  - GPU 3: `rank3_req2_judge_et.json` (336.90 MB)

✅ **Workers completed ranking**
- Server logs show: "Rank 1 sent ranking" and "Rank 2 sent ranking"

## Possible Issues:

1. **HTTP timeout** - Phase 2 takes too long (fixed with 300s timeout)
2. **GPU 0 not completing** - Haven't seen "Rank 0 sent ranking" in logs
3. **Response collection issue** - `all_gather` might be hanging
4. **Large response size** - With 88KB prompt, responses might be huge

## To Test:

```bash
# Terminal 2 (test client)
python test_t4_two_phase.py
```

You should now see the full error traceback which will tell us exactly what's failing!

