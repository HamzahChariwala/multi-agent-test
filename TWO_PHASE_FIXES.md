# Two-Phase Workflow Fixes

## Issues Found and Fixed

### Issue 1: "Collected 0 member responses"

**Problem:**  
FastAPI was validating the response against `GenerationOutput` schema, which doesn't have a `member_responses` field. This caused FastAPI to strip out that field from the response.

**Fix:**  
Changed the return type annotation from `-> GenerationOutput` to `-> dict` in the `generate_endpoint` method. This tells FastAPI to return the dict as-is without schema validation.

**File:** `serving/t4_cluster/prefill_server.py`
```python
# Before:
async def generate_endpoint(self, request: GenerationRequest) -> GenerationOutput:

# After:
async def generate_endpoint(self, request: GenerationRequest) -> dict:
```

---

### Issue 2: HTTP 422 "Field required" for /judge endpoint

**Problem:**  
The `/judge` endpoint was using `JudgingRequest` schema which requires `task_prompt` and `candidates` fields. These aren't needed since we're using stored state from Phase 1.

**Fix:**  
Created a simpler `SimpleJudgeRequest` schema with only `request_id` and `max_tokens` fields.

**File:** `serving/t4_cluster/prefill_server.py`
```python
class SimpleJudgeRequest(BaseModel):
    """Simple request for judging phase."""
    request_id: str = Field(..., description="Unique request identifier")
    max_tokens: int = Field(default=50, description="Max tokens for ranking")
```

---

## To Test

1. **Restart the T4 cluster** (Terminal 1):
```bash
# Press Ctrl+C to stop current process
cd /home/azureuser/multi-agent-test
source venv/bin/activate
bash run_t4_simple.sh
```

2. **Run the two-phase test** (Terminal 2):
```bash
cd /home/azureuser/multi-agent-test
source venv/bin/activate
python test_t4_two_phase.py
```

## Expected Output

```
PHASE 1: GENERATE
✓ Phase 1 SUCCESS (took ~40s)

Collected 4 member responses:
[A] member_0: ...
[B] member_1: ...
[C] member_2: ...
[D] member_3: ...

PHASE 2: PEER RANKING
✓ Phase 2 SUCCESS (took ~25s)

Collected 4 rankings:
member_0: B A D C
member_1: D B C A
member_2: B D A C
member_3: A B D C

✓ TEST PASSED!
```

## What Should Work Now

✅ Phase 1 collects all 4 member responses via `all_gather`  
✅ Phase 1 returns responses to client in `member_responses` field  
✅ Phase 2 uses stored KV cache from Phase 1  
✅ Phase 2 accepts simple request with just `request_id` and `max_tokens`  
✅ Phase 2 formats responses as A/B/C/D with ranking instruction  
✅ Phase 2 collects all 4 rankings via `all_gather`  
✅ Both phases generate profiling traces  

