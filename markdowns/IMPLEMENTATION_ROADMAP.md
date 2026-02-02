# Implementation Roadmap - Two-Phase Synthesis Architecture

## Status: Documentation Complete ✅, Implementation Pending

This document tracks the transition from the old architecture (KV forking + chairman voting) to the new architecture (4-way TP synthesis with KV cache reuse).

---

## Documentation Updates (COMPLETE)

### ✅ Created
- `markdowns/A100_SYNTHESIS_ARCHITECTURE.md` - Comprehensive technical specification
- `markdowns/IMPLEMENTATION_ROADMAP.md` - This file

### ✅ Updated
- `README.md` - New architecture overview and workflow diagram
- `DEPLOYMENT.md` - Updated deployment instructions for synthesis server
- `config/models.yaml` - New model configuration (synthesis_model)
- `config/endpoints.yaml` - New endpoint structure (synthesis instead of chairman)

---

## Implementation Tasks

### Phase 1: A100 Synthesis Server (Core Implementation)

#### Task 1.1: Create Tensor Parallelism Utilities
**File:** `serving/common/tp_utils.py` (NEW)

**Functions needed:**
```python
def shard_column(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor
def shard_row(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor
def all_reduce_logits(logits: torch.Tensor, world_size: int) -> torch.Tensor
def validate_kv_cache(kv_cache: Tuple) -> bool
def estimate_kv_cache_size(kv_cache: Tuple) -> float
```

**Reference:** Can adapt from `serving/a100_cluster/chairman_tp.py` but extend to 4 GPUs

**Estimated effort:** 2-3 hours

---

#### Task 1.2: Create Synthesis Server
**File:** `serving/a100_cluster/synthesis_server.py` (NEW)

**Core Components:**

1. **SynthesisWorker Class**
   - Initialize TP group (4 ranks)
   - Load model shard per rank
   - KV cache storage (dict indexed by request_id)
   - Cache cleanup background task
   
2. **Phase 1 Handler: `/prefill_initial`**
   ```python
   async def prefill_initial(request: InitialPrefillRequest):
       # Tokenize context
       # Perform TP forward pass (no generation)
       # Store KV cache with request_id
       # Return success
   ```

3. **Phase 2 Handler: `/synthesize_final`**
   ```python
   async def synthesize_final(request: FinalSynthesisRequest):
       # Retrieve KV cache by request_id
       # Tokenize appended text
       # Incremental TP forward (extend KV cache)
       # Generate synthesis
       # Delete KV cache
       # Return synthesis
   ```

4. **Supporting Methods**
   - `_shard_model()`: Partition model across 4 GPUs
   - `_tp_forward()`: Forward pass with all-reduce
   - `_tp_generate()`: Generation with TP
   - `_cleanup_expired_caches()`: Background task (5 min timeout)

**Reference:** 
- Structure from `serving/a100_cluster/chairman_tp.py`
- Model sharding from `serving/a100_cluster/large_model_twin.py`

**Estimated effort:** 1-2 days

---

#### Task 1.3: Create Test Scripts
**Files:** 
- `test_a100_synthesis_phase1.py` (NEW) - Test Phase 1 only
- `test_a100_synthesis_phase2.py` (NEW) - Test full workflow
- `test_a100_synthesis_timeout.py` (NEW) - Test cache expiration

**Test scenarios:**
1. Phase 1: Prefill and cache storage
2. Phase 1 + 2: Full two-phase flow
3. Cache expiration: Wait >5 min, verify cleanup
4. Session conflict: Try concurrent Phase 1 calls
5. Missing cache: Call Phase 2 without Phase 1

**Estimated effort:** 4-6 hours

---

### Phase 2: Orchestrator Integration

#### Task 2.1: Update Orchestrator for Two-Phase Workflow
**File:** `orchestrator/orchestrator.py` (MODIFY)

**Changes needed:**

1. **Remove chairman voting logic:**
   - Delete `_run_chairman_voting()`
   - Delete `_collect_chairman_votes()`

2. **Add two-phase synthesis:**
   ```python
   async def deliberate(self, question: str):
       request_id = generate_unique_id()
       
       # Phase 1a + 1b (parallel)
       t4_task = self._run_t4_council(question)
       a100_task = self._a100_prefill_initial(request_id, question)
       
       t4_results, _ = await asyncio.gather(t4_task, a100_task)
       
       # Phase 2: Synthesis with cache reuse
       synthesis = await self._a100_synthesize_final(
           request_id,
           t4_results
       )
       
       return synthesis
   ```

3. **Add synthesis API methods:**
   - `_a100_prefill_initial()`
   - `_a100_synthesize_final()`
   - `_format_synthesis_prompt()`

**Estimated effort:** 4-6 hours

---

#### Task 2.2: Update Orchestrator Tests
**File:** `orchestrator/test_orchestrator.py` (MODIFY)

Update to test two-phase flow instead of three-stage flow.

**Estimated effort:** 2-3 hours

---

### Phase 3: Cleanup and Migration

#### Task 3.1: Remove Deprecated Files
**Files to delete:**
- `serving/a100_cluster/large_model_twin.py`
- `serving/a100_cluster/kv_fork.py`
- `serving/a100_cluster/chairman_tp.py` (optional - can keep as reference)
- `test_a100_large_model.py` (old test)
- `test_a100_chairman.py` (old test)

**Estimated effort:** 30 minutes

---

#### Task 3.2: Update All Documentation Cross-References
**Files to check:**
- All `markdowns/*.md` files
- `A100_TESTING_GUIDE.md` - Rewrite for synthesis
- `A100_QUICKSTART.md` - Update for synthesis

**Estimated effort:** 2-3 hours

---

### Phase 4: Testing and Validation

#### Task 4.1: Integration Testing
**Tests:**
1. T4 cluster alone (existing, should still work)
2. A100 synthesis Phase 1 only
3. A100 synthesis Phase 1 + 2
4. Full end-to-end: T4 + A100 orchestration
5. Error cases (timeout, missing cache, etc.)

**Estimated effort:** 4-6 hours

---

#### Task 4.2: Performance Profiling
**Metrics to collect:**
- Phase 1 latency (target: <600ms)
- Phase 2 prefill latency (target: <200ms)
- Phase 2 decode latency (target: <2.5s)
- Total deliberation time (target: <25s)
- Memory usage per GPU
- NCCL communication overhead

**Estimated effort:** 3-4 hours

---

## Total Estimated Effort

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: A100 Core | 2-3 days |
| Phase 2: Orchestrator | 1 day |
| Phase 3: Cleanup | 3-4 hours |
| Phase 4: Testing | 1-2 days |
| **TOTAL** | **4-6 days** |

---

## Implementation Order (Recommended)

1. ✅ **Documentation** (COMPLETE)
2. `serving/common/tp_utils.py` - Build foundation
3. `serving/a100_cluster/synthesis_server.py` - Core server
4. `test_a100_synthesis_*.py` - Test in isolation
5. `orchestrator/orchestrator.py` - Integration
6. End-to-end testing
7. Cleanup deprecated code
8. Update remaining docs

---

## Dependencies

### External Libraries (Already Installed)
- ✅ PyTorch 2.1.0+
- ✅ Transformers
- ✅ NCCL (via PyTorch)
- ✅ FastAPI / Uvicorn

### Internal Dependencies
- ✅ `serving/common/model_loader.py` - Works as-is
- ✅ `serving/common/http_server.py` - Works as-is
- ✅ `serving/common/inference.py` - Works as-is
- ✅ `serving/common/profiling.py` - Works as-is
- ✅ `serving/t4_cluster/*` - No changes needed

---

## Risk Assessment

### Low Risk ✅
- **T4 cluster**: Unchanged, well-tested
- **TP implementation**: Similar to existing chairman_tp.py
- **KV cache mechanics**: Standard HuggingFace feature

### Medium Risk ⚠️
- **KV cache storage**: Need proper session management
- **NCCL stability**: 4-way TP instead of 2-way (more communication)
- **Memory management**: Single session simplifies, but need monitoring

### High Risk ⛔
- None identified with current design

**Mitigation:** Extensive testing at each phase before proceeding.

---

## Success Criteria

### Functional
- ✅ Phase 1 completes and stores KV cache
- ✅ Phase 2 retrieves cache and generates synthesis
- ✅ Cache expires after 5 minutes
- ✅ Session conflicts handled correctly
- ✅ End-to-end deliberation works

### Performance
- ✅ Phase 1 latency < 600ms
- ✅ Phase 2 total < 3s
- ✅ Memory usage < 45GB per GPU
- ✅ No NCCL failures over 100 deliberations

### Quality
- ✅ Synthesis quality equal or better than old chairman
- ✅ Error handling robust
- ✅ Logging comprehensive
- ✅ Documentation complete

---

## Next Steps

**To begin implementation:**
```bash
# 1. Review this roadmap
# 2. Read A100_SYNTHESIS_ARCHITECTURE.md thoroughly
# 3. Start with Task 1.1 (tp_utils.py)
# 4. Follow recommended implementation order
```

**Questions before starting?**
- Model sharding strategy clear?
- KV cache storage approach understood?
- Two-phase workflow makes sense?

If yes, you're ready to build! 🚀




