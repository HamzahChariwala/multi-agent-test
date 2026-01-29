"""
Test script for two-phase T4 cluster workflow:
Phase 1: Generate diverse answers
Phase 2: Peer ranking
"""

import asyncio
import httpx
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_two_phase_workflow():
    """Test the full two-phase workflow."""
    endpoint = "http://localhost:8000"
    
    logger.info("=" * 80)
    logger.info("T4 TWO-PHASE COUNCIL TEST")
    logger.info("=" * 80)
    logger.info("Phase 1: Generate diverse answers (4 members)")
    logger.info("Phase 2: Peer ranking of all answers")
    logger.info("=" * 80)
    
    # Check if service is ready
    logger.info("\nStep 1: Checking service health...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(30):
            try:
                response = await client.get(f"{endpoint}/health")
                if response.status_code == 200:
                    logger.info("✓ Service is ready")
                    break
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt == 0:
                    logger.info("Waiting for service to start...")
                await asyncio.sleep(2)
        else:
            logger.error("✗ Service failed to start after 60 seconds")
            return False
    
    # Read prompt from sample.txt
    sample_file = Path(__file__).parent / "sample.txt"
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            prompt_text = f.read()
        logger.info(f"Loaded prompt from sample.txt ({len(prompt_text)} characters)")
    else:
        prompt_text = "What is the capital of France?"
        logger.warning("sample.txt not found, using default prompt")
    
    # ========================================================================
    # PHASE 1: GENERATE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: GENERATE")
    logger.info("=" * 80)
    
    request = {
        "task_prompt": prompt_text,
        "max_tokens": 100,
        "request_id": "test_two_phase_001"
    }
    
    logger.info(f"Sending generation request with prompt: {prompt_text[:100]}...")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{endpoint}/generate",
                json=request
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"✗ Phase 1 FAIL: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
            
            result = response.json()
            logger.info(f"✓ Phase 1 SUCCESS (took {elapsed:.2f}s)")
            
            # Extract all member responses
            member_responses = result.get("member_responses", [])
            logger.info(f"\nCollected {len(member_responses)} member responses:")
            for i, resp in enumerate(member_responses):
                member_id = resp.get("member_id", f"member_{i}")
                answer = resp.get("answer", "")
                logger.info(f"\n[{chr(65+i)}] {member_id}:")
                logger.info(f"  {answer[:150]}...")
            
    except Exception as e:
        logger.error(f"✗ Phase 1 EXCEPTION: {e}")
        return False
    
    # ========================================================================
    # PHASE 2: JUDGE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: PEER RANKING")
    logger.info("=" * 80)
    
    judge_request = {
        "request_id": "test_two_phase_001_judge",
        "max_tokens": 50,
    }
    
    logger.info("Sending ranking request...")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes for Phase 2
            response = await client.post(
                f"{endpoint}/judge",
                json=judge_request
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"✗ Phase 2 FAIL: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
            
            result = response.json()
            logger.info(f"✓ Phase 2 SUCCESS (took {elapsed:.2f}s)")
            
            # Extract all rankings
            rankings = result.get("rankings", [])
            logger.info(f"\nCollected {len(rankings)} rankings:")
            for i, ranking in enumerate(rankings):
                judge_id = ranking.get("judge_id", f"judge_{i}")
                order = ranking.get("ranking", "")
                logger.info(f"\n{judge_id}: {order}")
            
            return True
            
    except Exception as e:
        import traceback
        logger.error(f"✗ Phase 2 EXCEPTION: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False


async def check_profiling_traces():
    """Check if profiling traces were generated."""
    logger.info("\n" + "=" * 80)
    logger.info("Checking Profiling Traces")
    logger.info("=" * 80)
    
    traces_dir = Path("profiling_traces")
    
    if not traces_dir.exists():
        logger.warning("✗ Profiling traces directory not found")
        return False
    
    all_traces_found = True
    
    # Check each GPU
    for gpu_id in range(4):
        gpu_dir = traces_dir / f"t4_gpu{gpu_id}"
        
        if not gpu_dir.exists():
            logger.warning(f"✗ No traces for GPU {gpu_id}")
            all_traces_found = False
            continue
        
        traces = list(gpu_dir.glob("*.json"))
        
        if traces:
            logger.info(f"✓ GPU {gpu_id}: Found {len(traces)} trace file(s)")
            for trace in traces[:3]:  # Show first 3
                size_mb = trace.stat().st_size / 1024 / 1024
                logger.info(f"  - {trace.name} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"✗ No traces for GPU {gpu_id}")
            all_traces_found = False
    
    return all_traces_found


async def main():
    """Run all tests."""
    logger.info("\n")
    
    # Test two-phase workflow
    workflow_success = await test_two_phase_workflow()
    
    # Wait a bit for traces to be written
    await asyncio.sleep(2)
    
    # Check traces
    traces_found = await check_profiling_traces()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    if workflow_success:
        logger.info("✓ Two-Phase Workflow: PASS")
    else:
        logger.error("✗ Two-Phase Workflow: FAIL")
    
    if traces_found:
        logger.info("✓ Profiling traces: FOUND")
    else:
        logger.warning("⚠ Profiling traces: INCOMPLETE")
    
    logger.info("=" * 80)
    
    if workflow_success:
        logger.info("\n✓ TEST PASSED!")
        logger.info("\nNext steps:")
        logger.info("1. Check profiling traces for both phases")
        logger.info("2. Analyze KV cache reuse in Phase 2")
        logger.info("3. Compare generation diversity across members")
    else:
        logger.error("\n✗ TEST FAILED")


if __name__ == "__main__":
    asyncio.run(main())

