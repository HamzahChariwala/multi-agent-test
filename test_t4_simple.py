"""
Test script for simple T4 cluster architecture.
Only one HTTP endpoint on port 8000 (rank 0).
"""

import asyncio
import httpx
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_generate_endpoint():
    """Test the generation endpoint on rank 0."""
    endpoint = "http://localhost:8000"
    
    logger.info("=" * 70)
    logger.info(f"Testing T4 Prefill Server at {endpoint}")
    logger.info("=" * 70)
    
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
    
    # Test generation
    logger.info("\nStep 2: Testing generation...")
    
    # Read prompt from sample.txt
    sample_file = Path(__file__).parent / "sample.txt"
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            prompt_text = f.read()
        logger.info(f"Loaded prompt from sample.txt ({len(prompt_text)} characters)")
    else:
        prompt_text = "What is the capital of France?"
        logger.warning("sample.txt not found, using default prompt")
    
    request = {
        "task_prompt": prompt_text,
        "max_tokens": 15,
        "request_id": "test_request_001"
    }
    
    logger.info(f"Sending request with prompt: {prompt_text[:100]}... (truncated)")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{endpoint}/generate",
                json=request
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ SUCCESS (took {elapsed:.2f}s)")
                logger.info(f"Generated text: {result['answer'][:200]}...")
                logger.info(f"Member ID: {result.get('member_id', 'N/A')}")
                logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
                return True
            else:
                logger.error(f"✗ FAIL: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"✗ EXCEPTION: {e}")
        return False


async def check_profiling_traces():
    """Check if profiling traces were generated."""
    logger.info("\n" + "=" * 70)
    logger.info("Checking Profiling Traces")
    logger.info("=" * 70)
    
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
    logger.info("=" * 70)
    logger.info("T4 SIMPLE CLUSTER TEST")
    logger.info("=" * 70)
    logger.info("Architecture: HTTP on Rank 0 only, broadcasts to Ranks 1-3")
    logger.info("=" * 70)
    
    # Test generation
    generation_success = await test_generate_endpoint()
    
    # Wait a bit for traces to be written
    await asyncio.sleep(2)
    
    # Check traces
    traces_found = await check_profiling_traces()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    if generation_success:
        logger.info("✓ Generation: PASS")
    else:
        logger.error("✗ Generation: FAIL")
    
    if traces_found:
        logger.info("✓ Profiling traces: FOUND")
    else:
        logger.warning("⚠ Profiling traces: INCOMPLETE")
    
    logger.info("=" * 70)
    
    if generation_success:
        logger.info("\n✓ TEST PASSED!")
        logger.info("\nNext steps:")
        logger.info("1. Check profiling traces in profiling_traces/")
        logger.info("2. Look for 'prefill', 'broadcast_kv', and 'decode_only' operations")
        logger.info("3. Compare prefill time vs KV transfer time")
    else:
        logger.error("\n✗ TEST FAILED")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check if all 4 GPU processes are running")
        logger.error("2. Check logs for errors")
        logger.error("3. Verify NCCL initialization succeeded")


if __name__ == "__main__":
    asyncio.run(main())

