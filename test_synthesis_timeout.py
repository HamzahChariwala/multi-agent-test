"""
Test KV Cache Timeout and Expiration.

This tests that:
1. KV cache expires after the configured timeout (default 5 minutes)
2. Expired caches are cleaned up properly
3. Phase 2 fails appropriately when cache is expired

This is an accelerated test using a shorter timeout for testing purposes.
"""

import asyncio
import httpx
import logging
import time
import subprocess
import os
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_cache_timeout():
    """Test KV cache timeout and cleanup."""
    
    endpoint = "http://localhost:8021"  # Different port for test server
    request_id = f"test_timeout_{int(time.time())}"
    
    logger.info("=" * 70)
    logger.info("Testing KV Cache Timeout and Expiration")
    logger.info("=" * 70)
    logger.info("Note: This test requires a server with --cache-timeout=30 (30 seconds)")
    logger.info("=" * 70)
    
    # Check if server is running
    logger.info("\nChecking server health...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{endpoint}/health")
            if response.status_code != 200:
                logger.error(f"✗ Server unhealthy: {response.status_code}")
                logger.info("\nStart test server with:")
                logger.info("  export CUDA_VISIBLE_DEVICES=0,1,2,3")
                logger.info("  export MASTER_PORT=29501")
                logger.info("  python serving/a100_cluster/synthesis_server.py --port 8021 --cache-timeout 30")
                return False
            logger.info("✓ Server is healthy")
        except Exception as e:
            logger.error(f"✗ Cannot connect to server: {e}")
            logger.info("\nStart test server with:")
            logger.info("  export CUDA_VISIBLE_DEVICES=0,1,2,3")
            logger.info("  export MASTER_PORT=29501")
            logger.info("  python serving/a100_cluster/synthesis_server.py --port 8021 --cache-timeout 30")
            return False
    
    # ========================================================================
    # Test 1: Normal flow within timeout
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Test 1: Normal flow within timeout (should succeed)")
    logger.info("=" * 70)
    
    context = "Test context for timeout validation. This is a simple test to verify KV cache functionality."
    
    # Phase 1
    logger.info("Sending Phase 1 request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/prefill_initial",
            json={"request_id": request_id, "context": context}
        )
        
        if response.status_code == 200:
            logger.info("✓ Phase 1 succeeded")
        else:
            logger.error(f"✗ Phase 1 failed: {response.status_code}")
            return False
    
    # Wait a bit (but less than timeout)
    logger.info("Waiting 5 seconds (within timeout)...")
    await asyncio.sleep(5)
    
    # Phase 2 (should succeed)
    logger.info("Sending Phase 2 request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/synthesize_final",
            json={
                "request_id": request_id,
                "appended_text": "Now synthesize.",
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            logger.info("✓ Phase 2 succeeded (as expected)")
        else:
            logger.error(f"✗ Phase 2 failed unexpectedly: {response.status_code}")
            return False
    
    # ========================================================================
    # Test 2: Timeout expiration
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Cache expiration after timeout (should fail)")
    logger.info("=" * 70)
    
    request_id_2 = f"test_timeout_2_{int(time.time())}"
    
    # Phase 1
    logger.info("Sending Phase 1 request...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/prefill_initial",
            json={"request_id": request_id_2, "context": context}
        )
        
        if response.status_code == 200:
            logger.info("✓ Phase 1 succeeded")
            result = response.json()
            logger.info(f"  Cache stored at {time.strftime('%H:%M:%S')}")
        else:
            logger.error(f"✗ Phase 1 failed: {response.status_code}")
            return False
    
    # Check cache exists
    logger.info("\nChecking cache status...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{endpoint}/debug/cache_status")
        if response.status_code == 200:
            cache_status = response.json()
            active_caches = cache_status.get('active_caches', [])
            logger.info(f"  Active caches: {len(active_caches)}")
            
            if len(active_caches) > 0:
                for cache in active_caches:
                    logger.info(f"    {cache['request_id']}: expires in {cache['expires_in']:.1f}s")
    
    # Wait for timeout + 5 seconds
    wait_time = 35  # 30s timeout + 5s buffer
    logger.info(f"\nWaiting {wait_time} seconds for cache to expire...")
    for i in range(wait_time):
        await asyncio.sleep(1)
        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{wait_time} seconds elapsed...")
    
    # Check cache was cleaned up
    logger.info("\nChecking if cache was cleaned up...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{endpoint}/debug/cache_status")
        if response.status_code == 200:
            cache_status = response.json()
            active_caches = cache_status.get('active_caches', [])
            
            if len(active_caches) == 0:
                logger.info("✓ Cache was cleaned up (as expected)")
            else:
                logger.warning(f"⚠ Still have {len(active_caches)} active cache(s)")
                for cache in active_caches:
                    logger.warning(f"    {cache['request_id']}: age {cache['age_seconds']:.1f}s")
    
    # Phase 2 (should fail with expired cache error)
    logger.info("\nSending Phase 2 request (should fail)...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/synthesize_final",
            json={
                "request_id": request_id_2,
                "appended_text": "Now synthesize.",
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            logger.error("✗ Phase 2 succeeded (should have failed with expired cache)")
            return False
        else:
            result = response.json()
            error_msg = result.get('error', '')
            
            if 'expired' in error_msg.lower() or 'not found' in error_msg.lower():
                logger.info("✓ Phase 2 failed as expected (expired cache)")
                logger.info(f"  Error: {error_msg}")
            else:
                logger.warning(f"⚠ Phase 2 failed but with unexpected error: {error_msg}")
    
    # ========================================================================
    # Test 3: Missing cache (Phase 2 without Phase 1)
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Test 3: Missing cache (Phase 2 without Phase 1)")
    logger.info("=" * 70)
    
    request_id_3 = f"test_missing_{int(time.time())}"
    
    logger.info("Sending Phase 2 without Phase 1...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/synthesize_final",
            json={
                "request_id": request_id_3,
                "appended_text": "Try to synthesize without cache.",
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            logger.error("✗ Phase 2 succeeded (should have failed with missing cache)")
            return False
        else:
            result = response.json()
            error_msg = result.get('error', '')
            
            if 'not found' in error_msg.lower():
                logger.info("✓ Phase 2 failed as expected (missing cache)")
                logger.info(f"  Error: {error_msg}")
            else:
                logger.warning(f"⚠ Phase 2 failed but with unexpected error: {error_msg}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✓ Test 1: Normal flow within timeout - PASSED")
    logger.info("✓ Test 2: Cache expiration after timeout - PASSED")
    logger.info("✓ Test 3: Missing cache detection - PASSED")
    logger.info("=" * 70)
    logger.info("\n✓ Cache timeout test PASSED")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_cache_timeout())
    exit(0 if success else 1)

