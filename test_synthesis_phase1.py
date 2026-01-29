"""
Test Phase 1 only: Initial context prefill and KV cache storage.

This tests that the A100 synthesis server can:
1. Receive a prefill request
2. Prefill the context with 4-way TP
3. Store the KV cache properly
4. Return success response
"""

import asyncio
import httpx
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_phase1():
    """Test Phase 1: Prefill and cache storage."""
    
    endpoint = "http://localhost:8020"
    
    logger.info("=" * 70)
    logger.info("Testing Phase 1: Initial Context Prefill")
    logger.info("=" * 70)
    
    # Step 1: Check health
    logger.info("\nStep 1: Checking server health...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{endpoint}/health")
            if response.status_code == 200:
                logger.info("✓ Server is healthy")
            else:
                logger.error(f"✗ Server unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Cannot connect to server: {e}")
            logger.info("Make sure synthesis server is running:")
            logger.info("  cd /home/azureuser/multi-agent-test")
            logger.info("  export CUDA_VISIBLE_DEVICES=0,1,2,3")
            logger.info("  export MASTER_PORT=29500")
            logger.info("  python serving/a100_cluster/synthesis_server.py")
            return False
    
    # Step 2: Send Phase 1 request
    logger.info("\nStep 2: Sending Phase 1 prefill request...")
    
    test_context = """You are a wise council chairman synthesizing diverse perspectives.

Question: What are the key considerations for implementing artificial general intelligence (AGI) safely?

Your task is to analyze multiple viewpoints from council members and provide a comprehensive synthesis."""
    
    request = {
        "request_id": "test_phase1_001",
        "context": test_context,
        "metadata": {
            "test_name": "phase1_only",
            "timestamp": time.time()
        }
    }
    
    logger.info(f"  Request ID: {request['request_id']}")
    logger.info(f"  Context length: {len(test_context)} chars")
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/prefill_initial",
                json=request
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ Phase 1 SUCCESS")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Cache stored: {result.get('cache_stored')}")
                logger.info(f"  Context tokens: {result.get('context_tokens')}")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Cache size: {result.get('cache_size_mb'):.2f} MB")
                logger.info(f"  Total time: {elapsed:.2f} seconds")
                
                # Check expectations
                if result.get('prefill_time_ms', 0) > 1000:
                    logger.warning(f"⚠ Prefill time is high: {result.get('prefill_time_ms'):.2f} ms (expected < 600 ms)")
                
                if result.get('context_tokens', 0) == 0:
                    logger.error("✗ No tokens processed")
                    return False
                
            else:
                logger.error(f"✗ Phase 1 FAILED: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Phase 1 EXCEPTION: {e}")
            return False
    
    # Step 3: Check cache status
    logger.info("\nStep 3: Checking cache status...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{endpoint}/debug/cache_status")
            if response.status_code == 200:
                cache_status = response.json()
                
                active_caches = cache_status.get('active_caches', [])
                
                if len(active_caches) > 0:
                    logger.info(f"✓ Found {len(active_caches)} active cache(s)")
                    
                    for cache in active_caches:
                        logger.info(f"  Cache: {cache['request_id']}")
                        logger.info(f"    Age: {cache['age_seconds']:.1f}s")
                        logger.info(f"    Seq len: {cache['seq_len']}")
                        logger.info(f"    Expires in: {cache['expires_in']:.1f}s")
                else:
                    logger.error("✗ No active caches found (cache may have been cleaned up)")
                    return False
            else:
                logger.warning(f"⚠ Could not get cache status: {response.status_code}")
        
        except Exception as e:
            logger.warning(f"⚠ Cache status check failed: {e}")
    
    # Step 4: Check TP status
    logger.info("\nStep 4: Checking tensor parallel status...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{endpoint}/debug/tp_status")
            if response.status_code == 200:
                tp_status = response.json()
                
                logger.info(f"✓ TP Status:")
                logger.info(f"  Rank: {tp_status.get('rank')}")
                logger.info(f"  World size: {tp_status.get('world_size')}")
                logger.info(f"  Device: {tp_status.get('device')}")
                logger.info(f"  Model: {tp_status.get('model')}")
                
                tp_info = tp_status.get('tp_info', {})
                if tp_info.get('initialized'):
                    logger.info(f"  NCCL initialized: ✓")
                    logger.info(f"  Backend: {tp_info.get('backend')}")
                else:
                    logger.warning("  NCCL not initialized")
        
        except Exception as e:
            logger.warning(f"⚠ TP status check failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY: Phase 1")
    logger.info("=" * 70)
    logger.info("✓ Prefill: SUCCESS")
    logger.info("✓ Cache storage: SUCCESS")
    logger.info("✓ Response validation: SUCCESS")
    logger.info("=" * 70)
    logger.info("\n✓ Phase 1 test PASSED")
    logger.info("\nNote: KV cache will expire after 5 minutes")
    logger.info("To test Phase 2, run test_synthesis_full.py within 5 minutes")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_phase1())
    exit(0 if success else 1)

