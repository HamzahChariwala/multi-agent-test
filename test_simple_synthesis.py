#!/usr/bin/env python3
"""
Test simplified synthesis: single-phase prefill + decode with profiling.
No KV cache reuse - just take full prompt and generate.
"""

import asyncio
import httpx
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple_synthesis():
    """Test single-phase synthesis with full prompt."""
    
    endpoint = "http://localhost:8020"
    request_id = f"synthesis_{int(time.time())}"
    
    logger.info("="*80)
    logger.info("SIMPLE SYNTHESIS TEST: Llama-2-70B with 4-way TP + Profiling")
    logger.info("="*80)
    logger.info(f"Request ID: {request_id}")
    logger.info("")
    
    # Load sample.txt
    logger.info("Loading sample.txt...")
    sample_file = Path("/home/azureuser/multi-agent-test/sample.txt")
    context = sample_file.read_text()
    logger.info(f"✓ Loaded context: {len(context)} characters, {len(context.split())} words")
    logger.info("")
    
    # Build full prompt (context + instruction like we'll get from T4)
    instruction = """

Based on the profiling documentation above, please provide a concise summary of:
1. The key profiling tools available in PyTorch
2. The main challenges identified for distributed tracing
3. Recommendations for profiling multi-GPU workloads

Please keep your response under 200 words."""
    
    full_prompt = context + instruction
    logger.info(f"Full prompt: {len(full_prompt)} characters")
    logger.info("")
    
    # Check server health
    logger.info("Checking server health...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{endpoint}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Server healthy")
                logger.info(f"  Model: {data.get('model')}")
                logger.info(f"  Ranks: {data.get('world_size')}")
            else:
                logger.error(f"✗ Server unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Cannot connect: {e}")
            return False
    
    logger.info("")
    logger.info("="*80)
    logger.info("SYNTHESIS: Prefill + Decode in One Go")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Send synthesis request
    logger.info(f"Sending {len(full_prompt)} characters for synthesis...")
    logger.info("This will profile: tokenization + prefill + decode across 4 GPUs")
    logger.info("")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/synthesize",
                json={
                    "request_id": request_id,
                    "prompt": full_prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "metadata": {
                        "test_name": "simple_synthesis",
                        "document": "sample.txt"
                    }
                }
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ SYNTHESIS SUCCESS")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Tokens generated: {result.get('num_tokens')}")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Decode time: {result.get('decode_time_ms'):.2f} ms")
                logger.info(f"  Total time: {result.get('total_time_ms'):.2f} ms")
                logger.info(f"  Wall time: {elapsed:.2f} seconds")
                logger.info("")
                
                logger.info("Generated Text:")
                logger.info("-" * 80)
                logger.info(result.get('text', 'No text returned'))
                logger.info("-" * 80)
                logger.info("")
                
                # Check profiling traces
                logger.info("Checking for profiling traces...")
                trace_dir = Path("/home/azureuser/multi-agent-test/profiling_traces")
                for gpu_id in range(4):
                    gpu_dir = trace_dir / f"a100_gpu{gpu_id}"
                    traces = list(gpu_dir.glob(f"{request_id}*"))
                    if traces:
                        logger.info(f"  ✓ GPU {gpu_id}: {len(traces)} trace files")
                        for trace in traces:
                            size = trace.stat().st_size / 1024  # KB
                            logger.info(f"    - {trace.name} ({size:.1f} KB)")
                    else:
                        logger.warning(f"  ⚠ GPU {gpu_id}: No traces found")
                
            else:
                logger.error(f"✗ SYNTHESIS FAILED: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except httpx.ReadTimeout:
            logger.error("✗ SYNTHESIS TIMEOUT (>300s)")
            return False
        except Exception as e:
            logger.error(f"✗ SYNTHESIS EXCEPTION: {e}")
            return False
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Total time: {elapsed:.2f}s")
    logger.info("")
    logger.info("Profiling traces saved to: profiling_traces/a100_gpu{0,1,2,3}/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review traces in Chrome://tracing")
    logger.info("  2. Begin T4-A100 integration")
    logger.info("="*80)
    logger.info("")
    logger.info("✓ SIMPLE SYNTHESIS TEST PASSED!")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_simple_synthesis())
    exit(0 if success else 1)

