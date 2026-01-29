#!/usr/bin/env python3
"""
Full end-to-end test of Llama-2-70B with 4-way TP using sample.txt.
Tests both Phase 1 (prefill) and Phase 2 (synthesis generation) with profiling.
"""

import asyncio
import httpx
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_full_pipeline():
    """Test full two-phase synthesis pipeline with profiling."""
    
    endpoint = "http://localhost:8020"
    request_id = f"full_pipeline_{int(time.time())}"
    
    logger.info("="*80)
    logger.info("FULL PIPELINE TEST: Llama-2-70B with 4-way TP + Profiling")
    logger.info("="*80)
    logger.info(f"Request ID: {request_id}")
    logger.info("")
    
    # Load sample.txt
    logger.info("Loading sample.txt...")
    sample_file = Path("/home/azureuser/multi-agent-test/sample.txt")
    if not sample_file.exists():
        logger.error("sample.txt not found!")
        return False
    
    context = sample_file.read_text()
    logger.info(f"✓ Loaded context: {len(context)} characters, {len(context.split())} words")
    logger.info("")
    
    # Wait for server
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
            logger.info("Make sure server is running with profiling enabled:")
            logger.info("  export ENABLE_PROFILING=true")
            logger.info("  python3 serving/a100_cluster/synthesis_server.py --profiling-enabled ...")
            return False
    
    logger.info("")
    logger.info("="*80)
    logger.info("PHASE 1: Initial Context Prefill")
    logger.info("="*80)
    
    # Phase 1: Prefill the full document
    logger.info(f"Sending {len(context)} characters for prefill...")
    logger.info("This will profile: tokenization + full prefill across 4 GPUs")
    logger.info("")
    
    phase1_start = time.time()
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/prefill_initial",
                json={
                    "request_id": request_id,
                    "context": context,
                    "metadata": {
                        "test_name": "full_pipeline_profiled",
                        "document": "sample.txt"
                    }
                }
            )
            
            phase1_elapsed = time.time() - phase1_start
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ Phase 1 SUCCESS")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Cache stored: {result.get('cache_stored')}")
                logger.info(f"  Context tokens: {result.get('context_tokens')}")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Cache size: {result.get('cache_size_mb'):.2f} MB")
                logger.info(f"  Total Phase 1 time: {phase1_elapsed:.2f} seconds")
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
                        logger.warning(f"  ⚠ GPU {gpu_id}: No traces found yet")
                
            else:
                logger.error(f"✗ Phase 1 FAILED: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except httpx.ReadTimeout:
            logger.error("✗ Phase 1 TIMEOUT (>180s)")
            logger.info("This might indicate an issue with the prefill or NCCL coordination")
            return False
        except Exception as e:
            logger.error(f"✗ Phase 1 EXCEPTION: {e}")
            return False
    
    logger.info("")
    logger.info("="*80)
    logger.info("PHASE 2: Final Synthesis with Generation")
    logger.info("="*80)
    
    # Phase 2: Append instruction and generate
    instruction = """
Based on the profiling documentation above, please provide a concise summary of:
1. The key profiling tools available in PyTorch
2. The main challenges identified for distributed tracing
3. Recommendations for profiling multi-GPU workloads

Please keep your response under 200 words.
"""
    
    logger.info(f"Appending instruction ({len(instruction)} chars)")
    logger.info("This will profile: incremental prefill + decode generation across 4 GPUs")
    logger.info("")
    
    phase2_start = time.time()
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/synthesize_final",
                json={
                    "request_id": request_id,
                    "appended_text": instruction,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "metadata": {
                        "test_name": "full_pipeline_profiled"
                    }
                }
            )
            
            phase2_elapsed = time.time() - phase2_start
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ Phase 2 SUCCESS")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Tokens generated: {result.get('num_tokens')}")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Decode time: {result.get('decode_time_ms'):.2f} ms")
                logger.info(f"  Total synthesis time: {result.get('total_time_ms'):.2f} ms")
                logger.info(f"  Total Phase 2 time: {phase2_elapsed:.2f} seconds")
                logger.info("")
                
                logger.info("Generated Synthesis:")
                logger.info("-" * 80)
                logger.info(result.get('synthesis', 'No synthesis returned'))
                logger.info("-" * 80)
                logger.info("")
                
                # Check Phase 2 traces
                logger.info("Checking for Phase 2 profiling traces...")
                trace_dir = Path("/home/azureuser/multi-agent-test/profiling_traces")
                for gpu_id in range(4):
                    gpu_dir = trace_dir / f"a100_gpu{gpu_id}"
                    traces = list(gpu_dir.glob(f"{request_id}*"))
                    logger.info(f"  ✓ GPU {gpu_id}: {len(traces)} total trace files")
                
            else:
                logger.error(f"✗ Phase 2 FAILED: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except httpx.ReadTimeout:
            logger.error("✗ Phase 2 TIMEOUT (>180s)")
            return False
        except Exception as e:
            logger.error(f"✗ Phase 2 EXCEPTION: {e}")
            return False
    
    # Final summary
    total_time = phase1_elapsed + phase2_elapsed
    
    logger.info("")
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Phase 1 (Prefill): {phase1_elapsed:.2f}s")
    logger.info(f"✓ Phase 2 (Synthesis): {phase2_elapsed:.2f}s")
    logger.info(f"✓ Total pipeline time: {total_time:.2f}s")
    logger.info("")
    logger.info("Profiling traces saved to: profiling_traces/a100_gpu{0,1,2,3}/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review traces in Chrome://tracing (load *_trace.json files)")
    logger.info("  2. Check summaries (*_summary.txt) for top operations")
    logger.info("  3. Analyze ExecutionTrace files (*_et.json) for graph structure")
    logger.info("="*80)
    logger.info("")
    logger.info("✓ FULL PIPELINE TEST PASSED!")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    exit(0 if success else 1)

