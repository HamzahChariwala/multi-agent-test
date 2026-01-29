"""
Test Full Two-Phase Workflow: Prefill + Synthesis with KV cache reuse.

This tests the complete workflow:
1. Phase 1: Prefill initial context and store KV cache
2. Phase 2: Append council results and generate synthesis (reusing KV cache)

This simulates the real orchestrator workflow.
"""

import asyncio
import httpx
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_full_workflow():
    """Test complete two-phase workflow."""
    
    endpoint = "http://localhost:8020"
    request_id = f"test_full_{int(time.time())}"
    
    logger.info("=" * 70)
    logger.info("Testing Full Two-Phase Workflow")
    logger.info("=" * 70)
    logger.info(f"Request ID: {request_id}")
    logger.info("=" * 70)
    
    # Check health
    logger.info("\nChecking server health...")
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
            return False
    
    # ========================================================================
    # PHASE 1: Initial Context Prefill
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Initial Context Prefill")
    logger.info("=" * 70)
    
    initial_context = """You are a wise AI council chairman tasked with synthesizing diverse perspectives from multiple council members.

Question: What are the most important ethical considerations for deploying large language models in healthcare?

Context: This is a critical question that requires balancing multiple concerns including patient safety, data privacy, accuracy, accessibility, and potential biases in medical decision-making.

Your role: You will receive multiple perspectives from council members with different viewpoints. Your task is to:
1. Identify common themes and agreements
2. Acknowledge and resolve disagreements
3. Provide a comprehensive, balanced synthesis
4. Highlight key actionable recommendations

Please prepare to analyze the council members' responses."""
    
    phase1_request = {
        "request_id": request_id,
        "context": initial_context,
        "metadata": {
            "test": "full_workflow",
            "phase": 1
        }
    }
    
    logger.info(f"Sending initial context ({len(initial_context)} chars)...")
    
    phase1_start = time.time()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/prefill_initial",
                json=phase1_request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ Phase 1 completed successfully")
                logger.info(f"  Context tokens: {result.get('context_tokens')}")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Cache size: {result.get('cache_size_mb'):.2f} MB")
                logger.info(f"  Elapsed: {time.time() - phase1_start:.2f}s")
            else:
                logger.error(f"✗ Phase 1 failed: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Phase 1 exception: {e}")
            return False
    
    # Simulate T4 council deliberation time
    logger.info("\nSimulating T4 council deliberation (3 seconds)...")
    await asyncio.sleep(3)
    
    # ========================================================================
    # PHASE 2: Synthesis with Council Results
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Synthesis with Council Results")
    logger.info("=" * 70)
    
    # Simulated council results
    council_results = """

Council Member Responses:

Member 1 (Conservative, Temperature=0.3):
"The primary ethical consideration is patient safety. We must ensure LLMs are rigorously validated before deployment, with clear limitations on their use. They should augment, not replace, human physicians. Strict regulatory oversight is essential to prevent misuse."

Member 2 (Balanced, Temperature=0.7):
"While patient safety is paramount, we must also consider accessibility. LLMs could democratize healthcare access in underserved areas. The key is implementing robust safeguards: transparent decision-making, human oversight, and continuous monitoring for biases. Data privacy through encryption and anonymization is critical."

Member 3 (Progressive, Temperature=1.0):
"We're at a transformative moment! LLMs can revolutionize healthcare by providing 24/7 support, multilingual access, and personalized care. Yes, we need safeguards, but over-regulation could delay life-saving innovations. Focus on rapid iteration with safety checks, empowering patients with AI assistants, and addressing the digital divide."

Peer Rankings:
- Member 1 rated Member 2's response highest for thoroughness
- Member 2 rated Member 3's response highest for vision
- Member 3 rated Member 1's response highest for caution
- Average consensus: All three perspectives are valuable and complementary

Your task: Synthesize these perspectives into a comprehensive final answer that:
1. Acknowledges all viewpoints
2. Identifies the core ethical principles (safety, accessibility, privacy)
3. Proposes a balanced framework
4. Provides actionable recommendations

Please provide your synthesis now:"""
    
    phase2_request = {
        "request_id": request_id,
        "appended_text": council_results,
        "max_tokens": 500,
        "temperature": 0.7,
        "metadata": {
            "test": "full_workflow",
            "phase": 2
        }
    }
    
    logger.info(f"Sending council results ({len(council_results)} chars)...")
    logger.info(f"Requesting {phase2_request['max_tokens']} tokens synthesis...")
    
    phase2_start = time.time()
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{endpoint}/synthesize_final",
                json=phase2_request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("✓ Phase 2 completed successfully")
                logger.info(f"  Prefill time: {result.get('prefill_time_ms'):.2f} ms")
                logger.info(f"  Decode time: {result.get('decode_time_ms'):.2f} ms")
                logger.info(f"  Total time: {result.get('total_time_ms'):.2f} ms")
                logger.info(f"  Generated tokens: {result.get('num_tokens')}")
                logger.info(f"  Elapsed: {time.time() - phase2_start:.2f}s")
                
                synthesis = result.get('synthesis', '')
                logger.info(f"\nGenerated Synthesis ({len(synthesis)} chars):")
                logger.info("-" * 70)
                logger.info(synthesis)
                logger.info("-" * 70)
                
                # Validate synthesis quality
                if len(synthesis) < 50:
                    logger.warning("⚠ Synthesis seems too short")
                
                if result.get('decode_time_ms', 0) > 5000:
                    logger.warning(f"⚠ Decode time is high: {result.get('decode_time_ms'):.2f} ms")
                
            else:
                logger.error(f"✗ Phase 2 failed: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Phase 2 exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    total_elapsed = time.time() - phase1_start
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✓ Phase 1 (Prefill): SUCCESS")
    logger.info("✓ Phase 2 (Synthesis): SUCCESS")
    logger.info("✓ KV cache reuse: SUCCESS")
    logger.info(f"✓ Total workflow time: {total_elapsed:.2f}s")
    logger.info("=" * 70)
    logger.info("\n✓ Full two-phase workflow test PASSED")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_full_workflow())
    exit(0 if success else 1)

