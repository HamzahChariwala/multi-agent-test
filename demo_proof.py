#!/usr/bin/env python3
"""
Simple demo to PROVE the synthesis server is working with actual text generation.
"""

import asyncio
import httpx
import json

async def main():
    print("\n" + "="*70)
    print("PROOF OF WORKING SYNTHESIS SERVER")
    print("="*70)
    
    client = httpx.AsyncClient(timeout=120.0)
    
    # Phase 1: Prefill
    print("\n[PHASE 1] Sending context for prefill...")
    context = "You are a helpful AI assistant. Please answer this question: What is 2+2?"
    
    phase1_req = {
        "request_id": "demo_proof",
        "context": context
    }
    
    response = await client.post("http://localhost:8020/prefill_initial", json=phase1_req)
    result = response.json()
    
    print(f"✓ Prefill completed in {result['prefill_time_ms']:.1f}ms")
    print(f"✓ Stored {result['context_tokens']} tokens in KV cache ({result['cache_size_mb']:.2f} MB)")
    
    # Phase 2: Generate synthesis
    print("\n[PHASE 2] Generating answer with KV cache reuse...")
    
    phase2_req = {
        "request_id": "demo_proof",
        "appended_text": "\n\nAnswer: ",
        "max_tokens": 30,
        "temperature": 0.7
    }
    
    response = await client.post("http://localhost:8020/synthesize_final", json=phase2_req)
    result = response.json()
    
    print(f"\n{'='*70}")
    print("GENERATED TEXT:")
    print(f"{'='*70}")
    print(f"{result['synthesis']}")
    print(f"{'='*70}")
    
    print(f"\n✓ Generated {result['num_tokens']} tokens")
    print(f"✓ Prefill time: {result['prefill_time_ms']:.1f}ms")
    print(f"✓ Decode time: {result['decode_time_ms']:.1f}ms")
    print(f"✓ Total time: {result['total_time_ms']:.1f}ms")
    
    print("\n" + "="*70)
    print("✓ PROOF COMPLETE: Server is generating actual text!")
    print("="*70 + "\n")
    
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())

