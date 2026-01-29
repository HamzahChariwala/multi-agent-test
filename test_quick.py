#!/usr/bin/env python3
"""Quick test with small input and limited output."""

import asyncio
import httpx
import time

async def test():
    endpoint = "http://localhost:8020"
    request_id = f"test_{int(time.time())}"
    
    # Small input
    with open("/home/azureuser/multi-agent-test/test_input_small.txt") as f:
        context = f.read()
    
    prompt = context + "\n\nSummarize the above in one sentence:"
    
    print(f"Testing with {len(prompt)} characters, max 50 tokens...")
    print()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.time()
        response = await client.post(
            f"{endpoint}/synthesize",
            json={
                "request_id": request_id,
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.7
            }
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ SUCCESS ({elapsed:.1f}s)")
            print(f"  Prefill: {result['prefill_time_ms']:.0f}ms")
            print(f"  Decode: {result['decode_time_ms']:.0f}ms")
            print(f"  Tokens: {result['num_tokens']}")
            print()
            print("Generated:")
            print(result['text'])
            print()
            print(f"Traces: profiling_traces/a100_gpu{{0,1,2,3}}/{request_id}*")
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(response.text)
            return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test())
    exit(0 if success else 1)

