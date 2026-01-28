"""
Prefill server that runs on GPU 0 - receives HTTP requests, does prefill,
broadcasts KV to decode workers, collects results.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.distributed as dist
import logging
from fastapi import FastAPI
import uvicorn
from typing import List

from serving.common.model_loader import load_model
from serving.common.inference import generate, format_prompt_for_model
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled
from serving.t4_cluster.kv_transfer import synchronize_ranks

from schemas.generation import GenerationRequest, GenerationOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrefillServer:
    """Server on GPU 0 that handles HTTP and coordinates decode workers."""
    
    def __init__(self, model_name: str, device: str = "cuda:0", precision: str = "bf16"):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.rank = 0
        self.world_size = 4
        
        self.model = None
        self.tokenizer = None
        
        self.profiling_enabled = is_profiling_enabled("t4_gpu0")
        
        # FastAPI app
        self.app = FastAPI(title="T4 Prefill Server")
        self.app.post("/generate")(self.generate_endpoint)
        self.app.get("/health")(self.health_check)
    
    def initialize(self):
        """Load model and initialize distributed."""
        logger.info(f"[Rank {self.rank}] Initializing prefill server")
        
        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Load model with profiling
        gpu_id = "t4_gpu0"
        with profile_operation("init_model_load", gpu_id, "init", self.profiling_enabled):
            self.model, self.tokenizer = load_model(
                model_name=self.model_name,
                device=self.device,
                precision=self.precision,
            )
        
        logger.info(f"[Rank {self.rank}] Model loaded on {self.device}")
        
        # Synchronize with all workers
        synchronize_ranks()
        logger.info(f"[Rank {self.rank}] All workers synchronized")
    
    async def health_check(self):
        """Health check endpoint."""
        return {"status": "healthy", "rank": self.rank}
    
    async def generate_endpoint(self, request: GenerationRequest) -> GenerationOutput:
        """
        HTTP endpoint that:
        1. Receives request
        2. Does prefill
        3. Broadcasts KV to decode workers
        4. Collects results from decode workers
        """
        logger.info(f"[Rank {self.rank}] Received generation request: {request.request_id}")
        
        gpu_id = "t4_gpu0"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request.request_id,
            enabled=self.profiling_enabled
        ):
            # Format prompt
            with profile_operation("format_prompt", gpu_id, request.request_id, self.profiling_enabled):
                prompt = format_prompt_for_model(request.task_prompt, self.model_name)
            
            # Tokenize
            with profile_operation("tokenize", gpu_id, request.request_id, self.profiling_enabled):
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            
            batch_size, seq_len = input_ids.shape
            
            # Signal decode workers that work is coming
            with profile_operation("signal_workers", gpu_id, request.request_id, self.profiling_enabled):
                metadata = torch.tensor(
                    [1, batch_size, seq_len, request.max_tokens],
                    dtype=torch.long,
                    device=self.device
                )
                dist.broadcast(metadata, src=0)
                logger.info(f"[Rank {self.rank}] Broadcasted metadata to workers")
                
                # Broadcast input_ids
                dist.broadcast(input_ids.contiguous(), src=0)
                logger.info(f"[Rank {self.rank}] Broadcasted input_ids")
            
            # Do prefill to generate KV cache
            with profile_operation("prefill", gpu_id, request.request_id, self.profiling_enabled):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    
                logger.info(f"[Rank {self.rank}] Prefill complete, got {len(past_key_values)} layers of KV cache")
            
            # Broadcast KV cache to all decode workers
            with profile_operation("broadcast_kv", gpu_id, request.request_id, self.profiling_enabled):
                # past_key_values can be tuple of tuples or DynamicCache
                # Convert to tuple format if needed
                if hasattr(past_key_values, 'key_cache'):
                    # It's a DynamicCache object
                    kv_list = [(past_key_values.key_cache[i], past_key_values.value_cache[i]) 
                               for i in range(len(past_key_values.key_cache))]
                else:
                    # Already in tuple format
                    kv_list = past_key_values
                
                for layer_idx, kv_pair in enumerate(kv_list):
                    # Handle different formats - Phi-2 returns (key, value, ...) tuples
                    if isinstance(kv_pair, (tuple, list)):
                        # Just take first two elements (key and value)
                        key_cache = kv_pair[0]
                        value_cache = kv_pair[1]
                    else:
                        # Log structure for debugging
                        logger.error(f"[Rank {self.rank}] Unexpected KV structure at layer {layer_idx}: {type(kv_pair)}")
                        raise ValueError(f"Cannot unpack KV cache at layer {layer_idx}")
                    
                    dist.broadcast(key_cache.contiguous(), src=0)
                    dist.broadcast(value_cache.contiguous(), src=0)
                    
                logger.info(f"[Rank {self.rank}] Broadcasted KV cache ({len(kv_list)} layers)")
            
            # Do decode on GPU 0 as well
            with profile_operation("decode", gpu_id, request.request_id, self.profiling_enabled):
                generated_ids, _ = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    max_tokens=request.max_tokens,
                    temperature=0.7,
                )
                
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                )
            
            logger.info(f"[Rank {self.rank}] Completed full request")
            logger.info(f"[Rank {self.rank}] Generated: {generated_text[:200]}...")
        
        return GenerationOutput(
            answer=generated_text,
            assumptions=[],
            confidence=0.8,
            member_id="prefill_gpu0",
            request_id=request.request_id,
        )
    
    def run(self, port: int = 8000):
        """Run HTTP server."""
        logger.info(f"[Rank {self.rank}] Starting HTTP server on port {port}")
        
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=False,
        )


def run_prefill_server(rank: int, model_name: str):
    """Entry point for spawned process."""
    # Fix import paths
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "4"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    server = PrefillServer(model_name=model_name)
    server.initialize()
    server.run(port=8000)


if __name__ == "__main__":
    run_prefill_server(0, "microsoft/phi-2")

