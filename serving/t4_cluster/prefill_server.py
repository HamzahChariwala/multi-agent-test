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
from serving.common.profiling import profile_operation, is_profiling_enabled
from serving.common.continuous_profiler import get_manager, record_request_activity
from serving.t4_cluster.kv_transfer import synchronize_ranks

from schemas.generation import GenerationRequest, GenerationOutput
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleJudgeRequest(BaseModel):
    """Simple request for judging phase."""
    request_id: str = Field(..., description="Unique request identifier")
    max_tokens: int = Field(default=50, description="Max tokens for ranking")


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
        
        # Store session data for two-phase workflow
        self.current_kv_cache = None
        self.current_input_ids = None
        self.member_responses = []
        
        # FastAPI app
        self.app = FastAPI(title="T4 Prefill Server")
        self.app.post("/generate")(self.generate_endpoint)
        self.app.post("/judge")(self.judge_endpoint)
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
        
        # Start continuous profiler monitoring
        if self.profiling_enabled:
            manager = get_manager(idle_timeout=30.0)
            manager.start_monitoring()
            logger.info(f"[Rank {self.rank}] Started continuous profiler monitoring")
    
    async def health_check(self):
        """Health check endpoint."""
        return {"status": "healthy", "rank": self.rank}
    
    async def judge_endpoint(self, request: SimpleJudgeRequest) -> dict:
        """
        Phase 2: Peer ranking endpoint.
        Uses stored KV cache from Phase 1 and appends ranking instruction.
        """
        logger.info(f"[Rank {self.rank}] Received judging request: {request.request_id}")
        
        if not self.member_responses:
            return {"error": "No member responses from Phase 1. Run /generate first."}
        
        if self.current_kv_cache is None:
            return {"error": "No KV cache from Phase 1. Run /generate first."}
        
        gpu_id = "t4_gpu0"
        
        # Mark request activity for continuous profiling
        if self.profiling_enabled:
            record_request_activity(gpu_id, "two_phase_session")
        
        # Format all responses as A, B, C, D
        with profile_operation("format_ranking_prompt", gpu_id, request.request_id, self.profiling_enabled):
            ranking_text = "\n\nHere are the responses from all council members:\n\n"
            for idx, resp in enumerate(self.member_responses):
                label = chr(65 + idx)  # A, B, C, D
                ranking_text += f"[{label}] {resp['answer']}\n\n"
            
            ranking_text += (
                "Rank these responses from best to worst by listing only the letters "
                "in order (e.g., 'B A D C'). Provide ONLY the ranking with no explanation:"
            )
            
            logger.info(f"[Rank {self.rank}] Ranking prompt: {ranking_text[:200]}...")
        
        # Tokenize the ranking instruction
        with profile_operation("tokenize_ranking", gpu_id, request.request_id, self.profiling_enabled):
            encoded = self.tokenizer(
                ranking_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            ranking_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(ranking_ids)).to(self.device)
        
        batch_size, new_seq_len = ranking_ids.shape
        
        # Broadcast judging metadata to workers
        with profile_operation("signal_judging", gpu_id, request.request_id, self.profiling_enabled):
            # [phase=2, batch_size, seq_len, max_tokens]
            metadata = torch.tensor(
                [2, batch_size, new_seq_len, request.max_tokens],
                dtype=torch.long,
                device=self.device
            )
            dist.broadcast(metadata, src=0)
            
            # Broadcast ranking instruction tokens
            dist.broadcast(ranking_ids.contiguous(), src=0)
            logger.info(f"[Rank {self.rank}] Broadcasted ranking instruction")
        
        # Append to existing KV cache by running prefill on new tokens
        with profile_operation("prefill_ranking", gpu_id, request.request_id, self.profiling_enabled):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ranking_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.current_kv_cache,
                    use_cache=True,
                )
                updated_kv_cache = outputs.past_key_values
                
            logger.info(f"[Rank {self.rank}] Extended KV cache with ranking instruction")
        
        # Broadcast the EXTENDED KV cache
        with profile_operation("broadcast_extended_kv", gpu_id, request.request_id, self.profiling_enabled):
            if hasattr(updated_kv_cache, 'key_cache'):
                kv_list = [(updated_kv_cache.key_cache[i], updated_kv_cache.value_cache[i]) 
                           for i in range(len(updated_kv_cache.key_cache))]
            else:
                kv_list = updated_kv_cache
            
            for layer_idx, kv_pair in enumerate(kv_list):
                if isinstance(kv_pair, (tuple, list)):
                    key_cache = kv_pair[0]
                    value_cache = kv_pair[1]
                else:
                    raise ValueError(f"Cannot unpack KV cache at layer {layer_idx}")
                
                dist.broadcast(key_cache.contiguous(), src=0)
                dist.broadcast(value_cache.contiguous(), src=0)
            
            logger.info(f"[Rank {self.rank}] Broadcasted extended KV cache")
        
        # Generate ranking on GPU 0
        with profile_operation("decode_ranking", gpu_id, request.request_id, self.profiling_enabled):
            # Concatenate original input with ranking instruction
            full_input_ids = torch.cat([self.current_input_ids, ranking_ids], dim=1)
            
            generated_ids, _ = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=full_input_ids,
                past_key_values=updated_kv_cache,
                max_tokens=request.max_tokens,
                temperature=0.3,  # Lower temp for more consistent rankings
            )
            
            # Decode ONLY the new tokens (skip the full input)
            input_length = full_input_ids.shape[1]
            new_tokens = generated_ids[0, input_length:]
            ranking_output_only = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
        
        logger.info(f"[Rank {self.rank}] Generated ranking (new tokens only): {ranking_output_only}")
        
        # Collect rankings from all workers
        with profile_operation("collect_rankings", gpu_id, request.request_id, self.profiling_enabled):
            max_text_len = 256  # Rankings are short
            text_tensor = torch.zeros(max_text_len, dtype=torch.long, device=self.device)
            encoded = self.tokenizer.encode(ranking_output_only, add_special_tokens=False)[:max_text_len]
            text_tensor[:len(encoded)] = torch.tensor(encoded, device=self.device)
            
            gathered_rankings = [torch.zeros_like(text_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_rankings, text_tensor)
            
            all_rankings = []
            for rank_idx, text_tensor in enumerate(gathered_rankings):
                nonzero = torch.nonzero(text_tensor)
                if len(nonzero) > 0:
                    actual_len = nonzero[-1].item() + 1
                else:
                    actual_len = 0
                
                decoded = self.tokenizer.decode(
                    text_tensor[:actual_len].cpu().tolist(),
                    skip_special_tokens=True
                )
                
                all_rankings.append({
                    "judge_id": f"member_{rank_idx}",
                    "ranking": decoded,
                })
                
                logger.info(f"[Rank {self.rank}] Collected ranking from rank {rank_idx}: {decoded}")
        
        logger.info(f"[Rank {self.rank}] Completed judging phase")
        
        return {
            "rankings": all_rankings,
            "request_id": request.request_id,
        }
    
    async def generate_endpoint(self, request: GenerationRequest) -> dict:
        """
        HTTP endpoint that:
        1. Receives request
        2. Does prefill
        3. Broadcasts KV to decode workers
        4. Collects results from decode workers
        """
        logger.info(f"[Rank {self.rank}] Received generation request: {request.request_id}")
        
        gpu_id = "t4_gpu0"
        
        # Mark request activity for continuous profiling
        if self.profiling_enabled:
            record_request_activity(gpu_id, "two_phase_session")
        
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
        
        # Signal decode workers that Phase 1 (Generate) is starting
        with profile_operation("signal_workers", gpu_id, request.request_id, self.profiling_enabled):
            # [phase=1 (generate), batch_size, seq_len, max_tokens]
            metadata = torch.tensor(
                [1, batch_size, seq_len, request.max_tokens],
                dtype=torch.long,
                device=self.device
            )
            dist.broadcast(metadata, src=0)
            logger.info(f"[Rank {self.rank}] Broadcasted Phase 1 metadata to workers")
            
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
            
            # Decode ONLY the new tokens (skip the input)
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[0, input_length:]
            generated_text_only = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
        
        logger.info(f"[Rank {self.rank}] Completed full request")
        logger.info(f"[Rank {self.rank}] Generated (new tokens only): {generated_text_only[:200]}...")
        
        # Collect responses from all workers
        with profile_operation("collect_responses", gpu_id, request.request_id, self.profiling_enabled):
            # Encode ONLY the new tokens to send (use fixed max length)
            max_text_len = 512  # Much smaller now since we're only sending generated portion
            text_tensor = torch.zeros(max_text_len, dtype=torch.long, device=self.device)
            encoded = self.tokenizer.encode(generated_text_only, add_special_tokens=False)[:max_text_len]
            text_tensor[:len(encoded)] = torch.tensor(encoded, device=self.device)
            
            # Gather from all ranks
            gathered_texts = [torch.zeros_like(text_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_texts, text_tensor)
            
            # Decode all responses
            self.member_responses = []
            for rank_idx, text_tensor in enumerate(gathered_texts):
                # Find actual length (stop at first padding/zero)
                nonzero = torch.nonzero(text_tensor)
                if len(nonzero) > 0:
                    actual_len = nonzero[-1].item() + 1
                else:
                    actual_len = 0
                
                decoded = self.tokenizer.decode(
                    text_tensor[:actual_len].cpu().tolist(),
                    skip_special_tokens=True
                )
                
                self.member_responses.append({
                    "member_id": f"member_{rank_idx}",
                    "answer": decoded,
                    "confidence": 0.8,
                })
                
                logger.info(f"[Rank {self.rank}] Collected from rank {rank_idx}: {decoded[:100]}...")
            
        # Store KV cache for Phase 2
        self.current_kv_cache = past_key_values
        self.current_input_ids = input_ids
        
        logger.info(f"[Rank {self.rank}] Stored KV cache for judging phase")
        
        return {
            "answer": generated_text_only,
            "member_id": "member_0",
            "confidence": 0.8,
            "request_id": request.request_id,
            "member_responses": self.member_responses,
        }
    
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

