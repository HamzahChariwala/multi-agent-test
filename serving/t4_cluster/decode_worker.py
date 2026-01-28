"""Decode worker for T4 cluster (Ranks 1-3)."""

import os
import torch
import torch.distributed as dist
import logging
from typing import Optional

from serving.common.model_loader import load_model, get_model_config
from serving.common.inference import generate, format_prompt_for_model
from serving.common.http_server import BaseModelServer
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled
from serving.t4_cluster.kv_transfer import broadcast_kv_cache, allocate_kv_buffer, synchronize_ranks

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput

logger = logging.getLogger(__name__)


class DecodeWorker(BaseModelServer):
    """
    Decode worker that receives KV cache and performs decoding.
    Runs on Ranks 1-3 of T4 cluster.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        port: int,
        temperature: float,
        model_name: str,
        device: str,
        precision: str = "bf16",
    ):
        """
        Initialize decode worker.
        
        Args:
            rank: Process rank (1-3)
            world_size: Total number of processes
            port: HTTP port to serve on
            temperature: Sampling temperature
            model_name: Model name
            device: Device to use
            precision: Model precision
        """
        assert rank > 0, "DecodeWorker must be rank > 0"
        
        self.rank = rank
        self.world_size = world_size
        self.temperature = temperature
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        gpu_id = f"t4_gpu{rank}"
        self.profiling_enabled = is_profiling_enabled(gpu_id)
        
        # Initialize base server
        super().__init__(
            gpu_id=gpu_id,
            port=port,
            model_name=model_name,
        )
        
        logger.info(f"[Rank {rank}] Initializing DecodeWorker on port {port}")
    
    def initialize(self):
        """Load model and setup."""
        logger.info(f"[Rank {self.rank}] Loading model {self.model_name}")
        
        # Load model
        self.model, self.tokenizer = load_model(
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
        )
        
        # Get model config
        self.model_config = get_model_config(self.model)
        
        self.model_loaded = True
        
        logger.info(f"[Rank {self.rank}] Model loaded successfully")
        
        # Synchronize with other ranks
        synchronize_ranks()
    
    def _receive_kv_cache(self, request_id: str) -> tuple:
        """
        Receive KV cache broadcast from prefill worker.
        
        Args:
            request_id: Request ID for profiling
        
        Returns:
            Tuple of (past_key_values, None) - placeholder for actual implementation
        """
        # In actual implementation, this would:
        # 1. Signal prefill worker to start
        # 2. Allocate KV buffer
        # 3. Receive broadcast
        
        # For now, return None (will do full prefill+decode on each worker)
        return None
    
    async def _handle_generate(self, request: GenerationRequest) -> GenerationOutput:
        """
        Handle generation request.
        
        Args:
            request: Generation request
        
        Returns:
            Generation output
        """
        gpu_id = f"t4_gpu{self.rank}"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request.request_id,
            enabled=self.profiling_enabled
        ):
            # Format prompt
            prompt = format_prompt_for_model(
                request.task_prompt,
                self.model_name,
            )
            
            # Tokenize
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            
            # In full implementation, would receive KV cache here
            # For now, do full generation
            with profile_operation("generate", gpu_id, request.request_id, self.profiling_enabled):
                generated_ids, _ = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    past_key_values=None,
                    max_tokens=request.max_tokens,
                    temperature=self.temperature,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Extract answer (remove prompt)
            answer = generated_text[len(prompt):].strip()
            
            # Create output
            output = GenerationOutput(
                answer=answer,
                assumptions=[],
                confidence=0.8,
                risks=[],
                member_id=f"member_{self.rank}",
            )
        
        return output
    
    async def _handle_judge(self, request: JudgingRequest) -> JudgingOutput:
        """
        Handle judging request.
        
        Args:
            request: Judging request
        
        Returns:
            Judging output
        """
        gpu_id = f"t4_gpu{self.rank}"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request.request_id,
            enabled=self.profiling_enabled
        ):
            # Format judging prompt
            candidates_text = "\n\n".join([
                f"Candidate {cid}:\n{answer}"
                for cid, answer in request.candidates.items()
            ])
            
            prompt = format_prompt_for_model(
                f"Task: {request.task_prompt}\n\n"
                f"Candidates to evaluate:\n{candidates_text}\n\n"
                f"Rubric: {request.rubric}\n\n"
                f"Provide scores (0.0-1.0) and ranking for each candidate. "
                f"Return as JSON with fields: scores, ranking, top_reasoning.",
                self.model_name,
            )
            
            # Tokenize and generate
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encoded["input_ids"].to(self.device)
            
            with profile_operation("judge", gpu_id, request.request_id, self.profiling_enabled):
                generated_ids, _ = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    max_tokens=512,
                    temperature=0.3,  # Lower temp for judging
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
            
            # Parse output (simplified - would need robust JSON parsing)
            # For now, create mock judgment
            candidate_ids = list(request.candidates.keys())
            scores = {cid: 0.5 + (hash(cid) % 50) / 100.0 for cid in candidate_ids}
            ranking = sorted(candidate_ids, key=lambda x: scores[x], reverse=True)
            
            output = JudgingOutput(
                scores=scores,
                ranking=ranking,
                top_reasoning={
                    ranking[0]: "Best overall solution",
                    ranking[1]: "Good but missing details" if len(ranking) > 1 else "",
                    ranking[2]: "Adequate solution" if len(ranking) > 2 else "",
                },
                judge_id=f"member_{self.rank}",
            )
        
        return output


def main():
    """Main entry point for decode worker."""
    # Get configuration from environment
    rank = int(os.environ.get("RANK", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Load config
    import yaml
    with open("./config/models.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    with open("./config/endpoints.yaml", "r") as f:
        endpoint_config = yaml.safe_load(f)
    
    model_name = model_config["small_model"]["name"]
    
    # Get port and temperature for this rank
    member_idx = rank - 1  # members are 0-indexed
    member_config = endpoint_config["members"][member_idx]
    port = int(member_config["url"].split(":")[-1])
    temperature = member_config["temperature"]
    
    # Create worker
    worker = DecodeWorker(
        rank=rank,
        world_size=world_size,
        port=port,
        temperature=temperature,
        model_name=model_name,
        device=f"cuda:{rank}",
    )
    
    # Initialize
    worker.initialize()
    
    # Run server
    worker.run()


if __name__ == "__main__":
    main()

