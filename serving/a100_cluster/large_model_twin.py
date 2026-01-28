"""Large model twin setup for A100 cluster (Members 4 & 5)."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from typing import Optional

from serving.common.model_loader import load_model, get_model_config
from serving.common.inference import generate, format_prompt_for_model
from serving.common.http_server import BaseModelServer
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled
from serving.a100_cluster.kv_fork import send_kv_cache, recv_kv_cache, synchronize_pair

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput

logger = logging.getLogger(__name__)


class LargeModelWorker(BaseModelServer):
    """
    Large model worker that can fork KV cache to another worker.
    Used for Members 4 and 5 on A100 cluster.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        port: int,
        temperature: float,
        model_name: str,
        device: str,
        is_primary: bool = True,
        precision: str = "bf16",
    ):
        """
        Initialize large model worker.
        
        Args:
            rank: Process rank (0 or 1 in this 2-GPU setup)
            world_size: Total processes (should be 2)
            port: HTTP port
            temperature: Sampling temperature
            model_name: Model to load
            device: CUDA device
            is_primary: If True, this worker prefills and forks KV
            precision: Model precision
        """
        self.rank = rank
        self.world_size = world_size
        self.temperature = temperature
        self.model_name = model_name
        self.device = device
        self.is_primary = is_primary
        self.precision = precision
        
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        gpu_id = f"a100_gpu{rank}"
        self.profiling_enabled = is_profiling_enabled(gpu_id)
        
        # Initialize base server
        super().__init__(
            gpu_id=gpu_id,
            port=port,
            model_name=model_name,
        )
        
        logger.info(
            f"[Rank {rank}] Initializing LargeModelWorker "
            f"({'primary' if is_primary else 'secondary'}) on port {port}"
        )
    
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
        
        # Synchronize with pair
        if dist.is_initialized():
            dist.barrier()
    
    async def _handle_generate(self, request: GenerationRequest) -> GenerationOutput:
        """Handle generation request."""
        gpu_id = f"a100_gpu{self.rank}"
        
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
            
            if self.is_primary:
                # Primary: do prefill and fork KV
                with profile_operation("prefill", gpu_id, request.request_id, self.profiling_enabled):
                    from serving.common.inference import prefill
                    logits, past_key_values = prefill(
                        model=self.model,
                        input_ids=input_ids,
                        use_cache=True,
                    )
                
                # Fork KV to secondary
                if dist.is_initialized() and self.world_size > 1:
                    with profile_operation("kv_fork", gpu_id, request.request_id, self.profiling_enabled):
                        send_kv_cache(past_key_values, dst_rank=1, tag=hash(request.request_id) % 1000)
                
                # Generate with own temperature
                with profile_operation("decode", gpu_id, request.request_id, self.profiling_enabled):
                    generated_ids, _ = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        max_tokens=request.max_tokens,
                        temperature=self.temperature,
                    )
            
            else:
                # Secondary: receive KV and decode
                past_key_values = None
                if dist.is_initialized():
                    with profile_operation("kv_receive", gpu_id, request.request_id, self.profiling_enabled):
                        seq_len = input_ids.shape[1]
                        past_key_values = recv_kv_cache(
                            src_rank=0,
                            model_config=self.model_config,
                            batch_size=1,
                            seq_len=seq_len,
                            device=self.device,
                            dtype=self.model.dtype,
                            tag=hash(request.request_id) % 1000,
                        )
                
                # Generate with own temperature
                with profile_operation("decode", gpu_id, request.request_id, self.profiling_enabled):
                    generated_ids, _ = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        max_tokens=request.max_tokens,
                        temperature=self.temperature,
                    )
            
            # Decode text
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Extract answer
            answer = generated_text[len(prompt):].strip()
            
            # Create output
            member_id = f"member_{4 if self.is_primary else 5}"
            output = GenerationOutput(
                answer=answer,
                assumptions=[],
                confidence=0.85,
                risks=[],
                member_id=member_id,
            )
        
        return output
    
    async def _handle_judge(self, request: JudgingRequest) -> JudgingOutput:
        """Handle judging request."""
        gpu_id = f"a100_gpu{self.rank}"
        
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
                f"Provide scores and ranking. Return as JSON.",
                self.model_name,
            )
            
            # Tokenize and generate
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encoded["input_ids"].to(self.device)
            
            generated_ids, _ = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                max_tokens=512,
                temperature=0.3,
            )
            
            # Create mock judgment (simplified)
            candidate_ids = list(request.candidates.keys())
            scores = {cid: 0.5 + (hash(cid + str(self.rank)) % 50) / 100.0 for cid in candidate_ids}
            ranking = sorted(candidate_ids, key=lambda x: scores[x], reverse=True)
            
            member_id = f"member_{4 if self.is_primary else 5}"
            output = JudgingOutput(
                scores=scores,
                ranking=ranking,
                top_reasoning={
                    ranking[0]: "Top choice based on criteria",
                    ranking[1]: "Strong alternative" if len(ranking) > 1 else "",
                    ranking[2]: "Acceptable solution" if len(ranking) > 2 else "",
                },
                judge_id=member_id,
            )
        
        return output


def setup_distributed(rank: int, world_size: int):
    """Setup distributed communication."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29501")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    dist.init_process_group(backend="nccl")


def worker_process(rank: int, world_size: int):
    """Worker process entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[A100-{rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Load config
    import yaml
    with open("./config/models.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    with open("./config/endpoints.yaml", "r") as f:
        endpoint_config = yaml.safe_load(f)
    
    model_name = model_config["large_model"]["name"]
    
    # Member 4 (rank 0) or Member 5 (rank 1)
    member_idx = 3 + rank  # Members 0-2 are T4, 3-4 are A100
    member_config = endpoint_config["members"][member_idx]
    port = int(member_config["url"].split(":")[-1])
    temperature = member_config["temperature"]
    
    # Create worker
    worker = LargeModelWorker(
        rank=rank,
        world_size=world_size,
        port=port,
        temperature=temperature,
        model_name=model_name,
        device=f"cuda:{rank}",
        is_primary=(rank == 0),
    )
    
    # Initialize
    worker.initialize()
    
    # Run server
    worker.run()


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    world_size = 2  # 2 GPUs for large model twin
    
    # Spawn processes
    mp.spawn(
        worker_process,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

