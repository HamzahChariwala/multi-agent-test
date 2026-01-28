"""Chairman model with Tensor Parallelism on GPUs 2-3."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from typing import Optional, List, Dict

from serving.common.model_loader import load_model
from serving.common.inference import generate, format_prompt_for_model
from serving.common.http_server import ChairmanServer
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled

from schemas.chairman import ChairmanRequest, ChairmanOutput

logger = logging.getLogger(__name__)


class ChairmanWorker(ChairmanServer):
    """
    Chairman model worker with Tensor Parallelism.
    Runs across GPU 2-3 of A100 cluster.
    """
    
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        port: int,
        model_name: str,
        device: str,
        precision: str = "bf16",
    ):
        """
        Initialize chairman worker.
        
        Args:
            tp_rank: Tensor parallel rank (0 or 1)
            tp_size: Tensor parallel size (should be 2)
            port: HTTP port (only rank 0 serves HTTP)
            model_name: Model to load
            device: CUDA device
            precision: Model precision
        """
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        self.model = None
        self.tokenizer = None
        
        self.profiling_enabled = is_profiling_enabled("chairman_tp")
        
        # Only rank 0 serves HTTP
        if tp_rank == 0:
            super().__init__(port=port, model_name=model_name)
        else:
            self.port = port
            self.app = None
        
        logger.info(f"[Chairman TP-{tp_rank}] Initializing on {device}")
    
    def initialize(self):
        """Load model with TP."""
        logger.info(f"[Chairman TP-{self.tp_rank}] Loading model {self.model_name}")
        
        # Load model with TP
        self.model, self.tokenizer = load_model(
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        
        if self.tp_rank == 0:
            self.model_loaded = True
        
        logger.info(f"[Chairman TP-{self.tp_rank}] Model loaded successfully")
        
        # Synchronize TP ranks
        if dist.is_initialized():
            dist.barrier()
    
    async def _handle_synthesize(self, request: ChairmanRequest) -> ChairmanOutput:
        """
        Handle synthesis request.
        
        Args:
            request: Chairman request
        
        Returns:
            Chairman output
        """
        with ProfilerContext(
            gpu_id="chairman_tp",
            request_id=request.request_id,
            enabled=self.profiling_enabled
        ):
            # Format synthesis prompt
            candidates_text = "\n\n".join([
                f"Candidate {cid}:\n{answer}"
                for cid, answer in request.candidates.items()
            ])
            
            judgments_text = "\n\n".join([
                f"Judge {i+1} Rankings: {', '.join(j.get('ranking', []))}"
                for i, j in enumerate(request.judgments)
            ])
            
            prompt = format_prompt_for_model(
                f"Original Task: {request.task_prompt}\n\n"
                f"Candidate Answers:\n{candidates_text}\n\n"
                f"Council Judgments:\n{judgments_text}\n\n"
                f"As the chairman, synthesize the best final answer by combining "
                f"insights from the top-ranked candidates. Explain your decision process.",
                self.model_name,
            )
            
            # Tokenize (only on rank 0 for simplicity)
            if self.tp_rank == 0:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                )
                input_ids = encoded["input_ids"].to(self.device)
            else:
                # For TP, all ranks need input
                # In production, broadcast from rank 0
                input_ids = None
            
            # Generate with TP
            with profile_operation("synthesis", "chairman_tp", request.request_id, self.profiling_enabled):
                if self.tp_rank == 0:
                    generated_ids, _ = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_ids=input_ids,
                        max_tokens=1024,
                        temperature=0.7,
                    )
                else:
                    # Secondary TP rank participates in computation
                    # but doesn't return results
                    # In real TP implementation, this would be part of distributed forward pass
                    pass
            
            # Only rank 0 decodes and returns
            if self.tp_rank == 0:
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                # Extract answer
                final_answer = generated_text[len(prompt):].strip()
                
                # Determine which candidates were selected
                # (simplified - would parse from generated text in production)
                all_rankings = [j.get("ranking", []) for j in request.judgments]
                top_candidates = []
                for ranking in all_rankings:
                    if ranking:
                        top_candidates.append(ranking[0])
                
                # Get unique top candidates
                selected_candidates = list(set(top_candidates[:3]))
                
                # Create output
                output = ChairmanOutput(
                    final_answer=final_answer,
                    decision_trace=(
                        f"Analyzed {len(request.candidates)} candidates and "
                        f"{len(request.judgments)} judgments. Selected top-ranked "
                        f"candidates for synthesis: {', '.join(selected_candidates)}"
                    ),
                    selected_candidate_ids=selected_candidates,
                    confidence=0.90,
                )
                
                return output
            
            # Non-rank-0 shouldn't reach here in normal flow
            return None
    
    def run_worker(self):
        """Run worker (rank 0 runs HTTP server, others participate in TP)."""
        if self.tp_rank == 0:
            # Rank 0: run HTTP server
            self.run()
        else:
            # Other ranks: wait for work (would participate in TP operations)
            logger.info(f"[Chairman TP-{self.tp_rank}] Ready for TP operations")
            # In real implementation, would listen for TP work
            # For now, just barrier to keep alive
            while dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    break


def setup_distributed(tp_rank: int, tp_size: int):
    """Setup distributed for TP."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29502")
    os.environ["RANK"] = str(tp_rank)
    os.environ["WORLD_SIZE"] = str(tp_size)
    
    dist.init_process_group(backend="nccl")


def worker_process(tp_rank: int, tp_size: int):
    """Worker process entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Chairman-TP{tp_rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    # GPU mapping: chairman uses GPUs 2-3
    gpu_id = 2 + tp_rank
    torch.cuda.set_device(gpu_id)
    
    # Setup distributed
    setup_distributed(tp_rank, tp_size)
    
    # Load config
    import yaml
    with open("./config/models.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    with open("./config/endpoints.yaml", "r") as f:
        endpoint_config = yaml.safe_load(f)
    
    model_name = model_config["chairman_model"]["name"]
    port = int(endpoint_config["chairman"]["url"].split(":")[-1])
    
    # Create worker
    worker = ChairmanWorker(
        tp_rank=tp_rank,
        tp_size=tp_size,
        port=port,
        model_name=model_name,
        device=f"cuda:{gpu_id}",
    )
    
    # Initialize
    worker.initialize()
    
    # Run
    worker.run_worker()


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    tp_size = 2  # 2 GPUs for tensor parallelism
    
    logger.info(f"Launching Chairman with TP={tp_size}")
    
    # Spawn processes
    mp.spawn(
        worker_process,
        args=(tp_size,),
        nprocs=tp_size,
        join=True,
    )


if __name__ == "__main__":
    main()

