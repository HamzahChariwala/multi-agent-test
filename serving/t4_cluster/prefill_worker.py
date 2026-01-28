"""Prefill worker for T4 cluster (Rank 0)."""

import os
import torch
import torch.distributed as dist
from queue import Queue
from threading import Thread
import logging

from serving.common.model_loader import load_model, get_model_config
from serving.common.inference import prefill
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled
from serving.t4_cluster.kv_transfer import broadcast_kv_cache, synchronize_ranks

logger = logging.getLogger(__name__)


class PrefillWorker:
    """
    Prefill worker that runs on Rank 0 of T4 cluster.
    Handles prefill computation and broadcasts KV cache to decode workers.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        model_name: str,
        device: str = "cuda:0",
        precision: str = "bf16",
    ):
        """
        Initialize prefill worker.
        
        Args:
            rank: Process rank (should be 0)
            world_size: Total number of processes
            model_name: Model to load
            device: Device to use
            precision: Model precision
        """
        assert rank == 0, "PrefillWorker must be rank 0"
        
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        self.request_queue = Queue()
        self.profiling_enabled = is_profiling_enabled("t4_gpu0")
        
        logger.info(f"[Rank {rank}] Initializing PrefillWorker")
    
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
        
        logger.info(f"[Rank {self.rank}] Model loaded successfully")
        
        # Synchronize with other ranks
        synchronize_ranks()
    
    def run_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        request_id: str,
    ) -> tuple:
        """
        Run prefill and broadcast KV cache.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            request_id: Request ID for profiling
        
        Returns:
            Tuple of (logits, past_key_values, seq_len)
        """
        logger.info(f"[Rank {self.rank}] Running prefill for request {request_id}")
        
        # Profile the prefill operation
        with ProfilerContext(
            gpu_id="t4_gpu0",
            request_id=request_id,
            enabled=self.profiling_enabled
        ):
            # Prefill
            with profile_operation("prefill", "t4_gpu0", request_id, self.profiling_enabled):
                logits, past_key_values = prefill(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            
            # Broadcast KV cache to decode workers
            with profile_operation("kv_broadcast", "t4_gpu0", request_id, self.profiling_enabled):
                broadcast_kv_cache(past_key_values, src_rank=0)
        
        seq_len = input_ids.shape[1]
        
        logger.info(
            f"[Rank {self.rank}] Prefill complete for request {request_id}, "
            f"seq_len={seq_len}"
        )
        
        return logits, past_key_values, seq_len
    
    def run(self):
        """Main worker loop."""
        logger.info(f"[Rank {self.rank}] Starting prefill worker loop")
        
        # In a real implementation, this would listen for prefill requests
        # from decode workers via IPC (shared memory, pipes, or message passing)
        # For now, this is a placeholder
        
        synchronize_ranks()
        
        logger.info(f"[Rank {self.rank}] Prefill worker ready")


def main():
    """Main entry point for prefill worker."""
    # Get rank and world size from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Load config
    import yaml
    with open("./config/models.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config["small_model"]["name"]
    
    # Create and run worker
    worker = PrefillWorker(
        rank=rank,
        world_size=world_size,
        model_name=model_name,
        device=f"cuda:{rank}",
    )
    
    worker.initialize()
    worker.run()


if __name__ == "__main__":
    main()

