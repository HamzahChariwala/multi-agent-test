"""
Simple decode worker that waits for broadcasts from rank 0 and does decode.
No HTTP server - just a loop waiting for work.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.distributed as dist
import logging
from typing import Tuple
from transformers import DynamicCache

from serving.common.model_loader import load_model
from serving.common.inference import generate
from serving.common.profiling import ProfilerContext, profile_operation, is_profiling_enabled
from serving.t4_cluster.kv_transfer import synchronize_ranks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDecodeWorker:
    """Decode worker that waits for broadcasts and processes decode."""
    
    def __init__(self, rank: int, model_name: str, temperature: float, device: str, precision: str = "bf16"):
        self.rank = rank
        self.model_name = model_name
        self.temperature = temperature
        self.device = device
        self.precision = precision
        self.world_size = 4
        
        self.model = None
        self.tokenizer = None
        
        self.profiling_enabled = is_profiling_enabled(f"t4_gpu{rank}")
    
    def initialize(self):
        """Load model and initialize distributed."""
        logger.info(f"[Rank {self.rank}] Initializing decode worker")
        
        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Load model
        gpu_id = f"t4_gpu{self.rank}"
        with profile_operation("init_model_load", gpu_id, "init", self.profiling_enabled):
            self.model, self.tokenizer = load_model(
                model_name=self.model_name,
                device=self.device,
                precision=self.precision,
            )
        
        logger.info(f"[Rank {self.rank}] Model loaded on {self.device}")
        
        # Synchronize
        synchronize_ranks()
        logger.info(f"[Rank {self.rank}] Synchronized with all workers")
    
    def receive_kv_cache(self, num_layers: int, key_shape: Tuple, value_shape: Tuple):
        """Receive KV cache broadcast from rank 0 and return DynamicCache."""
        # Use the model's actual dtype (usually bfloat16 or float16)
        dtype = next(self.model.parameters()).dtype
        
        # Build DynamicCache object
        cache = DynamicCache()
        
        for layer_idx in range(num_layers):
            # Allocate buffers with correct dtype
            key = torch.zeros(key_shape, dtype=dtype, device=self.device)
            value = torch.zeros(value_shape, dtype=dtype, device=self.device)
            
            # Receive broadcasts
            dist.broadcast(key, src=0)
            dist.broadcast(value, src=0)
            
            # Update cache
            cache.update(key, value, layer_idx)
        
        return cache
    
    def run(self):
        """Main loop - wait for work and process."""
        logger.info(f"[Rank {self.rank}] Decode worker ready - listening for broadcasts")
        
        request_count = 0
        
        try:
            while True:
                # Wait for metadata broadcast from rank 0
                # [has_work, batch_size, seq_len, max_tokens]
                metadata = torch.zeros(4, dtype=torch.long, device=self.device)
                dist.broadcast(metadata, src=0)
                
                has_work = metadata[0].item()
                
                if has_work == 0:
                    # Keep-alive ping, no work
                    continue
                
                batch_size = metadata[1].item()
                seq_len = metadata[2].item()
                max_tokens = metadata[3].item()
                
                request_count += 1
                request_id = f"rank{self.rank}_req{request_count}"
                
                logger.info(
                    f"[Rank {self.rank}] Received work: "
                    f"batch_size={batch_size}, seq_len={seq_len}, max_tokens={max_tokens}"
                )
                
                gpu_id = f"t4_gpu{self.rank}"
                
                with ProfilerContext(
                    gpu_id=gpu_id,
                    request_id=request_id,
                    enabled=self.profiling_enabled
                ):
                    # Receive input_ids
                    with profile_operation("receive_inputs", gpu_id, request_id, self.profiling_enabled):
                        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
                        dist.broadcast(input_ids, src=0)
                    
                    # Receive KV cache
                    with profile_operation("receive_kv", gpu_id, request_id, self.profiling_enabled):
                        # Get model config to know number of layers and shapes
                        num_layers = self.model.config.num_hidden_layers
                        num_heads = self.model.config.num_attention_heads
                        head_dim = self.model.config.hidden_size // num_heads
                        
                        key_shape = (batch_size, num_heads, seq_len, head_dim)
                        value_shape = (batch_size, num_heads, seq_len, head_dim)
                        
                        past_key_values = self.receive_kv_cache(num_layers, key_shape, value_shape)
                    
                    logger.info(f"[Rank {self.rank}] Received KV cache ({num_layers} layers)")
                    
                    # Do decode
                    with profile_operation("decode_only", gpu_id, request_id, self.profiling_enabled):
                        generated_ids, _ = generate(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                        )
                        
                        generated_text = self.tokenizer.decode(
                            generated_ids[0],
                            skip_special_tokens=True,
                        )
                    
                    logger.info(f"[Rank {self.rank}] Completed decode #{request_count}")
                    logger.info(f"[Rank {self.rank}] Generated: {generated_text[:200]}...")
        
        except KeyboardInterrupt:
            logger.info(f"[Rank {self.rank}] Processed {request_count} requests, shutting down")


def run_simple_decode_worker(rank: int, model_name: str, temperature: float):
    """Entry point for spawned process."""
    # Fix import paths
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "4"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    worker = SimpleDecodeWorker(
        rank=rank,
        model_name=model_name,
        temperature=temperature,
        device=f"cuda:{rank}",
    )
    worker.initialize()
    worker.run()


if __name__ == "__main__":
    import sys
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_simple_decode_worker(rank, "microsoft/phi-2", 0.7)

