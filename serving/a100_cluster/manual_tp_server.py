"""
Manual Tensor Parallelism with tensor_parallel library

4 processes, one per GPU, with PROPER tensor parallelism.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import logging
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import tensor_parallel as tp

from serving.common.profiling import ProfilerContext, profile_operation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class SynthesisRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None


class SynthesisResponse(BaseModel):
    status: str
    request_id: str
    text: str
    num_tokens: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    rank: int
    world_size: int
    memory_allocated_gb: float


class TPWorker:
    """Worker with tensor_parallel library."""
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        model_name: str,
        precision: str,
        port: int,
        profiling_enabled: bool
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.precision = precision
        self.port = port
        self.profiling_enabled = profiling_enabled
        
        self.device = f"cuda:{rank}"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"[Rank {rank}] Initializing TP worker with tensor_parallel")
    
    def initialize(self):
        """Initialize with tensor_parallel."""
        # Init distributed
        dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)
        
        logger.info(f"[Rank {self.rank}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with tensor_parallel
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        logger.info(f"[Rank {self.rank}] Loading model on CPU...")
        
        # Load on CPU first
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info(f"[Rank {self.rank}] Wrapping with tensor_parallel...")
        # Wrap with tensor_parallel - it will handle the sharding
        # When distributed=True, each rank specifies only its own device
        self.model = tp.tensor_parallel(
            model,
            device_ids=[self.device],  # Only this rank's device
            distributed=True,  # Use torch.distributed
        )
        
        self.model.eval()
        
        # Sync
        dist.barrier()
        logger.info(f"[Rank {self.rank}] Model loaded with tensor_parallel!")
        
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(self.rank) / 1024**3
            logger.info(f"[Rank {self.rank}] GPU memory: {mem:.2f} GB")
    
    def synthesize(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        metadata: Optional[Dict] = None
    ) -> Optional[SynthesisResponse]:
        """Generate with TP."""
        start_time = time.time()
        gpu_id = f"a100_gpu{self.rank}"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request_id,
            enabled=self.profiling_enabled,
            split_by_gpu=False,  # Already separate!
            num_gpus=self.world_size
        ):
            # Tokenize
            with profile_operation("tokenize", gpu_id, request_id, self.profiling_enabled):
                encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
            
            if self.rank == 0:
                logger.info(f"[{request_id}] Input tokens: {input_ids.shape[1]}")
            
            # Generate with TP
            with profile_operation("generate", gpu_id, request_id, self.profiling_enabled):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            
            # Decode
            with profile_operation("decode", gpu_id, request_id, self.profiling_enabled):
                text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            num_generated = outputs.shape[1] - input_ids.shape[1]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if self.rank == 0:
            logger.info(f"[{request_id}] Generated {num_generated} tokens in {elapsed_ms:.0f}ms")
            return SynthesisResponse(
                status="success",
                request_id=request_id,
                text=text,
                num_tokens=num_generated,
                total_time_ms=elapsed_ms
            )
        return None
    
    def get_health(self) -> HealthResponse:
        mem = torch.cuda.memory_allocated(self.rank) / 1024**3 if torch.cuda.is_available() else 0
        return HealthResponse(
            status="healthy",
            model=self.model_name,
            rank=self.rank,
            world_size=self.world_size,
            memory_allocated_gb=mem
        )


def broadcast_work(worker: TPWorker, request: Optional[SynthesisRequest] = None) -> Optional[Dict]:
    """Broadcast work from rank 0 to all ranks."""
    if worker.rank == 0:
        # Rank 0: prepare and broadcast work
        if request is None:
            work = [None]  # Shutdown signal
        else:
            work = [{
                "request_id": request.request_id,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "metadata": request.metadata
            }]
        dist.broadcast_object_list(work, src=0)
        return work[0]
    else:
        # Other ranks: receive work
        work = [None]
        dist.broadcast_object_list(work, src=0)
        return work[0]


def create_app(worker: TPWorker) -> FastAPI:
    """FastAPI app (rank 0 only)."""
    app = FastAPI(title="Manual TP Server")
    
    @app.get("/health")
    async def health():
        return worker.get_health()
    
    @app.post("/synthesize")
    async def synthesize(request: SynthesisRequest):
        try:
            # Broadcast work to all ranks
            broadcast_work(worker, request)
            
            # All ranks execute (but only rank 0 returns response)
            response = worker.synthesize(
                request_id=request.request_id,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                metadata=request.metadata
            )
            if response is None:
                raise HTTPException(500, "Non-rank-0 received request")
            return response
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise HTTPException(500, str(e))
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--profiling-enabled", action="store_true")
    args = parser.parse_args()
    
    # Get rank from env (set by torchrun)
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    profiling = args.profiling_enabled or os.getenv("ENABLE_PROFILING", "").lower() == "true"
    
    if rank == 0:
        logger.info("="*80)
        logger.info("Tensor Parallel Server - WITH tensor_parallel library")
        logger.info("="*80)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Profiling: {profiling}")
        logger.info("="*80)
    
    worker = TPWorker(rank, world_size, args.model_name, args.precision, args.port, profiling)
    worker.initialize()
    
    if rank == 0:
        # Rank 0: run FastAPI server
        app = create_app(worker)
        import threading
        
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info(f"[Rank 0] FastAPI server started on port {args.port}")
    
    # All ranks: listen for work
    logger.info(f"[Rank {rank}] Listening for work...")
    while True:
        if rank != 0:
            # Non-rank-0: wait for broadcast
            work = broadcast_work(worker, None)
            if work is None:
                break  # Shutdown
            
            # Execute the work
            worker.synthesize(
                request_id=work["request_id"],
                prompt=work["prompt"],
                max_tokens=work["max_tokens"],
                temperature=work["temperature"],
                metadata=work.get("metadata")
            )
        else:
            # Rank 0: just keep the thread alive
            time.sleep(0.1)


if __name__ == "__main__":
    main()
