"""Launcher for T4 cluster multi-GPU setup."""

import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "29500"):
    """
    Setup distributed process group.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Initialize NCCL
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    
    logger.info(f"Rank {rank}/{world_size} initialized")


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_prefill_worker(rank: int, world_size: int):
    """
    Run prefill worker (rank 0).
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    try:
        # Import and run prefill worker
        from serving.t4_cluster.prefill_worker import PrefillWorker
        import yaml
        
        # Load config
        with open("./config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        model_name = config["small_model"]["name"]
        
        worker = PrefillWorker(
            rank=rank,
            world_size=world_size,
            model_name=model_name,
            device=f"cuda:{rank}",
        )
        
        worker.initialize()
        worker.run()
        
    except Exception as e:
        logger.error(f"Prefill worker error: {e}", exc_info=True)
        raise
    finally:
        cleanup_distributed()


def run_decode_worker(rank: int, world_size: int):
    """
    Run decode worker (ranks 1-3).
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    try:
        # Import and run decode worker
        from serving.t4_cluster.decode_worker import DecodeWorker
        import yaml
        
        # Load config
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
        
        worker = DecodeWorker(
            rank=rank,
            world_size=world_size,
            port=port,
            temperature=temperature,
            model_name=model_name,
            device=f"cuda:{rank}",
        )
        
        worker.initialize()
        worker.run()
        
    except Exception as e:
        logger.error(f"Decode worker error: {e}", exc_info=True)
        raise
    finally:
        cleanup_distributed()


def worker_process(rank: int, world_size: int):
    """
    Worker process entry point.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    if rank == 0:
        run_prefill_worker(rank, world_size)
    else:
        run_decode_worker(rank, world_size)


def main():
    """Main entry point for T4 cluster launcher."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    world_size = 4  # 1 prefill + 3 decode workers
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)
    
    if torch.cuda.device_count() < world_size:
        logger.error(f"Need {world_size} GPUs, but only {torch.cuda.device_count()} available")
        sys.exit(1)
    
    logger.info(f"Launching T4 cluster with {world_size} workers")
    
    # Setup NCCL environment
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "INFO")
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable P2P
    
    # Spawn processes
    mp.spawn(
        worker_process,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
    
    logger.info("T4 cluster shutdown complete")


if __name__ == "__main__":
    main()

