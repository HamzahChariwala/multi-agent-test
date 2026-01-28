"""
Simple launcher for T4 cluster:
- Rank 0: HTTP server that does prefill and broadcasts to decode workers
- Ranks 1-3: Decode workers that wait for broadcasts
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch.multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_prefill_server(rank: int, model_name: str):
    """Run prefill server on rank 0."""
    from serving.t4_cluster.prefill_server import run_prefill_server as run_server
    run_server(rank, model_name)


def run_decode_worker(rank: int, model_name: str, temperature: float):
    """Run decode worker on ranks 1-3."""
    from serving.t4_cluster.simple_decode_worker import run_simple_decode_worker
    run_simple_decode_worker(rank, model_name, temperature)


def main():
    """Launch all workers."""
    # Load config
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    endpoint_path = Path(__file__).parent.parent.parent / "config" / "endpoints.yaml"
    with open(endpoint_path, "r") as f:
        endpoint_config = yaml.safe_load(f)
    
    model_name = model_config["small_model"]["name"]
    
    # Get temperatures for each member
    temperatures = [
        endpoint_config["members"][0]["temperature"],  # Rank 1
        endpoint_config["members"][1]["temperature"],  # Rank 2
        endpoint_config["members"][2]["temperature"],  # Rank 3
    ]
    
    logger.info("="  * 60)
    logger.info("Starting T4 Cluster with Simple Architecture")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Rank 0: Prefill server (HTTP on port 8000)")
    logger.info(f"Rank 1: Decode worker (temp={temperatures[0]})")
    logger.info(f"Rank 2: Decode worker (temp={temperatures[1]})")
    logger.info(f"Rank 3: Decode worker (temp={temperatures[2]})")
    logger.info("=" * 60)
    
    # Set environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "4"
    
    # Spawn processes
    mp.spawn(
        launch_worker,
        args=(model_name, temperatures),
        nprocs=4,
        join=True,
    )


def launch_worker(rank: int, model_name: str, temperatures: list):
    """Launch appropriate worker based on rank."""
    os.environ["RANK"] = str(rank)
    
    if rank == 0:
        # Prefill server with HTTP
        run_prefill_server(rank, model_name)
    else:
        # Decode worker
        temperature = temperatures[rank - 1]
        run_decode_worker(rank, model_name, temperature)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

