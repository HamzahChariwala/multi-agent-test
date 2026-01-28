"""Point-to-point KV cache forking utilities for A100 cluster."""

import torch
import torch.distributed as dist
from typing import Any, Tuple


def send_kv_cache(kv_cache: Any, dst_rank: int, tag: int = 0, group: Any = None):
    """
    Send KV cache to destination rank using point-to-point communication.
    
    Args:
        kv_cache: KV cache to send
        dst_rank: Destination rank
        tag: Message tag for matching send/recv
        group: Process group (None for default)
    """
    if kv_cache is None:
        return
    
    # Send number of layers first
    num_layers = torch.tensor([len(kv_cache)], dtype=torch.int64, device="cuda")
    dist.send(num_layers, dst=dst_rank, tag=tag, group=group)
    
    # Send each layer's KV cache
    for layer_idx, layer_cache in enumerate(kv_cache):
        if layer_cache is None:
            continue
        
        key_cache, value_cache = layer_cache
        
        # Send key cache
        dist.send(key_cache, dst=dst_rank, tag=tag + layer_idx * 2 + 1, group=group)
        
        # Send value cache
        dist.send(value_cache, dst=dst_rank, tag=tag + layer_idx * 2 + 2, group=group)


def recv_kv_cache(
    src_rank: int,
    model_config: dict,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tag: int = 0,
    group: Any = None,
) -> Any:
    """
    Receive KV cache from source rank.
    
    Args:
        src_rank: Source rank
        model_config: Model configuration for buffer allocation
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to receive on
        dtype: Data type
        tag: Message tag for matching send/recv
        group: Process group
    
    Returns:
        Received KV cache
    """
    # Receive number of layers
    num_layers_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    dist.recv(num_layers_tensor, src=src_rank, tag=tag, group=group)
    num_layers = int(num_layers_tensor.item())
    
    # Allocate and receive KV cache
    num_heads = model_config.get("num_attention_heads", 32)
    hidden_size = model_config.get("hidden_size", 4096)
    head_dim = hidden_size // num_heads
    
    kv_cache = []
    
    for layer_idx in range(num_layers):
        # Allocate key cache buffer
        key_cache = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Allocate value cache buffer
        value_cache = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Receive key cache
        dist.recv(key_cache, src=src_rank, tag=tag + layer_idx * 2 + 1, group=group)
        
        # Receive value cache
        dist.recv(value_cache, src=src_rank, tag=tag + layer_idx * 2 + 2, group=group)
        
        kv_cache.append((key_cache, value_cache))
    
    return tuple(kv_cache)


def synchronize_pair(src_rank: int, dst_rank: int, group: Any = None):
    """
    Synchronize a pair of ranks before KV transfer.
    
    Args:
        src_rank: Source rank
        dst_rank: Destination rank
        group: Process group
    """
    if dist.is_initialized():
        # Create sync tensor
        sync_tensor = torch.tensor([1.0], device="cuda")
        
        current_rank = dist.get_rank(group)
        
        if current_rank == src_rank:
            dist.send(sync_tensor, dst=dst_rank, group=group)
        elif current_rank == dst_rank:
            dist.recv(sync_tensor, src=src_rank, group=group)


def estimate_transfer_time(kv_cache: Any, bandwidth_gbps: float = 100.0) -> float:
    """
    Estimate KV cache transfer time.
    
    Args:
        kv_cache: KV cache to transfer
        bandwidth_gbps: Estimated bandwidth in GB/s (NVLink ~100-200 GB/s)
    
    Returns:
        Estimated transfer time in seconds
    """
    if kv_cache is None:
        return 0.0
    
    # Calculate total size in bytes
    total_bytes = 0
    for layer_cache in kv_cache:
        if layer_cache is None:
            continue
        
        key_cache, value_cache = layer_cache
        total_bytes += key_cache.numel() * key_cache.element_size()
        total_bytes += value_cache.numel() * value_cache.element_size()
    
    # Convert to GB
    total_gb = total_bytes / (1024 ** 3)
    
    # Estimate time
    transfer_time = total_gb / bandwidth_gbps
    
    return transfer_time


def verify_kv_match(kv_cache_1: Any, kv_cache_2: Any, tolerance: float = 1e-5) -> bool:
    """
    Verify that two KV caches match (for debugging).
    
    Args:
        kv_cache_1: First KV cache
        kv_cache_2: Second KV cache
        tolerance: Tolerance for floating point comparison
    
    Returns:
        True if caches match within tolerance
    """
    if kv_cache_1 is None or kv_cache_2 is None:
        return kv_cache_1 == kv_cache_2
    
    if len(kv_cache_1) != len(kv_cache_2):
        return False
    
    for layer1, layer2 in zip(kv_cache_1, kv_cache_2):
        if layer1 is None or layer2 is None:
            if layer1 != layer2:
                return False
            continue
        
        key1, value1 = layer1
        key2, value2 = layer2
        
        # Check shapes
        if key1.shape != key2.shape or value1.shape != value2.shape:
            return False
        
        # Check values
        if not torch.allclose(key1, key2, rtol=tolerance, atol=tolerance):
            return False
        
        if not torch.allclose(value1, value2, rtol=tolerance, atol=tolerance):
            return False
    
    return True

