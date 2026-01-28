"""NCCL-based KV cache transfer utilities for T4 cluster."""

import torch
import torch.distributed as dist
from typing import Any, Tuple, List


def broadcast_kv_cache(
    kv_cache: Any,
    src_rank: int = 0,
    group: Any = None
) -> Any:
    """
    Broadcast KV cache from source rank to all ranks in group.
    
    Args:
        kv_cache: KV cache to broadcast (tuple of tensors)
        src_rank: Source rank (typically 0 for prefill worker)
        group: Process group (None for default)
    
    Returns:
        Broadcasted KV cache
    """
    if kv_cache is None:
        return None
    
    # KV cache is typically a tuple of layer caches
    # Each layer cache is a tuple of (key_cache, value_cache)
    broadcasted_cache = []
    
    for layer_idx, layer_cache in enumerate(kv_cache):
        if layer_cache is None:
            broadcasted_cache.append(None)
            continue
        
        # Each layer has (key, value) tuple
        key_cache, value_cache = layer_cache
        
        # Broadcast key cache
        dist.broadcast(key_cache, src=src_rank, group=group)
        
        # Broadcast value cache
        dist.broadcast(value_cache, src=src_rank, group=group)
        
        broadcasted_cache.append((key_cache, value_cache))
    
    return tuple(broadcasted_cache)


def allocate_kv_buffer(
    model_config: dict,
    batch_size: int,
    max_seq_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-allocate KV cache buffer for receiving broadcasts.
    
    Args:
        model_config: Model configuration dict with layer info
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        device: Device to allocate on
        dtype: Data type
    
    Returns:
        Pre-allocated KV cache structure
    """
    num_layers = model_config.get("num_layers", 32)
    num_heads = model_config.get("num_attention_heads", 32)
    hidden_size = model_config.get("hidden_size", 4096)
    head_dim = hidden_size // num_heads
    
    kv_buffer = []
    
    for _ in range(num_layers):
        # Allocate key cache [batch_size, num_heads, seq_len, head_dim]
        key_cache = torch.empty(
            (batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Allocate value cache [batch_size, num_heads, seq_len, head_dim]
        value_cache = torch.empty(
            (batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        kv_buffer.append((key_cache, value_cache))
    
    return kv_buffer


def verify_kv_integrity(kv_cache: Any) -> bool:
    """
    Verify KV cache integrity after transfer.
    
    Args:
        kv_cache: KV cache to verify
    
    Returns:
        True if valid, False otherwise
    """
    if kv_cache is None:
        return False
    
    try:
        for layer_cache in kv_cache:
            if layer_cache is None:
                continue
            
            key_cache, value_cache = layer_cache
            
            # Check for NaN or Inf
            if torch.isnan(key_cache).any() or torch.isinf(key_cache).any():
                return False
            
            if torch.isnan(value_cache).any() or torch.isinf(value_cache).any():
                return False
            
            # Check shapes match
            if key_cache.shape != value_cache.shape:
                return False
        
        return True
    
    except Exception:
        return False


def get_kv_cache_size(kv_cache: Any) -> int:
    """
    Calculate total size of KV cache in bytes.
    
    Args:
        kv_cache: KV cache structure
    
    Returns:
        Size in bytes
    """
    if kv_cache is None:
        return 0
    
    total_size = 0
    
    for layer_cache in kv_cache:
        if layer_cache is None:
            continue
        
        key_cache, value_cache = layer_cache
        total_size += key_cache.numel() * key_cache.element_size()
        total_size += value_cache.numel() * value_cache.element_size()
    
    return total_size


def trim_kv_cache(kv_cache: Any, actual_seq_len: int) -> Any:
    """
    Trim KV cache to actual sequence length (remove padding).
    
    Args:
        kv_cache: KV cache with padding
        actual_seq_len: Actual sequence length to keep
    
    Returns:
        Trimmed KV cache
    """
    if kv_cache is None:
        return None
    
    trimmed_cache = []
    
    for layer_cache in kv_cache:
        if layer_cache is None:
            trimmed_cache.append(None)
            continue
        
        key_cache, value_cache = layer_cache
        
        # Trim sequence dimension (typically dim=2)
        trimmed_key = key_cache[:, :, :actual_seq_len, :]
        trimmed_value = value_cache[:, :, :actual_seq_len, :]
        
        trimmed_cache.append((trimmed_key, trimmed_value))
    
    return tuple(trimmed_cache)


def synchronize_ranks(group: Any = None):
    """
    Synchronize all ranks in the group.
    
    Args:
        group: Process group (None for default)
    """
    if dist.is_initialized():
        dist.barrier(group=group)

