"""
Tensor Parallelism utilities for distributed model serving.

This module provides utilities for manual tensor parallelism across multiple GPUs,
including weight sharding and distributed communication operations.
"""

import torch
import torch.distributed as dist
from typing import Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def shard_column(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Shard a weight tensor along the column (output) dimension for column-parallel layers.
    
    Used for layers like q_proj, k_proj, v_proj, gate_proj, up_proj where each rank
    computes a portion of the output features.
    
    Args:
        weight: Weight tensor to shard, shape [out_features, in_features]
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
        
    Returns:
        Sharded weight tensor, shape [out_features//world_size, in_features]
        
    Example:
        >>> weight = torch.randn(4096, 4096)  # Original weight
        >>> shard = shard_column(weight, rank=0, world_size=4)
        >>> shard.shape
        torch.Size([1024, 4096])
    """
    if len(weight.shape) < 2:
        raise ValueError(f"Weight must be at least 2D, got shape {weight.shape}")
    
    out_features = weight.shape[0]
    shard_size = out_features // world_size
    
    if out_features % world_size != 0:
        logger.warning(
            f"Output features {out_features} not evenly divisible by world_size {world_size}. "
            f"Last rank will have fewer parameters."
        )
    
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size if rank < world_size - 1 else out_features
    
    sharded = weight[start_idx:end_idx, ...].contiguous()
    
    logger.debug(
        f"Column shard: rank={rank}, out_features={out_features}, "
        f"shard_size={shard_size}, shard_shape={sharded.shape}"
    )
    
    return sharded


def shard_row(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Shard a weight tensor along the row (input) dimension for row-parallel layers.
    
    Used for layers like o_proj, down_proj where the input is already partitioned
    across ranks and we need to split the input dimension.
    
    Args:
        weight: Weight tensor to shard, shape [out_features, in_features]
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
        
    Returns:
        Sharded weight tensor, shape [out_features, in_features//world_size]
        
    Example:
        >>> weight = torch.randn(4096, 4096)  # Original weight
        >>> shard = shard_row(weight, rank=0, world_size=4)
        >>> shard.shape
        torch.Size([4096, 1024])
    """
    if len(weight.shape) < 2:
        raise ValueError(f"Weight must be at least 2D, got shape {weight.shape}")
    
    in_features = weight.shape[1]
    shard_size = in_features // world_size
    
    if in_features % world_size != 0:
        logger.warning(
            f"Input features {in_features} not evenly divisible by world_size {world_size}. "
            f"Last rank will have fewer parameters."
        )
    
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size if rank < world_size - 1 else in_features
    
    sharded = weight[:, start_idx:end_idx].contiguous()
    
    logger.debug(
        f"Row shard: rank={rank}, in_features={in_features}, "
        f"shard_size={shard_size}, shard_shape={sharded.shape}"
    )
    
    return sharded


def all_reduce_tensor(
    tensor: torch.Tensor,
    world_size: int,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[Any] = None,
    average: bool = True
) -> torch.Tensor:
    """
    Perform all-reduce operation on a tensor across all ranks.
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes
        op: Reduction operation (SUM, MAX, MIN, etc.)
        group: Process group (None for default)
        average: If True and op is SUM, divide result by world_size
        
    Returns:
        Reduced tensor (in-place operation)
    """
    if not dist.is_initialized():
        logger.warning("torch.distributed not initialized, skipping all-reduce")
        return tensor
    
    dist.all_reduce(tensor, op=op, group=group)
    
    if average and op == dist.ReduceOp.SUM:
        tensor.div_(world_size)
    
    return tensor


def all_reduce_logits(
    logits: torch.Tensor,
    world_size: int,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    All-reduce and average logits across ranks.
    
    This is typically called after row-parallel layers to combine partial results.
    
    Args:
        logits: Logits tensor, shape [batch, seq_len, vocab_size]
        world_size: Number of processes
        group: Process group (None for default)
        
    Returns:
        Averaged logits tensor (in-place operation)
    """
    return all_reduce_tensor(
        logits,
        world_size=world_size,
        op=dist.ReduceOp.SUM,
        group=group,
        average=True
    )


def validate_kv_cache(kv_cache: Tuple) -> bool:
    """
    Validate KV cache structure.
    
    Args:
        kv_cache: Tuple of (key, value) pairs for each layer
        
    Returns:
        True if valid, False otherwise
    """
    if kv_cache is None:
        return False
    
    if not isinstance(kv_cache, tuple):
        logger.error(f"KV cache must be tuple, got {type(kv_cache)}")
        return False
    
    if len(kv_cache) == 0:
        logger.error("KV cache is empty")
        return False
    
    # Check each layer
    for layer_idx, layer_cache in enumerate(kv_cache):
        if not isinstance(layer_cache, tuple) or len(layer_cache) < 2:
            logger.error(
                f"Layer {layer_idx} cache must be tuple of (key, value), "
                f"got {type(layer_cache)} with length {len(layer_cache) if isinstance(layer_cache, tuple) else 'N/A'}"
            )
            return False
        
        key_cache, value_cache = layer_cache[0], layer_cache[1]
        
        if not isinstance(key_cache, torch.Tensor) or not isinstance(value_cache, torch.Tensor):
            logger.error(f"Layer {layer_idx} key/value must be tensors")
            return False
        
        if key_cache.shape != value_cache.shape:
            logger.error(
                f"Layer {layer_idx} key and value shapes don't match: "
                f"{key_cache.shape} vs {value_cache.shape}"
            )
            return False
    
    return True


def estimate_kv_cache_size(kv_cache: Tuple, dtype: torch.dtype = torch.bfloat16) -> float:
    """
    Estimate memory size of KV cache in GB.
    
    Args:
        kv_cache: Tuple of (key, value) pairs for each layer OR DynamicCache object
        dtype: Data type of tensors (default: bfloat16)
        
    Returns:
        Estimated size in GB
    """
    if kv_cache is None:
        return 0.0
    
    # Handle DynamicCache objects
    if hasattr(kv_cache, 'get_seq_length'):
        # It's a Cache object (DynamicCache, etc.)
        try:
            seq_len = kv_cache.get_seq_length()
            if seq_len == 0:
                return 0.0
            
            # For DynamicCache, estimate based on first key tensor if available
            if hasattr(kv_cache, 'key_cache') and len(kv_cache.key_cache) > 0:
                first_key = kv_cache.key_cache[0]
                bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
                layer_size = first_key.numel() * bytes_per_element * 2  # key + value
                total_bytes = layer_size * len(kv_cache.key_cache)
                return total_bytes / (1024 ** 3)
        except:
            pass
        return 0.0  # Fallback
    
    # Handle tuple format
    if len(kv_cache) == 0:
        return 0.0
    
    total_bytes = 0
    num_layers = len(kv_cache)
    
    # Get size from first layer and multiply
    first_layer = kv_cache[0]
    if isinstance(first_layer, tuple) and len(first_layer) >= 2:
        key_cache, value_cache = first_layer[0], first_layer[1]
        
        # Calculate bytes per element
        bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        
        # Each layer has key and value
        layer_size = key_cache.numel() * bytes_per_element  # key
        layer_size += value_cache.numel() * bytes_per_element  # value
        
        total_bytes = layer_size * num_layers
    
    # Convert to GB
    size_gb = total_bytes / (1024 ** 3)
    
    return size_gb


def get_tp_group_info() -> dict:
    """
    Get information about the current tensor parallel group.
    
    Returns:
        Dictionary with rank, world_size, and initialization status
    """
    if not dist.is_initialized():
        return {
            'initialized': False,
            'rank': 0,
            'world_size': 1,
            'backend': None
        }
    
    return {
        'initialized': True,
        'rank': dist.get_rank(),
        'world_size': dist.get_world_size(),
        'backend': dist.get_backend()
    }


def shard_model_weights(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    column_parallel_layers: Tuple[str, ...] = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'),
    row_parallel_layers: Tuple[str, ...] = ('o_proj', 'down_proj'),
    verbose: bool = False
) -> int:
    """
    Shard model weights in-place for tensor parallelism.
    
    Args:
        model: Model to shard
        rank: Current process rank
        world_size: Total number of processes
        column_parallel_layers: Names of layers to shard column-wise
        row_parallel_layers: Names of layers to shard row-wise
        verbose: If True, log each sharded parameter
        
    Returns:
        Number of parameters sharded
    """
    num_sharded = 0
    
    for name, param in model.named_parameters():
        # Skip if not a weight parameter or less than 2D
        if 'weight' not in name or len(param.shape) < 2:
            continue
        
        # Check if this is a column-parallel layer
        is_column = any(layer_name in name for layer_name in column_parallel_layers)
        is_row = any(layer_name in name for layer_name in row_parallel_layers)
        
        if is_column:
            original_shape = param.data.shape
            param.data = shard_column(param.data, rank, world_size)
            num_sharded += 1
            
            if verbose:
                logger.info(
                    f"[Rank {rank}] Sharded column-parallel {name}: "
                    f"{original_shape} -> {param.data.shape}"
                )
        
        elif is_row:
            original_shape = param.data.shape
            param.data = shard_row(param.data, rank, world_size)
            num_sharded += 1
            
            if verbose:
                logger.info(
                    f"[Rank {rank}] Sharded row-parallel {name}: "
                    f"{original_shape} -> {param.data.shape}"
                )
    
    logger.info(f"[Rank {rank}] Sharded {num_sharded} parameters for TP")
    
    return num_sharded


def synchronize_ranks(group: Optional[Any] = None):
    """
    Synchronize all ranks (barrier).
    
    Args:
        group: Process group (None for default)
    """
    if dist.is_initialized():
        dist.barrier(group=group)

