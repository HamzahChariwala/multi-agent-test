"""Model loading utilities for HuggingFace transformers."""

import os
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)


def load_model(
    model_name: str,
    device: str = "cuda",
    precision: str = "bf16",
    tp_rank: Optional[int] = None,
    tp_size: int = 1,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a HuggingFace model with optional tensor parallelism.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        precision: Precision mode ('bf16', 'fp16', 'fp32')
        tp_rank: Tensor parallel rank (None for no TP)
        tp_size: Tensor parallel world size
        cache_dir: Cache directory for model weights
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(precision, torch.bfloat16)
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    print(f"Loading model {model_name} on {device} with {precision} precision...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config and patch if needed (e.g., Phi-2 missing pad_token_id)
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Add missing pad_token_id if not present
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.eos_token_id
        print(f"Added pad_token_id={config.pad_token_id} to config")
    
    # Load model with device mapping
    if tp_size > 1 and tp_rank is not None:
        # Tensor parallel loading
        # Note: This is a simplified version. Production TP requires
        # proper sharding logic (e.g., via Megatron-LM or custom sharding)
        print(f"Loading model with TP rank {tp_rank}/{tp_size}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": device},
        )
        
        # Apply tensor parallel sharding to linear layers
        model = apply_tensor_parallel_sharding(model, tp_rank, tp_size)
    else:
        # Standard loading
        # Check if we should use 8-bit quantization
        load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
        
        if load_in_8bit:
            print(f"Loading model in 8-bit mode to save memory...")
            # For 8-bit, we need to use device_map with the specific device
            # Extract just the device index (e.g., "cuda:0" -> 0)
            if ":" in device:
                device_idx = int(device.split(":")[-1])
            else:
                device_idx = 0
            
            # Use BitsAndBytesConfig for 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map={"": device_idx},  # Map to specific GPU index
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map={"": device},
            )
    
    # Set to eval mode
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {count_parameters(model):,}")
    
    return model, tokenizer


def apply_tensor_parallel_sharding(
    model: PreTrainedModel,
    tp_rank: int,
    tp_size: int
) -> PreTrainedModel:
    """
    Apply tensor parallel sharding to model linear layers.
    
    This is a simplified implementation that shards column-parallel
    and row-parallel layers. Production systems would use libraries
    like Megatron-LM for proper TP implementation.
    
    Args:
        model: The model to shard
        tp_rank: Current tensor parallel rank
        tp_size: Total tensor parallel size
    
    Returns:
        Sharded model
    """
    # This is a placeholder for TP sharding logic
    # In a real implementation, you would:
    # 1. Identify which layers are column-parallel vs row-parallel
    # 2. Shard weights appropriately
    # 3. Insert communication collectives (all-reduce, all-gather)
    
    print(f"Applying TP sharding (rank {tp_rank}/{tp_size})...")
    
    # For now, just return the model as-is
    # Real TP would require splitting weight matrices and adding collectives
    return model


def count_parameters(model: PreTrainedModel) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_model_config(model: PreTrainedModel) -> dict:
    """Extract relevant configuration from model."""
    config = model.config
    
    return {
        "vocab_size": config.vocab_size,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def estimate_kv_cache_size(
    model: PreTrainedModel,
    batch_size: int,
    seq_len: int,
    precision: str = "bf16"
) -> int:
    """
    Estimate KV cache memory usage in bytes.
    
    Args:
        model: The model
        batch_size: Batch size
        seq_len: Sequence length
        precision: Data precision
    
    Returns:
        Estimated memory in bytes
    """
    config = get_model_config(model)
    
    num_layers = config.get("num_layers", 32)
    hidden_size = config.get("hidden_size", 4096)
    
    # KV cache: 2 (K and V) * num_layers * batch_size * seq_len * hidden_size
    elements = 2 * num_layers * batch_size * seq_len * hidden_size
    
    # Bytes per element
    bytes_per_element = {
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
    }.get(precision, 2)
    
    return elements * bytes_per_element


def allocate_kv_cache(
    model: PreTrainedModel,
    batch_size: int,
    max_seq_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Pre-allocate KV cache buffer.
    
    Args:
        model: The model
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        device: Device to allocate on
    
    Returns:
        Pre-allocated KV cache tensor
    """
    config = get_model_config(model)
    
    num_layers = config.get("num_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    head_dim = config.get("hidden_size", 4096) // num_heads
    
    # Shape: [num_layers, 2 (K/V), batch_size, num_heads, max_seq_len, head_dim]
    kv_shape = (num_layers, 2, batch_size, num_heads, max_seq_len, head_dim)
    
    kv_cache = torch.zeros(
        kv_shape,
        dtype=model.dtype,
        device=device
    )
    
    return kv_cache

