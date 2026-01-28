"""Tests for KV cache transfer utilities."""

import pytest
import torch

from serving.t4_cluster.kv_transfer import (
    allocate_kv_buffer,
    get_kv_cache_size,
    verify_kv_integrity,
    trim_kv_cache,
)


def test_allocate_kv_buffer():
    """Test KV buffer allocation."""
    model_config = {
        "num_layers": 4,
        "num_attention_heads": 8,
        "hidden_size": 256,
    }
    
    batch_size = 2
    max_seq_len = 128
    
    kv_buffer = allocate_kv_buffer(
        model_config,
        batch_size,
        max_seq_len,
        device="cpu",
        dtype=torch.float32,
    )
    
    # Check structure
    assert len(kv_buffer) == 4  # num_layers
    
    for key_cache, value_cache in kv_buffer:
        # Check shapes [batch_size, num_heads, seq_len, head_dim]
        assert key_cache.shape[0] == batch_size
        assert key_cache.shape[1] == 8  # num_heads
        assert key_cache.shape[2] == max_seq_len
        
        assert value_cache.shape == key_cache.shape


def test_get_kv_cache_size():
    """Test KV cache size calculation."""
    # Create mock KV cache
    kv_cache = [
        (torch.randn(1, 8, 64, 32), torch.randn(1, 8, 64, 32))
        for _ in range(4)
    ]
    
    size = get_kv_cache_size(kv_cache)
    
    # Each tensor: 1 * 8 * 64 * 32 * 4 bytes (float32) = 65536 bytes
    # 2 tensors per layer * 4 layers = 8 tensors
    expected = 8 * 65536
    
    assert size == expected


def test_verify_kv_integrity():
    """Test KV cache integrity verification."""
    # Valid KV cache
    kv_cache = [
        (torch.randn(1, 8, 64, 32), torch.randn(1, 8, 64, 32))
        for _ in range(4)
    ]
    
    assert verify_kv_integrity(kv_cache) is True
    
    # Invalid - with NaN
    kv_cache_with_nan = [
        (torch.tensor([[[[float('nan')]]]]), torch.randn(1, 1, 1, 1))
    ]
    
    assert verify_kv_integrity(kv_cache_with_nan) is False
    
    # Invalid - None
    assert verify_kv_integrity(None) is False


def test_trim_kv_cache():
    """Test KV cache trimming."""
    # Create KV cache with padding
    kv_cache = [
        (torch.randn(1, 8, 128, 32), torch.randn(1, 8, 128, 32))
        for _ in range(4)
    ]
    
    # Trim to actual length
    actual_seq_len = 64
    trimmed = trim_kv_cache(kv_cache, actual_seq_len)
    
    # Check trimmed shapes
    for key_cache, value_cache in trimmed:
        assert key_cache.shape[2] == actual_seq_len
        assert value_cache.shape[2] == actual_seq_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

