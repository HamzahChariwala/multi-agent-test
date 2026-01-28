"""Tests for profiling infrastructure."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from serving.common.profiling import (
    ProfilerContext,
    profile_operation,
    get_profiler_config,
    is_profiling_enabled,
)


def test_profiler_config_loading():
    """Test loading profiler configuration."""
    config = get_profiler_config("./config/profiling.yaml")
    
    assert isinstance(config, dict)
    assert "enabled" in config or len(config) == 0  # Empty if file doesn't exist


def test_profiler_context_disabled():
    """Test profiler context when disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not fail even when disabled
        with ProfilerContext(
            gpu_id="test_gpu",
            request_id="test_req",
            enabled=False
        ) as ctx:
            # Do some dummy computation
            x = torch.randn(10, 10)
            y = x @ x.T
        
        # Profiler should be None when disabled
        assert ctx.profiler is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profiler_context_enabled():
    """Test profiler context when enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary config
        import yaml
        config = {
            "enabled": True,
            "output_base_dir": tmpdir,
            "gpus": {
                "test_gpu": {
                    "enabled": True,
                    "operations": ["test"],
                    "with_stack": False,
                    "record_shapes": False,
                }
            },
            "schedule": {
                "wait": 0,
                "warmup": 0,
                "active": 1,
                "repeat": 1,
            }
        }
        
        config_path = Path(tmpdir) / "profiling.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run with profiling
        with ProfilerContext(
            gpu_id="test_gpu",
            request_id="test_req",
            config_path=str(config_path),
            enabled=True
        ) as ctx:
            # Do computation
            x = torch.randn(100, 100, device="cuda")
            for _ in range(3):
                y = x @ x.T
                ctx.step()
        
        # Check that profiler was created
        assert ctx.profiler is not None


def test_profile_operation_decorator():
    """Test profile_operation context manager."""
    # Should work without errors when disabled
    with profile_operation("test_op", "test_gpu", "test_req", enabled=False):
        x = torch.randn(10, 10)
        y = x * 2
    
    # Should also work when enabled (creates record_function)
    with profile_operation("test_op", "test_gpu", "test_req", enabled=True):
        x = torch.randn(10, 10)
        y = x * 2


def test_is_profiling_enabled():
    """Test checking if profiling is enabled for a GPU."""
    # Should return bool without error
    enabled = is_profiling_enabled("t4_gpu0", "./config/profiling.yaml")
    assert isinstance(enabled, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

