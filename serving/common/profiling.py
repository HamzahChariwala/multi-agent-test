"""PyTorch profiling utilities with ExecutionTraceObserver."""

import os
import json
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

import torch
import yaml


class ProfilerContext:
    """Context manager for PyTorch profiler with ExecutionTraceObserver."""
    
    def __init__(
        self,
        gpu_id: str,
        request_id: str,
        config_path: str = "./config/profiling.yaml",
        enabled: Optional[bool] = None
    ):
        """
        Initialize profiler context.
        
        Args:
            gpu_id: GPU identifier (e.g., 't4_gpu0', 'a100_gpu1')
            request_id: Unique request identifier
            config_path: Path to profiling configuration file
            enabled: Override config to enable/disable profiling
        """
        self.gpu_id = gpu_id
        self.request_id = request_id
        self.config = self._load_config(config_path)
        
        # Check if profiling is enabled
        env_enabled = os.getenv("ENABLE_PROFILING", "").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else (
            self.config.get("enabled", False) or env_enabled
        )
        
        # Get GPU-specific config
        self.gpu_config = self.config.get("gpus", {}).get(gpu_id, {})
        self.gpu_enabled = self.gpu_config.get("enabled", True)
        
        # Only enable if both global and GPU-specific are enabled
        self.enabled = self.enabled and self.gpu_enabled
        
        # Setup output directory
        base_dir = self.config.get("output_base_dir", "./profiling_traces")
        self.output_dir = Path(base_dir) / gpu_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiler = None
        self.et_file = None
        self.et_observer = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load profiling configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
    
    def __enter__(self):
        """Start profiling."""
        if not self.enabled:
            return self
        
        # Setup ExecutionTraceObserver if enabled
        et_config = self.config.get("execution_trace", {})
        self.et_observer = None
        if et_config.get("enabled", False):
            et_format = et_config.get("export_format", "json")
            self.et_file = str(self.output_dir / f"{self.request_id}_et.{et_format}")
            
            # Create and register ExecutionTraceObserver
            # This captures CPU-side execution graph
            self.et_observer = torch.profiler.ExecutionTraceObserver()
            self.et_observer.register_callback(self.et_file)
            self.et_observer.start()
        
        # Configure what to record
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        # Create profiler WITHOUT schedule for minimal overhead
        # Just capture the entire request in one go
        profiler_kwargs = {
            "activities": activities,
            "record_shapes": self.gpu_config.get("record_shapes", False),
            "profile_memory": False,  # Disable for lower overhead
            "with_stack": self.gpu_config.get("with_stack", False),
            "with_flops": False,  # Disable for lower overhead
            "with_modules": False,
        }
        
        # Add experimental config if needed for better trace detail
        if self.et_observer:
            profiler_kwargs["experimental_config"] = torch._C._profiler._ExperimentalConfig(
                verbose=True
            )
        
        self.profiler = torch.profiler.profile(**profiler_kwargs)
        
        self.profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and export traces."""
        if not self.enabled or self.profiler is None:
            return False
        
        # Stop ExecutionTraceObserver first
        if self.et_observer is not None:
            try:
                self.et_observer.stop()
                self.et_observer.unregister_callback()
                print(f"Exported ExecutionTrace to: {self.et_file}")
            except Exception as e:
                print(f"Warning: Could not export ExecutionTrace: {e}")
        
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # Export Chrome trace directly (no schedule means no on_trace_ready callback)
        trace_file = self.output_dir / f"{self.request_id}_trace.json"
        self.profiler.export_chrome_trace(str(trace_file))
        
        return False
    
    def step(self):
        """Advance profiler to next step (for iterative operations)."""
        # No-op when not using schedule - profiler runs continuously
        pass


@contextmanager
def profile_operation(operation_name: str, gpu_id: str, request_id: str, enabled: bool = True):
    """
    Context manager for profiling a specific operation.
    
    Args:
        operation_name: Name of the operation being profiled
        gpu_id: GPU identifier
        request_id: Request identifier
        enabled: Whether profiling is enabled
    
    Example:
        with profile_operation("prefill", "t4_gpu0", "req_123"):
            kv_cache = model.prefill(input_ids)
    """
    if enabled:
        with torch.profiler.record_function(operation_name):
            yield
    else:
        yield


def get_profiler_config(config_path: str = "./config/profiling.yaml") -> dict:
    """Load and return profiling configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {"enabled": False}


def is_profiling_enabled(gpu_id: str, config_path: str = "./config/profiling.yaml") -> bool:
    """Check if profiling is enabled for a specific GPU."""
    config = get_profiler_config(config_path)
    
    # Check environment variable
    env_enabled = os.getenv("ENABLE_PROFILING", "").lower() in ("true", "1", "yes")
    
    # Check global and GPU-specific config
    global_enabled = config.get("enabled", False) or env_enabled
    gpu_config = config.get("gpus", {}).get(gpu_id, {})
    gpu_enabled = gpu_config.get("enabled", True)
    
    return global_enabled and gpu_enabled

