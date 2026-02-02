"""PyTorch profiling utilities with ExecutionTraceObserver."""

import os
import json
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager
import logging as logging_module
import threading

import torch
import yaml

logger = logging_module.getLogger(__name__)


class ProfilerContext:
    """Context manager for PyTorch profiler with ExecutionTraceObserver."""
    
    def __init__(
        self,
        gpu_id: str,
        request_id: str,
        config_path: str = "./config/profiling.yaml",
        enabled: Optional[bool] = None,
        split_by_gpu: bool = False,
        num_gpus: int = 4
    ):
        """
        Initialize profiler context.
        
        Args:
            gpu_id: GPU identifier (e.g., 't4_gpu0', 'a100_gpu1', 'a100_synthesis')
            request_id: Unique request identifier
            config_path: Path to profiling configuration file
            enabled: Override config to enable/disable profiling
            split_by_gpu: If True, split combined trace into per-GPU traces
            num_gpus: Number of GPUs to split across (if split_by_gpu=True)
        """
        self.gpu_id = gpu_id
        self.request_id = request_id
        self.config = self._load_config(config_path)
        self.split_by_gpu = split_by_gpu
        self.num_gpus = num_gpus
        
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
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / gpu_id
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
        et_created = False
        if self.et_observer is not None:
            try:
                self.et_observer.stop()
                self.et_observer.unregister_callback()
                # Check if file was actually created
                if os.path.exists(self.et_file):
                    file_size = os.path.getsize(self.et_file) / (1024 * 1024)  # MB
                    print(f"✓ Exported ExecutionTrace to: {self.et_file} ({file_size:.1f} MB)")
                    et_created = True
                else:
                    print(f"⚠ ExecutionTrace file not created: {self.et_file}")
            except Exception as e:
                print(f"Warning: Could not export ExecutionTrace: {e}")
        
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # Export Chrome trace directly (no schedule means no on_trace_ready callback)
        trace_file = self.output_dir / f"{self.request_id}_trace.json"
        trace_created = False
        try:
            self.profiler.export_chrome_trace(str(trace_file))
            if os.path.exists(trace_file):
                file_size = os.path.getsize(trace_file) / (1024 * 1024)  # MB
                print(f"✓ Exported Chrome trace to: {trace_file} ({file_size:.1f} MB)")
                trace_created = True
            else:
                print(f"⚠ Chrome trace file not created: {trace_file}")
        except Exception as e:
            print(f"Warning: Could not export Chrome trace: {e}")
        
        # Split traces by GPU if requested (in background to avoid blocking)
        if self.split_by_gpu and (trace_created or et_created):
            def _split_in_background():
                try:
                    from serving.common.trace_splitter import split_traces
                    logger.info(f"Splitting traces into per-GPU directories (background)...")
                    split_traces(
                        combined_trace_dir=str(self.output_dir),
                        request_id=self.request_id,
                        output_base_dir=str(self.base_dir),
                        num_gpus=self.num_gpus,
                        cleanup_combined=True  # Delete combined traces after splitting
                    )
                    logger.info(f"✓ Traces split into a100_gpu{{0..{self.num_gpus-1}}} directories")
                except Exception as e:
                    logger.error(f"Trace splitting failed: {e}")
                    logger.exception("Trace splitting exception")
            
            # Run splitting in background thread to not block HTTP response
            split_thread = threading.Thread(target=_split_in_background, daemon=True)
            split_thread.start()
            print(f"Trace splitting started in background (check logs for completion)")
        
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

