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
        
        # Setup ExecutionTraceObserver
        et_config = self.config.get("execution_trace", {})
        if et_config.get("enabled", True):
            et_format = et_config.get("export_format", "json")
            self.et_file = str(self.output_dir / f"{self.request_id}_et.{et_format}")
        
        # Configure profiling schedule
        schedule_config = self.config.get("schedule", {})
        schedule = torch.profiler.schedule(
            wait=schedule_config.get("wait", 1),
            warmup=schedule_config.get("warmup", 1),
            active=schedule_config.get("active", 3),
            repeat=schedule_config.get("repeat", 1)
        )
        
        # Configure what to record
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        # Create profiler
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=self.gpu_config.get("record_shapes", True),
            profile_memory=True,
            with_stack=self.gpu_config.get("with_stack", True),
            with_flops=True,
            with_modules=False,
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True
            ) if self.et_file else None
        )
        
        self.profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and export traces."""
        if not self.enabled or self.profiler is None:
            return False
        
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # Export summary
        self._export_summary()
        
        return False
    
    def _trace_handler(self, prof):
        """Handle trace export."""
        # Export Chrome trace
        trace_file = self.output_dir / f"{self.request_id}_trace.json"
        prof.export_chrome_trace(str(trace_file))
        
        # Export ExecutionTrace if configured
        if self.et_file:
            try:
                # ExecutionTraceObserver export
                et_config = self.config.get("execution_trace", {})
                include_inputs = et_config.get("include_operator_inputs", False)
                
                # Export using the profiler's export functionality
                if hasattr(prof, 'export_execution_trace'):
                    prof.export_execution_trace(self.et_file)
            except Exception as e:
                print(f"Warning: Could not export ExecutionTrace: {e}")
    
    def _export_summary(self):
        """Export profiling summary statistics."""
        if self.profiler is None:
            return
        
        summary_file = self.output_dir / f"{self.request_id}_summary.txt"
        
        try:
            # Get key averages
            key_averages = self.profiler.key_averages(group_by_input_shape=True)
            
            with open(summary_file, 'w') as f:
                f.write(f"Profiling Summary for {self.gpu_id} - Request {self.request_id}\n")
                f.write("=" * 80 + "\n\n")
                
                # Top operations by CUDA time
                f.write("Top 10 Operations by CUDA Time:\n")
                f.write("-" * 80 + "\n")
                sorted_by_cuda = sorted(
                    key_averages,
                    key=lambda x: x.cuda_time_total,
                    reverse=True
                )[:10]
                for evt in sorted_by_cuda:
                    f.write(f"{evt.key:60s} {evt.cuda_time_total/1000:.2f} ms\n")
                
                f.write("\n")
                
                # Top operations by CPU time
                f.write("Top 10 Operations by CPU Time:\n")
                f.write("-" * 80 + "\n")
                sorted_by_cpu = sorted(
                    key_averages,
                    key=lambda x: x.cpu_time_total,
                    reverse=True
                )[:10]
                for evt in sorted_by_cpu:
                    f.write(f"{evt.key:60s} {evt.cpu_time_total/1000:.2f} ms\n")
                
                f.write("\n")
                
                # Memory statistics
                f.write("Memory Statistics:\n")
                f.write("-" * 80 + "\n")
                sorted_by_mem = sorted(
                    [e for e in key_averages if e.cpu_memory_usage > 0],
                    key=lambda x: x.cpu_memory_usage,
                    reverse=True
                )[:10]
                for evt in sorted_by_mem:
                    mem_mb = evt.cpu_memory_usage / (1024 * 1024)
                    f.write(f"{evt.key:60s} {mem_mb:.2f} MB\n")
        
        except Exception as e:
            print(f"Warning: Could not export summary: {e}")
    
    def step(self):
        """Advance profiler to next step (for iterative operations)."""
        if self.enabled and self.profiler is not None:
            self.profiler.step()


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

