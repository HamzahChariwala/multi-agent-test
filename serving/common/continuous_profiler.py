"""
Continuous profiler that spans multiple requests with idle timeout.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
import threading

import torch
import yaml

logger = logging.getLogger(__name__)


class ContinuousProfiler:
    """
    Profiler that runs continuously across requests with idle timeout.
    Exports traces only after idle_timeout seconds of inactivity.
    """
    
    def __init__(
        self,
        gpu_id: str,
        session_id: str,
        config_path: str = "./config/profiling.yaml",
        idle_timeout: float = 30.0,
        enabled: Optional[bool] = None
    ):
        self.gpu_id = gpu_id
        self.session_id = session_id
        self.idle_timeout = idle_timeout
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
        
        # Profiler state
        self.profiler = None
        self.et_observer = None
        self.et_file = None
        self.is_active = False
        self.last_activity = time.time()
        
        # Timeout task
        self.timeout_task = None
        self.timeout_lock = threading.Lock()
        
        logger.info(f"[{gpu_id}] ContinuousProfiler initialized (enabled={self.enabled}, timeout={idle_timeout}s)")
    
    def _load_config(self, config_path: str) -> dict:
        """Load profiling configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
    
    def start(self):
        """Start the profiler (called on first request)."""
        if not self.enabled or self.is_active:
            return
        
        logger.info(f"[{self.gpu_id}] Starting continuous profiler for session {self.session_id}")
        
        # Setup ExecutionTraceObserver if enabled
        et_config = self.config.get("execution_trace", {})
        if et_config.get("enabled", False):
            et_format = et_config.get("export_format", "json")
            self.et_file = str(self.output_dir / f"{self.session_id}_et.{et_format}")
            
            self.et_observer = torch.profiler.ExecutionTraceObserver()
            self.et_observer.register_callback(self.et_file)
            self.et_observer.start()
        
        # Configure profiler
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        profiler_kwargs = {
            "activities": activities,
            "record_shapes": self.gpu_config.get("record_shapes", False),
            "profile_memory": False,
            "with_stack": self.gpu_config.get("with_stack", False),
            "with_flops": False,
            "with_modules": False,
        }
        
        if self.et_observer:
            profiler_kwargs["experimental_config"] = torch._C._profiler._ExperimentalConfig(
                verbose=True
            )
        
        self.profiler = torch.profiler.profile(**profiler_kwargs)
        self.profiler_context = self.profiler.__enter__()
        
        self.is_active = True
        self.last_activity = time.time()
        
        logger.info(f"[{self.gpu_id}] Continuous profiler started")
    
    def record_activity(self):
        """Mark that a request is active (resets idle timer)."""
        if not self.enabled:
            return
        
        self.last_activity = time.time()
        
        # Start profiler if not already running
        if not self.is_active:
            self.start()
    
    def check_and_export_if_idle(self):
        """Check if idle timeout expired and export if so."""
        if not self.enabled or not self.is_active:
            return False
        
        idle_time = time.time() - self.last_activity
        
        if idle_time >= self.idle_timeout:
            logger.info(f"[{self.gpu_id}] Idle timeout ({self.idle_timeout}s) reached, exporting traces")
            self._export()
            return True
        
        return False
    
    def _export(self):
        """Export traces and stop profiler."""
        if not self.is_active:
            return
        
        logger.info(f"[{self.gpu_id}] Exporting continuous profiler traces")
        
        # Stop ExecutionTraceObserver FIRST
        if self.et_observer is not None:
            try:
                self.et_observer.stop()
                self.et_observer.unregister_callback()
                logger.info(f"[{self.gpu_id}] Exported ExecutionTrace to: {self.et_file}")
            except Exception as e:
                logger.warning(f"[{self.gpu_id}] Could not export ExecutionTrace: {e}")
        
        # Stop profiler SECOND
        if self.profiler is not None:
            try:
                self.profiler.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"[{self.gpu_id}] Could not stop profiler: {e}")
        
        # Export Chrome trace LAST (after __exit__, this is required when using ET Observer)
        if self.profiler is not None:
            try:
                trace_file = self.output_dir / f"{self.session_id}_trace.json"
                self.profiler.export_chrome_trace(str(trace_file))
                logger.info(f"[{self.gpu_id}] Exported Chrome trace to: {trace_file}")
            except Exception as e:
                logger.warning(f"[{self.gpu_id}] Could not export Chrome trace: {e}")
                import traceback
                logger.warning(f"[{self.gpu_id}] Full traceback: {traceback.format_exc()}")
        
        self.is_active = False
        self.profiler = None
        self.et_observer = None
    
    def force_export(self):
        """Force export immediately (for cleanup)."""
        logger.info(f"[{self.gpu_id}] Force exporting traces")
        self._export()


class ContinuousProfilerManager:
    """
    Manages continuous profilers with background idle monitoring using threading.
    """
    
    def __init__(self, idle_timeout: float = 30.0):
        self.profilers = {}  # gpu_id -> ContinuousProfiler
        self.idle_timeout = idle_timeout
        self.monitor_thread = None
        self.running = False
        
        logger.info(f"ContinuousProfilerManager initialized (idle_timeout={idle_timeout}s)")
    
    def get_or_create(self, gpu_id: str, session_id: str, config_path: str = "./config/profiling.yaml") -> ContinuousProfiler:
        """Get existing profiler or create new one."""
        if gpu_id not in self.profilers:
            self.profilers[gpu_id] = ContinuousProfiler(
                gpu_id=gpu_id,
                session_id=session_id,
                config_path=config_path,
                idle_timeout=self.idle_timeout
            )
        return self.profilers[gpu_id]
    
    def start_monitoring(self):
        """Start background monitoring for idle timeouts."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started continuous profiler monitoring (thread-based)")
    
    def _monitor_loop(self):
        """Background thread that checks for idle profilers."""
        while self.running:
            try:
                # Check each profiler
                for gpu_id, profiler in list(self.profilers.items()):
                    profiler.check_and_export_if_idle()
                
                # Check every 5 seconds
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in profiler monitor loop: {e}")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
    
    def shutdown(self):
        """Export all active profilers and shutdown."""
        logger.info("Shutting down ContinuousProfilerManager")
        self.stop_monitoring()
        
        for gpu_id, profiler in self.profilers.items():
            profiler.force_export()


# Global manager instance
_global_manager = None


def get_manager(idle_timeout: float = 30.0) -> ContinuousProfilerManager:
    """Get the global profiler manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ContinuousProfilerManager(idle_timeout=idle_timeout)
    return _global_manager


def record_request_activity(gpu_id: str, session_id: str):
    """Mark that a request is active on this GPU."""
    manager = get_manager()
    profiler = manager.get_or_create(gpu_id, session_id)
    profiler.record_activity()

