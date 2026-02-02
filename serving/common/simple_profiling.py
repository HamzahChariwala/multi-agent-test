"""
Simple profiling wrapper that creates one trace file per HTTP request phase.
No continuous profiling, no complexity - just works.
"""

import torch
from pathlib import Path
from contextlib import contextmanager
import logging
import threading

logger = logging.getLogger(__name__)


@contextmanager
def simple_profile(gpu_id: str, phase_name: str, enabled: bool = True):
    """
    Simple profiler that creates one trace per phase.
    
    Args:
        gpu_id: GPU identifier (e.g., "t4_gpu0")
        phase_name: Phase name (e.g., "phase1_generate", "phase2_judge")
        enabled: Whether to profile
    
    Usage:
        with simple_profile("t4_gpu0", "phase1_generate"):
            # ... do work ...
            pass
    
    This creates:
        - profiling_traces/{gpu_id}/{phase_name}_trace.json (Chrome trace)
        - profiling_traces/{gpu_id}/{phase_name}_et.json (ExecutionTrace)
    """
    if not enabled:
        yield
        return
    
    # Setup output directory
    output_dir = Path("./profiling_traces") / gpu_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup ExecutionTraceObserver
    et_file = str(output_dir / f"{phase_name}_et.json")
    et_observer = torch.profiler.ExecutionTraceObserver()
    et_observer.register_callback(et_file)
    et_observer.start()
    
    # Create profiler
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    )
    
    profiler.__enter__()
    
    try:
        yield
    finally:
        logger.info(f"[{gpu_id}] Phase {phase_name} complete, stopping profiler...")
        
        # Stop ET Observer (must be synchronous)
        try:
            et_observer.stop()
            et_observer.unregister_callback()
            import os
            if os.path.exists(et_file):
                size_mb = os.path.getsize(et_file) / (1024 * 1024)
                logger.info(f"[{gpu_id}] Exported ExecutionTrace: {et_file} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"[{gpu_id}] Error exporting ET trace: {e}", exc_info=True)
        
        # Stop profiler (must be synchronous to capture events)
        try:
            profiler.__exit__(None, None, None)
            logger.info(f"[{gpu_id}] Profiler stopped")
        except Exception as e:
            logger.error(f"[{gpu_id}] Error stopping profiler: {e}", exc_info=True)
        
        # Export Chrome trace in background thread (this is the slow part)
        def export_chrome_trace():
            try:
                trace_file = output_dir / f"{phase_name}_trace.json"
                logger.info(f"[{gpu_id}] Background: Exporting Chrome trace to {trace_file}...")
                profiler.export_chrome_trace(str(trace_file))
                import os
                if os.path.exists(trace_file):
                    size_mb = os.path.getsize(trace_file) / (1024 * 1024)
                    logger.info(f"[{gpu_id}] Background: Exported Chrome trace: {trace_file} ({size_mb:.1f} MB)")
            except Exception as e:
                logger.error(f"[{gpu_id}] Background: Error exporting Chrome trace: {e}", exc_info=True)
        
        # Start background thread for Chrome trace export
        export_thread = threading.Thread(target=export_chrome_trace, daemon=True)
        export_thread.start()
        logger.info(f"[{gpu_id}] Started background thread for Chrome trace export")

