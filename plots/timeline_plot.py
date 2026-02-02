"""
Timeline plot for GPU profiling traces.

Parses Chrome trace JSON files and creates a timeline showing ONLY categorized GPU kernel executions:
- Filters out all CPU-side operations, Python functions, kernel launches, user annotations
- Shows ONLY kernels matching these categories: GEMM, attention, elementwise, argmax, comms
- Uncategorized kernels are filtered out (no 'other' category)
- One row per GPU (0-4)
- Color-coded using magma colormap
- Communications labeled with kernel names
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re


# Operation categories (removed 'other' - only showing actual categorized kernels)
CATEGORIES = {
    'gemm': 0,
    'attention': 1,
    'elementwise': 2,
    'argmax': 3,
    'comms': 4,
}

# Magma colormap colors for each category
COLORS = plt.cm.magma(np.linspace(0.1, 0.9, 5))


def categorize_kernel(kernel_name: str) -> str:
    """
    Categorize a GPU kernel based on its name (based on actual trace analysis).
    Returns None for kernels that don't match any category (will be filtered out).
    
    Args:
        kernel_name: Name of the CUDA kernel
    
    Returns:
        Category name (gemm, attention, elementwise, argmax, comms) or None
    """
    kernel_lower = kernel_name.lower()
    
    # CHECK COMMS FIRST (including NCCL and c10d operations)
    if any(x in kernel_lower for x in [
        # NCCL kernels - actual device kernels
        'ncclkernel', 'nccldevkernel',
        # c10d collective wrapper
        'c10d::broadcast', 'c10d::allreduce', 'c10d::allgather',
        # NCCL namespace
        'nccl:',
        # Memory transfers
        'memcpy', 'dtoh', 'htod', 'd2h', 'h2d', 'd2d',
    ]):
        return 'comms'
    
    # GEMM operations - matrix multiplication (CuBLAS patterns)
    if any(x in kernel_lower for x in [
        'gemm', 'gemv',  # General matrix mult and matrix-vector mult
        'matmul', 'mm_', 'bmm',  # PyTorch patterns
        'sgemm', 'dgemm', 'hgemm',  # CuBLAS typed GEMMs
        'wgmma', 'mma_',  # Tensor core operations
        'addmm', 'baddbmm',  # PyTorch fused add+matmul
    ]):
        return 'gemm'
    
    # Attention operations - softmax and attention kernels
    if any(x in kernel_lower for x in [
        'softmax_warp',  # PyTorch softmax kernels
        'cunn_softmax',  # CUDA NN softmax
        'attention', 'flash_', 'fmha', 'mha',  # Attention patterns
        '_safe_softmax', '_softmax',  # PyTorch softmax variants
    ]):
        return 'attention'
    
    # Argmax/sampling operations
    if any(x in kernel_lower for x in [
        'argmax', 'argmin',  # Argmax/min reductions
        'topk', 'mbtopk',  # Top-k selection
        'multinomial',  # Sampling
        'radixsort',  # Sorting kernels
        'argmaxops',  # ArgMax reduction ops
    ]):
        return 'argmax'
    
    # Elementwise operations - check LAST to avoid false positives
    if any(x in kernel_lower for x in [
        'elementwise_kernel',  # PyTorch elementwise kernels
        'vectorized_elementwise',  # Vectorized variants
        'unrolled_elementwise',  # Unrolled variants
        'vectorized_layer_norm',  # Layer norm
        'fillfu',  # Fill operations
        'cudafunctor_add',  # Element-wise add
        'binaryfunctor',  # Binary operations
        # Don't include generic 'copy' or 'reduce' here
    ]):
        return 'elementwise'
    
    # Filter out everything else
    return None


def is_gpu_kernel(event: Dict) -> bool:
    """
    Check if event is an actual GPU kernel execution (not CPU, launch, etc.).
    
    Args:
        event: Trace event dictionary
    
    Returns:
        True if this is a GPU kernel execution
    """
    name = event.get('name', '')
    cat = event.get('cat', '')
    
    # Must be a complete event
    if event.get('ph') != 'X':
        return False
    
    # Skip if no name
    if not name:
        return False
    
    # Category must be kernel-related or CUDA runtime
    # Actual GPU kernels typically have categories like 'kernel', 'cuda_runtime', or stream info
    cat_lower = cat.lower()
    if not any(x in cat_lower for x in ['kernel', 'cuda', 'gpu', 'stream']):
        # If no kernel-related category, it's likely CPU-side
        return False
    
    # Skip CPU-side operations and user annotations
    cpu_patterns = [
        'python',           # Python function calls
        'torch::',          # PyTorch C++ API calls
        'aten::',           # ATen operations (CPU-side dispatch)
        'c10::',            # C10 core operations
        'Optimizer',        # Optimizer steps
        'autograd',         # Autograd operations
        'cudaLaunch',       # Kernel launch overhead
        'cudaMemcpy',       # Memory copies (unless you want to track these)
        'cudaMemset',       # Memory set operations
        'cudaDeviceSync',   # Synchronization calls
        'cudaStreamSync',   # Stream synchronization
        'cudaEventRecord',  # Event recording
        'cudaEventSync',    # Event synchronization
        'cudaGetDevice',    # Device queries
        'cudaMalloc',       # Memory allocation
        'cudaFree',         # Memory deallocation
        'ProfilerStep',     # Profiler overhead
        'record_function',  # PyTorch record_function overhead
        'user_annotation',  # User annotations
        'cpu_op',           # Generic CPU operations
        'cudaOccupancy',    # Occupancy queries
        'cudaThread',       # Thread management
        # User-defined markers that aren't actual kernels
        'prefill',          # User annotation (not a kernel name)
        'decode',           # User annotation (not a kernel name)
        'decode_only',      # User annotation
        'receive_kv',       # User annotation
        'kv_',              # KV-related annotations
    ]
    
    name_lower = name.lower()
    for pattern in cpu_patterns:
        if pattern.lower() in name_lower:
            return False
    
    # Skip memcpy/memset unless it's actually a GPU operation
    if 'memcpy' in name_lower or 'memset' in name_lower:
        # Only keep if it has H2D/D2H/D2D indicators (actual transfers)
        if not any(x in name_lower for x in ['h2d', 'd2h', 'd2d', 'dtoh', 'htod']):
            return False
    
    # Skip anything that looks like a simple marker (too short, no special chars)
    # Real kernel names usually have underscores, template params, etc.
    if len(name) < 10 and '_' not in name and '<' not in name:
        return False
    
    return True


def parse_trace_file(trace_path: Path, fallback_device_id: int = None) -> List[Dict]:
    """
    Parse Chrome trace JSON file and extract only GPU kernel executions.
    Preserves absolute timestamps to maintain relative timing across traces.
    
    Args:
        trace_path: Path to trace JSON file
        fallback_device_id: Device ID to use if not found in event metadata (extracted from directory name)
    
    Returns:
        List of GPU kernel events (no CPU operations)
    """
    with open(trace_path, 'r') as f:
        trace_data = json.load(f)
    
    events = []
    
    for event in trace_data.get('traceEvents', []):
        # Filter: only actual GPU kernel executions
        if not is_gpu_kernel(event):
            continue
        
        name = event.get('name', '')
        
        # Prioritize fallback_device_id (from directory/trace file location)
        # This is the most reliable - if an operation appears in GPU 1's trace,
        # it means GPU 1 executed it, regardless of metadata
        # (e.g., NCCL collectives appear in all participants' traces)
        if fallback_device_id is not None:
            device_id = fallback_device_id
        else:
            # Extract GPU ID from event metadata as fallback
            args = event.get('args', {})
            device_id = args.get('device', None)
            
            # Try to get device from other fields
            if device_id is None:
                # Check correlation ID or stream info
                device_id = args.get('Device', None)
            
            if device_id is None:
                # Check if it's in the category
                cat = event.get('cat', '')
                if 'stream' in cat.lower():
                    # Extract device from stream info
                    match = re.search(r'device (\d+)', cat.lower())
                    if match:
                        device_id = int(match.group(1))
        
        # Get timing info (in microseconds) - keep absolute timestamps
        ts = event.get('ts', 0)  # timestamp in microseconds
        dur = event.get('dur', 0)  # duration in microseconds
        
        if dur <= 0:
            continue
        
        # Categorize
        category = categorize_kernel(name)
        
        # Skip if doesn't match any category
        if category is None:
            continue
        
        events.append({
            'name': name,
            'category': category,
            'device': device_id if device_id is not None else 0,
            'start': ts / 1000.0,  # Convert to milliseconds but preserve absolute time
            'duration': dur / 1000.0,  # Convert to milliseconds
            'ts_us': ts,  # Keep original for debugging
        })
    
    return events


def create_timeline_plot(events: List[Dict], output_path: Path):
    """
    Create timeline plot from events.
    
    Args:
        events: List of event dictionaries
        output_path: Where to save the plot
    """
    if not events:
        print("No events to plot")
        return
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get time range - preserve absolute timestamps to show relative timing
    min_time = min(e['start'] for e in events)
    max_time = max(e['start'] + e['duration'] for e in events)
    
    # Normalize times to start at 0 (only shift, don't change relative timing)
    for event in events:
        event['start'] -= min_time
    
    time_range = max_time - min_time
    
    print(f"\nTime range: {time_range:.2f} ms")
    print(f"Absolute time window: {min_time:.2f} to {max_time:.2f} microseconds")
    
    # Group events by GPU
    gpu_events = {}
    for event in events:
        gpu_id = event['device']
        if gpu_id not in gpu_events:
            gpu_events[gpu_id] = []
        gpu_events[gpu_id].append(event)
    
    # Get GPU IDs (0-4)
    gpu_ids = sorted([g for g in gpu_events.keys() if 0 <= g <= 4])
    num_gpus = len(gpu_ids)
    
    if num_gpus == 0:
        print("No GPU events found")
        return
    
    # Create plot with stacking for overlaps
    y_positions = {gpu_id: i for i, gpu_id in enumerate(gpu_ids)}
    
    # Plot events with stacking for overlaps
    for gpu_id in gpu_ids:
        y_base = y_positions[gpu_id]
        gpu_ops = sorted(gpu_events[gpu_id], key=lambda e: e['start'])
        
        # Track lanes: list of (end_time) for each lane
        lanes = []
        
        # Assign each operation to a lane (deterministic greedy algorithm)
        for event in gpu_ops:
            # Find first available lane (where this op doesn't overlap)
            assigned_lane = None
            for lane_idx, lane_end in enumerate(lanes):
                if event['start'] >= lane_end:
                    # This lane is free
                    assigned_lane = lane_idx
                    lanes[lane_idx] = event['start'] + event['duration']
                    break
            
            if assigned_lane is None:
                # Need a new lane
                assigned_lane = len(lanes)
                lanes.append(event['start'] + event['duration'])
            
            event['lane'] = assigned_lane
        
        # Total number of lanes for this GPU (maximum concurrency)
        num_lanes = len(lanes)
        lane_height = 0.8 / num_lanes if num_lanes > 0 else 0.8
        
        # Now plot - all operations use the same lane_height for consistent stacking
        for event in gpu_ops:
            category = event['category']
            color_idx = CATEGORIES[category]
            color = COLORS[color_idx]
            
            lane = event['lane']
            y_bottom = y_base - 0.4 + (lane * lane_height)
            
            # Create rectangle - no border, translucent
            rect = mpatches.Rectangle(
                (event['start'], y_bottom),
                event['duration'],
                lane_height,
                facecolor=color,
                edgecolor='none',
                linewidth=0,
                alpha=0.6
            )
            ax.add_patch(rect)
            
            # Add label for communications (only if there's enough space)
            if category == 'comms' and lane_height > 0.1:
                # Truncate long kernel names
                label = event['name']
                if len(label) > 30:
                    label = label[:27] + '...'
                
                # Only label if rectangle is wide enough
                if event['duration'] > time_range * 0.02:  # At least 2% of time range
                    fontsize = max(4, min(6, lane_height * 10))
                    ax.text(
                        event['start'] + event['duration'] / 2,
                        y_bottom + lane_height / 2,
                        label,
                        fontsize=fontsize,
                        ha='center',
                        va='center',
                        rotation=0,
                        clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                    )
    
    # Setup axes
    ax.set_ylim(-0.5, num_gpus - 0.5)
    ax.set_xlim(0, time_range)
    
    ax.set_yticks(range(num_gpus))
    ax.set_yticklabels([f'GPU {gpu_id}' for gpu_id in gpu_ids])
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('GPU', fontsize=12)
    ax.set_title('GPU Operations Timeline', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Create legend - no borders on legend patches either
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[CATEGORIES[cat]], 
                      edgecolor='none', 
                      alpha=0.6,
                      label=cat.upper())
        for cat in ['gemm', 'attention', 'elementwise', 'argmax', 'comms']
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved timeline plot to: {output_path}")
    
    # Also save as PDF for better quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved timeline plot to: {pdf_path}")
    
    plt.close()


def print_statistics(events: List[Dict]):
    """Print statistics about the events."""
    if not events:
        return
    
    print("\n" + "="*80)
    print("PROFILING STATISTICS")
    print("="*80)
    
    # Count by category
    category_counts = {}
    category_times = {}
    
    for event in events:
        cat = event['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
        category_times[cat] = category_times.get(cat, 0.0) + event['duration']
    
    print("\nOperation Counts:")
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        time_ms = category_times[cat]
        print(f"  {cat.upper():15s}: {count:5d} ops, {time_ms:10.2f} ms total")
    
    # Total time
    total_time = sum(e['duration'] for e in events)
    print(f"\nTotal operation time: {total_time:.2f} ms")
    
    # Per-GPU breakdown
    print("\nPer-GPU breakdown:")
    gpu_events = {}
    for event in events:
        gpu_id = event['device']
        if gpu_id not in gpu_events:
            gpu_events[gpu_id] = []
        gpu_events[gpu_id].append(event)
    
    for gpu_id in sorted(gpu_events.keys()):
        if 0 <= gpu_id <= 4:
            count = len(gpu_events[gpu_id])
            time_ms = sum(e['duration'] for e in gpu_events[gpu_id])
            print(f"  GPU {gpu_id}: {count:5d} ops, {time_ms:10.2f} ms")
    
    # Find longest operations
    print("\nTop 10 longest operations:")
    sorted_events = sorted(events, key=lambda e: e['duration'], reverse=True)[:10]
    for i, event in enumerate(sorted_events, 1):
        print(f"  {i:2d}. [{event['category'].upper():10s}] {event['duration']:8.2f} ms - {event['name'][:60]}")
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create timeline plot from profiling traces")
    parser.add_argument(
        '--trace-dir',
        type=Path,
        default=Path('./traces/new'),
        help='Directory containing trace files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./plots/timeline.png'),
        help='Output plot path'
    )
    parser.add_argument(
        '--trace-file',
        type=str,
        help='Specific trace file to plot (e.g., t4_gpu0/<request_id>_trace.json)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Find trace files
    if args.trace_file:
        trace_path = args.trace_dir / args.trace_file
        if not trace_path.exists():
            print(f"Error: Trace file not found: {trace_path}")
            return
        trace_files = [trace_path]
    else:
        # Find all trace files in subdirectories - multiple patterns for different naming conventions
        trace_files = []
        
        # Pattern 1: Our original format (rank*_trace.json, test_*_trace.json)
        trace_files.extend(args.trace_dir.glob('**/*_trace.json'))
        
        # Pattern 2: vllm format (*-rank-*.pt.trace.json)
        trace_files.extend(args.trace_dir.glob('**/*-rank-*.pt.trace.json'))
        
        # Pattern 3: General *.trace.json (catch-all)
        trace_files.extend(args.trace_dir.glob('**/*.trace.json'))
        
        # Remove duplicates (in case files match multiple patterns)
        trace_files = list(set(trace_files))
        
        # Exclude execution trace files (.et.json)
        trace_files = [f for f in trace_files if not f.name.endswith('.et.json')]
    
    if not trace_files:
        print(f"No trace files found in {args.trace_dir}")
        return
    
    print(f"Found {len(trace_files)} trace file(s)")
    
    # Parse all traces and preserve timestamps
    all_events = []
    for trace_file in trace_files:
        print(f"Parsing: {trace_file.relative_to(args.trace_dir)}")
        
        # Extract device ID from directory name (e.g., t4_gpu0 -> 0, t4_gpu1 -> 1)
        device_id_from_dir = None
        parent_dir = trace_file.parent.name
        
        # Try to extract from patterns like "t4_gpu0", "a100_gpu1", "gpu0", etc.
        match = re.search(r'gpu(\d+)', parent_dir.lower())
        if match:
            device_id_from_dir = int(match.group(1))
            print(f"  Inferred device ID {device_id_from_dir} from directory name: {parent_dir}")
        
        events = parse_trace_file(trace_file, fallback_device_id=device_id_from_dir)
        print(f"  Found {len(events)} GPU kernel events")
        
        # Show timestamp range for this trace
        if events:
            min_ts = min(e['ts_us'] for e in events)
            max_ts = max(e['ts_us'] for e in events)
            print(f"  Timestamp range: {min_ts:.2f} to {max_ts:.2f} μs")
        
        all_events.extend(events)
    
    print(f"Total events: {len(all_events)}")
    
    if not all_events:
        print("No events found to plot")
        return
    
    # Print statistics
    print_statistics(all_events)
    
    # Create plot
    print(f"\nCreating timeline plot...")
    create_timeline_plot(all_events, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

