"""Tools for analyzing profiling traces and identifying bottlenecks."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


def load_trace(trace_file: Path) -> dict:
    """Load Chrome trace JSON."""
    with open(trace_file, 'r') as f:
        return json.load(f)


def analyze_kv_transfer_latency(trace_data: dict) -> Dict[str, float]:
    """
    Analyze KV transfer latency from profiling trace.
    
    Args:
        trace_data: Chrome trace data
    
    Returns:
        Dictionary with transfer statistics
    """
    kv_events = []
    
    # Find KV transfer events
    for event in trace_data.get('traceEvents', []):
        name = event.get('name', '')
        if 'kv_broadcast' in name.lower() or 'kv_fork' in name.lower() or 'kv_receive' in name.lower():
            if event.get('ph') == 'X':  # Complete events
                duration_us = event.get('dur', 0)
                kv_events.append({
                    'name': name,
                    'duration_ms': duration_us / 1000.0,
                    'timestamp': event.get('ts', 0),
                })
    
    if not kv_events:
        return {'error': 'No KV transfer events found'}
    
    durations = [e['duration_ms'] for e in kv_events]
    
    return {
        'count': len(kv_events),
        'total_ms': sum(durations),
        'mean_ms': statistics.mean(durations),
        'median_ms': statistics.median(durations),
        'min_ms': min(durations),
        'max_ms': max(durations),
        'stdev_ms': statistics.stdev(durations) if len(durations) > 1 else 0.0,
    }


def analyze_prefill_throughput(trace_data: dict) -> Dict[str, float]:
    """
    Analyze prefill throughput from profiling trace.
    
    Args:
        trace_data: Chrome trace data
    
    Returns:
        Dictionary with prefill statistics
    """
    prefill_events = []
    
    # Find prefill events
    for event in trace_data.get('traceEvents', []):
        name = event.get('name', '')
        if 'prefill' in name.lower():
            if event.get('ph') == 'X':
                duration_us = event.get('dur', 0)
                prefill_events.append({
                    'name': name,
                    'duration_ms': duration_us / 1000.0,
                    'timestamp': event.get('ts', 0),
                })
    
    if not prefill_events:
        return {'error': 'No prefill events found'}
    
    durations = [e['duration_ms'] for e in prefill_events]
    
    return {
        'count': len(prefill_events),
        'total_ms': sum(durations),
        'mean_ms': statistics.mean(durations),
        'median_ms': statistics.median(durations),
        'min_ms': min(durations),
        'max_ms': max(durations),
        'tokens_per_second': None,  # Would need token count info
    }


def analyze_memory_usage(trace_data: dict) -> Dict[str, any]:
    """
    Analyze memory usage from profiling trace.
    
    Args:
        trace_data: Chrome trace data
    
    Returns:
        Dictionary with memory statistics
    """
    memory_events = []
    
    # Find memory allocation events
    for event in trace_data.get('traceEvents', []):
        if event.get('ph') == 'i':  # Instant events
            args = event.get('args', {})
            if 'Bytes' in str(args) or 'bytes' in str(args):
                memory_events.append(event)
    
    return {
        'total_events': len(memory_events),
        'note': 'Detailed memory analysis requires ExecutionTrace artifacts',
    }


def compare_gpu_performance(trace_dir: Path) -> Dict[str, Dict]:
    """
    Compare performance across different GPUs.
    
    Args:
        trace_dir: Directory containing profiling traces
    
    Returns:
        Dictionary comparing GPU performance
    """
    gpu_stats = {}
    
    for gpu_dir in trace_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        
        gpu_id = gpu_dir.name
        traces = list(gpu_dir.glob('*_trace.json'))
        
        if not traces:
            continue
        
        # Analyze first trace for this GPU
        trace_file = traces[0]
        trace_data = load_trace(trace_file)
        
        gpu_stats[gpu_id] = {
            'kv_transfer': analyze_kv_transfer_latency(trace_data),
            'prefill': analyze_prefill_throughput(trace_data),
            'num_traces': len(traces),
        }
    
    return gpu_stats


def print_analysis(analysis: dict, title: str):
    """Pretty print analysis results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)
    
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze profiling traces")
    parser.add_argument(
        '--trace-dir',
        type=Path,
        default='./profiling_traces',
        help='Directory containing profiling traces'
    )
    parser.add_argument(
        '--gpu-id',
        type=str,
        help='Analyze specific GPU (e.g., t4_gpu0)'
    )
    parser.add_argument(
        '--request-id',
        type=str,
        help='Analyze specific request'
    )
    
    args = parser.parse_args()
    
    if not args.trace_dir.exists():
        print(f"Error: Trace directory {args.trace_dir} not found")
        return
    
    if args.gpu_id and args.request_id:
        # Analyze specific trace
        trace_file = args.trace_dir / args.gpu_id / f"{args.request_id}_trace.json"
        
        if not trace_file.exists():
            print(f"Error: Trace file {trace_file} not found")
            return
        
        trace_data = load_trace(trace_file)
        
        kv_analysis = analyze_kv_transfer_latency(trace_data)
        print_analysis(kv_analysis, f"KV Transfer Analysis - {args.gpu_id}/{args.request_id}")
        
        prefill_analysis = analyze_prefill_throughput(trace_data)
        print_analysis(prefill_analysis, f"Prefill Analysis - {args.gpu_id}/{args.request_id}")
        
        memory_analysis = analyze_memory_usage(trace_data)
        print_analysis(memory_analysis, f"Memory Analysis - {args.gpu_id}/{args.request_id}")
    
    else:
        # Compare all GPUs
        comparison = compare_gpu_performance(args.trace_dir)
        
        if not comparison:
            print("No profiling traces found")
            return
        
        print("\n" + "="*80)
        print("GPU Performance Comparison")
        print("="*80)
        
        for gpu_id, stats in comparison.items():
            print(f"\n{gpu_id}:")
            print(f"  Traces analyzed: {stats['num_traces']}")
            
            kv = stats.get('kv_transfer', {})
            if 'mean_ms' in kv:
                print(f"  KV Transfer (avg): {kv['mean_ms']:.3f} ms")
            
            prefill = stats.get('prefill', {})
            if 'mean_ms' in prefill:
                print(f"  Prefill (avg): {prefill['mean_ms']:.3f} ms")


if __name__ == "__main__":
    main()

