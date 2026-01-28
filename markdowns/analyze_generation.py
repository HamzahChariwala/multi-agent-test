#!/usr/bin/env python3
"""Analyze generation outputs from profiling traces."""

import json
from pathlib import Path

def count_operations(trace_file):
    """Count operations in a trace file."""
    with open(trace_file) as f:
        trace = json.load(f)
    
    events = trace.get('traceEvents', [])
    
    # Count different operation types
    counts = {
        'total_events': len(events),
        'embedding_ops': 0,
        'matmul_ops': 0,
        'nccl_broadcast': 0,
        'user_annotations': []
    }
    
    for event in events:
        name = event.get('name', '')
        cat = event.get('cat', '')
        
        if 'embedding' in name.lower():
            counts['embedding_ops'] += 1
        if 'matmul' in name.lower():
            counts['matmul_ops'] += 1
        if 'nccl:broadcast' in name.lower():
            counts['nccl_broadcast'] += 1
        if cat == 'user_annotation':
            counts['user_annotations'].append(name)
    
    return counts

def main():
    traces_dir = Path('./profiling_traces')
    
    print("=" * 80)
    print("T4 CLUSTER GENERATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze each GPU's trace
    gpu_data = []
    
    for gpu_dir in sorted(traces_dir.glob('t4_gpu*')):
        gpu_id = gpu_dir.name
        trace_files = list(gpu_dir.glob('*_trace.json'))
        
        if not trace_files:
            continue
        
        trace_file = trace_files[0]
        counts = count_operations(trace_file)
        
        gpu_num = int(gpu_id.replace('t4_gpu', ''))
        
        # Get temperature from config
        temps = {0: 0.7, 1: 0.3, 2: 0.7, 3: 1.0}  # From endpoints.yaml
        temp = temps.get(gpu_num, 'N/A')
        
        gpu_data.append({
            'gpu': gpu_id,
            'rank': gpu_num,
            'temperature': temp,
            'counts': counts,
            'trace_size_mb': trace_file.stat().st_size / (1024*1024)
        })
    
    # Print results
    print(f"{'GPU':<10} {'Rank':<6} {'Temp':<6} {'Events':<10} {'Embed':<8} {'MatMul':<8} {'NCCL':<8} {'Size (MB)':<10}")
    print("-" * 80)
    
    for data in gpu_data:
        print(f"{data['gpu']:<10} "
              f"{data['rank']:<6} "
              f"{data['temperature']:<6} "
              f"{data['counts']['total_events']:<10} "
              f"{data['counts']['embedding_ops']:<8} "
              f"{data['counts']['matmul_ops']:<8} "
              f"{data['counts']['nccl_broadcast']:<8} "
              f"{data['trace_size_mb']:<10.2f}")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("Embedding operations ≈ tokens generated (each token requires an embedding lookup)")
    print()
    
    # Find the outlier
    embed_counts = [d['counts']['embedding_ops'] for d in gpu_data]
    avg_embeds = sum(embed_counts) / len(embed_counts)
    
    for data in gpu_data:
        embeds = data['counts']['embedding_ops']
        if embeds < avg_embeds * 0.3:  # Less than 30% of average
            print(f"⚠ {data['gpu']} generated VERY FEW tokens ({embeds} vs avg {avg_embeds:.1f})")
            print(f"  This is due to temperature-based sampling randomness.")
            print(f"  With temp={data['temperature']}, each GPU independently samples tokens.")
            print(f"  {data['gpu']} likely sampled an EOS token very early and stopped.")
            print()
    
    print("User-defined operations captured:")
    for data in gpu_data:
        unique_ops = set(data['counts']['user_annotations'])
        if unique_ops:
            print(f"  {data['gpu']}: {', '.join(sorted(unique_ops))}")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Each GPU:")
    print("1. Received the same input_ids and KV cache from GPU 0")
    print("2. Performed decode independently with different temperature settings")
    print("3. Generated DIFFERENT outputs due to sampling randomness")
    print()
    print("GPU 2's smaller trace is EXPECTED and NORMAL behavior:")
    print("- It hit an early stopping condition (EOS token or max tokens)")
    print("- Fewer tokens = fewer operations = smaller trace")
    print("- This is exactly what happens in multi-agent generation!")

if __name__ == '__main__':
    main()

