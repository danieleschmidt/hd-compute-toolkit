"""Benchmark command-line interface."""

import argparse
import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Dict, Any

from ..torch import HDComputeTorch
from ..jax import HDComputeJAX


def benchmark_pytorch(dim: int, device: str, iterations: int = 1000) -> Dict[str, float]:
    """Benchmark PyTorch HDC operations."""
    hdc = HDComputeTorch(dim=dim, device=device)
    results = {}
    
    # Random hypervector generation
    start_time = time.time()
    for _ in range(iterations):
        hv = hdc.random_hv()
    results['random_hv_ms'] = (time.time() - start_time) * 1000 / iterations
    
    # Bundle operation
    hvs = [hdc.random_hv() for _ in range(1000)]
    start_time = time.time()
    for _ in range(iterations // 10):  # Fewer iterations for bundling
        bundled = hdc.bundle(hvs)
    results['bundle_1000_ms'] = (time.time() - start_time) * 1000 / (iterations // 10)
    
    # Bind operation
    hv1 = hdc.random_hv()
    hv2 = hdc.random_hv()
    start_time = time.time()
    for _ in range(iterations):
        bound = hdc.bind(hv1, hv2)
    results['bind_ms'] = (time.time() - start_time) * 1000 / iterations
    
    # Hamming distance
    start_time = time.time()
    for _ in range(iterations):
        dist = hdc.hamming_distance(hv1, hv2)
    results['hamming_distance_ms'] = (time.time() - start_time) * 1000 / iterations
    
    return results


def benchmark_jax(dim: int, iterations: int = 1000) -> Dict[str, float]:
    """Benchmark JAX HDC operations."""
    hdc = HDComputeJAX(dim=dim)
    results = {}
    
    # Random hypervector generation
    start_time = time.time()
    for _ in range(iterations):
        hv = hdc.random_hv()
    results['random_hv_ms'] = (time.time() - start_time) * 1000 / iterations
    
    # Bundle operation
    hvs = jnp.stack([hdc.random_hv() for _ in range(1000)])
    start_time = time.time()
    for _ in range(iterations // 10):
        bundled = hdc.bundle(hvs)
    results['bundle_1000_ms'] = (time.time() - start_time) * 1000 / (iterations // 10)
    
    # Bind operation
    hv1 = hdc.random_hv()
    hv2 = hdc.random_hv()
    start_time = time.time()
    for _ in range(iterations):
        bound = hdc.bind(hv1, hv2)
    results['bind_ms'] = (time.time() - start_time) * 1000 / iterations
    
    # Hamming distance
    start_time = time.time()
    for _ in range(iterations):
        dist = hdc.hamming_distance(hv1, hv2)
    results['hamming_distance_ms'] = (time.time() - start_time) * 1000 / iterations
    
    return results


def benchmark():
    """Main benchmark CLI function."""
    parser = argparse.ArgumentParser(description='Benchmark HD-Compute-Toolkit performance')
    parser.add_argument('--dim', type=int, default=10000, help='Hypervector dimension')
    parser.add_argument('--device', type=str, default='auto', help='Computing device')
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'jax', 'both'],
                       help='Backend to benchmark')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    
    args = parser.parse_args()
    
    print(f"HD-Compute-Toolkit Benchmark")
    print(f"Dimension: {args.dim}")
    print(f"Iterations: {args.iterations}")
    print("-" * 50)
    
    if args.backend in ['pytorch', 'both']:
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\nPyTorch Backend (device: {device})")
        print("-" * 30)
        
        try:
            pytorch_results = benchmark_pytorch(args.dim, device, args.iterations)
            for operation, time_ms in pytorch_results.items():
                print(f"{operation:20}: {time_ms:.3f} ms")
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
    
    if args.backend in ['jax', 'both']:
        print(f"\nJAX Backend")
        print("-" * 30)
        
        try:
            jax_results = benchmark_jax(args.dim, args.iterations)
            for operation, time_ms in jax_results.items():
                print(f"{operation:20}: {time_ms:.3f} ms")
        except Exception as e:
            print(f"JAX benchmark failed: {e}")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    benchmark()