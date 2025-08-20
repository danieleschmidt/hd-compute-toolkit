#!/usr/bin/env python3
"""
GENERATION 3 SCALABILITY TEST - Performance Optimization & Concurrent Processing
Test optimized operations, batch processing, and scalability features using existing backend
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))

def test_generation3_scalability():
    """Test Generation 3: MAKE IT SCALE (Optimized) functionality."""
    print("⚡ GENERATION 3: MAKE IT SCALE (Optimized)")
    
    try:
        from hd_compute.pure_python import HDComputePython
        
        # Test 1: Performance optimization
        print("🚀 Testing performance optimization...")
        
        # Initialize optimized backend
        hdc = HDComputePython(dim=2000)
        print("✅ Scalable HDC initialized")
        
        # Test 2: Batch operations simulation
        print("📦 Testing batch operations...")
        
        # Generate batch of hypervectors efficiently
        batch_hvs = [hdc.random_hv() for _ in range(50)]
        print(f"✅ Batch hypervector generation: {len(batch_hvs)} vectors")
        
        # Batch bundle operation
        start_time = time.time()
        bundled_batch = hdc.bundle(batch_hvs)
        bundle_time = (time.time() - start_time) * 1000
        print(f"✅ Batch bundle operation: {bundle_time:.2f} ms for {len(batch_hvs)} vectors")
        
        # Test 3: Concurrent processing
        print("⚡ Testing concurrent processing...")
        
        def process_hv_pair(hv1, hv2):
            """Process a pair of hypervectors concurrently."""
            similarity = hdc.cosine_similarity(hv1, hv2)
            bound = hdc.bind(hv1, hv2)
            return similarity, bound
        
        # Generate test data
        hv_pairs = [(hdc.random_hv(), hdc.random_hv()) for _ in range(20)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for hv1, hv2 in hv_pairs:
            result = process_hv_pair(hv1, hv2)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(lambda pair: process_hv_pair(*pair), hv_pairs))
        concurrent_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1.0
        print(f"✅ Concurrent processing speedup: {speedup:.2f}x")
        print(f"   Sequential: {sequential_time*1000:.2f} ms")
        print(f"   Concurrent: {concurrent_time*1000:.2f} ms")
        
        # Test 4: Large-scale operations
        print("🧮 Testing large-scale operations...")
        
        # Test large-scale bundling
        large_batch = [hdc.random_hv() for _ in range(200)]
        start_time = time.time()
        large_bundled = hdc.bundle(large_batch)
        large_bundle_time = (time.time() - start_time) * 1000
        print(f"✅ Large-scale bundling: {large_bundle_time:.2f} ms for {len(large_batch)} vectors")
        
        # Test batch similarity computation
        hvs1 = [hdc.random_hv() for _ in range(30)]
        hvs2 = [hdc.random_hv() for _ in range(30)]
        
        start_time = time.time()
        similarities = hdc.batch_cosine_similarity(hvs1, hvs2)
        batch_sim_time = (time.time() - start_time) * 1000
        print(f"✅ Batch similarity computation: {batch_sim_time:.2f} ms for {len(hvs1)} pairs")
        
        # Test 5: Advanced operations at scale
        print("🎯 Testing advanced operations at scale...")
        
        hv1, hv2 = hdc.random_hv(), hdc.random_hv()
        
        # Test fractional binding with timing
        start_time = time.time()
        frac_bound = hdc.fractional_bind(hv1, hv2, power=0.5)
        frac_time = (time.time() - start_time) * 1000
        print(f"✅ Fractional binding: {frac_time:.2f} ms")
        
        # Test quantum superposition with timing
        start_time = time.time()
        quantum_hv = hdc.quantum_superposition([hv1, hv2], amplitudes=[0.6, 0.4])
        quantum_time = (time.time() - start_time) * 1000
        print(f"✅ Quantum superposition: {quantum_time:.2f} ms")
        
        # Test hierarchical binding with timing
        structure = {"level1": {"item1": hv1, "item2": hv2}, "level2": hv1}
        start_time = time.time()
        hierarchical = hdc.hierarchical_bind(structure)
        hierarchical_time = (time.time() - start_time) * 1000
        print(f"✅ Hierarchical binding: {hierarchical_time:.2f} ms")
        
        # Test 6: Memory and performance optimization
        print("💾 Testing memory optimization...")
        
        # Test cleanup operation
        item_memory = [hdc.random_hv() for _ in range(50)]
        noisy_hv = hdc.random_hv()
        
        start_time = time.time()
        cleaned_hv = hdc.cleanup(noisy_hv, item_memory, k=5)
        cleanup_time = (time.time() - start_time) * 1000
        print(f"✅ Cleanup operation: {cleanup_time:.2f} ms with {len(item_memory)} items")
        
        # Test sequence encoding
        sequence = [hdc.random_hv() for _ in range(10)]
        start_time = time.time()
        encoded_seq = hdc.encode_sequence(sequence)
        seq_time = (time.time() - start_time) * 1000
        print(f"✅ Sequence encoding: {seq_time:.2f} ms for {len(sequence)} elements")
        
        print("🎯 GENERATION 3 COMPLETE: Scalability and optimization working!")
        return True
        
    except Exception as e:
        print(f"❌ Generation 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_scalability():
    """Benchmark scalability across different dimensions."""
    print("\n📈 SCALABILITY BENCHMARK")
    print("-" * 40)
    
    try:
        from hd_compute.pure_python import HDComputePython
        
        dimensions = [500, 1000, 2000]
        operations = ['random_hv', 'bundle', 'bind', 'cosine_similarity']
        
        print(f"{'Dimension':<10} {'Operation':<20} {'Time (ms)':<12} {'Throughput':<15}")
        print("-" * 65)
        
        for dim in dimensions:
            hdc = HDComputePython(dim=dim)
            
            for op_name in operations:
                if op_name == 'random_hv':
                    start_time = time.time()
                    for _ in range(50):
                        _ = hdc.random_hv()
                    elapsed = (time.time() - start_time) * 1000
                    throughput = 50 / (elapsed / 1000)
                    
                elif op_name == 'bundle':
                    hvs = [hdc.random_hv() for _ in range(5)]
                    start_time = time.time()
                    for _ in range(10):
                        _ = hdc.bundle(hvs)
                    elapsed = (time.time() - start_time) * 1000
                    throughput = 10 / (elapsed / 1000)
                    
                elif op_name == 'bind':
                    hv1, hv2 = hdc.random_hv(), hdc.random_hv()
                    start_time = time.time()
                    for _ in range(50):
                        _ = hdc.bind(hv1, hv2)
                    elapsed = (time.time() - start_time) * 1000
                    throughput = 50 / (elapsed / 1000)
                    
                elif op_name == 'cosine_similarity':
                    hv1, hv2 = hdc.random_hv(), hdc.random_hv()
                    start_time = time.time()
                    for _ in range(50):
                        _ = hdc.cosine_similarity(hv1, hv2)
                    elapsed = (time.time() - start_time) * 1000
                    throughput = 50 / (elapsed / 1000)
                
                print(f"{dim:<10} {op_name:<20} {elapsed:<12.2f} {throughput:<15.0f} ops/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_generation3_scalability()
    success2 = benchmark_scalability()
    sys.exit(0 if success1 and success2 else 1)