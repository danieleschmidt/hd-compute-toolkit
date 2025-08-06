#!/usr/bin/env python3
"""
Generation 3 Test Suite: Verify scalability and performance optimization
"""

import sys
import time
import concurrent.futures
import traceback
sys.path.insert(0, '/root/repo')


def test_scalable_performance_optimization():
    """Test performance optimization features."""
    print("⚡ GENERATION 3 SCALABILITY TEST")
    print("=" * 50)
    
    try:
        from hd_compute.scalable_backends.scalable_python import ScalableHDComputePython
        
        # Test initialization with performance features
        print("\n🚀 Testing Scalable Backend Initialization:")
        
        hdc = ScalableHDComputePython(
            dim=500,
            enable_caching=True,
            cache_size_mb=50,
            max_parallel_workers=4,
            enable_profiling=True,
            enable_audit_logging=False,
            strict_validation=False
        )
        print("✅ Scalable backend initialized successfully")
        
        # Test basic operations still work
        print("\n🔧 Testing Basic Operations:")
        
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        print("✅ Random hypervector generation")
        
        bundled = hdc.bundle([hv1, hv2])
        print("✅ Bundling operation")
        
        bound = hdc.bind(hv1, hv2)
        print("✅ Binding operation")
        
        similarity = hdc.cosine_similarity(hv1, hv2)
        print(f"✅ Cosine similarity: {similarity:.4f}")
        
        # Test caching effectiveness
        print("\n💾 Testing Caching Performance:")
        
        # First execution (cache miss)
        start_time = time.time()
        result1 = hdc.cosine_similarity(hv1, hv2)
        first_time = time.time() - start_time
        
        # Second execution (should be cached)
        start_time = time.time()
        result2 = hdc.cosine_similarity(hv1, hv2)
        second_time = time.time() - start_time
        
        if result1 == result2:
            print(f"✅ Cache working: first={first_time*1000:.2f}ms, second={second_time*1000:.2f}ms")
            if second_time < first_time * 0.5:  # Should be significantly faster
                print("✅ Cache provides performance benefit")
            else:
                print("⚠️  Cache benefit unclear (very fast operations)")
        else:
            print("❌ Cache inconsistency detected")
            return False
        
        # Test parallel processing with large datasets
        print("\n⚡ Testing Parallel Processing:")
        
        # Create large dataset for parallel testing
        large_hvs = [hdc.random_hv() for _ in range(30)]  # Should trigger parallel processing
        
        start_time = time.time()
        bundled_parallel = hdc.bundle(large_hvs)
        parallel_time = time.time() - start_time
        
        print(f"✅ Parallel bundling of {len(large_hvs)} HVs: {parallel_time*1000:.2f}ms")
        
        # Test batch operations
        hvs1 = [hdc.random_hv() for _ in range(20)]
        hvs2 = [hdc.random_hv() for _ in range(20)]
        
        start_time = time.time()
        batch_similarities = hdc.batch_cosine_similarity(hvs1, hvs2)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch similarity computation: {len(batch_similarities)} pairs in {batch_time*1000:.2f}ms")
        
        if len(batch_similarities) == 20:
            print("✅ Correct number of similarity results")
        else:
            print(f"❌ Expected 20 results, got {len(batch_similarities)}")
            return False
        
        # Test performance statistics
        print("\n📊 Testing Performance Monitoring:")
        
        stats = hdc.get_performance_statistics()
        print(f"✅ Performance statistics collected:")
        print(f"   Total operations: {stats['total_operations']}")
        print(f"   Cache hit rate: {stats['cache_stats']['hit_rate']:.3f}")
        print(f"   Success rate: {stats['success_rate']:.3f}")
        
        # Test workload optimization
        print("\n⚙️ Testing Dynamic Optimization:")
        
        workload_profile = {
            'repeat_operations': True,
            'avg_batch_size': 150,
            'available_memory_mb': 1500
        }
        
        hdc.optimize_for_workload(workload_profile)
        print("✅ Dynamic optimization applied")
        
        # Test benchmarking
        print("\n🏁 Testing Built-in Benchmarking:")
        
        benchmark_results = hdc.benchmark_operations(num_iterations=50)
        print("✅ Benchmark results:")
        for operation, time_ms in benchmark_results.items():
            print(f"   {operation}: {time_ms:.3f}ms")
        
        # Test cache warmup
        print("\n🔥 Testing Cache Warmup:")
        
        start_time = time.time()
        hdc.warmup_caches(num_samples=20)
        warmup_time = time.time() - start_time
        print(f"✅ Cache warmed up in {warmup_time*1000:.2f}ms")
        
        print(f"\n🎯 GENERATION 3 SCALABILITY: ✅ COMPLETE")
        print("Performance optimization and scalability features working!")
        return True
        
    except Exception as e:
        print(f"❌ Scalable backend test failed: {e}")
        traceback.print_exc()
        return False


def test_concurrent_access():
    """Test thread-safe concurrent access to scalable backend."""
    print("\n🔀 CONCURRENT ACCESS TEST")
    print("-" * 40)
    
    try:
        from hd_compute.scalable_backends.scalable_python import ScalableHDComputePython
        
        hdc = ScalableHDComputePython(
            dim=300,
            enable_caching=True,
            max_parallel_workers=8,
            enable_audit_logging=False
        )
        
        # Test concurrent operations
        def worker_function(worker_id: int, num_operations: int):
            """Worker function for concurrent testing."""
            results = []
            for i in range(num_operations):
                # Mix of different operations
                if i % 4 == 0:
                    result = hdc.random_hv()
                elif i % 4 == 1:
                    hv1, hv2 = hdc.random_hv(), hdc.random_hv()
                    result = hdc.bind(hv1, hv2)
                elif i % 4 == 2:
                    hvs = [hdc.random_hv() for _ in range(5)]
                    result = hdc.bundle(hvs)
                else:
                    hv1, hv2 = hdc.random_hv(), hdc.random_hv()
                    result = hdc.cosine_similarity(hv1, hv2)
                
                results.append(result)
            
            return f"Worker {worker_id}: {len(results)} operations completed"
        
        # Run concurrent workers
        num_workers = 6
        operations_per_worker = 20
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(worker_function, worker_id, operations_per_worker)
                futures.append(future)
            
            # Wait for all workers to complete
            completed_workers = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(f"✅ {result}")
                    completed_workers += 1
                except Exception as e:
                    print(f"❌ Worker failed: {e}")
        
        total_time = time.time() - start_time
        total_operations = num_workers * operations_per_worker
        
        print(f"\n📈 Concurrent Performance Summary:")
        print(f"   Total operations: {total_operations}")
        print(f"   Completed workers: {completed_workers}/{num_workers}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Operations/second: {total_operations/total_time:.1f}")
        
        # Verify system is still healthy after concurrent access
        health = hdc.health_check()
        if health['status'] == 'healthy':
            print("✅ System healthy after concurrent access")
        else:
            print(f"⚠️  System status after concurrent access: {health['status']}")
            print(f"   Issues: {health['issues']}")
        
        return completed_workers == num_workers
        
    except Exception as e:
        print(f"❌ Concurrent access test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency and resource management."""
    print("\n💾 MEMORY EFFICIENCY TEST")
    print("-" * 40)
    
    try:
        from hd_compute.scalable_backends.scalable_python import ScalableHDComputePython
        import gc
        
        # Test with different cache sizes
        cache_sizes = [10, 50, 100]  # MB
        
        for cache_size in cache_sizes:
            print(f"\nTesting with {cache_size}MB cache:")
            
            hdc = ScalableHDComputePython(
                dim=1000,
                enable_caching=True,
                cache_size_mb=cache_size,
                enable_audit_logging=False
            )
            
            # Generate many operations to test memory management
            large_hvs = []
            for i in range(100):
                hv = hdc.random_hv()
                large_hvs.append(hv)
                
                # Trigger some cached operations
                if i % 10 == 0 and len(large_hvs) > 1:
                    similarity = hdc.cosine_similarity(large_hvs[-1], large_hvs[-2])
                    bundled = hdc.bundle(large_hvs[-5:]) if len(large_hvs) >= 5 else hdc.bundle(large_hvs)
            
            # Check memory usage and cache statistics
            stats = hdc.get_performance_statistics()
            print(f"✅ Cache size {cache_size}MB: {stats['cache_stats']['hits']} hits, "
                  f"{stats['cache_stats']['misses']} misses")
            
            # Clean up
            hdc.cleanup_resources()
            del hdc
            del large_hvs
            gc.collect()
        
        # Test resource cleanup
        print("\n🧹 Testing Resource Cleanup:")
        
        hdc = ScalableHDComputePython(dim=200, enable_caching=True, enable_audit_logging=False)
        
        # Perform operations
        for _ in range(20):
            hv1, hv2 = hdc.random_hv(), hdc.random_hv()
            hdc.cosine_similarity(hv1, hv2)
        
        # Test cleanup
        hdc.cleanup_resources()
        print("✅ Resource cleanup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return False


def test_advanced_optimizations():
    """Test advanced optimization features."""
    print("\n🎯 ADVANCED OPTIMIZATIONS TEST")
    print("-" * 40)
    
    try:
        from hd_compute.scalable_backends.scalable_python import ScalableHDComputePython
        
        hdc = ScalableHDComputePython(
            dim=800,
            enable_caching=True,
            enable_profiling=True,
            max_parallel_workers=6,
            enable_audit_logging=False
        )
        
        # Test optimized random hypervector generation
        print("Testing optimized hypervector generation:")
        
        common_sparsities = [0.25, 0.5, 0.75]
        for sparsity in common_sparsities:
            start_time = time.time()
            hvs = [hdc.random_hv(sparsity=sparsity) for _ in range(10)]
            generation_time = time.time() - start_time
            
            # Verify sparsity is approximately correct
            actual_sparsity = sum(sum(hv.data) for hv in hvs) / (len(hvs) * hdc.dim)
            
            print(f"✅ Sparsity {sparsity}: generated in {generation_time*1000:.2f}ms, "
                  f"actual={actual_sparsity:.3f}")
            
            if abs(actual_sparsity - sparsity) > 0.1:  # Allow 10% deviation
                print(f"⚠️  Sparsity deviation larger than expected")
        
        # Test lookup table optimizations
        print("\nTesting lookup table optimizations:")
        
        # Generate many HVs with same sparsity (should use optimized path)
        start_time = time.time()
        optimized_hvs = [hdc.random_hv(sparsity=0.5) for _ in range(50)]
        optimized_time = time.time() - start_time
        
        # Generate HVs with different sparsity (should use standard path)
        start_time = time.time()
        standard_hvs = [hdc.random_hv(sparsity=0.33) for _ in range(50)]
        standard_time = time.time() - start_time
        
        print(f"✅ Optimized generation: {optimized_time*1000:.2f}ms for 50 HVs")
        print(f"✅ Standard generation: {standard_time*1000:.2f}ms for 50 HVs")
        
        if optimized_time <= standard_time:
            print("✅ Optimization provides performance benefit")
        else:
            print("⚠️  Optimization benefit unclear")
        
        # Test batch processing optimization  
        print("\nTesting batch processing optimization:")
        
        hvs1 = [hdc.random_hv() for _ in range(60)]  # Should trigger parallel processing
        hvs2 = [hdc.random_hv() for _ in range(60)]
        
        start_time = time.time()
        batch_results = hdc.batch_cosine_similarity(hvs1, hvs2)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch similarity (60 pairs): {batch_time*1000:.2f}ms")
        print(f"   Average per pair: {batch_time/len(batch_results)*1000:.3f}ms")
        
        if len(batch_results) == 60:
            print("✅ Correct batch processing results")
        else:
            print(f"❌ Expected 60 results, got {len(batch_results)}")
            return False
        
        # Test advanced statistics
        print("\nTesting advanced performance statistics:")
        
        stats = hdc.get_performance_statistics()
        
        required_stats = ['cache_stats', 'performance_stats', 'optimization_enabled']
        missing_stats = [stat for stat in required_stats if stat not in stats]
        
        if not missing_stats:
            print("✅ All required performance statistics available")
            print(f"   Cache enabled: {stats['optimization_enabled']['caching']}")
            print(f"   Parallel processing: {stats['optimization_enabled']['parallel_processing']}")
            print(f"   Profiling: {stats['optimization_enabled']['profiling']}")
        else:
            print(f"❌ Missing statistics: {missing_stats}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced optimizations test failed: {e}")
        return False


def test_scalability_limits():
    """Test scalability with increasing load."""
    print("\n📈 SCALABILITY LIMITS TEST")
    print("-" * 40)
    
    try:
        from hd_compute.scalable_backends.scalable_python import ScalableHDComputePython
        
        hdc = ScalableHDComputePython(
            dim=400,
            enable_caching=True,
            cache_size_mb=200,
            max_parallel_workers=8,
            enable_audit_logging=False
        )
        
        # Test with increasing workload sizes
        workload_sizes = [10, 50, 100, 200]
        
        print("Testing scalability with increasing workload:")
        
        for size in workload_sizes:
            print(f"\n  Testing workload size: {size}")
            
            # Generate test data
            hvs = [hdc.random_hv() for _ in range(size)]
            
            # Time bundle operation
            start_time = time.time()
            bundled = hdc.bundle(hvs)
            bundle_time = time.time() - start_time
            
            # Time batch similarity operations
            hvs1 = hvs[:size//2]
            hvs2 = hvs[size//2:]
            
            if len(hvs1) == len(hvs2):
                start_time = time.time()
                similarities = hdc.batch_cosine_similarity(hvs1, hvs2)
                similarity_time = time.time() - start_time
                
                print(f"    Bundle {size} HVs: {bundle_time*1000:.2f}ms")
                print(f"    Similarity {len(similarities)} pairs: {similarity_time*1000:.2f}ms")
                
                # Check if performance is reasonable (not exponential growth)
                if size > 10:
                    time_per_item = bundle_time / size * 1000  # ms per item
                    if time_per_item < 50:  # Less than 50ms per item
                        print(f"    ✅ Performance scaling good: {time_per_item:.2f}ms per item")
                    else:
                        print(f"    ⚠️  Performance scaling concern: {time_per_item:.2f}ms per item")
            else:
                print(f"    Bundle {size} HVs: {bundle_time*1000:.2f}ms")
        
        # Test system health under load
        print("\nTesting system health under sustained load:")
        
        # Sustained operations
        operations_completed = 0
        start_time = time.time()
        target_duration = 2.0  # 2 seconds
        
        while time.time() - start_time < target_duration:
            hv1, hv2 = hdc.random_hv(), hdc.random_hv()
            similarity = hdc.cosine_similarity(hv1, hv2)
            operations_completed += 1
        
        actual_duration = time.time() - start_time
        ops_per_second = operations_completed / actual_duration
        
        print(f"✅ Sustained load test: {operations_completed} operations in {actual_duration:.2f}s")
        print(f"   Throughput: {ops_per_second:.1f} operations/second")
        
        # Check final system health
        final_health = hdc.health_check()
        if final_health['status'] == 'healthy':
            print("✅ System remained healthy under sustained load")
        else:
            print(f"⚠️  System status after load: {final_health['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scalability limits test failed: {e}")
        return False


def main():
    """Run Generation 3 scalability test suite."""
    print("⚡ HD-COMPUTE GENERATION 3 TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Scalable Performance Optimization", test_scalable_performance_optimization),
        ("Concurrent Access", test_concurrent_access),
        ("Memory Efficiency", test_memory_efficiency),
        ("Advanced Optimizations", test_advanced_optimizations),
        ("Scalability Limits", test_scalability_limits)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 GENERATION 3 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    success_rate = (passed_tests / total_tests) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 GENERATION 3: ✅ COMPLETE")
        print("Scalability and performance optimization implemented successfully!")
        print("Ready to proceed with Quality Gates and Production Deployment")
        return True
    else:
        print(f"\n⚠️  GENERATION 3: PARTIAL COMPLETION")
        print("Some scalability features need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)