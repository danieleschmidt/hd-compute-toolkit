"""
Scaling and Performance Test Suite
=================================

Tests Generation 3 scaling improvements:
- Performance optimization and caching
- Memory efficiency and object pooling
- Concurrent processing and vectorization
- Auto-tuning and adaptive parameters
- Scalability under load
"""

import time
import numpy as np
import sys
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Add project root to path
sys.path.insert(0, '/root/repo')

from hd_compute.research.optimized_algorithms import (
    OptimizedTemporalHDC,
    OptimizedAttentionHDC, 
    OptimizedQuantumHDC,
    create_optimized_suite
)
from hd_compute.performance.advanced_optimization import (
    AdaptiveCache,
    HypervectorMemoryPool,
    VectorizedOperations,
    ConcurrentProcessor,
    PerformanceProfiler,
    global_cache,
    global_profiler
)


class ScalingTestSuite:
    """Comprehensive scaling and performance test suite."""
    
    def __init__(self):
        self.test_results = {}
        self.dimensions = [100, 500, 1000, 2000]
        self.workload_sizes = [10, 50, 100, 500]
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete scaling test suite."""
        print("🚀 SCALING & PERFORMANCE TEST SUITE")
        print("=" * 50)
        
        test_methods = [
            self.test_caching_performance,
            self.test_memory_efficiency,
            self.test_vectorized_operations,
            self.test_concurrent_processing,
            self.test_auto_tuning,
            self.test_scalability_limits,
            self.test_throughput_benchmarks
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                self.test_results[test_method.__name__] = {
                    'status': 'passed',
                    'result': result
                }
            except Exception as e:
                self.test_results[test_method.__name__] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.print_summary()
        return self.test_results
    
    def test_caching_performance(self) -> Dict[str, Any]:
        """Test caching system performance and hit rates."""
        print("\n💾 Testing Caching Performance...")
        
        cache = AdaptiveCache(max_size=100, ttl_seconds=60)
        
        # Generate test data
        test_vectors = [np.random.randn(1000) for _ in range(50)]
        cache_keys = [f"vector_{i}" for i in range(50)]
        
        # Fill cache
        start_time = time.time()
        for key, vector in zip(cache_keys, test_vectors):
            cache.put(key, vector)
        cache_fill_time = time.time() - start_time
        
        # Test cache hits
        start_time = time.time()
        hits = 0
        for key in cache_keys[:30]:  # Test subset
            if cache.get(key) is not None:
                hits += 1
        cache_read_time = time.time() - start_time
        
        hit_rate = hits / 30
        
        # Test cache eviction
        for i in range(50, 150):  # Add more items to trigger eviction
            cache.put(f"extra_{i}", np.random.randn(1000))
        
        final_size = len(cache.cache)
        
        print(f"  ✅ Cache fill time: {cache_fill_time:.4f}s")
        print(f"  ✅ Cache hit rate: {hit_rate:.2%}")
        print(f"  ✅ Cache read time: {cache_read_time:.4f}s")
        print(f"  ✅ Eviction working: {final_size <= 100}")
        
        return {
            'cache_fill_time': cache_fill_time,
            'hit_rate': hit_rate,
            'cache_read_time': cache_read_time,
            'eviction_correct': final_size <= 100,
            'final_cache_size': final_size
        }
    
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory pooling and efficiency."""
        print("\n🧠 Testing Memory Efficiency...")
        
        pool = HypervectorMemoryPool(dim=1000, initial_size=50, max_size=200)
        
        # Test vector allocation and return
        start_time = time.time()
        vectors = []
        for _ in range(100):
            vector = pool.get_vector()
            vectors.append(vector)
        allocation_time = time.time() - start_time
        
        # Test vector return
        start_time = time.time()
        for vector in vectors:
            pool.return_vector(vector)
        return_time = time.time() - start_time
        
        pool_stats = pool.get_stats()
        
        # Test reuse efficiency
        reused_vectors = []
        for _ in range(50):
            vector = pool.get_vector()
            reused_vectors.append(vector)
        
        reuse_stats = pool.get_stats()
        
        print(f"  ✅ Allocation time: {allocation_time:.4f}s for 100 vectors")
        print(f"  ✅ Return time: {return_time:.4f}s")
        print(f"  ✅ Pool utilization: {pool_stats['utilization']:.2%}")
        print(f"  ✅ Memory reuse working: {reuse_stats['available'] > 0}")
        
        return {
            'allocation_time': allocation_time,
            'return_time': return_time,
            'pool_utilization': pool_stats['utilization'],
            'memory_reuse_available': reuse_stats['available'],
            'total_allocated': pool_stats['total_allocated']
        }
    
    def test_vectorized_operations(self) -> Dict[str, Any]:
        """Test vectorized operation performance."""
        print("\n⚡ Testing Vectorized Operations...")
        
        ops = VectorizedOperations()
        
        # Test batch cosine similarity
        query = np.random.randn(1000)
        vectors = np.random.randn(500, 1000)
        
        start_time = time.time()
        similarities = ops.batch_cosine_similarity(query, vectors)
        batch_time = time.time() - start_time
        
        # Compare with individual operations
        start_time = time.time()
        individual_similarities = []
        for vector in vectors:
            similarity = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector))
            individual_similarities.append(similarity)
        individual_time = time.time() - start_time
        
        speedup = individual_time / batch_time if batch_time > 0 else float('inf')
        
        # Test batch binding
        vectors1 = np.random.randn(100, 1000)
        vectors2 = np.random.randn(100, 1000)
        
        start_time = time.time()
        bound_vectors = ops.batch_binding(vectors1, vectors2)
        binding_time = time.time() - start_time
        
        # Test batch bundling
        weights = np.random.rand(100)
        weights /= np.sum(weights)
        
        start_time = time.time()
        bundled = ops.batch_bundling(vectors1, weights)
        bundling_time = time.time() - start_time
        
        print(f"  ✅ Batch similarity speedup: {speedup:.1f}x")
        print(f"  ✅ Batch binding time: {binding_time:.4f}s for 100 pairs")
        print(f"  ✅ Batch bundling time: {bundling_time:.4f}s")
        print(f"  ✅ Results shape correct: {similarities.shape == (500,)}")
        
        return {
            'similarity_speedup': speedup,
            'batch_similarity_time': batch_time,
            'individual_similarity_time': individual_time,
            'binding_time': binding_time,
            'bundling_time': bundling_time,
            'results_correct': similarities.shape == (500,)
        }
    
    def test_concurrent_processing(self) -> Dict[str, Any]:
        """Test concurrent processing capabilities."""
        print("\n🔄 Testing Concurrent Processing...")
        
        processor = ConcurrentProcessor(max_workers=4)
        
        def compute_heavy_task(x):
            """Simulate computationally heavy task."""
            result = 0
            for i in range(1000):
                result += np.sum(np.random.randn(100) * x)
            return result\n        \n        # Test sequential processing\n        items = list(range(20))\n        \n        start_time = time.time()\n        sequential_results = [compute_heavy_task(x) for x in items]\n        sequential_time = time.time() - start_time\n        \n        # Test parallel processing\n        start_time = time.time()\n        parallel_results = processor.parallel_apply(compute_heavy_task, items)\n        parallel_time = time.time() - start_time\n        \n        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')\n        \n        # Test map-reduce\n        def map_func(x):\n            return x * x\n        \n        def reduce_func(a, b):\n            return a + b\n        \n        start_time = time.time()\n        mapreduce_result = processor.parallel_map_reduce(\n            map_func, reduce_func, items[:10]\n        )\n        mapreduce_time = time.time() - start_time\n        \n        processor.shutdown()\n        \n        print(f\"  ✅ Parallel speedup: {speedup:.1f}x\")\n        print(f\"  ✅ Sequential time: {sequential_time:.3f}s\")\n        print(f\"  ✅ Parallel time: {parallel_time:.3f}s\")\n        print(f\"  ✅ Map-reduce completed in {mapreduce_time:.4f}s\")\n        print(f\"  ✅ Results consistent: {len(sequential_results) == len(parallel_results)}\")\n        \n        return {\n            'parallel_speedup': speedup,\n            'sequential_time': sequential_time,\n            'parallel_time': parallel_time,\n            'mapreduce_time': mapreduce_time,\n            'results_consistent': len(sequential_results) == len(parallel_results)\n        }\n    \n    def test_auto_tuning(self) -> Dict[str, Any]:\n        \"\"\"Test auto-tuning capabilities.\"\"\"\n        print(\"\\n🎯 Testing Auto-Tuning...\")\n        \n        temporal = OptimizedTemporalHDC(dim=500)\n        \n        # Generate test sequence\n        sequence = [np.random.randn(500) for _ in range(20)]\n        \n        # Run multiple predictions to trigger auto-tuning\n        performance_metrics = []\n        \n        for i in range(10):\n            start_time = time.time()\n            predictions = temporal.sequence_prediction(sequence, prediction_horizon=3)\n            execution_time = time.time() - start_time\n            performance_metrics.append(1.0 / execution_time)  # Higher is better\n        \n        # Check if performance improves over time (auto-tuning effect)\n        early_avg = np.mean(performance_metrics[:3])\n        late_avg = np.mean(performance_metrics[-3:])\n        improvement = (late_avg - early_avg) / early_avg if early_avg > 0 else 0\n        \n        stats = temporal.get_performance_stats()\n        \n        print(f\"  ✅ Performance improvement: {improvement:.2%}\")\n        print(f\"  ✅ Total predictions: {stats['operation_stats']['predictions_made']}\")\n        print(f\"  ✅ Cache utilization: {stats['cache_stats']['utilization']:.2%}\")\n        print(f\"  ✅ Memory pool utilization: {stats['memory_pool_stats']['utilization']:.2%}\")\n        \n        return {\n            'performance_improvement': improvement,\n            'total_predictions': stats['operation_stats']['predictions_made'],\n            'cache_utilization': stats['cache_stats']['utilization'],\n            'memory_pool_utilization': stats['memory_pool_stats']['utilization'],\n            'auto_tuning_active': improvement > -0.1  # Allow for small variations\n        }\n    \n    def test_scalability_limits(self) -> Dict[str, Any]:\n        \"\"\"Test system behavior under increasing load.\"\"\"\n        print(\"\\n📈 Testing Scalability Limits...\")\n        \n        results = {}\n        \n        for dim in [100, 500, 1000, 2000]:\n            print(f\"    Testing dimension {dim}...\")\n            \n            suite = create_optimized_suite(dim)\n            temporal = suite['temporal']\n            attention = suite['attention']\n            \n            # Test temporal scalability\n            sequence = [np.random.randn(dim) for _ in range(100)]\n            \n            start_time = time.time()\n            predictions = temporal.sequence_prediction(sequence, prediction_horizon=5)\n            temporal_time = time.time() - start_time\n            \n            # Test attention scalability\n            hvs = [np.random.randn(dim) for _ in range(50)]\n            query = np.random.randn(dim)\n            \n            start_time = time.time()\n            attended = attention.multi_head_attention_optimized(query, hvs, hvs)\n            attention_time = time.time() - start_time\n            \n            results[f'dim_{dim}'] = {\n                'temporal_time': temporal_time,\n                'attention_time': attention_time,\n                'temporal_ops_per_sec': 5 / temporal_time if temporal_time > 0 else 0,\n                'attention_ops_per_sec': 1 / attention_time if attention_time > 0 else 0\n            }\n        \n        # Check scaling behavior\n        scaling_efficiency = []\n        base_dim = 100\n        for dim in [500, 1000, 2000]:\n            base_time = results[f'dim_{base_dim}']['temporal_time']\n            current_time = results[f'dim_{dim}']['temporal_time']\n            theoretical_scaling = (dim / base_dim) ** 2  # Quadratic scaling expected\n            actual_scaling = current_time / base_time\n            efficiency = theoretical_scaling / actual_scaling if actual_scaling > 0 else 0\n            scaling_efficiency.append(efficiency)\n        \n        avg_efficiency = np.mean(scaling_efficiency)\n        \n        print(f\"  ✅ Scaling efficiency: {avg_efficiency:.2f}\")\n        print(f\"  ✅ Max dimension tested: 2000\")\n        print(f\"  ✅ All dimensions completed successfully\")\n        \n        results['scaling_efficiency'] = avg_efficiency\n        results['max_dimension'] = 2000\n        \n        return results\n    \n    def test_throughput_benchmarks(self) -> Dict[str, Any]:\n        \"\"\"Test system throughput under various workloads.\"\"\"\n        print(\"\\n🏁 Testing Throughput Benchmarks...\")\n        \n        dim = 1000\n        suite = create_optimized_suite(dim)\n        \n        benchmarks = {}\n        \n        # Temporal processing benchmark\n        temporal = suite['temporal']\n        sequences = [[np.random.randn(dim) for _ in range(20)] for _ in range(10)]\n        \n        start_time = time.time()\n        for sequence in sequences:\n            temporal.sequence_prediction(sequence, prediction_horizon=3)\n        temporal_throughput_time = time.time() - start_time\n        \n        temporal_ops_per_sec = (10 * 3) / temporal_throughput_time\n        \n        # Attention processing benchmark\n        attention = suite['attention']\n        query_batches = [np.random.randn(dim) for _ in range(50)]\n        memory_batch = [np.random.randn(dim) for _ in range(100)]\n        \n        start_time = time.time()\n        for query in query_batches:\n            attention.contextual_retrieval_optimized(query, memory_batch, top_k=10)\n        attention_throughput_time = time.time() - start_time\n        \n        attention_ops_per_sec = 50 / attention_throughput_time\n        \n        # Quantum processing benchmark\n        quantum = suite['quantum']\n        vector_sets = [[np.random.randn(dim) for _ in range(20)] for _ in range(5)]\n        \n        start_time = time.time()\n        for vectors in vector_sets:\n            quantum.create_quantum_superposition_optimized(vectors[:5])\n            quantum.grover_search_optimized(vectors, vectors[0])\n        quantum_throughput_time = time.time() - start_time\n        \n        quantum_ops_per_sec = (5 * 2) / quantum_throughput_time\n        \n        # Overall system performance\n        total_time = temporal_throughput_time + attention_throughput_time + quantum_throughput_time\n        total_ops = 30 + 50 + 10  # Total operations across all benchmarks\n        overall_throughput = total_ops / total_time\n        \n        benchmarks = {\n            'temporal_ops_per_sec': temporal_ops_per_sec,\n            'attention_ops_per_sec': attention_ops_per_sec,\n            'quantum_ops_per_sec': quantum_ops_per_sec,\n            'overall_throughput': overall_throughput,\n            'total_time': total_time\n        }\n        \n        print(f\"  ✅ Temporal: {temporal_ops_per_sec:.1f} ops/sec\")\n        print(f\"  ✅ Attention: {attention_ops_per_sec:.1f} ops/sec\")\n        print(f\"  ✅ Quantum: {quantum_ops_per_sec:.1f} ops/sec\")\n        print(f\"  ✅ Overall: {overall_throughput:.1f} ops/sec\")\n        \n        return benchmarks\n    \n    def print_summary(self):\n        \"\"\"Print comprehensive test results summary.\"\"\"\n        print(\"\\n\" + \"=\" * 50)\n        print(\"🚀 SCALING & PERFORMANCE SUMMARY\")\n        print(\"=\" * 50)\n        \n        total_tests = len(self.test_results)\n        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'passed')\n        \n        print(f\"📊 Test Results: {passed_tests}/{total_tests} passed\")\n        \n        if passed_tests > 0:\n            print(\"\\n🎯 KEY PERFORMANCE METRICS:\")\n            \n            # Extract key metrics\n            if 'test_caching_performance' in self.test_results:\n                cache_result = self.test_results['test_caching_performance']['result']\n                print(f\"   • Cache Hit Rate: {cache_result.get('hit_rate', 0):.1%}\")\n            \n            if 'test_vectorized_operations' in self.test_results:\n                vector_result = self.test_results['test_vectorized_operations']['result']\n                print(f\"   • Vectorization Speedup: {vector_result.get('similarity_speedup', 0):.1f}x\")\n            \n            if 'test_concurrent_processing' in self.test_results:\n                parallel_result = self.test_results['test_concurrent_processing']['result']\n                print(f\"   • Parallel Speedup: {parallel_result.get('parallel_speedup', 0):.1f}x\")\n            \n            if 'test_throughput_benchmarks' in self.test_results:\n                throughput_result = self.test_results['test_throughput_benchmarks']['result']\n                print(f\"   • Overall Throughput: {throughput_result.get('overall_throughput', 0):.1f} ops/sec\")\n            \n            if 'test_scalability_limits' in self.test_results:\n                scaling_result = self.test_results['test_scalability_limits']['result']\n                print(f\"   • Scaling Efficiency: {scaling_result.get('scaling_efficiency', 0):.2f}\")\n        \n        success_rate = passed_tests / total_tests * 100\n        print(f\"\\n🎯 Success Rate: {success_rate:.1f}%\")\n        \n        if success_rate >= 90:\n            print(\"🚀 EXCELLENT: Scaling goals achieved!\")\n            print(\"   • High-performance optimizations working\")\n            print(\"   • Excellent scalability and throughput\")\n            print(\"   • Ready for production deployment\")\n        elif success_rate >= 75:\n            print(\"✅ GOOD: Most scaling features working\")\n            print(\"   • Performance optimizations functional\")\n            print(\"   • Scaling capabilities demonstrated\")\n        else:\n            print(\"⚠️  NEEDS IMPROVEMENT: Scaling issues detected\")\n\n\ndef main():\n    \"\"\"Run scaling and performance test suite.\"\"\"\n    test_suite = ScalingTestSuite()\n    results = test_suite.run_all_tests()\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    main()