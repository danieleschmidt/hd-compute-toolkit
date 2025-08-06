"""Tests for performance optimization components."""

import pytest
import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hd_compute.pure_python import HDComputePython
from hd_compute.performance import HDCProfiler, BenchmarkSuite
from hd_compute.performance.optimization import PerformanceOptimizer, LRUCache, OperationCache


class TestHDCProfiler:
    """Test HDC profiler functionality."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = HDCProfiler()
        
        assert profiler.sampling_interval == 0.1
        assert len(profiler.metrics_history) == 0
        assert not profiler._monitoring
    
    def test_profile_operation_context(self):
        """Test profiling operation with context manager."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=100)
        
        with profiler.profile_operation('test_operation', 100, 5):
            for _ in range(5):
                hv = hdc.random_hv()
        
        assert len(profiler.metrics_history) == 1
        metrics = profiler.metrics_history[0]
        
        assert metrics.operation_name == 'test_operation'
        assert metrics.dimension == 100
        assert metrics.iterations == 5
        assert metrics.execution_time_ms > 0
        assert metrics.operations_per_second > 0
    
    def test_profile_function(self):
        """Test profiling individual function."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=50)
        
        def test_function():
            return hdc.random_hv()
        
        result, metrics = profiler.profile_function(test_function)
        
        assert len(result.data) == 50
        assert metrics.operation_name == 'test_function'
        assert metrics.execution_time_ms > 0
    
    def test_benchmark_operation(self):
        """Test benchmarking operation."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=100)
        
        def benchmark_func():
            return hdc.random_hv()
        
        metrics = profiler.benchmark_operation(
            benchmark_func,
            iterations=10,
            warmup_iterations=2,
            operation_name='random_generation'
        )
        
        assert metrics.operation_name == 'random_generation'
        assert metrics.iterations == 10
        assert metrics.execution_time_ms > 0
        assert metrics.operations_per_second > 0
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=50)
        
        # Profile multiple operations
        with profiler.profile_operation('op1', 50, 3):
            for _ in range(3):
                hdc.random_hv()
        
        with profiler.profile_operation('op2', 50, 2):
            for _ in range(2):
                hdc.random_hv()
        
        summary = profiler.get_metrics_summary()
        
        assert summary['total_operations'] == 2
        assert 'op1' in summary['operations_by_name']
        assert 'op2' in summary['operations_by_name']
        
        op1_stats = summary['operations_by_name']['op1']
        assert op1_stats['count'] == 1
        assert op1_stats['avg_time_ms'] > 0
    
    def test_export_metrics(self):
        """Test metrics export."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=30)
        
        with profiler.profile_operation('export_test', 30, 1):
            hdc.random_hv()
        
        exported = profiler.export_metrics()
        
        assert len(exported) == 1
        assert exported[0]['operation_name'] == 'export_test'
        assert 'execution_time_ms' in exported[0]
        assert 'timestamp' in exported[0]
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        profiler = HDCProfiler()
        hdc = HDComputePython(dim=30)
        
        with profiler.profile_operation('clear_test', 30, 1):
            hdc.random_hv()
        
        assert len(profiler.metrics_history) == 1
        
        profiler.clear_metrics()
        
        assert len(profiler.metrics_history) == 0


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(maxsize=10)
        
        assert cache.maxsize == 10
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_get(self):
        """Test putting and getting items."""
        cache = LRUCache(maxsize=3)
        
        # Put items
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        
        # Get items
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') is None
        
        # Check hit/miss counts
        assert cache.hits == 2
        assert cache.misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(maxsize=2)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # Should evict key1
        
        assert cache.get('key1') is None  # Evicted
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'
    
    def test_cache_update_existing(self):
        """Test updating existing key."""
        cache = LRUCache(maxsize=2)
        
        cache.put('key1', 'value1')
        cache.put('key1', 'new_value1')  # Update
        
        assert cache.get('key1') == 'new_value1'
        assert len(cache.cache) == 1
    
    def test_cache_access_order(self):
        """Test that access updates order."""
        cache = LRUCache(maxsize=2)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        
        # Access key1 to make it most recent
        cache.get('key1')
        
        # Add key3, should evict key2 (least recently used)
        cache.put('key3', 'value3')
        
        assert cache.get('key1') == 'value1'  # Still there
        assert cache.get('key2') is None     # Evicted
        assert cache.get('key3') == 'value3' # New item
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = LRUCache(maxsize=5)
        
        cache.put('a', 1)
        cache.put('b', 2)
        
        cache.get('a')  # Hit
        cache.get('c')  # Miss
        cache.get('b')  # Hit
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['maxsize'] == 5
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = LRUCache(maxsize=5)
        
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')
        
        assert len(cache.cache) == 2
        assert cache.hits > 0
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestOperationCache:
    """Test operation cache functionality."""
    
    def test_operation_cache_initialization(self):
        """Test operation cache initialization."""
        cache = OperationCache(maxsize=100)
        
        assert cache.cache.maxsize == 100
        assert len(cache.cache_by_dimension) == 0
    
    def test_cache_result(self):
        """Test caching and retrieving results."""
        cache = OperationCache()
        
        # Cache a result
        result = [1, 0, 1, 0]
        cache.cache_result('random_hv', 100, result, sparsity=0.5)
        
        # Retrieve result
        cached = cache.get_cached_result('random_hv', 100, sparsity=0.5)
        
        assert cached == result
    
    def test_dimension_specific_caching(self):
        """Test dimension-specific caching."""
        cache = OperationCache()
        
        # Cache results for different dimensions
        cache.cache_result('test_op', 100, 'result100')
        cache.cache_result('test_op', 200, 'result200')
        
        # Retrieve results
        assert cache.get_cached_result('test_op', 100) == 'result100'
        assert cache.get_cached_result('test_op', 200) == 'result200'
        assert cache.get_cached_result('test_op', 300) is None
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = OperationCache()
        
        # Same parameters should generate same key
        key1 = cache._generate_key('op', 'arg1', kwarg1='value1')
        key2 = cache._generate_key('op', 'arg1', kwarg1='value1')
        key3 = cache._generate_key('op', 'arg2', kwarg1='value1')
        
        assert key1 == key2
        assert key1 != key3
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = OperationCache()
        
        cache.cache_result('op1', 100, 'result1')
        cache.cache_result('op2', 200, 'result2')
        
        # Clear specific dimension
        cache.clear(dimension=100)
        
        assert cache.get_cached_result('op1', 100) is None
        assert cache.get_cached_result('op2', 200) == 'result2'
        
        # Clear all
        cache.clear()
        
        assert cache.get_cached_result('op2', 200) is None


class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.enable_caching is True
        assert optimizer.operation_cache is not None
        assert len(optimizer.operation_stats) == 0
    
    def test_cached_operation_decorator(self):
        """Test cached operation decorator."""
        optimizer = PerformanceOptimizer()
        
        call_count = 0
        
        @optimizer.cached_operation('test_op', 100)
        def expensive_operation(value):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return value * 2
        
        # First call - should execute function
        result1 = expensive_operation(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again
    
    def test_optimizer_without_caching(self):
        """Test optimizer with caching disabled."""
        optimizer = PerformanceOptimizer(enable_caching=False)
        
        assert optimizer.enable_caching is False
        assert optimizer.operation_cache is None
        
        call_count = 0
        
        @optimizer.cached_operation('test_op', 100)
        def test_function():
            nonlocal call_count
            call_count += 1
            return 'result'
        
        # Both calls should execute function
        test_function()
        test_function()
        
        assert call_count == 2
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        optimizer = PerformanceOptimizer()
        
        # Simulate some operations with poor performance
        optimizer._record_operation('slow_op', 0.2, cache_miss=True)
        for _ in range(15):
            optimizer._record_operation('slow_op', 0.15, cache_miss=True)
        
        suggestions = optimizer.suggest_optimizations()
        
        assert len(suggestions) > 0
        # Should suggest optimization for slow operation
        slow_op_suggestion = any('slow_op' in s for s in suggestions)
        assert slow_op_suggestion
    
    def test_performance_report(self):
        """Test performance report generation."""
        optimizer = PerformanceOptimizer()
        
        # Record some operations
        optimizer._record_operation('op1', 0.01)
        optimizer._record_operation('op2', 0.05)
        optimizer._record_cache_hit('op1')
        
        report = optimizer.get_performance_report()
        
        assert 'timestamp' in report
        assert 'caching_enabled' in report
        assert 'operation_statistics' in report
        assert 'optimization_suggestions' in report
        
        assert 'op1' in report['operation_statistics']
        assert 'op2' in report['operation_statistics']
    
    def test_clear_statistics(self):
        """Test clearing statistics."""
        optimizer = PerformanceOptimizer()
        
        optimizer._record_operation('test_op', 0.01)
        
        assert len(optimizer.operation_stats) > 0
        
        optimizer.clear_statistics()
        
        assert len(optimizer.operation_stats) == 0


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        profiler = HDCProfiler()
        benchmark = BenchmarkSuite(profiler)
        
        assert benchmark.profiler is profiler
        assert len(benchmark.results) == 0
    
    def test_benchmark_random_generation(self):
        """Test benchmarking random generation."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=50)
        
        results = benchmark.benchmark_random_generation(
            hdc,
            dimensions=[50, 100],
            sparsity_levels=[0.5],
            iterations=5
        )
        
        assert len(results) == 2  # 2 dimensions
        
        for result in results:
            assert result.operation.startswith('random_hv_sparsity')
            assert result.backend == 'HDComputePython'
            assert result.dimension in [50, 100]
            assert result.iterations == 5
            assert result.avg_time_ms > 0
            assert result.operations_per_second > 0
    
    def test_benchmark_bind_operations(self):
        """Test benchmarking bind operations."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=50)
        
        results = benchmark.benchmark_bind_operations(
            hdc,
            dimensions=[50, 100],
            iterations=5
        )
        
        assert len(results) == 2  # 2 dimensions
        
        for result in results:
            assert result.operation == 'bind'
            assert result.dimension in [50, 100]
            assert result.avg_time_ms > 0
    
    def test_benchmark_bundle_operations(self):
        """Test benchmarking bundle operations."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=50)
        
        results = benchmark.benchmark_bundle_operations(
            hdc,
            dimensions=[50],
            bundle_sizes=[2, 3],
            iterations=5
        )
        
        assert len(results) == 2  # 2 bundle sizes
        
        for result in results:
            assert result.operation.startswith('bundle_size')
            assert result.dimension == 50
    
    def test_benchmark_similarity_operations(self):
        """Test benchmarking similarity operations."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=50)
        
        results = benchmark.benchmark_similarity_operations(
            hdc,
            dimensions=[50, 100],
            iterations=5
        )
        
        assert len(results) == 2
        
        for result in results:
            assert result.operation == 'cosine_similarity'
            assert result.avg_time_ms > 0
    
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmark."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=30)
        
        results = benchmark.run_comprehensive_benchmark(
            hdc,
            dimensions=[30, 50],
            iterations=3
        )
        
        assert 'random_generation' in results
        assert 'bundle_operations' in results
        assert 'bind_operations' in results
        assert 'similarity_operations' in results
        
        # Check that all operations produced results
        total_results = sum(len(r) for r in results.values())
        assert total_results > 0
    
    def test_benchmark_summary(self):
        """Test benchmark summary."""
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=30)
        
        # Run small benchmark
        benchmark.benchmark_bind_operations(hdc, dimensions=[30], iterations=3)
        
        summary = benchmark.get_summary()
        
        assert summary['total_benchmarks'] > 0
        assert 'operation_averages_ms' in summary
        assert 'backend_averages_ms' in summary
    
    def test_export_results_json(self):
        """Test exporting results to JSON."""
        import tempfile
        import json
        
        benchmark = BenchmarkSuite()
        hdc = HDComputePython(dim=30)
        
        # Run small benchmark
        benchmark.benchmark_bind_operations(hdc, dimensions=[30], iterations=3)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        benchmark.export_results(temp_file, format='json')
        
        # Verify file contents
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        assert 'benchmark_results' in data
        assert 'summary' in data
        assert 'timestamp' in data
        
        # Cleanup
        os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])