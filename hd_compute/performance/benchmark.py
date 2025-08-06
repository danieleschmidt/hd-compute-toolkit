"""Comprehensive benchmark suite for HDC operations."""

import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging

from .profiler import HDCProfiler, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    operation: str
    backend: str
    dimension: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    operations_per_second: float
    memory_usage_mb: float
    success_rate: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkSuite:
    """Comprehensive benchmark suite for HDC operations."""
    
    def __init__(self, profiler: Optional[HDCProfiler] = None):
        """Initialize benchmark suite.
        
        Args:
            profiler: HDC profiler instance
        """
        self.profiler = profiler or HDCProfiler()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_random_generation(
        self,
        hdc_backend: Any,
        dimensions: List[int] = None,
        sparsity_levels: List[float] = None,
        iterations: int = 100
    ) -> List[BenchmarkResult]:
        """Benchmark random hypervector generation.
        
        Args:
            hdc_backend: HDC backend instance
            dimensions: List of dimensions to test
            sparsity_levels: List of sparsity levels to test
            iterations: Number of iterations per test
            
        Returns:
            List of benchmark results
        """
        if dimensions is None:
            dimensions = [500, 1000, 5000, 10000]
        
        if sparsity_levels is None:
            sparsity_levels = [0.3, 0.5, 0.7]
        
        results = []
        backend_name = type(hdc_backend).__name__
        
        logger.info(f"Benchmarking random generation with {backend_name}")
        
        for dim in dimensions:
            for sparsity in sparsity_levels:
                logger.debug(f"Testing dimension {dim}, sparsity {sparsity}")
                
                # Create backend with specific dimension
                try:
                    test_backend = type(hdc_backend)(dim=dim)
                except Exception as e:
                    logger.error(f"Failed to create backend for dim {dim}: {e}")
                    continue
                
                # Benchmark the operation
                times = []
                successes = 0
                
                with self.profiler.profile_operation(f"random_hv_d{dim}_s{sparsity}", dim, iterations):
                    for i in range(iterations):
                        try:
                            start_time = time.perf_counter()
                            hv = test_backend.random_hv(sparsity=sparsity)
                            end_time = time.perf_counter()
                            
                            times.append((end_time - start_time) * 1000)  # Convert to ms
                            successes += 1
                            
                        except Exception as e:
                            logger.warning(f"Iteration {i} failed: {e}")
                
                # Calculate statistics
                if times:
                    total_time = sum(times)
                    avg_time = statistics.mean(times)
                    min_time = min(times)
                    max_time = max(times)
                    std_dev = statistics.stdev(times) if len(times) > 1 else 0
                    ops_per_sec = 1000 / avg_time if avg_time > 0 else 0
                    success_rate = successes / iterations
                    
                    # Get memory usage from profiler
                    memory_usage = 0
                    if self.profiler.metrics_history:
                        memory_usage = self.profiler.metrics_history[-1].memory_usage_mb
                    
                    result = BenchmarkResult(
                        operation=f"random_hv_sparsity_{sparsity}",
                        backend=backend_name,
                        dimension=dim,
                        iterations=iterations,
                        total_time_ms=total_time,
                        avg_time_ms=avg_time,
                        min_time_ms=min_time,
                        max_time_ms=max_time,
                        std_dev_ms=std_dev,
                        operations_per_second=ops_per_sec,
                        memory_usage_mb=memory_usage,
                        success_rate=success_rate,
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    logger.debug(f"Result: {avg_time:.3f}ms avg, {ops_per_sec:.1f} ops/sec")
        
        return results
    
    def benchmark_bundle_operations(
        self,
        hdc_backend: Any,
        dimensions: List[int] = None,
        bundle_sizes: List[int] = None,
        iterations: int = 100
    ) -> List[BenchmarkResult]:
        """Benchmark bundle operations.
        
        Args:
            hdc_backend: HDC backend instance
            dimensions: List of dimensions to test
            bundle_sizes: List of bundle sizes to test
            iterations: Number of iterations per test
            
        Returns:
            List of benchmark results
        """
        if dimensions is None:
            dimensions = [1000, 5000, 10000]
        
        if bundle_sizes is None:
            bundle_sizes = [2, 5, 10, 20]
        
        results = []
        backend_name = type(hdc_backend).__name__
        
        logger.info(f"Benchmarking bundle operations with {backend_name}")
        
        for dim in dimensions:
            for bundle_size in bundle_sizes:
                logger.debug(f"Testing dimension {dim}, bundle size {bundle_size}")
                
                try:
                    test_backend = type(hdc_backend)(dim=dim)
                    
                    # Pre-generate hypervectors for bundling
                    hvs = [test_backend.random_hv() for _ in range(bundle_size)]
                    
                except Exception as e:
                    logger.error(f"Failed to prepare bundle test for dim {dim}: {e}")
                    continue
                
                # Benchmark bundling
                times = []
                successes = 0
                
                with self.profiler.profile_operation(f"bundle_d{dim}_n{bundle_size}", dim, iterations):
                    for i in range(iterations):
                        try:
                            start_time = time.perf_counter()
                            bundled = test_backend.bundle(hvs)
                            end_time = time.perf_counter()
                            
                            times.append((end_time - start_time) * 1000)
                            successes += 1
                            
                        except Exception as e:
                            logger.warning(f"Bundle iteration {i} failed: {e}")
                
                # Calculate statistics
                if times:
                    result = self._calculate_benchmark_result(
                        f"bundle_size_{bundle_size}",
                        backend_name,
                        dim,
                        iterations,
                        times,
                        successes
                    )
                    
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    def benchmark_bind_operations(
        self,
        hdc_backend: Any,
        dimensions: List[int] = None,
        iterations: int = 100
    ) -> List[BenchmarkResult]:
        """Benchmark bind operations.
        
        Args:
            hdc_backend: HDC backend instance
            dimensions: List of dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            List of benchmark results
        """
        if dimensions is None:
            dimensions = [1000, 5000, 10000]
        
        results = []
        backend_name = type(hdc_backend).__name__
        
        logger.info(f"Benchmarking bind operations with {backend_name}")
        
        for dim in dimensions:
            logger.debug(f"Testing bind with dimension {dim}")
            
            try:
                test_backend = type(hdc_backend)(dim=dim)
                hv1 = test_backend.random_hv()
                hv2 = test_backend.random_hv()
                
            except Exception as e:
                logger.error(f"Failed to prepare bind test for dim {dim}: {e}")
                continue
            
            # Benchmark binding
            times = []
            successes = 0
            
            with self.profiler.profile_operation(f"bind_d{dim}", dim, iterations):
                for i in range(iterations):
                    try:
                        start_time = time.perf_counter()
                        bound = test_backend.bind(hv1, hv2)
                        end_time = time.perf_counter()
                        
                        times.append((end_time - start_time) * 1000)
                        successes += 1
                        
                    except Exception as e:
                        logger.warning(f"Bind iteration {i} failed: {e}")
            
            # Calculate statistics
            if times:
                result = self._calculate_benchmark_result(
                    "bind",
                    backend_name,
                    dim,
                    iterations,
                    times,
                    successes
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def benchmark_similarity_operations(
        self,
        hdc_backend: Any,
        dimensions: List[int] = None,
        iterations: int = 100
    ) -> List[BenchmarkResult]:
        """Benchmark similarity operations.
        
        Args:
            hdc_backend: HDC backend instance
            dimensions: List of dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            List of benchmark results
        """
        if dimensions is None:
            dimensions = [1000, 5000, 10000]
        
        results = []
        backend_name = type(hdc_backend).__name__
        
        logger.info(f"Benchmarking similarity operations with {backend_name}")
        
        for dim in dimensions:
            logger.debug(f"Testing similarity with dimension {dim}")
            
            try:
                test_backend = type(hdc_backend)(dim=dim)
                hv1 = test_backend.random_hv()
                hv2 = test_backend.random_hv()
                
            except Exception as e:
                logger.error(f"Failed to prepare similarity test for dim {dim}: {e}")
                continue
            
            # Benchmark similarity calculation
            times = []
            successes = 0
            
            with self.profiler.profile_operation(f"similarity_d{dim}", dim, iterations):
                for i in range(iterations):
                    try:
                        start_time = time.perf_counter()
                        sim = test_backend.cosine_similarity(hv1, hv2)
                        end_time = time.perf_counter()
                        
                        times.append((end_time - start_time) * 1000)
                        successes += 1
                        
                    except Exception as e:
                        logger.warning(f"Similarity iteration {i} failed: {e}")
            
            # Calculate statistics
            if times:
                result = self._calculate_benchmark_result(
                    "cosine_similarity",
                    backend_name,
                    dim,
                    iterations,
                    times,
                    successes
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def run_comprehensive_benchmark(
        self,
        hdc_backend: Any,
        dimensions: List[int] = None,
        iterations: int = 100
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark suite.
        
        Args:
            hdc_backend: HDC backend instance
            dimensions: List of dimensions to test
            iterations: Number of iterations per test
            
        Returns:
            Dictionary mapping operation types to results
        """
        if dimensions is None:
            dimensions = [500, 1000, 2000, 5000]
        
        logger.info("Starting comprehensive HDC benchmark suite")
        
        benchmark_results = {}
        
        # Benchmark all operations
        benchmark_results['random_generation'] = self.benchmark_random_generation(
            hdc_backend, dimensions, iterations=iterations
        )
        
        benchmark_results['bundle_operations'] = self.benchmark_bundle_operations(
            hdc_backend, dimensions, iterations=iterations
        )
        
        benchmark_results['bind_operations'] = self.benchmark_bind_operations(
            hdc_backend, dimensions, iterations=iterations
        )
        
        benchmark_results['similarity_operations'] = self.benchmark_similarity_operations(
            hdc_backend, dimensions, iterations=iterations
        )
        
        logger.info("Comprehensive benchmark suite completed")
        return benchmark_results
    
    def _calculate_benchmark_result(
        self,
        operation: str,
        backend: str,
        dimension: int,
        iterations: int,
        times: List[float],
        successes: int
    ) -> BenchmarkResult:
        """Calculate benchmark result from timing data."""
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        ops_per_sec = 1000 / avg_time if avg_time > 0 else 0
        success_rate = successes / iterations
        
        # Get memory usage from profiler
        memory_usage = 0
        if self.profiler.metrics_history:
            memory_usage = self.profiler.metrics_history[-1].memory_usage_mb
        
        return BenchmarkResult(
            operation=operation,
            backend=backend,
            dimension=dimension,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            operations_per_second=ops_per_sec,
            memory_usage_mb=memory_usage,
            success_rate=success_rate,
            timestamp=time.time()
        )
    
    def export_results(self, filename: str, format: str = 'json'):
        """Export benchmark results to file.
        
        Args:
            filename: Output filename
            format: Output format ('json' or 'csv')
        """
        if format == 'json':
            results_data = [result.to_dict() for result in self.results]
            
            with open(filename, 'w') as f:
                json.dump({
                    'benchmark_results': results_data,
                    'summary': self.get_summary(),
                    'timestamp': time.time()
                }, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            if self.results:
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                    writer.writeheader()
                    
                    for result in self.results:
                        writer.writerow(result.to_dict())
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Benchmark results exported to {filename}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {"total_benchmarks": 0}
        
        # Group results by operation and backend
        by_operation = {}
        by_backend = {}
        
        for result in self.results:
            # By operation
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result.avg_time_ms)
            
            # By backend
            if result.backend not in by_backend:
                by_backend[result.backend] = []
            by_backend[result.backend].append(result.avg_time_ms)
        
        # Calculate averages
        operation_averages = {
            op: statistics.mean(times) for op, times in by_operation.items()
        }
        
        backend_averages = {
            backend: statistics.mean(times) for backend, times in by_backend.items()
        }
        
        return {
            "total_benchmarks": len(self.results),
            "operation_averages_ms": operation_averages,
            "backend_averages_ms": backend_averages,
            "fastest_operation": min(operation_averages.items(), key=lambda x: x[1]),
            "slowest_operation": max(operation_averages.items(), key=lambda x: x[1])
        }