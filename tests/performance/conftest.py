"""
Performance test configuration for HD-Compute-Toolkit.

This module provides specialized fixtures and configuration for performance
testing, including benchmarking utilities and performance monitoring.
"""

import time
import psutil
import gc
from typing import Dict, Any, List, Generator, Callable
from pathlib import Path

import pytest
import numpy as np
import torch


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.gpu_start_memory = None
        self.gpu_end_memory = None
        
    def start(self):
        """Start monitoring."""
        gc.collect()  # Clean up before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.gpu_start_memory = torch.cuda.memory_allocated()
        
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Stop monitoring and return metrics."""
        self.end_time = time.perf_counter()
        
        process = psutil.Process()
        self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.gpu_end_memory = torch.cuda.memory_allocated()
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            'duration_seconds': self.end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': self.end_memory,
            'memory_delta_mb': self.end_memory - self.start_memory,
        }
        
        if self.gpu_start_memory is not None:
            gpu_delta = (self.gpu_end_memory - self.gpu_start_memory) / 1024 / 1024  # MB
            metrics.update({
                'gpu_start_memory_mb': self.gpu_start_memory / 1024 / 1024,
                'gpu_end_memory_mb': self.gpu_end_memory / 1024 / 1024,
                'gpu_memory_delta_mb': gpu_delta,
            })
        
        return metrics


@pytest.fixture
def performance_monitor() -> Generator[PerformanceMonitor, None, None]:
    """Provide a performance monitor for benchmarking."""
    monitor = PerformanceMonitor()
    yield monitor


@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        'warmup_rounds': 3,
        'timing_rounds': 10,
        'min_time': 0.001,  # 1ms minimum
        'max_time': 60.0,   # 60s maximum
        'memory_limit_mb': 4096,  # 4GB memory limit
        'timeout_seconds': 120,   # 2 minute timeout
        'gc_between_runs': True,
    }


@pytest.fixture
def benchmark_dimensions() -> List[int]:
    """Standard dimensions for benchmarking."""
    return [1000, 4096, 10000, 16000, 32000]


@pytest.fixture
def benchmark_batch_sizes() -> List[int]:
    """Standard batch sizes for benchmarking."""
    return [1, 10, 100, 1000]


@pytest.fixture
def stress_test_config() -> Dict[str, Any]:
    """Configuration for stress testing."""
    return {
        'max_dimension': 100000,
        'max_batch_size': 10000,
        'max_runtime_seconds': 600,  # 10 minutes
        'memory_pressure_mb': 8192,  # 8GB
    }


@pytest.fixture
def benchmark_runner(benchmark_config: Dict[str, Any], performance_monitor: PerformanceMonitor):
    """Utility for running benchmarks with consistent methodology."""
    
    def run_benchmark(
        func: Callable, 
        *args, 
        name: str = "benchmark",
        **kwargs
    ) -> Dict[str, Any]:
        """Run a function as a benchmark with timing and memory monitoring."""
        
        # Warmup rounds
        for _ in range(benchmark_config['warmup_rounds']):
            if benchmark_config['gc_between_runs']:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            func(*args, **kwargs)
        
        # Timing rounds
        times = []
        memory_deltas = []
        
        for _ in range(benchmark_config['timing_rounds']):
            if benchmark_config['gc_between_runs']:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            performance_monitor.start()
            result = func(*args, **kwargs)
            metrics = performance_monitor.stop()
            
            times.append(metrics['duration_seconds'])
            memory_deltas.append(metrics['memory_delta_mb'])
        
        # Compute statistics
        times = np.array(times)
        memory_deltas = np.array(memory_deltas)
        
        return {
            'name': name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'mean_memory_delta': np.mean(memory_deltas),
            'max_memory_delta': np.max(memory_deltas),
            'num_runs': len(times),
            'result': result,  # Last result for validation
        }
    
    return run_benchmark


@pytest.fixture
def benchmark_results_file(tmp_path: Path) -> Path:
    """File path for saving benchmark results."""
    return tmp_path / "benchmark_results.json"


@pytest.fixture
def performance_regression_checker():
    """Check for performance regressions against baseline."""
    
    def check_regression(
        current_time: float,
        baseline_time: float,
        tolerance: float = 0.2  # 20% tolerance
    ) -> bool:
        """Check if current performance is within acceptable range of baseline."""
        regression_threshold = baseline_time * (1 + tolerance)
        return current_time <= regression_threshold
    
    return check_regression


# GPU-specific performance fixtures
@pytest.fixture
def gpu_benchmark_config(benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-specific benchmark configuration."""
    config = benchmark_config.copy()
    config.update({
        'cuda_sync': True,
        'measure_gpu_memory': True,
        'empty_cache_between_runs': True,
    })
    return config


@pytest.fixture
def multi_gpu_config() -> Dict[str, Any]:
    """Configuration for multi-GPU testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        pytest.skip("Multi-GPU testing requires at least 2 GPUs")
    
    return {
        'device_count': device_count,
        'devices': [f'cuda:{i}' for i in range(device_count)],
        'parallel_streams': True,
    }


# Memory profiling fixtures
@pytest.fixture
def memory_profiler():
    """Memory profiling utilities."""
    try:
        from memory_profiler import profile
        return profile
    except ImportError:
        pytest.skip("memory_profiler not available")


@pytest.fixture
def line_profiler():
    """Line-by-line profiling utilities."""
    try:
        from line_profiler import LineProfiler
        return LineProfiler()
    except ImportError:
        pytest.skip("line_profiler not available")


# Performance data generators
@pytest.fixture
def large_hypervector_batch(benchmark_dimensions: List[int]):
    """Generate large batches of hypervectors for stress testing."""
    
    def generate_batch(dim: int, batch_size: int, device: str = 'cpu'):
        """Generate a batch of binary hypervectors."""
        if device.startswith('cuda'):
            torch.cuda.manual_seed(42)
            return torch.randint(0, 2, (batch_size, dim), 
                               dtype=torch.int8, device=device)
        else:
            torch.manual_seed(42)
            return torch.randint(0, 2, (batch_size, dim), dtype=torch.int8)
    
    return generate_batch


@pytest.fixture
def scalability_test_sizes() -> List[tuple]:
    """Test sizes for scalability analysis."""
    return [
        (1000, 10),
        (1000, 100),
        (1000, 1000),
        (10000, 10),
        (10000, 100),
        (10000, 1000),
        (16000, 10),
        (16000, 100),
        (32000, 10),
        (32000, 100),
    ]


# Benchmark result analysis
@pytest.fixture
def performance_analyzer():
    """Analyze and compare performance results."""
    
    def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results and compute statistics."""
        if not results:
            return {}
        
        times = [r['mean_time'] for r in results]
        memory_deltas = [r['mean_memory_delta'] for r in results]
        
        return {
            'total_time': sum(times),
            'average_time': np.mean(times),
            'fastest_test': min(results, key=lambda x: x['mean_time'])['name'],
            'slowest_test': max(results, key=lambda x: x['mean_time'])['name'],
            'total_memory_usage': sum(memory_deltas),
            'peak_memory_test': max(results, key=lambda x: x['max_memory_delta'])['name'],
        }
    
    return analyze_results