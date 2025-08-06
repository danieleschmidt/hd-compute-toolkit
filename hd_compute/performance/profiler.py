"""Performance profiling utilities for HDC operations."""

import time
import threading

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    operations_per_second: float
    dimension: int
    iterations: int = 1
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class HDCProfiler:
    """Performance profiler for HDC operations."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """Initialize profiler.
        
        Args:
            sampling_interval: How often to sample system metrics (seconds)
        """
        self.sampling_interval = sampling_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread = None
        self._current_metrics = {}
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start system monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_system(self):
        """Monitor system metrics in background thread."""
        if not PSUTIL_AVAILABLE:
            # Fallback monitoring without psutil
            while self._monitoring:
                with self._lock:
                    self._current_metrics['memory_mb'] = 0.0
                    self._current_metrics['cpu_percent'] = 0.0
                time.sleep(self.sampling_interval)
            return
        
        process = psutil.Process()
        
        while self._monitoring:
            try:
                with self._lock:
                    self._current_metrics['memory_mb'] = process.memory_info().rss / 1024 / 1024
                    self._current_metrics['cpu_percent'] = process.cpu_percent()
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.warning(f"Error monitoring system metrics: {e}")
    
    @contextmanager
    def profile_operation(self, operation_name: str, dimension: int = 0, iterations: int = 1):
        """Context manager for profiling HDC operations.
        
        Args:
            operation_name: Name of the operation being profiled
            dimension: Hypervector dimension
            iterations: Number of iterations
            
        Yields:
            Context for the profiled operation
        """
        # Start monitoring if not already started
        if not self._monitoring:
            self.start_monitoring()
        
        # Record initial state
        start_time = time.perf_counter()
        initial_memory = 0
        peak_memory = 0
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = initial_memory
            except:
                pass
        
        try:
            yield self
            
            # Monitor peak memory during operation
            if self._monitoring:
                with self._lock:
                    current_memory = self._current_metrics.get('memory_mb', initial_memory)
                    peak_memory = max(peak_memory, current_memory)
        
        finally:
            # Calculate metrics
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            final_memory = initial_memory
            cpu_usage = 0.0
            
            try:
                with self._lock:
                    final_memory = self._current_metrics.get('memory_mb', initial_memory)
                    cpu_usage = self._current_metrics.get('cpu_percent', 0.0)
            except:
                pass
            
            memory_usage_mb = final_memory - initial_memory
            operations_per_second = (iterations * 1000.0) / execution_time_ms if execution_time_ms > 0 else 0
            
            # Create metrics record
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage,
                peak_memory_mb=peak_memory,
                operations_per_second=operations_per_second,
                dimension=dimension,
                iterations=iterations
            )
            
            self.metrics_history.append(metrics)
            
            logger.debug(f"Profiled {operation_name}: {execution_time_ms:.2f}ms, "
                        f"{operations_per_second:.1f} ops/sec")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Profile a function call and return results with metrics.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, metrics)
        """
        operation_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
        dimension = kwargs.get('dimension', 0)
        
        with self.profile_operation(operation_name, dimension, 1):
            result = func(*args, **kwargs)
        
        return result, self.metrics_history[-1]
    
    def benchmark_operation(
        self, 
        func: Callable, 
        iterations: int = 100,
        warmup_iterations: int = 10,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> PerformanceMetrics:
        """Benchmark an operation multiple times.
        
        Args:
            func: Function to benchmark
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            operation_name: Name for the operation
            **kwargs: Additional arguments for the function
            
        Returns:
            Performance metrics
        """
        if operation_name is None:
            operation_name = func.__name__ if hasattr(func, '__name__') else 'benchmark'
        
        dimension = kwargs.get('dimension', 0)
        
        # Warmup
        logger.debug(f"Warming up {operation_name} for {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            try:
                func(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Actual benchmark
        logger.debug(f"Benchmarking {operation_name} for {iterations} iterations")
        
        with self.profile_operation(operation_name, dimension, iterations):
            for _ in range(iterations):
                func(**kwargs)
        
        return self.metrics_history[-1]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics.
        
        Returns:
            Summary dictionary with statistics
        """
        if not self.metrics_history:
            return {"total_operations": 0}
        
        operations_by_name = {}
        total_time = 0
        total_memory = 0
        
        for metrics in self.metrics_history:
            if metrics.operation_name not in operations_by_name:
                operations_by_name[metrics.operation_name] = []
            operations_by_name[metrics.operation_name].append(metrics)
            total_time += metrics.execution_time_ms
            total_memory += metrics.memory_usage_mb
        
        summary = {
            "total_operations": len(self.metrics_history),
            "total_execution_time_ms": total_time,
            "total_memory_usage_mb": total_memory,
            "operations_by_name": {}
        }
        
        for op_name, op_metrics in operations_by_name.items():
            times = [m.execution_time_ms for m in op_metrics]
            ops_per_sec = [m.operations_per_second for m in op_metrics]
            
            summary["operations_by_name"][op_name] = {
                "count": len(op_metrics),
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "avg_ops_per_sec": sum(ops_per_sec) / len(ops_per_sec),
                "total_time_ms": sum(times)
            }
        
        return summary
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            self.metrics_history.clear()
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics as list of dictionaries.
        
        Returns:
            List of metric dictionaries
        """
        return [
            {
                'operation_name': m.operation_name,
                'execution_time_ms': m.execution_time_ms,
                'memory_usage_mb': m.memory_usage_mb,
                'cpu_usage_percent': m.cpu_usage_percent,
                'peak_memory_mb': m.peak_memory_mb,
                'operations_per_second': m.operations_per_second,
                'dimension': m.dimension,
                'iterations': m.iterations,
                'timestamp': m.timestamp
            }
            for m in self.metrics_history
        ]
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()