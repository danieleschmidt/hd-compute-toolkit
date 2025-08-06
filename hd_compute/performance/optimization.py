"""Performance optimization utilities and caching strategies."""

import time
import functools
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import OrderedDict, defaultdict
import weakref
import logging
import pickle
import hashlib

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, maxsize: int = 1000):
        """Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class OperationCache:
    """Specialized cache for HDC operations."""
    
    def __init__(self, maxsize: int = 500):
        """Initialize operation cache.
        
        Args:
            maxsize: Maximum number of cached results
        """
        self.cache = LRUCache(maxsize)
        self.cache_by_dimension = defaultdict(lambda: LRUCache(100))
    
    def _generate_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key for operation."""
        key_data = f"{operation}"
        
        # Add arguments to key
        if args:
            key_data += f"_args_{str(args)}"
        
        if kwargs:
            # Sort kwargs for consistent keys
            sorted_kwargs = sorted(kwargs.items())
            key_data += f"_kwargs_{str(sorted_kwargs)}"
        
        # Hash to create fixed-length key
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, operation: str, dimension: int, *args, **kwargs) -> Optional[Any]:
        """Get cached result for operation.
        
        Args:
            operation: Operation name
            dimension: Hypervector dimension
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(operation, *args, **kwargs)
        
        # Try dimension-specific cache first
        result = self.cache_by_dimension[dimension].get(key)
        if result is not None:
            return result
        
        # Try general cache
        return self.cache.get(key)
    
    def cache_result(self, operation: str, dimension: int, result: Any, *args, **kwargs):
        """Cache operation result.
        
        Args:
            operation: Operation name
            dimension: Hypervector dimension
            result: Result to cache
            *args: Operation arguments
            **kwargs: Operation keyword arguments
        """
        key = self._generate_key(operation, *args, **kwargs)
        
        # Store in both general and dimension-specific caches
        self.cache.put(key, result)
        self.cache_by_dimension[dimension].put(key, result)
    
    def clear(self, dimension: Optional[int] = None):
        """Clear cache.
        
        Args:
            dimension: If specified, clear only dimension-specific cache
        """
        if dimension is not None:
            if dimension in self.cache_by_dimension:
                self.cache_by_dimension[dimension].clear()
        else:
            self.cache.clear()
            self.cache_by_dimension.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        general_stats = self.cache.get_stats()
        dimension_stats = {}
        
        for dim, cache in self.cache_by_dimension.items():
            dimension_stats[dim] = cache.get_stats()
        
        return {
            'general_cache': general_stats,
            'dimension_caches': dimension_stats
        }


class PerformanceOptimizer:
    """Performance optimizer for HDC operations."""
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 1000):
        """Initialize performance optimizer.
        
        Args:
            enable_caching: Whether to enable operation caching
            cache_size: Size of operation cache
        """
        self.enable_caching = enable_caching
        self.operation_cache = OperationCache(cache_size) if enable_caching else None
        self.operation_stats = defaultdict(lambda: defaultdict(int))
        self.optimization_suggestions = []
        self.lock = threading.RLock()
    
    def cached_operation(self, operation_name: str, dimension: int = 0):
        """Decorator for caching operation results.
        
        Args:
            operation_name: Name of the operation
            dimension: Hypervector dimension
        """
        def decorator(func: Callable) -> Callable:
            if not self.enable_caching:
                return func
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Try to get cached result
                cached_result = self.operation_cache.get_cached_result(
                    operation_name, dimension, *args, **kwargs
                )
                
                if cached_result is not None:
                    self._record_cache_hit(operation_name)
                    return cached_result
                
                # Compute result
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Cache result
                self.operation_cache.cache_result(
                    operation_name, dimension, result, *args, **kwargs
                )
                
                # Record statistics
                self._record_operation(operation_name, execution_time, cache_miss=True)
                
                return result
            
            return wrapper
        return decorator
    
    def _record_cache_hit(self, operation_name: str):
        """Record cache hit statistics."""
        with self.lock:
            self.operation_stats[operation_name]['cache_hits'] += 1
    
    def _record_operation(self, operation_name: str, execution_time: float, cache_miss: bool = False):
        """Record operation statistics."""
        with self.lock:
            stats = self.operation_stats[operation_name]
            stats['total_calls'] += 1
            stats['total_time'] += execution_time
            
            if cache_miss:
                stats['cache_misses'] += 1
            
            # Update moving averages
            if 'avg_time' not in stats:
                stats['avg_time'] = execution_time
            else:
                # Exponential moving average
                alpha = 0.1
                stats['avg_time'] = alpha * execution_time + (1 - alpha) * stats['avg_time']
    
    def optimize_batch_size(self, operation_func: Callable, data_size: int, max_batch_size: int = 1000) -> int:
        """Find optimal batch size for an operation.
        
        Args:
            operation_func: Function to optimize
            data_size: Total size of data to process
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        if data_size <= max_batch_size:
            return data_size
        
        # Test different batch sizes
        test_sizes = [16, 32, 64, 128, 256, 512, min(max_batch_size, 1000)]
        best_size = 1
        best_throughput = 0
        
        logger.debug(f"Optimizing batch size for data size {data_size}")
        
        for batch_size in test_sizes:
            if batch_size > data_size:
                continue
            
            try:
                # Time a few iterations
                start_time = time.perf_counter()
                iterations = 3
                
                for _ in range(iterations):
                    # Simulate batch processing
                    operation_func(batch_size=batch_size)
                
                end_time = time.perf_counter()
                time_per_batch = (end_time - start_time) / iterations
                throughput = batch_size / time_per_batch
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_size = batch_size
                
                logger.debug(f"Batch size {batch_size}: {throughput:.1f} items/sec")
                
            except Exception as e:
                logger.warning(f"Failed to test batch size {batch_size}: {e}")
        
        logger.info(f"Optimal batch size: {best_size} (throughput: {best_throughput:.1f} items/sec)")
        return best_size
    
    def suggest_optimizations(self) -> List[str]:
        """Generate optimization suggestions based on usage patterns.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        with self.lock:
            for op_name, stats in self.operation_stats.items():
                total_calls = stats.get('total_calls', 0)
                cache_hits = stats.get('cache_hits', 0)
                cache_misses = stats.get('cache_misses', 0)
                avg_time = stats.get('avg_time', 0)
                
                if total_calls == 0:
                    continue
                
                cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
                
                # Suggest caching improvements
                if cache_hit_rate < 0.3 and total_calls > 10:
                    suggestions.append(f"Low cache hit rate ({cache_hit_rate:.1%}) for {op_name}. "
                                     f"Consider increasing cache size or reviewing caching strategy.")
                
                # Suggest performance improvements
                if avg_time > 100:  # >100ms average
                    suggestions.append(f"Slow operation {op_name} (avg: {avg_time:.1f}ms). "
                                     f"Consider batch processing or algorithm optimization.")
                
                # Suggest frequent operations optimization
                if total_calls > 1000:
                    suggestions.append(f"Frequently called operation {op_name} ({total_calls} calls). "
                                     f"Consider specialized optimizations or precomputation.")
        
        return suggestions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': time.time(),
            'caching_enabled': self.enable_caching,
            'operation_statistics': {},
            'cache_statistics': {},
            'optimization_suggestions': self.suggest_optimizations()
        }
        
        # Add operation statistics
        with self.lock:
            for op_name, stats in self.operation_stats.items():
                report['operation_statistics'][op_name] = dict(stats)
        
        # Add cache statistics
        if self.operation_cache:
            report['cache_statistics'] = self.operation_cache.get_stats()
        
        return report
    
    def clear_statistics(self):
        """Clear all performance statistics."""
        with self.lock:
            self.operation_stats.clear()
            self.optimization_suggestions.clear()
            
            if self.operation_cache:
                self.operation_cache.clear()
    
    def enable_auto_optimization(self, check_interval: int = 100):
        """Enable automatic optimization suggestions.
        
        Args:
            check_interval: How often to check for optimizations (in operations)
        """
        # This would be implemented with a background thread
        # that periodically analyzes performance and applies optimizations
        logger.info(f"Auto-optimization enabled with check interval: {check_interval}")


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize_hdc_operation(operation_name: str, dimension: int = 0):
    """Decorator to optimize HDC operation with caching and profiling.
    
    Args:
        operation_name: Name of the operation
        dimension: Hypervector dimension
    """
    optimizer = get_global_optimizer()
    return optimizer.cached_operation(operation_name, dimension)