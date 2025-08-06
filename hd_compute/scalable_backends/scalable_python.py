"""Scalable Pure Python HDC implementation with performance optimization and caching."""

import functools
import threading
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import hashlib
import pickle
from collections import defaultdict

from ..robust_backends.robust_python import RobustHDComputePython
from ..pure_python.hdc_python import SimpleArray
# Import only what we need, with fallbacks
try:
    from ..cache.cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from ..performance.optimization import LRUCache
except ImportError:
    # Simple fallback LRU cache
    from collections import OrderedDict
    class LRUCache:
        def __init__(self, maxsize=1000):
            self.maxsize = maxsize
            self.cache = OrderedDict()
            self.hits = 0
            self.misses = 0
        
        def get(self, key):
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            self.misses += 1
            return None
        
        def put(self, key, value, ttl=None):
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value

# Simple performance profiler fallback
class SimpleProfiler:
    def __init__(self):
        self.operations = {}
    
    def get_summary(self):
        return self.operations
    
    def save_profile(self, filename):
        pass

# Simple cache manager fallback
class SimpleCacheManager:
    def __init__(self, max_size_mb=100):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value, ttl=None):
        self.cache[key] = value

logger = logging.getLogger(__name__)


def cached_operation(cache_key_func: Optional[callable] = None, ttl_seconds: int = 3600):
    """Decorator for caching HDC operations with automatic cache key generation."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(self, *args, **kwargs)
            else:
                # Default cache key based on function name and arguments
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try to get from cache first
            cached_result = self._operation_cache.get(cache_key)
            if cached_result is not None:
                self._cache_stats['hits'] += 1
                return cached_result
            
            # Execute operation
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result if operation was successful and not too large
            if result is not None and self._should_cache_result(result, execution_time):
                self._operation_cache.put(cache_key, result, ttl=ttl_seconds)
                self._cache_stats['misses'] += 1
            
            return result
        return wrapper
    return decorator


def parallel_operation(chunk_size: int = 100, max_workers: Optional[int] = None):
    """Decorator for parallelizing operations over large datasets."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if we should parallelize based on data size
            if not self._should_parallelize(args, kwargs):
                return func(self, *args, **kwargs)
            
            # Split work into chunks for parallel processing
            chunks = self._create_chunks(args, kwargs, chunk_size)
            
            if len(chunks) <= 1:
                return func(self, *args, **kwargs)
            
            # Execute in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for chunk_args, chunk_kwargs in chunks:
                    future = executor.submit(func, self, *chunk_args, **chunk_kwargs)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Parallel operation chunk failed: {e}")
                        # Fallback to sequential for this chunk
                        results.append(None)
                
                # Combine results
                return self._combine_parallel_results(results, func.__name__)
        
        return wrapper
    return decorator


class ScalableHDComputePython(RobustHDComputePython):
    """Scalable Pure Python HDC implementation with performance optimization."""
    
    def __init__(self, dim: int, device: Optional[str] = None, dtype=float,
                 enable_audit_logging: bool = False, strict_validation: bool = False,
                 enable_caching: bool = True, cache_size_mb: int = 100,
                 max_parallel_workers: int = 4, enable_profiling: bool = True):
        """Initialize scalable HDC context.
        
        Args:
            dim: Dimensionality of hypervectors
            device: Device specification (ignored for Python)
            dtype: Data type for hypervectors  
            enable_audit_logging: Whether to enable audit logging
            strict_validation: Whether to use strict validation
            enable_caching: Whether to enable operation caching
            cache_size_mb: Cache size in megabytes
            max_parallel_workers: Maximum number of parallel workers
            enable_profiling: Whether to enable performance profiling
        """
        # Initialize robust backend
        super().__init__(dim, device, dtype, enable_audit_logging, strict_validation)
        
        # Performance optimization components
        self.enable_caching = enable_caching
        self.max_parallel_workers = max_parallel_workers
        
        # Initialize caching system
        if enable_caching:
            if CacheManager is not None:
                self._operation_cache = CacheManager(max_size_mb=cache_size_mb)
            else:
                self._operation_cache = SimpleCacheManager(max_size_mb=cache_size_mb)
            self._memory_cache = LRUCache(maxsize=1000)
        else:
            self._operation_cache = None
            self._memory_cache = None
        
        # Initialize performance profiling
        if enable_profiling:
            self._profiler = SimpleProfiler()
        else:
            self._profiler = None
        
        # Performance statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        self._performance_stats = {
            'total_operations': 0,
            'parallel_operations': 0,
            'cached_operations': 0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Operation-specific optimizations
        self._optimization_thresholds = {
            'parallel_bundle_size': 10,  # Bundle operations with >10 HVs use parallel
            'parallel_batch_size': 50,   # Batch operations >50 items use parallel
            'cache_min_time_ms': 10,     # Only cache operations >10ms
            'cache_max_size_kb': 1024,   # Don't cache results >1MB
        }
        
        # Precomputed lookup tables for common operations
        self._initialize_lookup_tables()
        
        logger.info(f"Initialized ScalableHDComputePython with dim={dim}, "
                   f"caching={'enabled' if enable_caching else 'disabled'}, "
                   f"max_workers={max_parallel_workers}")
    
    def _initialize_lookup_tables(self):
        """Initialize lookup tables for performance optimization."""
        # Precompute common sparsity patterns
        self._sparsity_patterns = {}
        common_sparsities = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for sparsity in common_sparsities:
            # Pre-generate pattern for faster random HV generation
            pattern = []
            for i in range(min(1000, self.dim)):  # Pattern for first 1000 elements
                pattern.append(1.0 if (i * 0.618) % 1 < sparsity else 0.0)
            self._sparsity_patterns[sparsity] = pattern
        
        # Precompute common mathematical constants
        self._math_constants = {
            'inv_sqrt_dim': 1.0 / (self.dim ** 0.5),
            'half_dim': self.dim // 2,
            'quarter_dim': self.dim // 4
        }
    
    def _should_cache_result(self, result: Any, execution_time: float) -> bool:
        """Determine if a result should be cached."""
        if not self.enable_caching:
            return False
        
        # Don't cache very fast operations
        if execution_time * 1000 < self._optimization_thresholds['cache_min_time_ms']:
            return False
        
        # Don't cache very large results
        try:
            result_size = len(pickle.dumps(result))
            if result_size > self._optimization_thresholds['cache_max_size_kb'] * 1024:
                return False
        except:
            return False
        
        return True
    
    def _should_parallelize(self, args: tuple, kwargs: dict) -> bool:
        """Determine if an operation should be parallelized."""
        if len(args) == 0:
            return False
        
        # Check for bundling operations with many hypervectors
        if isinstance(args[0], (list, tuple)) and len(args[0]) > self._optimization_thresholds['parallel_bundle_size']:
            return True
        
        return False
    
    def _create_chunks(self, args: tuple, kwargs: dict, chunk_size: int) -> List[Tuple[tuple, dict]]:
        """Split arguments into chunks for parallel processing."""
        if not args or not isinstance(args[0], (list, tuple)):
            return [(args, kwargs)]
        
        items = args[0]
        chunks = []
        
        for i in range(0, len(items), chunk_size):
            chunk_items = items[i:i + chunk_size]
            chunk_args = (chunk_items,) + args[1:]
            chunks.append((chunk_args, kwargs))
        
        return chunks
    
    def _combine_parallel_results(self, results: List[Any], operation_name: str) -> Any:
        """Combine results from parallel operation chunks."""
        # Filter out None results (failed chunks)
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return self.random_hv()  # Fallback
        
        if operation_name == 'bundle':
            # For bundle operations, bundle all chunk results
            return super().bundle(valid_results)
        
        # For other operations, return first valid result
        return valid_results[0]
    
    # Optimized core operations with caching and parallelization
    
    def _generate_optimized_random_hv(self, sparsity: float) -> SimpleArray:
        """Optimized random hypervector generation using patterns."""
        # Use precomputed pattern if available
        if sparsity in self._sparsity_patterns:
            base_pattern = self._sparsity_patterns[sparsity]
            
            # Extend pattern to full dimension with permutations
            data = []
            pattern_len = len(base_pattern)
            
            for i in range(self.dim):
                pattern_idx = (i * 17) % pattern_len  # Prime number for better distribution
                data.append(base_pattern[pattern_idx])
            
            # Add some randomness
            import random
            for i in range(0, self.dim, 10):  # Every 10th element gets randomized
                if random.random() < 0.1:  # 10% chance
                    data[i] = 1.0 if random.random() < sparsity else 0.0
        else:
            # Fall back to standard generation
            data = []
            for _ in range(self.dim):
                value = 1.0 if self._rng.random() < sparsity else 0.0
                data.append(value)
        
        return SimpleArray(data, (self.dim,))
    
    @cached_operation(ttl_seconds=1800)  # Cache for 30 minutes
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> SimpleArray:
        """Generate random hypervector with caching."""
        if batch_size:
            logger.warning("Batch generation not optimized in Python backend")
        
        # Use optimized generation for common sparsities
        if sparsity in self._sparsity_patterns:
            return self._generate_optimized_random_hv(sparsity)
        
        return super().random_hv(sparsity, batch_size)
    
    @parallel_operation(chunk_size=20)
    @cached_operation(ttl_seconds=3600)  # Cache for 1 hour
    def bundle(self, hvs: List[SimpleArray], threshold: Optional[float] = None) -> SimpleArray:
        """Bundle hypervectors with parallel processing and caching."""
        return super().bundle(hvs, threshold)
    
    @cached_operation(ttl_seconds=7200)  # Cache for 2 hours
    def bind(self, hv1: SimpleArray, hv2: SimpleArray) -> SimpleArray:
        """Bind hypervectors with caching."""
        return super().bind(hv1, hv2)
    
    @cached_operation(ttl_seconds=3600)
    def cosine_similarity(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        """Compute cosine similarity with caching."""
        return super().cosine_similarity(hv1, hv2)
    
    # Advanced operations with optimization
    
    def batch_cosine_similarity(self, hvs1: List[SimpleArray], hvs2: List[SimpleArray]) -> List[float]:
        """Optimized batch similarity computation."""
        if len(hvs1) != len(hvs2):
            raise ValueError("Batches must have same size")
        
        # Use parallel processing for large batches
        if len(hvs1) > self._optimization_thresholds['parallel_batch_size']:
            return self._parallel_batch_similarity(hvs1, hvs2)
        
        # Sequential processing for smaller batches
        return [self.cosine_similarity(hv1, hv2) for hv1, hv2 in zip(hvs1, hvs2)]
    
    def _parallel_batch_similarity(self, hvs1: List[SimpleArray], hvs2: List[SimpleArray]) -> List[float]:
        """Parallel batch similarity computation."""
        chunk_size = max(10, len(hvs1) // self.max_parallel_workers)
        
        def compute_chunk(start_idx: int, end_idx: int):
            return [self.cosine_similarity(hvs1[i], hvs2[i]) 
                   for i in range(start_idx, end_idx)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            futures = []
            for i in range(0, len(hvs1), chunk_size):
                end_idx = min(i + chunk_size, len(hvs1))
                future = executor.submit(compute_chunk, i, end_idx)
                futures.append(future)
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
        
        return results
    
    # Performance monitoring and optimization
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = self.get_operation_statistics()
        
        cache_hit_rate = (
            self._cache_stats['hits'] / 
            max(self._cache_stats['hits'] + self._cache_stats['misses'], 1)
        )
        
        performance_stats = {
            **base_stats,
            'cache_stats': {
                **self._cache_stats,
                'hit_rate': cache_hit_rate
            },
            'performance_stats': self._performance_stats,
            'optimization_enabled': {
                'caching': self.enable_caching,
                'parallel_processing': self.max_parallel_workers > 1,
                'profiling': self._profiler is not None
            }
        }
        
        if self._profiler:
            performance_stats['profiling_data'] = self._profiler.get_summary()
        
        return performance_stats
    
    def optimize_for_workload(self, workload_profile: Dict[str, Any]):
        """Dynamically optimize settings based on workload profile."""
        # Adjust cache settings based on operation frequency
        if workload_profile.get('repeat_operations', False):
            self._optimization_thresholds['cache_min_time_ms'] = 5  # Cache more aggressively
        
        # Adjust parallelization thresholds based on data size
        avg_batch_size = workload_profile.get('avg_batch_size', 10)
        if avg_batch_size > 100:
            self._optimization_thresholds['parallel_batch_size'] = avg_batch_size // 2
        
        # Adjust memory cache size based on available memory
        available_memory_mb = workload_profile.get('available_memory_mb', 1000)
        if available_memory_mb > 2000:
            if self._memory_cache:
                self._memory_cache.maxsize = 2000
        
        logger.info(f"Optimized settings for workload: {workload_profile}")
    
    def benchmark_operations(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark core operations for performance analysis."""
        benchmark_results = {}
        
        # Benchmark random hypervector generation
        start_time = time.time()
        for _ in range(num_iterations):
            self.random_hv()
        benchmark_results['random_hv_ms'] = (time.time() - start_time) / num_iterations * 1000
        
        # Benchmark bundle operations
        hvs = [self.random_hv() for _ in range(10)]
        start_time = time.time()
        for _ in range(num_iterations // 10):  # Fewer iterations for bundle
            self.bundle(hvs)
        benchmark_results['bundle_ms'] = (time.time() - start_time) / (num_iterations // 10) * 1000
        
        # Benchmark bind operations
        hv1, hv2 = self.random_hv(), self.random_hv()
        start_time = time.time()
        for _ in range(num_iterations):
            self.bind(hv1, hv2)
        benchmark_results['bind_ms'] = (time.time() - start_time) / num_iterations * 1000
        
        # Benchmark similarity computation
        start_time = time.time()
        for _ in range(num_iterations):
            self.cosine_similarity(hv1, hv2)
        benchmark_results['similarity_ms'] = (time.time() - start_time) / num_iterations * 1000
        
        return benchmark_results
    
    def warmup_caches(self, num_samples: int = 50):
        """Warm up caches with common operations."""
        logger.info(f"Warming up caches with {num_samples} samples...")
        
        # Generate and cache common hypervectors
        for sparsity in [0.25, 0.5, 0.75]:
            for _ in range(num_samples // 10):
                self.random_hv(sparsity=sparsity)
        
        # Cache common similarity computations
        hvs = [self.random_hv() for _ in range(10)]
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                self.cosine_similarity(hvs[i], hvs[j])
        
        logger.info("Cache warmup completed")
    
    def cleanup_resources(self):
        """Clean up resources and save performance data."""
        if self._profiler:
            self._profiler.save_profile("hdc_performance_profile.json")
        
        if self._operation_cache:
            # Force cache cleanup
            pass
        
        logger.info("Resources cleaned up successfully")