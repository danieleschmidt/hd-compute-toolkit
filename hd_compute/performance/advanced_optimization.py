"""
Advanced Performance Optimization for HDC Research
=================================================

Implements cutting-edge optimization techniques:
- Adaptive caching with LRU and frequency-based eviction
- Vectorized operations with SIMD-like optimizations
- Memory pool management and object recycling
- Concurrent processing with thread and process pools
- Performance profiling and auto-tuning
"""

import time
import threading
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
from functools import wraps, lru_cache
import hashlib
import pickle


class AdaptiveCache:
    """Advanced caching system with multiple eviction strategies."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.creation_times[key] > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            # Update access statistics
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
            
            # Move to end (LRU behavior)
            value = self.cache.pop(key)
            self.cache[key] = value
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            current_time = time.time()
            
            # Update existing item
            if key in self.cache:
                self.cache[key] = value
                self.access_times[key] = current_time
                return
            
            # Add new item
            if len(self.cache) >= self.max_size:
                self._evict_item()
            
            self.cache[key] = value
            self.creation_times[key] = current_time
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict_item(self) -> None:
        """Evict item using hybrid LRU + LFU strategy."""
        if not self.cache:
            return
        
        current_time = time.time()
        
        # Remove expired items first
        expired_keys = [
            key for key, create_time in self.creation_times.items()
            if current_time - create_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_key(key)
        
        if len(self.cache) < self.max_size:
            return
        
        # Hybrid eviction: consider both recency and frequency
        scores = {}
        for key in self.cache:
            recency_score = current_time - self.access_times[key]  # Higher = older
            frequency_score = 1.0 / (self.access_counts[key] + 1)  # Higher = less frequent
            scores[key] = recency_score * 0.7 + frequency_score * 0.3
        
        # Evict item with highest score (oldest + least frequent)
        worst_key = max(scores.keys(), key=lambda k: scores[k])
        self._remove_key(worst_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key and all associated metadata."""
        self.cache.pop(key, None)
        self.access_counts.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'total_accesses': total_accesses,
                'average_access_count': total_accesses / max(1, len(self.cache)),
                'ttl_seconds': self.ttl_seconds
            }


class HypervectorMemoryPool:
    """Memory pool for efficient hypervector allocation and reuse."""
    
    def __init__(self, dim: int, initial_size: int = 100, max_size: int = 1000):
        self.dim = dim
        self.initial_size = initial_size
        self.max_size = max_size
        self.available_vectors = []
        self.in_use_count = 0
        self.total_allocated = 0
        self.lock = threading.Lock()
        
        # Pre-allocate initial vectors
        self._allocate_vectors(initial_size)
    
    def _allocate_vectors(self, count: int) -> None:
        """Allocate new vectors to the pool."""
        for _ in range(count):
            vector = np.zeros(self.dim, dtype=np.float32)
            self.available_vectors.append(vector)
            self.total_allocated += 1
    
    def get_vector(self) -> np.ndarray:
        """Get a vector from the pool."""
        with self.lock:
            if not self.available_vectors:
                if self.total_allocated < self.max_size:
                    self._allocate_vectors(min(10, self.max_size - self.total_allocated))
                else:
                    # Pool exhausted, create new vector
                    return np.zeros(self.dim, dtype=np.float32)
            
            if self.available_vectors:
                vector = self.available_vectors.pop()
                self.in_use_count += 1
                return vector
            else:
                return np.zeros(self.dim, dtype=np.float32)
    
    def return_vector(self, vector: np.ndarray) -> None:
        """Return a vector to the pool."""
        if vector.shape != (self.dim,):
            return  # Invalid vector, don't add to pool
        
        with self.lock:
            if len(self.available_vectors) < self.max_size:
                # Reset vector and return to pool
                vector.fill(0.0)
                self.available_vectors.append(vector)
                if self.in_use_count > 0:
                    self.in_use_count -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'available': len(self.available_vectors),
                'in_use': self.in_use_count,
                'total_allocated': self.total_allocated,
                'max_size': self.max_size,
                'utilization': self.in_use_count / max(1, self.total_allocated)
            }


class VectorizedOperations:
    """Optimized vectorized operations for hyperdimensional computing."""
    
    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and batch of vectors efficiently."""
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(vectors.shape[0])
        
        query_normalized = query / query_norm
        
        # Normalize vectors
        vector_norms = np.linalg.norm(vectors, axis=1)
        vector_norms[vector_norms == 0] = 1  # Avoid division by zero
        vectors_normalized = vectors / vector_norms[:, np.newaxis]
        
        # Compute dot products (cosine similarities)
        similarities = np.dot(vectors_normalized, query_normalized)
        
        return similarities
    
    @staticmethod
    def batch_binding(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """Perform batch binding operation (element-wise XOR for binary vectors)."""
        if vectors1.dtype == np.bool_ or np.issubdtype(vectors1.dtype, np.integer):
            # Binary binding using XOR
            return np.logical_xor(vectors1, vectors2).astype(vectors1.dtype)
        else:
            # Real-valued binding using circular convolution approximation
            return np.multiply(vectors1, vectors2)
    
    @staticmethod
    def batch_bundling(vectors: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform batch bundling operation (weighted sum)."""
        if weights is None:
            return np.mean(vectors, axis=0)
        else:
            weights = weights / np.sum(weights)  # Normalize weights
            return np.average(vectors, axis=0, weights=weights)
    
    @staticmethod
    def batch_permutation(vectors: np.ndarray, shift: int) -> np.ndarray:
        """Perform batch circular permutation."""
        return np.roll(vectors, shift, axis=1)
    
    @staticmethod
    def parallel_distance_matrix(vectors: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        """Compute distance matrix between all vector pairs."""
        n = vectors.shape[0]
        
        if metric == 'cosine':
            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1)
            norms[norms == 0] = 1
            normalized = vectors / norms[:, np.newaxis]
            
            # Compute cosine similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)
            return 1 - similarity_matrix  # Convert to distance
        
        elif metric == 'hamming':
            # For binary vectors
            distances = np.zeros((n, n))
            for i in range(n):
                distances[i] = np.sum(vectors != vectors[i], axis=1) / vectors.shape[1]
            return distances
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class ConcurrentProcessor:
    """Concurrent processing manager for HDC operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.thread_pool = None
        self.process_pool = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    def parallel_apply(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Apply function to items in parallel."""
        if len(items) <= 1:
            return [func(item) for item in items]
        
        if use_processes:
            if not self.process_pool:
                from concurrent.futures import ProcessPoolExecutor
                self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            executor = self.process_pool
        else:
            if not self.thread_pool:
                from concurrent.futures import ThreadPoolExecutor
                self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            executor = self.thread_pool
        
        futures = [executor.submit(func, item) for item in items]
        results = [future.result() for future in futures]
        
        return results
    
    def parallel_map_reduce(self, map_func: Callable, reduce_func: Callable, 
                           items: List[Any], chunk_size: Optional[int] = None) -> Any:
        """Perform parallel map-reduce operation."""
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Apply map function to each chunk
        def map_chunk(chunk):
            return [map_func(item) for item in chunk]
        
        mapped_chunks = self.parallel_apply(map_chunk, chunks, use_processes=True)
        
        # Flatten and reduce
        all_mapped = [item for chunk in mapped_chunks for item in chunk]
        
        # Apply reduce function
        result = all_mapped[0] if all_mapped else None
        for item in all_mapped[1:]:
            result = reduce_func(result, item)
        
        return result


class PerformanceProfiler:
    """Lightweight performance profiler for HDC operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.memory_usage = {}
        
    def time_operation(self, operation_name: str):
        """Decorator to time operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.timings[operation_name].append(execution_time)
                    self.operation_counts[operation_name] += 1
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.timings[f"{operation_name}_error"].append(execution_time)
                    raise
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times),
                    'ops_per_second': len(times) / sum(times) if sum(times) > 0 else 0
                }
        
        return stats
    
    def get_top_operations(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get operations with highest total time."""
        operation_times = [
            (op, sum(times)) for op, times in self.timings.items()
        ]
        operation_times.sort(key=lambda x: x[1], reverse=True)
        return operation_times[:limit]


class AutoTuner:
    """Automatic parameter tuning for HDC algorithms."""
    
    def __init__(self):
        self.performance_history = {}
        self.parameter_sets = {}
        self.best_parameters = {}
        
    def register_tunable_function(self, func_name: str, parameter_ranges: Dict[str, tuple]):
        """Register a function for auto-tuning with parameter ranges."""
        self.parameter_sets[func_name] = parameter_ranges
        self.performance_history[func_name] = []
    
    def suggest_parameters(self, func_name: str) -> Dict[str, Any]:
        """Suggest parameters for a function based on performance history."""
        if func_name not in self.parameter_sets:
            return {}
        
        if func_name in self.best_parameters:
            # Return best known parameters with small random variations
            best = self.best_parameters[func_name]
            suggested = {}
            
            for param, value in best.items():
                if isinstance(value, (int, float)):
                    # Add 5% random variation
                    variation = value * 0.05 * (np.random.random() - 0.5)
                    suggested[param] = type(value)(value + variation)
                else:
                    suggested[param] = value
            
            return suggested
        else:
            # Random sampling from parameter ranges
            ranges = self.parameter_sets[func_name]
            suggested = {}
            
            for param, (min_val, max_val) in ranges.items():
                if isinstance(min_val, int):
                    suggested[param] = np.random.randint(min_val, max_val + 1)
                else:
                    suggested[param] = np.random.uniform(min_val, max_val)
            
            return suggested
    
    def record_performance(self, func_name: str, parameters: Dict[str, Any], 
                         performance_metric: float) -> None:
        """Record performance for given parameters."""
        if func_name not in self.performance_history:
            self.performance_history[func_name] = []
        
        self.performance_history[func_name].append({
            'parameters': parameters.copy(),
            'performance': performance_metric,
            'timestamp': time.time()
        })
        
        # Update best parameters if this is the best performance so far
        history = self.performance_history[func_name]
        best_record = max(history, key=lambda x: x['performance'])
        
        if (func_name not in self.best_parameters or 
            best_record['performance'] > self._get_best_performance(func_name)):
            self.best_parameters[func_name] = best_record['parameters'].copy()
    
    def _get_best_performance(self, func_name: str) -> float:
        """Get best performance for a function."""
        if func_name not in self.performance_history:
            return -float('inf')
        
        history = self.performance_history[func_name]
        return max(record['performance'] for record in history) if history else -float('inf')


# Global instances for easy access
global_cache = AdaptiveCache(max_size=5000, ttl_seconds=7200)
global_profiler = PerformanceProfiler()
global_tuner = AutoTuner()


def cached_operation(cache_key_func: Optional[Callable] = None, ttl: Optional[float] = None):
    """Decorator for caching operation results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            global_cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def performance_monitored(operation_name: str):
    """Decorator for performance monitoring."""
    return global_profiler.time_operation(operation_name)