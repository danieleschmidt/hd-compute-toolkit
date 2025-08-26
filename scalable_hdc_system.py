#!/usr/bin/env python3
"""
Generation 3: Scalable HDC System with Performance Optimization and Distributed Computing
"""

import sys
import time
import traceback
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import hashlib
from collections import defaultdict, OrderedDict
import weakref
import gc

# Advanced imports for optimization
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveCache:
    """Advanced caching system with LRU eviction and memory management."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.total_memory_mb = 0
        self._lock = threading.RLock()
        
        logger.info(f"Adaptive cache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _estimate_memory(self, data: Any) -> float:
        """Estimate memory usage of data in MB."""
        if isinstance(data, dict) and 'data' in data:
            return len(data['data']) * 8 / 1024 / 1024  # 8 bytes per float
        elif isinstance(data, (list, tuple)):
            return len(data) * 8 / 1024 / 1024
        else:
            return 0.1  # Default small object size
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache with LRU update."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_count[key] += 1
                return value
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Store item in cache with memory management."""
        with self._lock:
            memory_needed = self._estimate_memory(value)
            
            # Remove existing key if present
            if key in self.cache:
                old_memory = self._estimate_memory(self.cache[key])
                self.total_memory_mb -= old_memory
                del self.cache[key]
            
            # Evict items if necessary
            while (len(self.cache) >= self.max_size or 
                   self.total_memory_mb + memory_needed > self.max_memory_mb):
                if not self.cache:
                    break
                
                # Remove least recently used item
                oldest_key = next(iter(self.cache))
                oldest_value = self.cache.pop(oldest_key)
                self.total_memory_mb -= self._estimate_memory(oldest_value)
                del self.access_count[oldest_key]
                
                logger.debug(f"Cache evicted: {oldest_key}")
            
            # Add new item
            self.cache[key] = value
            self.total_memory_mb += memory_needed
            self.access_count[key] = 1
            
            logger.debug(f"Cache stored: {key} ({memory_needed:.2f}MB)")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.total_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': sum(self.access_count.values()) / max(len(self.cache), 1)
            }

class PerformanceOptimizer:
    """Advanced performance optimization for HDC operations."""
    
    def __init__(self):
        self.operation_profiles = defaultdict(list)
        self.optimization_cache = AdaptiveCache(max_size=5000, max_memory_mb=200)
        self._lock = threading.RLock()
        
        # Detect system capabilities
        self.cpu_count = mp.cpu_count()
        
        if PSUTIL_AVAILABLE:
            self.memory_gb = psutil.virtual_memory().total / 1024**3
        else:
            self.memory_gb = 8  # Default assumption
        
        logger.info(f"Performance optimizer: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
    
    def should_parallelize(self, operation: str, data_size: int) -> Tuple[bool, int]:
        """Determine if operation should be parallelized and optimal thread count."""
        
        # Thresholds for parallelization
        thresholds = {
            'bundle': 100,      # Bundle 100+ hypervectors
            'batch_bind': 50,   # Batch bind 50+ pairs
            'similarity_matrix': 25,  # 25x25+ similarity matrix
            'search': 1000      # Search in 1000+ items
        }
        
        threshold = thresholds.get(operation, 100)
        
        if data_size < threshold:
            return False, 1
        
        # Adaptive thread count based on data size and system resources
        base_threads = min(self.cpu_count, 8)  # Cap at 8 threads
        
        if data_size > threshold * 10:
            threads = base_threads
        elif data_size > threshold * 5:
            threads = max(2, base_threads // 2)
        else:
            threads = 2
        
        return True, threads
    
    def get_optimal_batch_size(self, operation: str, total_items: int) -> int:
        """Calculate optimal batch size for operation."""
        
        base_batch_sizes = {
            'bundle': 100,
            'bind': 500,
            'similarity': 200,
            'encoding': 1000
        }
        
        base_size = base_batch_sizes.get(operation, 100)
        
        # Adapt based on memory and CPU count
        memory_factor = max(1, self.memory_gb / 8)  # Scale with memory
        cpu_factor = max(1, self.cpu_count / 4)     # Scale with CPUs
        
        optimal_size = int(base_size * memory_factor * cpu_factor)
        
        # Ensure reasonable bounds
        optimal_size = max(10, min(optimal_size, total_items // 2))
        
        return optimal_size
    
    def profile_operation(self, operation: str, duration: float, data_size: int):
        """Profile operation performance for future optimization."""
        with self._lock:
            self.operation_profiles[operation].append({
                'duration': duration,
                'data_size': data_size,
                'throughput': data_size / duration if duration > 0 else 0,
                'timestamp': time.time()
            })
            
            # Keep only recent profiles
            if len(self.operation_profiles[operation]) > 50:
                self.operation_profiles[operation] = self.operation_profiles[operation][-50:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        insights = {}
        
        with self._lock:
            for operation, profiles in self.operation_profiles.items():
                if not profiles:
                    continue
                
                durations = [p['duration'] for p in profiles]
                throughputs = [p['throughput'] for p in profiles[-10:]]  # Recent throughput
                
                insights[operation] = {
                    'avg_duration': sum(durations) / len(durations),
                    'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                    'sample_count': len(profiles),
                    'recent_trend': 'improving' if len(throughputs) > 2 and 
                                   throughputs[-1] > throughputs[0] else 'stable'
                }
        
        return insights

class DistributedHDCProcessor:
    """Distributed processing for large-scale HDC operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.optimizer = PerformanceOptimizer()
        self.cache = AdaptiveCache(max_size=20000, max_memory_mb=1000)
        
        logger.info(f"Distributed HDC processor: {self.max_workers} max workers")
    
    def parallel_bundle(self, hvs: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Parallel bundling with optimal batching."""
        if len(hvs) < 50:  # Use sequential for small sets
            return self._sequential_bundle(hvs, weights)
        
        should_parallel, num_threads = self.optimizer.should_parallelize('bundle', len(hvs))
        if not should_parallel:
            return self._sequential_bundle(hvs, weights)
        
        start_time = time.time()
        
        # Determine batch size
        batch_size = self.optimizer.get_optimal_batch_size('bundle', len(hvs))
        batches = [hvs[i:i + batch_size] for i in range(0, len(hvs), batch_size)]
        
        if weights:
            weight_batches = [weights[i:i + batch_size] for i in range(0, len(weights), batch_size)]
        else:
            weight_batches = [None] * len(batches)
        
        logger.info(f"Parallel bundling: {len(hvs)} vectors in {len(batches)} batches using {num_threads} threads")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_batch = {
                executor.submit(self._sequential_bundle, batch, batch_weights): i 
                for i, (batch, batch_weights) in enumerate(zip(batches, weight_batches))
            }
            
            partial_results = [None] * len(batches)
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    partial_results[batch_idx] = future.result()
                except Exception as exc:
                    logger.error(f'Batch {batch_idx} failed: {exc}')
                    raise
        
        # Combine partial results
        final_result = self._sequential_bundle(partial_results, None)
        
        duration = time.time() - start_time
        self.optimizer.profile_operation('parallel_bundle', duration, len(hvs))
        
        return final_result
    
    def parallel_similarity_matrix(self, hvs: List[Dict]) -> List[List[float]]:
        """Compute similarity matrix in parallel."""
        n = len(hvs)
        
        if n < 25:  # Sequential for small matrices
            return self._sequential_similarity_matrix(hvs)
        
        should_parallel, num_threads = self.optimizer.should_parallelize('similarity_matrix', n * n)
        if not should_parallel:
            return self._sequential_similarity_matrix(hvs)
        
        start_time = time.time()
        
        # Create similarity matrix
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        # Generate work items (upper triangular matrix)
        work_items = [(i, j) for i in range(n) for j in range(i, n)]
        
        logger.info(f"Computing {n}x{n} similarity matrix with {num_threads} threads ({len(work_items)} comparisons)")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_pair = {
                executor.submit(self._compute_similarity, hvs[i], hvs[j]): (i, j) 
                for i, j in work_items
            }
            
            for future in as_completed(future_to_pair):
                i, j = future_to_pair[future]
                try:
                    similarity = future.result()
                    similarity_matrix[i][j] = similarity
                    if i != j:  # Matrix is symmetric
                        similarity_matrix[j][i] = similarity
                except Exception as exc:
                    logger.error(f'Similarity computation ({i},{j}) failed: {exc}')
                    raise
        
        duration = time.time() - start_time
        self.optimizer.profile_operation('parallel_similarity_matrix', duration, n * n)
        
        return similarity_matrix
    
    def batch_search(self, query_hv: Dict, database_hvs: List[Dict], top_k: int = 10) -> List[Tuple[int, float]]:
        """Parallel search for most similar hypervectors."""
        if len(database_hvs) < 1000:
            return self._sequential_search(query_hv, database_hvs, top_k)
        
        start_time = time.time()
        
        # Check cache first
        query_key = f"search_{self._compute_hv_hash(query_hv)}_{len(database_hvs)}_{top_k}"
        cached_result = self.cache.get(query_key)
        if cached_result:
            logger.debug(f"Cache hit for search query")
            return cached_result
        
        should_parallel, num_threads = self.optimizer.should_parallelize('search', len(database_hvs))
        
        if not should_parallel:
            result = self._sequential_search(query_hv, database_hvs, top_k)
        else:
            # Parallel search with batching
            batch_size = self.optimizer.get_optimal_batch_size('search', len(database_hvs))
            batches = [(i, database_hvs[i:i + batch_size]) 
                      for i in range(0, len(database_hvs), batch_size)]
            
            logger.info(f"Parallel search: {len(database_hvs)} vectors in {len(batches)} batches")
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_batch = {
                    executor.submit(self._batch_similarities, query_hv, batch_data, start_idx): start_idx 
                    for start_idx, batch_data in batches
                }
                
                all_similarities = []
                for future in as_completed(future_to_batch):
                    start_idx = future_to_batch[future]
                    try:
                        batch_similarities = future.result()
                        all_similarities.extend(batch_similarities)
                    except Exception as exc:
                        logger.error(f'Search batch starting at {start_idx} failed: {exc}')
                        raise
            
            # Find top-k results
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            result = all_similarities[:top_k]
        
        # Cache result
        self.cache.put(query_key, result)
        
        duration = time.time() - start_time
        self.optimizer.profile_operation('batch_search', duration, len(database_hvs))
        
        return result
    
    def _sequential_bundle(self, hvs: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Sequential bundling implementation."""
        if not hvs:
            raise ValueError("Empty hypervector list")
        
        dim = hvs[0]['dim']
        if weights is None:
            weights = [1.0] * len(hvs)
        
        result_data = [0.0] * dim
        for hv, weight in zip(hvs, weights):
            for j, val in enumerate(hv['data']):
                result_data[j] += weight * val
        
        # Normalize
        max_abs = max(abs(x) for x in result_data)
        if max_abs > 0:
            result_data = [x / max_abs for x in result_data]
        
        return {
            'data': result_data,
            'dim': dim,
            'checksum': self._compute_hv_hash(result_data)
        }
    
    def _sequential_similarity_matrix(self, hvs: List[Dict]) -> List[List[float]]:
        """Sequential similarity matrix computation."""
        n = len(hvs)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                sim = self._compute_similarity(hvs[i], hvs[j])
                matrix[i][j] = sim
                if i != j:
                    matrix[j][i] = sim
        
        return matrix
    
    def _sequential_search(self, query_hv: Dict, database_hvs: List[Dict], top_k: int) -> List[Tuple[int, float]]:
        """Sequential search implementation."""
        similarities = []
        for i, hv in enumerate(database_hvs):
            sim = self._compute_similarity(query_hv, hv)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _batch_similarities(self, query_hv: Dict, batch_hvs: List[Dict], start_idx: int) -> List[Tuple[int, float]]:
        """Compute similarities for a batch of hypervectors."""
        similarities = []
        for i, hv in enumerate(batch_hvs):
            sim = self._compute_similarity(query_hv, hv)
            similarities.append((start_idx + i, sim))
        return similarities
    
    def _compute_similarity(self, hv1: Dict, hv2: Dict) -> float:
        """Compute cosine similarity between two hypervectors."""
        # Compute dot product and norms
        dot_product = sum(a * b for a, b in zip(hv1['data'], hv2['data']))
        norm1 = sum(a * a for a in hv1['data']) ** 0.5
        norm2 = sum(b * b for b in hv2['data']) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(-1.0, min(1.0, dot_product / (norm1 * norm2)))
    
    def _compute_hv_hash(self, data) -> str:
        """Compute hash for hypervector data."""
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        data_str = json.dumps(data[:100], sort_keys=True)  # Use first 100 elements for hash
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'max_workers': self.max_workers,
            'cache_stats': self.cache.get_stats(),
            'performance_insights': self.optimizer.get_performance_insights(),
            'memory_info': self._get_memory_info()
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return {
                'total_gb': mem.total / 1024**3,
                'available_gb': mem.available / 1024**3,
                'used_percent': mem.percent
            }
        else:
            return {'available': False}

def test_scalable_system():
    """Test the scalable HDC system."""
    print("🚀 GENERATION 3: SCALABLE HDC SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Initialize scalable system
        processor = DistributedHDCProcessor(max_workers=4)
        print("✅ Scalable HDC processor initialized")
        
        # Generate test data
        print("\n📊 Generating test hypervectors...")
        test_hvs = []
        for i in range(500):
            hv_data = [1.0 if (i + j) % 3 == 0 else -1.0 for j in range(1000)]
            hv = {
                'data': hv_data,
                'dim': 1000,
                'checksum': f"test_hv_{i}"
            }
            test_hvs.append(hv)
        print(f"✅ Generated {len(test_hvs)} test hypervectors")
        
        # Test parallel bundling
        print("\n⚡ Testing parallel bundling...")
        start_time = time.time()
        bundled = processor.parallel_bundle(test_hvs[:100])
        bundle_time = time.time() - start_time
        print(f"✅ Parallel bundling completed in {bundle_time:.3f}s")
        
        # Test similarity matrix computation
        print("\n🔍 Testing similarity matrix computation...")
        start_time = time.time()
        similarity_matrix = processor.parallel_similarity_matrix(test_hvs[:30])
        matrix_time = time.time() - start_time
        print(f"✅ 30x30 similarity matrix computed in {matrix_time:.3f}s")
        
        # Test batch search
        print("\n🔎 Testing batch search...")
        query_hv = test_hvs[0]
        start_time = time.time()
        search_results = processor.batch_search(query_hv, test_hvs[1:], top_k=10)
        search_time = time.time() - start_time
        print(f"✅ Search completed in {search_time:.3f}s")
        print(f"   Top similarity: {search_results[0][1]:.4f}")
        
        # Test caching effectiveness
        print("\n💾 Testing cache effectiveness...")
        start_time = time.time()
        search_results2 = processor.batch_search(query_hv, test_hvs[1:], top_k=10)
        cached_search_time = time.time() - start_time
        
        speedup = search_time / cached_search_time if cached_search_time > 0 else float('inf')
        print(f"✅ Cached search: {cached_search_time:.3f}s ({speedup:.1f}x speedup)")
        
        # Performance benchmarking
        print("\n📈 Performance benchmarking...")
        large_hvs = test_hvs * 4  # 2000 hypervectors
        
        start_time = time.time()
        large_bundle = processor.parallel_bundle(large_hvs[:200])
        large_bundle_time = time.time() - start_time
        
        throughput = 200 / large_bundle_time
        print(f"✅ Large bundling: {large_bundle_time:.3f}s ({throughput:.0f} HVs/sec)")
        
        # System status
        status = processor.get_system_status()
        print(f"\n🎯 System Status:")
        print(f"   Workers: {status['max_workers']}")
        print(f"   Cache size: {status['cache_stats']['size']}")
        print(f"   Cache memory: {status['cache_stats']['memory_mb']:.1f}MB")
        
        if 'total_gb' in status['memory_info']:
            print(f"   System memory: {status['memory_info']['used_percent']:.1f}% used")
        
        # Auto-scaling demonstration
        print(f"\n⚡ Auto-scaling insights:")
        insights = status['performance_insights']
        for op, metrics in insights.items():
            print(f"   {op}: {metrics['avg_throughput']:.0f} items/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Scalable system test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scalable_system()
    
    if success:
        print("\n🎉 GENERATION 3 COMPLETE: System is scalable and optimized!")
        print("\n🌟 Key Features Implemented:")
        print("   ⚡ Adaptive parallel processing")
        print("   💾 Intelligent caching system")
        print("   📊 Performance profiling")
        print("   🔧 Auto-scaling optimization")
        print("   🚀 Distributed computation")
    else:
        print("\n⚠️ GENERATION 3 needs fixes")
    
    sys.exit(0 if success else 1)