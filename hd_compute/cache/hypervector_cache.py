"""Specialized cache for hypervectors with optimized storage."""

import hashlib
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class HypervectorCache:
    """Specialized cache for hypervectors with compression and deduplication."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize hypervector cache.
        
        Args:
            cache_manager: Cache manager instance. Creates new one if None.
        """
        self.cache_manager = cache_manager or CacheManager()
        self.namespace = "hypervectors"
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._stores = 0
    
    def _generate_hypervector_key(
        self, 
        dimension: int, 
        sparsity: float, 
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate unique key for hypervector based on parameters."""
        key_components = [
            f"dim_{dimension}",
            f"sparsity_{sparsity:.6f}"
        ]
        
        if seed is not None:
            key_components.append(f"seed_{seed}")
        
        if metadata:
            sorted_metadata = sorted(metadata.items())
            metadata_str = "_".join(f"{k}_{v}" for k, v in sorted_metadata)
            key_components.append(f"meta_{metadata_str}")
        
        key = "_".join(key_components)
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_hypervector(
        self,
        dimension: int,
        sparsity: float = 0.5,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """Get hypervector from cache.
        
        Args:
            dimension: Hypervector dimension
            sparsity: Sparsity level (fraction of 1s)
            seed: Random seed used for generation
            metadata: Additional metadata for cache key
            
        Returns:
            Cached hypervector or None if not found
        """
        cache_key = self._generate_hypervector_key(dimension, sparsity, seed, metadata)
        
        try:
            cached_data = self.cache_manager.get(cache_key, self.namespace)
            
            if cached_data is not None:
                self._hits += 1
                
                # Cached data includes the hypervector and metadata
                hypervector = cached_data['hypervector']
                
                # Validate cached hypervector
                if hypervector.shape[0] == dimension:
                    return hypervector
                else:
                    logger.warning(f"Cached hypervector dimension mismatch: {hypervector.shape[0]} != {dimension}")
                    # Remove invalid cache entry
                    self.cache_manager.delete(cache_key, self.namespace)
            
            self._misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving hypervector from cache: {e}")
            self._misses += 1
            return None
    
    def store_hypervector(
        self,
        hypervector: np.ndarray,
        dimension: int,
        sparsity: float = 0.5,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store hypervector in cache.
        
        Args:
            hypervector: Hypervector to cache
            dimension: Hypervector dimension
            sparsity: Sparsity level
            seed: Random seed used for generation
            metadata: Additional metadata
            
        Returns:
            Cache key for the stored hypervector
        """
        cache_key = self._generate_hypervector_key(dimension, sparsity, seed, metadata)
        
        try:
            # Validate hypervector
            if hypervector.shape[0] != dimension:
                raise ValueError(f"Hypervector dimension {hypervector.shape[0]} != expected {dimension}")
            
            # Compress hypervector for storage
            compressed_hv = self._compress_hypervector(hypervector)
            
            # Store with metadata
            cache_data = {
                'hypervector': compressed_hv,
                'dimension': dimension,
                'sparsity': sparsity,
                'seed': seed,
                'metadata': metadata or {},
                'original_dtype': str(hypervector.dtype),
                'compressed': True
            }
            
            self.cache_manager.set(cache_key, cache_data, self.namespace)
            self._stores += 1
            
            return cache_key
            
        except Exception as e:
            logger.error(f"Error storing hypervector in cache: {e}")
            raise
    
    def _compress_hypervector(self, hypervector: np.ndarray) -> np.ndarray:
        """Compress binary hypervector for storage.
        
        For binary hypervectors, we can pack 8 bits into each byte.
        """
        if hypervector.dtype == bool:
            # Pack boolean array into bytes
            return np.packbits(hypervector)
        elif hypervector.dtype in [np.float32, np.float64]:
            # For float hypervectors, convert to boolean first
            binary_hv = (hypervector > 0.5).astype(bool)
            return np.packbits(binary_hv)
        else:
            # For other types, store as-is
            return hypervector
    
    def _decompress_hypervector(self, compressed_hv: np.ndarray, dimension: int, original_dtype: str) -> np.ndarray:
        """Decompress hypervector from storage."""
        if len(compressed_hv) * 8 >= dimension:
            # Unpack bits and take only the required dimension
            unpacked = np.unpackbits(compressed_hv)[:dimension]
            
            if original_dtype == 'bool':
                return unpacked.astype(bool)
            elif original_dtype in ['float32', 'float64']:
                return unpacked.astype(np.float32)
            else:
                return unpacked
        else:
            # Not compressed, return as-is
            return compressed_hv
    
    def get_cached_hypervector_or_generate(
        self,
        hdc_backend,
        dimension: int,
        sparsity: float = 0.5,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Get hypervector from cache or generate and cache it.
        
        Args:
            hdc_backend: HDC backend instance for generation
            dimension: Hypervector dimension
            sparsity: Sparsity level
            seed: Random seed
            metadata: Additional metadata
            
        Returns:
            Hypervector (cached or newly generated)
        """
        # Try to get from cache first
        cached_hv = self.get_hypervector(dimension, sparsity, seed, metadata)
        
        if cached_hv is not None:
            return cached_hv
        
        # Generate new hypervector
        if seed is not None:
            # Set seed in backend if supported
            if hasattr(hdc_backend, '_generator'):
                hdc_backend._generator.manual_seed(seed)
            elif hasattr(hdc_backend, 'key'):
                import jax.random as random
                hdc_backend.key = random.PRNGKey(seed)
        
        new_hv = hdc_backend.random_hv(sparsity=sparsity)
        
        # Store in cache
        self.store_hypervector(new_hv, dimension, sparsity, seed, metadata)
        
        return new_hv
    
    def clear_cache(self):
        """Clear all cached hypervectors."""
        self.cache_manager.clear(self.namespace)
        self._reset_stats()
    
    def _reset_stats(self):
        """Reset cache statistics."""
        self._hits = 0
        self._misses = 0
        self._stores = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self.cache_manager.get_cache_stats()
        
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'stores': self._stores,
            'hit_rate': hit_rate,
            'total_cache_size_mb': cache_stats['total_size_mb'],
            'hypervector_namespace_stats': cache_stats
        }
    
    def cleanup_old_entries(self, max_age_days: int = 7):
        """Clean up old hypervector cache entries.
        
        Args:
            max_age_days: Maximum age in days for cache entries
        """
        # This would require modification of CacheManager to support TTL
        # For now, we can clear the entire cache
        logger.info("Cleaning up hypervector cache (clearing all entries)")
        self.clear_cache()
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """Estimate memory usage of cached hypervectors."""
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Rough estimation based on cache size
        estimated_hypervectors = cache_stats.get('file_cache_size', 0)
        avg_hypervector_size_mb = 0.01  # Rough estimate for compressed binary hypervectors
        
        return {
            'estimated_hypervectors_count': estimated_hypervectors,
            'estimated_total_memory_mb': estimated_hypervectors * avg_hypervector_size_mb,
            'cache_file_size_mb': cache_stats['total_size_mb']
        }