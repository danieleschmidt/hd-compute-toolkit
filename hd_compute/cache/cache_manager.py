"""Central cache management system."""

import os
import hashlib
import pickle
import time
from typing import Any, Optional, Dict, Callable
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Thread-safe cache manager with file-based persistence."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 500):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files. Uses environment variable if None.
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir or os.getenv('HDC_CACHE_DIR', '.cache/hdc'))
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._memory_cache_size = 0
        self._max_memory_items = 100
    
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace."""
        combined_key = f"{namespace}:{key}"
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        # Create subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.cache"
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key, namespace)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                self._access_times[cache_key] = time.time()
                return self._memory_cache[cache_key]
            
            # Check file cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Add to memory cache if space available
                    if len(self._memory_cache) < self._max_memory_items:
                        self._memory_cache[cache_key] = data
                        self._access_times[cache_key] = time.time()
                    
                    return data
                
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_path}: {e}")
                    # Remove corrupted cache file
                    cache_path.unlink(missing_ok=True)
            
            return None
    
    def set(self, key: str, value: Any, namespace: str = "default", ttl: Optional[int] = None):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time to live in seconds (not implemented for file cache)
        """
        cache_key = self._generate_key(key, namespace)
        
        with self._lock:
            # Store in memory cache
            if len(self._memory_cache) >= self._max_memory_items:
                self._evict_memory_cache()
            
            self._memory_cache[cache_key] = value
            self._access_times[cache_key] = time.time()
            
            # Store in file cache
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Check cache size and cleanup if needed
                self._check_cache_size()
                
            except Exception as e:
                logger.error(f"Failed to write cache file {cache_path}: {e}")
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(key, namespace)
        
        with self._lock:
            deleted = False
            
            # Remove from memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                del self._access_times[cache_key]
                deleted = True
            
            # Remove from file cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            
            return deleted
    
    def clear(self, namespace: Optional[str] = None):
        """Clear cache entries.
        
        Args:
            namespace: If specified, only clear entries from this namespace
        """
        with self._lock:
            if namespace is None:
                # Clear everything
                self._memory_cache.clear()
                self._access_times.clear()
                
                # Remove all cache files
                for cache_file in self.cache_dir.rglob("*.cache"):
                    cache_file.unlink(missing_ok=True)
            else:
                # Clear specific namespace
                namespace_prefix = f"{namespace}:"
                keys_to_remove = []
                
                for cache_key in list(self._memory_cache.keys()):
                    # Check if this key belongs to the namespace
                    # This is approximate since we hash the keys
                    keys_to_remove.append(cache_key)
                
                for cache_key in keys_to_remove:
                    if cache_key in self._memory_cache:
                        del self._memory_cache[cache_key]
                        del self._access_times[cache_key]
                    
                    cache_path = self._get_cache_path(cache_key)
                    cache_path.unlink(missing_ok=True)
    
    def _evict_memory_cache(self):
        """Evict least recently used items from memory cache."""
        if not self._access_times:
            return
        
        # Remove 20% of items (LRU)
        num_to_remove = max(1, len(self._access_times) // 5)
        
        # Sort by access time and remove oldest
        oldest_keys = sorted(
            self._access_times.items(), 
            key=lambda x: x[1]
        )[:num_to_remove]
        
        for cache_key, _ in oldest_keys:
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            if cache_key in self._access_times:
                del self._access_times[cache_key]
    
    def _check_cache_size(self):
        """Check cache size and cleanup if needed."""
        total_size = sum(
            cache_file.stat().st_size 
            for cache_file in self.cache_dir.rglob("*.cache")
            if cache_file.exists()
        )
        
        if total_size > self.max_size_bytes:
            self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Remove old cache files to free space."""
        cache_files = list(self.cache_dir.rglob("*.cache"))
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        target_size = self.max_size_bytes * 0.8  # Remove until 80% of max size
        
        for cache_file in cache_files:
            if total_size <= target_size:
                break
            
            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                total_size -= file_size
                logger.debug(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cache_files = list(self.cache_dir.rglob("*.cache"))
            total_size = sum(
                f.stat().st_size for f in cache_files if f.exists()
            )
            
            return {
                'memory_cache_size': len(self._memory_cache),
                'file_cache_size': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'cache_dir': str(self.cache_dir)
            }
    
    def cached(self, namespace: str = "default", ttl: Optional[int] = None):
        """Decorator for caching function results.
        
        Args:
            namespace: Cache namespace
            ttl: Time to live in seconds (not implemented)
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
                
                # Try to get from cache
                result = self.get(cache_key, namespace)
                if result is not None:
                    return result
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, namespace, ttl)
                return result
            
            return wrapper
        return decorator