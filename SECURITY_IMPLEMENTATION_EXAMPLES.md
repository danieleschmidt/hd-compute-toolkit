# Security Implementation Examples

This document provides specific examples of how to update the vulnerable code identified in the security scan with secure alternatives.

## 1. Updating Cache Manager (High Priority)

### Before: Unsafe Pickle Usage

```python
# /root/repo/hd_compute/cache/cache_manager.py:69
def _load_from_disk(self, cache_key: str) -> Optional[Any]:
    cache_path = self._get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)  # VULNERABLE - no validation
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_path}: {e}")
    return None
```

### After: Secure Pickle with Validation

```python
# Updated /root/repo/hd_compute/cache/cache_manager.py
from ..security.secure_serialization import safe_pickle_load, safe_pickle_dump, migrate_legacy_pickle, is_secure_pickle_file

def _load_from_disk(self, cache_key: str) -> Optional[Any]:
    """Load data from disk cache with security validation."""
    cache_path = self._get_cache_path(cache_key)
    
    if not cache_path.exists():
        return None
    
    try:
        # Try secure format first
        if is_secure_pickle_file(str(cache_path)):
            data = safe_pickle_load(str(cache_path))
        else:
            # Migrate legacy format
            logger.info(f"Migrating legacy cache file to secure format: {cache_path}")
            data = migrate_legacy_pickle(str(cache_path))
        
        # Add to memory cache if space available
        if len(self._memory_cache) < self._max_memory_items:
            self._memory_cache[cache_key] = data
        
        logger.debug(f"Cache hit (disk): {cache_key}")
        return data
        
    except Exception as e:
        logger.warning(f"Failed to load cache file {cache_path}: {e}")
        # Remove corrupted cache file
        try:
            cache_path.unlink()
        except OSError:
            pass
    
    return None

def _save_to_disk(self, cache_key: str, data: Any) -> bool:
    """Save data to disk cache with security protection."""
    try:
        cache_path = self._get_cache_path(cache_key)
        safe_pickle_dump(data, str(cache_path))
        logger.debug(f"Saved to disk cache: {cache_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to save cache file {cache_path}: {e}")
        return False
```

## 2. Updating Environment Utils (Medium Priority)

### Before: Unsafe Dynamic Import

```python
# /root/repo/hd_compute/utils/environment.py:97
def _check_dependencies(self) -> Dict[str, Any]:
    # ... code ...
    for dep_name, version_spec in deps.items():
        try:
            module = __import__(dep_name)  # VULNERABLE - no validation
            version = getattr(module, '__version__', 'unknown')
            # ... rest of code
```

### After: Secure Dynamic Import

```python
# Updated /root/repo/hd_compute/utils/environment.py
from ..security.secure_imports import SecureImporter

def _check_dependencies(self) -> Dict[str, Any]:
    """Check required and optional dependencies securely."""
    dependencies = {
        'required': {
            'numpy': '>=1.21.0',
            'torch': '>=1.12.0',
            'jax': '>=0.3.0',
            'jaxlib': '>=0.3.0'
        },
        'optional': {
            'librosa': '>=0.9.0',
            'psutil': '>=5.8.0',
            'wandb': '>=0.13.0',
            'tensorboard': '>=2.10.0',
            'matplotlib': '>=3.5.0',
            'seaborn': '>=0.11.0',
        },
        'development': {
            'pytest': '>=7.0.0',
            'black': '>=22.0.0',
            'flake8': '>=4.0.0',
            'mypy': '>=0.991',
            'pre-commit': '>=2.20.0'
        }
    }
    
    results = {}
    importer = SecureImporter()
    
    for category, deps in dependencies.items():
        results[category] = {}
        
        for dep_name, version_spec in deps.items():
            # Use secure dependency checking
            dep_result = importer.check_dependency(dep_name, version_spec)
            results[category][dep_name] = {
                'available': dep_result['available'],
                'version': dep_result['version'],
                'required': version_spec,
                'version_ok': dep_result.get('version_compatible', True)
            }
            
            if not dep_result['available']:
                if category == 'required':
                    error_msg = f"Required dependency '{dep_name}' not found"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                else:
                    warning_msg = f"Optional dependency '{dep_name}' not found"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
    
    return results
```

## 3. Updating Database Repository (High Priority)

### Before: Unsafe Pickle in Database

```python
# /root/repo/hd_compute/database/repository.py:284
def get_cached_hypervector(self, cache_key: str) -> Optional[Any]:
    query = "SELECT data FROM hypervector_cache WHERE cache_key = ?"
    results = self.db.execute_query(query, (cache_key,))
    
    if results:
        return pickle.loads(results[0]['data'])  # VULNERABLE
    return None
```

### After: Secure Database Deserialization

```python
# Updated /root/repo/hd_compute/database/repository.py
from ..security.secure_serialization import SecureSerializer

def __init__(self, db_connection):
    self.db = db_connection
    self.serializer = SecureSerializer()
    # ... rest of init

def get_cached_hypervector(self, cache_key: str) -> Optional[Any]:
    """Get cached hypervector with secure deserialization."""
    query = "SELECT data, is_secure_format FROM hypervector_cache WHERE cache_key = ?"
    results = self.db.execute_query(query, (cache_key,))
    
    if not results:
        return None
    
    data_blob = results[0]['data']
    is_secure = results[0].get('is_secure_format', False)
    
    try:
        if is_secure:
            # New secure format
            return self.serializer.deserialize(data_blob)
        else:
            # Legacy format - use restricted unpickler for safety
            logger.warning(f"Loading legacy cache format: {cache_key}")
            from ..security.secure_serialization import RestrictedUnpickler
            import io
            return RestrictedUnpickler(io.BytesIO(data_blob)).load()
    
    except Exception as e:
        logger.error(f"Failed to deserialize cached data for key {cache_key}: {e}")
        # Clean up corrupted cache entry
        self._remove_cache_entry(cache_key)
        return None

def store_cached_hypervector(self, cache_key: str, data: Any) -> bool:
    """Store hypervector with secure serialization."""
    try:
        # Serialize securely
        serialized_data = self.serializer.serialize(data)
        
        query = """
        INSERT OR REPLACE INTO hypervector_cache 
        (cache_key, data, is_secure_format, created_at) 
        VALUES (?, ?, ?, ?)
        """
        
        self.db.execute_query(query, (
            cache_key, 
            serialized_data, 
            True,  # Mark as secure format
            datetime.utcnow()
        ))
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store cached data for key {cache_key}: {e}")
        return False

def _remove_cache_entry(self, cache_key: str):
    """Remove corrupted cache entry."""
    try:
        query = "DELETE FROM hypervector_cache WHERE cache_key = ?"
        self.db.execute_query(query, (cache_key,))
    except Exception as e:
        logger.error(f"Failed to remove corrupted cache entry {cache_key}: {e}")
```

## 4. Updating Cryptographic Practices (Medium Priority)

### Before: Weak MD5 Hashing

```python
# /root/repo/hd_compute/cache/cache_manager.py:44
def _generate_cache_key(self, namespace: str, key: str) -> str:
    combined_key = f"{namespace}:{key}"
    return hashlib.md5(combined_key.encode()).hexdigest()  # WEAK
```

### After: Secure Hashing with Context

```python
# Updated cache key generation
from ..security.security_config import get_crypto_utils

def _generate_cache_key(self, namespace: str, key: str) -> str:
    """Generate cache key with appropriate hashing.
    
    Note: For cache keys (non-security purpose), MD5 is acceptable for performance.
    The fast_hash method uses MD5 with clear documentation of its purpose.
    """
    crypto = get_crypto_utils()
    combined_key = f"{namespace}:{key}"
    
    # Use fast hash for cache keys (performance over security)
    # This is clearly documented as non-security use case
    return crypto.fast_hash(combined_key)

def _generate_secure_key(self, namespace: str, key: str, purpose: str = "general") -> str:
    """Generate secure key for security-sensitive purposes."""
    crypto = get_crypto_utils()
    combined_key = f"{namespace}:{key}:{purpose}"
    
    # Use secure hash for security-sensitive purposes
    return crypto.secure_hash(combined_key)
```

### Context-Aware Hashing Strategy

```python
# /root/repo/hd_compute/security/hash_strategy.py
from enum import Enum
from .security_config import get_crypto_utils

class HashPurpose(Enum):
    """Define different hash purposes for appropriate algorithm selection."""
    CACHE_KEY = "cache_key"           # Performance over security - MD5 OK
    DATA_FINGERPRINT = "fingerprint" # Data integrity - SHA-256
    SECURITY_TOKEN = "security"      # Security critical - SHA-256
    PASSWORD_HASH = "password"       # Use proper password hashing (not implemented here)

def get_hash_for_purpose(data: str, purpose: HashPurpose) -> str:
    """Get appropriate hash based on purpose."""
    crypto = get_crypto_utils()
    
    if purpose in [HashPurpose.CACHE_KEY]:
        # Fast hash for non-security purposes
        return crypto.fast_hash(data)
    else:
        # Secure hash for security-sensitive purposes  
        return crypto.secure_hash(data)
```

## 5. Updating Security Scanner (Low Priority)

### Before: Potentially Unsafe Import

```python
# /root/repo/security_scan.py:350
def generate_security_report(self, results: Dict) -> str:
    report.append(f"**Scan Date**: {__import__('datetime').datetime.now().isoformat()}")
```

### After: Standard Import

```python
# Updated security_scan.py
import datetime  # Standard import at top of file

def generate_security_report(self, results: Dict) -> str:
    report.append(f"**Scan Date**: {datetime.datetime.now().isoformat()}")
```

## 6. Updating Test Files (Low Priority)

### Before: Hardcoded Test Credentials

```python
# Test files with hardcoded passwords
password="test_password"
password="valid_password"
password="secure_password"
```

### After: Environment-Based Test Credentials

```python
# Updated test files
import os

# Use environment variables or secure test fixtures
TEST_PASSWORD = os.getenv('HDC_TEST_PASSWORD', 'test_default_password')
VALID_PASSWORD = os.getenv('HDC_TEST_VALID_PASSWORD', 'valid_default_password')
SECURE_PASSWORD = os.getenv('HDC_TEST_SECURE_PASSWORD', 'secure_default_password')

# Or use pytest fixtures
@pytest.fixture
def test_credentials():
    """Provide test credentials from environment or defaults."""
    return {
        'password': os.getenv('HDC_TEST_PASSWORD', 'test_default_password'),
        'valid_password': os.getenv('HDC_TEST_VALID_PASSWORD', 'valid_default_password'),
        'secure_password': os.getenv('HDC_TEST_SECURE_PASSWORD', 'secure_default_password')
    }
```

## 7. Configuration-Based Security Migration

### Gradual Migration Strategy

```python
# /root/repo/hd_compute/security/migration.py
from typing import Any, Optional
from pathlib import Path
import logging
from .security_config import get_security_config
from .secure_serialization import safe_pickle_load, migrate_legacy_pickle, is_secure_pickle_file

logger = logging.getLogger(__name__)

class SecurityMigrationManager:
    """Manages gradual migration to secure practices."""
    
    def __init__(self):
        self.config = get_security_config()
    
    def load_pickle_with_migration(self, file_path: str) -> Optional[Any]:
        """Load pickle with automatic migration support.
        
        This function provides a migration path from legacy to secure pickle:
        1. Try secure format first
        2. Fall back to legacy with restricted unpickler
        3. Optionally migrate to secure format
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        try:
            # Check if already in secure format
            if is_secure_pickle_file(file_path):
                logger.debug(f"Loading secure pickle: {file_path}")
                return safe_pickle_load(file_path)
            
            # Legacy format detected
            logger.info(f"Legacy pickle detected: {file_path}")
            
            if self.config.development_mode:
                # In development mode, auto-migrate
                logger.info(f"Auto-migrating legacy pickle: {file_path}")
                return migrate_legacy_pickle(file_path)
            else:
                # In production, use restricted unpickler without migration
                logger.warning(f"Using restricted unpickler for legacy format: {file_path}")
                from .secure_serialization import RestrictedUnpickler
                import io
                
                with open(file_path, 'rb') as f:
                    return RestrictedUnpickler(f).load()
                
        except Exception as e:
            logger.error(f"Failed to load pickle file {file_path}: {e}")
            return None
    
    def should_use_secure_feature(self, feature_name: str) -> bool:
        """Check if secure feature should be enabled based on configuration."""
        feature_flags = {
            'secure_serialization': self.config.enable_secure_serialization,
            'input_validation': self.config.enable_input_validation,
            'import_restrictions': self.config.restrict_dynamic_imports,
            'audit_logging': self.config.enable_audit_logging,
        }
        
        return feature_flags.get(feature_name, True)
```

## 8. Performance Considerations

### Benchmarking Secure vs Legacy Operations

```python
# /root/repo/hd_compute/security/benchmarks.py
import time
import pickle
from typing import Any, Dict
from .secure_serialization import SecureSerializer, safe_pickle_dump, safe_pickle_load

def benchmark_serialization_methods(test_data: Any, iterations: int = 100) -> Dict[str, float]:
    """Benchmark different serialization methods."""
    
    results = {}
    
    # Benchmark legacy pickle
    start_time = time.time()
    for _ in range(iterations):
        pickled = pickle.dumps(test_data)
        unpickled = pickle.loads(pickled)
    legacy_time = time.time() - start_time
    results['legacy_pickle'] = legacy_time
    
    # Benchmark secure serialization
    serializer = SecureSerializer()
    start_time = time.time()
    for _ in range(iterations):
        secure_data = serializer.serialize(test_data)
        unpickled = serializer.deserialize(secure_data)
    secure_time = time.time() - start_time
    results['secure_serialization'] = secure_time
    
    # Calculate overhead
    overhead = ((secure_time - legacy_time) / legacy_time) * 100
    results['overhead_percent'] = overhead
    
    return results
```

## Summary

These examples demonstrate how to systematically replace vulnerable patterns with secure alternatives while maintaining functionality and providing migration paths. The key principles are:

1. **Gradual Migration**: Support both legacy and secure formats during transition
2. **Clear Documentation**: Document why certain choices are made (e.g., MD5 for cache keys)
3. **Configuration-Driven**: Allow security features to be controlled via configuration
4. **Performance Awareness**: Balance security with performance requirements
5. **Comprehensive Logging**: Log security events for monitoring and debugging
6. **Backwards Compatibility**: Provide migration utilities for existing data

The implementation prioritizes the most critical vulnerabilities (unsafe pickle operations) while providing a framework for addressing all identified security issues systematically.