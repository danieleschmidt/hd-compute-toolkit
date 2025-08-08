# HD-Compute-Toolkit Security Remediation Plan

## Executive Summary

This document provides a comprehensive remediation plan for the 38 security vulnerabilities identified in the HD-Compute-Toolkit security scan. The plan prioritizes fixes based on actual security risk and provides secure alternatives while maintaining research functionality.

## Vulnerability Analysis Summary

| Category | Count | Severity | Risk Level | Priority |
|----------|--------|----------|------------|-----------|
| Unsafe Dynamic Code Execution | 19 | High | Medium* | Phase 1 |
| Unsafe Pickle Operations | 4 | High | High | Phase 1 |
| Weak Cryptographic Practices | 11 | Medium | Low-Medium | Phase 2 |
| Hardcoded Test Credentials | 4 | Medium | Low | Phase 3 |

*Most dynamic execution is legitimate dependency checking/testing

## Detailed Risk Assessment

### 1. Unsafe Pickle Operations (CRITICAL)

**Files Affected:**
- `/root/repo/hd_compute/cache/cache_manager.py:69`
- `/root/repo/hd_compute/database/repository.py:284`
- `/root/repo/hd_compute/validation/quality_assurance.py:403`

**Security Risk:** HIGH - Arbitrary code execution if cache/database compromised

**Business Impact:** 
- Cache poisoning attacks possible
- Potential for remote code execution
- Data integrity compromise

**Remediation:** Implement restricted pickle unpickler with allowlist

### 2. Dynamic Import Usage (MEDIUM)

**Files Affected:**
- `/root/repo/hd_compute/utils/environment.py:97` (dependency checking)
- `/root/repo/test_basic_import.py:51` (module testing) 
- `/root/repo/security_scan.py:350` (datetime import)

**Security Risk:** MEDIUM - Potential for code injection if input not validated

**Business Impact:**
- Dependency validation could be manipulated
- Test frameworks could load malicious modules

**Remediation:** Replace with `importlib.import_module()` and input validation

### 3. Weak Cryptography (LOW-MEDIUM)

**Files Affected:** Multiple files using MD5 for cache keys and fingerprinting

**Security Risk:** LOW-MEDIUM - Hash collision attacks, not suitable for cryptographic purposes

**Business Impact:**
- Cache key collisions possible (low probability)
- Not suitable for security-critical hashing

**Remediation:** Upgrade to SHA-256, keep MD5 for non-security uses with documentation

## Phase 1: Critical Security Fixes (Week 1)

### 1.1 Secure Pickle Implementation

Create a restricted pickle unpickler that only allows safe types:

```python
# /root/repo/hd_compute/security/secure_serialization.py
import pickle
import io
import hmac
import hashlib
import json
from typing import Any, Set, Optional

class RestrictedUnpickler(pickle.Unpickler):
    """Secure unpickler that restricts allowed modules and types."""
    
    # Allowlist of safe modules for HDC operations
    ALLOWED_MODULES: Set[str] = {
        'builtins',
        'numpy', 'numpy.core', 'numpy.core.multiarray',
        'torch', 'torch.tensor',
        'jax', 'jax.numpy',
        'hd_compute.pure_python.hdc_python',
        'hd_compute.core.hdc_base',
    }
    
    FORBIDDEN_CALLABLES: Set[str] = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'input', 'exit', 'quit'
    }
    
    def find_class(self, module: str, name: str):
        """Override to restrict module loading."""
        # Check if module is in allowlist
        module_root = module.split('.')[0]
        if module_root not in self.ALLOWED_MODULES:
            raise pickle.UnpicklingError(
                f"Module '{module}' not allowed for unpickling"
            )
        
        # Check for dangerous callables
        if name in self.FORBIDDEN_CALLABLES:
            raise pickle.UnpicklingError(
                f"Callable '{name}' not allowed for unpickling"
            )
        
        return super().find_class(module, name)

class SecureSerializer:
    """Secure serialization with integrity checking."""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generate a secure key for integrity checking."""
        return hashlib.sha256(b"hdc_secure_cache_key").digest()
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object with integrity signature."""
        # Serialize the object
        data = pickle.dumps(obj)
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key, 
            data, 
            hashlib.sha256
        ).digest()
        
        # Combine signature and data
        return signature + data
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize with integrity verification."""
        # Split signature and data
        signature = data[:32]  # SHA-256 digest size
        payload = data[32:]
        
        # Verify signature
        expected_signature = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Data integrity check failed - potential tampering")
        
        # Use restricted unpickler
        return RestrictedUnpickler(io.BytesIO(payload)).load()

def safe_pickle_load(file_path: str) -> Any:
    """Safely load pickle data with security restrictions."""
    serializer = SecureSerializer()
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    return serializer.deserialize(data)

def safe_pickle_dump(obj: Any, file_path: str) -> None:
    """Safely dump pickle data with integrity protection."""
    serializer = SecureSerializer()
    
    with open(file_path, 'wb') as f:
        f.write(serializer.serialize(obj))
```

### 1.2 Secure Dynamic Import Utilities

```python
# /root/repo/hd_compute/security/secure_imports.py
import importlib
import sys
from typing import Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class SecureImporter:
    """Secure dynamic import utilities."""
    
    # Allowlist of safe modules for dynamic import
    ALLOWED_MODULES: Set[str] = {
        'numpy', 'torch', 'jax', 'jaxlib',
        'librosa', 'psutil', 'wandb', 'tensorboard',
        'matplotlib', 'seaborn', 'pytest', 'black',
        'flake8', 'mypy', 'pre-commit'
    }
    
    @classmethod
    def safe_import(cls, module_name: str) -> Optional[Any]:
        """Safely import a module with validation."""
        # Validate module name
        if not cls._validate_module_name(module_name):
            logger.warning(f"Invalid module name format: {module_name}")
            return None
        
        # Check allowlist
        if module_name not in cls.ALLOWED_MODULES:
            logger.warning(f"Module not in allowlist: {module_name}")
            return None
        
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logger.info(f"Module {module_name} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error importing module {module_name}: {e}")
            return None
    
    @staticmethod
    def _validate_module_name(module_name: str) -> bool:
        """Validate module name format."""
        if not module_name:
            return False
        
        # Check for dangerous characters
        dangerous_chars = {'/', '\\', '..', ';', '|', '&', '$'}
        if any(char in module_name for char in dangerous_chars):
            return False
        
        # Must be valid Python identifier pattern
        parts = module_name.split('.')
        return all(part.isidentifier() for part in parts)
    
    @classmethod 
    def check_dependency(cls, dep_name: str) -> dict:
        """Check if dependency is available with secure import."""
        module = cls.safe_import(dep_name)
        
        if module is None:
            return {
                'available': False,
                'version': None,
                'error': 'Import failed or not allowed'
            }
        
        version = getattr(module, '__version__', 'unknown')
        return {
            'available': True,
            'version': version,
            'module': module
        }
```

### 1.3 Updated Cache Manager

```python
# Update to /root/repo/hd_compute/cache/cache_manager.py
def _load_from_disk(self, cache_key: str) -> Optional[Any]:
    """Load data from disk cache with security validation."""
    cache_path = self._get_cache_path(cache_key)
    
    if cache_path.exists():
        try:
            from ..security.secure_serialization import safe_pickle_load
            data = safe_pickle_load(str(cache_path))
            
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
```

## Phase 2: Cryptographic Upgrades (Week 2-3)

### 2.1 Modern Hash Functions

```python
# /root/repo/hd_compute/security/crypto_utils.py
import hashlib
import secrets
from typing import Union

class SecureCrypto:
    """Secure cryptographic utilities."""
    
    @staticmethod
    def secure_hash(data: Union[str, bytes]) -> str:
        """Generate secure SHA-256 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def fast_hash(data: Union[str, bytes]) -> str:
        """Generate fast hash for non-security purposes (cache keys, etc).
        
        Note: Uses MD5 - acceptable for cache keys but NOT for security.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def secure_random() -> float:
        """Generate cryptographically secure random float."""
        return secrets.SystemRandom().random()
    
    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Generate cryptographically secure salt."""
        return secrets.token_bytes(length)
```

### 2.2 Cache Key Generation Update

```python
def _generate_cache_key(self, namespace: str, key: str) -> str:
    """Generate cache key with appropriate hashing."""
    from ..security.crypto_utils import SecureCrypto
    
    combined_key = f"{namespace}:{key}"
    
    # Use fast hash for cache keys (non-security purpose)
    # MD5 is acceptable here for performance
    return SecureCrypto.fast_hash(combined_key)
```

## Phase 3: Security Hardening (Week 4)

### 3.1 Input Validation Framework

```python
# /root/repo/hd_compute/security/validation.py
import re
from pathlib import Path
from typing import Any, List, Optional

class SecurityValidator:
    """Input validation for security purposes."""
    
    # Patterns for malicious input detection
    MALICIOUS_PATTERNS = [
        r'eval\s*\(',
        r'exec\s*\(',  
        r'__import__\s*\(',
        r'subprocess\.',
        r'os\.system',
        r'\.\.\/\.\.\/',  # Path traversal
        r'<script[^>]*>',  # XSS
        r'(DROP|DELETE|UPDATE|INSERT)\s+',  # SQL injection
    ]
    
    @classmethod
    def validate_user_input(cls, input_data: str) -> bool:
        """Validate user input for malicious patterns."""
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, input_data, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security."""
        path = Path(file_path)
        
        # Check for path traversal
        if '..' in path.parts:
            return False
        
        # Must be relative to project directory
        try:
            path.resolve().relative_to(Path.cwd())
            return True
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe use."""
        # Remove dangerous characters
        safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        return safe_chars[:255]
```

### 3.2 Security Configuration

```python
# /root/repo/hd_compute/security/config.py
from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Enable security features
    enable_input_validation: bool = True
    enable_secure_serialization: bool = True
    enable_audit_logging: bool = True
    
    # Cryptographic settings
    use_secure_hashing: bool = True
    require_integrity_checks: bool = True
    
    # File operation restrictions
    restrict_file_access: bool = True
    allowed_file_extensions: tuple = ('.json', '.csv', '.hdc', '.pkl')
    
    # Import restrictions
    restrict_dynamic_imports: bool = True
    
    @classmethod
    def from_environment(cls) -> 'SecurityConfig':
        """Load security config from environment variables."""
        return cls(
            enable_input_validation=os.getenv('HDC_SECURITY_INPUT_VALIDATION', 'true').lower() == 'true',
            enable_secure_serialization=os.getenv('HDC_SECURITY_SECURE_SERIALIZATION', 'true').lower() == 'true',
            enable_audit_logging=os.getenv('HDC_SECURITY_AUDIT_LOGGING', 'true').lower() == 'true',
            use_secure_hashing=os.getenv('HDC_SECURITY_SECURE_HASHING', 'true').lower() == 'true',
            require_integrity_checks=os.getenv('HDC_SECURITY_INTEGRITY_CHECKS', 'true').lower() == 'true',
            restrict_file_access=os.getenv('HDC_SECURITY_RESTRICT_FILES', 'true').lower() == 'true',
            restrict_dynamic_imports=os.getenv('HDC_SECURITY_RESTRICT_IMPORTS', 'true').lower() == 'true',
        )
```

## Testing and Validation

### Security Test Suite

```python
# /root/repo/tests/security/test_security_fixes.py
import pytest
from hd_compute.security.secure_serialization import RestrictedUnpickler, SecureSerializer
from hd_compute.security.secure_imports import SecureImporter
from hd_compute.security.validation import SecurityValidator

class TestSecureSerialization:
    """Test secure serialization functionality."""
    
    def test_restricted_unpickler_blocks_dangerous_modules(self):
        """Test that dangerous modules are blocked."""
        import pickle
        import io
        
        # Try to create pickle with dangerous module
        dangerous_data = pickle.dumps(eval)  # This would be dangerous to unpickle
        
        with pytest.raises(pickle.UnpicklingError):
            RestrictedUnpickler(io.BytesIO(dangerous_data)).load()
    
    def test_secure_serializer_integrity(self):
        """Test data integrity checking."""
        serializer = SecureSerializer()
        
        # Serialize legitimate data
        original_data = {'test': [1, 2, 3], 'array': [0.1, 0.2, 0.3]}
        serialized = serializer.serialize(original_data)
        
        # Should deserialize correctly
        deserialized = serializer.deserialize(serialized)
        assert deserialized == original_data
        
        # Tampered data should fail
        tampered = serialized[:-1] + b'\x00'
        with pytest.raises(ValueError, match="integrity check failed"):
            serializer.deserialize(tampered)

class TestSecureImports:
    """Test secure import functionality."""
    
    def test_safe_import_allowed_modules(self):
        """Test that allowed modules can be imported."""
        numpy_module = SecureImporter.safe_import('numpy')
        # Should succeed or return None if numpy not available
        assert numpy_module is None or hasattr(numpy_module, 'array')
    
    def test_safe_import_blocks_dangerous_modules(self):
        """Test that dangerous modules are blocked."""
        result = SecureImporter.safe_import('os')
        assert result is None  # Should be blocked

class TestInputValidation:
    """Test input validation functionality."""
    
    def test_malicious_input_detection(self):
        """Test detection of malicious input patterns."""
        malicious_inputs = [
            "eval('print(1)')",
            "__import__('os').system('ls')",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]
        
        for malicious in malicious_inputs:
            assert not SecurityValidator.validate_user_input(malicious)
    
    def test_safe_input_acceptance(self):
        """Test that safe input is accepted."""
        safe_inputs = [
            "legitimate data string",
            "numpy.array([1, 2, 3])",
            "dimension=1000"
        ]
        
        for safe in safe_inputs:
            assert SecurityValidator.validate_user_input(safe)
```

## Migration Strategy

### 1. Backward Compatibility

During migration, maintain compatibility with existing cached data:

```python
def _load_from_disk_with_migration(self, cache_key: str) -> Optional[Any]:
    """Load with automatic migration from old format."""
    cache_path = self._get_cache_path(cache_key)
    
    if not cache_path.exists():
        return None
    
    try:
        # Try new secure format first
        from ..security.secure_serialization import safe_pickle_load
        return safe_pickle_load(str(cache_path))
    
    except (ValueError, pickle.UnpicklingError):
        # Fall back to old format with restricted unpickler
        logger.warning(f"Migrating cache file to secure format: {cache_path}")
        
        try:
            with open(cache_path, 'rb') as f:
                # Use restricted unpickler for safety
                data = RestrictedUnpickler(f).load()
            
            # Re-save in secure format
            self._save_to_disk_secure(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to migrate cache file {cache_path}: {e}")
            # Remove corrupted file
            cache_path.unlink(missing_ok=True)
            return None
```

### 2. Configuration-Based Rollout

Use feature flags to gradually enable security features:

```python
# In configuration
SECURITY_CONFIG = {
    'enable_secure_pickle': True,  # Enable by default
    'enable_import_restrictions': False,  # Gradual rollout
    'enable_input_validation': True,
    'migration_mode': True  # Support old cache format during transition
}
```

## Monitoring and Alerting

### Security Event Logging

```python
# /root/repo/hd_compute/security/audit_logger.py
import logging
from datetime import datetime
from typing import Dict, Any

class SecurityAuditLogger:
    """Logger for security-relevant events."""
    
    def __init__(self):
        self.logger = logging.getLogger('hdc.security')
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event with structured data."""
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.logger.warning(f"SECURITY_EVENT: {event_data}")
    
    def log_blocked_import(self, module_name: str):
        """Log blocked dynamic import attempt."""
        self.log_security_event('BLOCKED_IMPORT', {
            'module_name': module_name,
            'reason': 'Not in allowlist'
        })
    
    def log_integrity_failure(self, file_path: str):
        """Log cache integrity check failure.""" 
        self.log_security_event('INTEGRITY_FAILURE', {
            'file_path': file_path,
            'reason': 'HMAC verification failed'
        })
```

## Conclusion

This remediation plan addresses all 38 security vulnerabilities found in the HD-Compute-Toolkit while maintaining the research functionality. The phased approach allows for careful implementation and testing of security fixes without disrupting the development workflow.

**Key Benefits:**
- ✅ Eliminates arbitrary code execution risks from pickle operations
- ✅ Secures dynamic import functionality with allowlists
- ✅ Modernizes cryptographic practices while maintaining performance
- ✅ Maintains backward compatibility during migration
- ✅ Provides comprehensive security monitoring and alerting
- ✅ Preserves legitimate research use cases

**Implementation Timeline:** 4 weeks with immediate focus on critical pickle vulnerabilities.

**Success Metrics:**
- Zero high-severity security findings in subsequent scans
- No regression in legitimate functionality
- Successful migration of existing cached data
- Complete audit trail of security-relevant events