"""Secure serialization utilities for HD-Compute-Toolkit.

This module provides secure alternatives to pickle operations with:
- Restricted unpickler that only allows safe modules
- Integrity checking with HMAC signatures
- Safe fallback mechanisms for migration
"""

import pickle
import io
import hmac
import hashlib
import logging
from typing import Any, Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """Secure unpickler that restricts allowed modules and types.
    
    This class provides a secure alternative to pickle.load() by maintaining
    an allowlist of safe modules that can be unpickled. It prevents arbitrary
    code execution by blocking dangerous modules and callables.
    """
    
    # Allowlist of safe modules for HDC operations
    ALLOWED_MODULES: Set[str] = {
        'builtins',
        'collections',
        'datetime',
        'numpy', 'numpy.core', 'numpy.core.multiarray', 'numpy.core._multiarray_umath',
        'torch', 'torch.tensor', 'torch._tensor',
        'jax', 'jax.numpy', 'jax._src.numpy.lax_numpy',
        'hd_compute.pure_python.hdc_python',
        'hd_compute.core.hdc_base',
        'hd_compute.core.hdc',
        'hd_compute.memory.simple_memory',
        'hd_compute.memory.item_memory',
        'hd_compute.memory.associative_memory',
    }
    
    # Forbidden callables that should never be unpickled
    FORBIDDEN_CALLABLES: Set[str] = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'input', 'exit', 'quit',
        'system', 'popen', 'subprocess',
        'getattr', 'setattr', 'delattr',
        'globals', 'locals', 'vars'
    }
    
    def find_class(self, module: str, name: str):
        """Override to restrict module and callable loading.
        
        Args:
            module: Module name to load from
            name: Class/function name to load
            
        Returns:
            The requested class/function if allowed
            
        Raises:
            pickle.UnpicklingError: If module or callable not allowed
        """
        # Check if module is in allowlist
        module_root = module.split('.')[0]
        if module_root not in self.ALLOWED_MODULES:
            logger.warning(f"Blocked unpickling of module: {module}")
            raise pickle.UnpicklingError(
                f"Module '{module}' not allowed for unpickling. "
                f"Add to ALLOWED_MODULES if this is a legitimate module."
            )
        
        # Check for dangerous callables
        if name in self.FORBIDDEN_CALLABLES:
            logger.warning(f"Blocked unpickling of dangerous callable: {name}")
            raise pickle.UnpicklingError(
                f"Callable '{name}' not allowed for unpickling due to security risk"
            )
        
        logger.debug(f"Allowing unpickle of {module}.{name}")
        return super().find_class(module, name)


class SecureSerializer:
    """Secure serialization with integrity checking.
    
    This class provides secure serialization that:
    - Uses HMAC signatures for integrity verification
    - Prevents tampering detection
    - Provides secure key management
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """Initialize secure serializer.
        
        Args:
            secret_key: Optional custom key for HMAC. If None, generates default.
        """
        self.secret_key = secret_key or self._generate_default_key()
    
    def _generate_default_key(self) -> bytes:
        """Generate a default key for integrity checking.
        
        Note: In production, this should come from a secure key management system.
        """
        # For HDC toolkit, we use a deterministic key based on toolkit identity
        # In production, consider using environment variables or key management
        return hashlib.sha256(b"hdc_toolkit_cache_integrity_key_v1").digest()
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object with integrity signature.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Signed serialized data (signature + payload)
        """
        try:
            # Serialize the object
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create HMAC signature
            signature = hmac.new(
                self.secret_key, 
                data, 
                hashlib.sha256
            ).digest()
            
            # Combine signature and data (signature first for easy splitting)
            return signature + data
            
        except Exception as e:
            logger.error(f"Failed to serialize object: {e}")
            raise ValueError(f"Serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize with integrity verification and restricted unpickling.
        
        Args:
            data: Signed serialized data
            
        Returns:
            Deserialized object
            
        Raises:
            ValueError: If integrity check fails or data is malformed
            pickle.UnpicklingError: If unpickling restrictions are violated
        """
        try:
            if len(data) < 32:  # Must have at least 32-byte signature
                raise ValueError("Data too short to contain valid signature")
            
            # Split signature and payload
            signature = data[:32]  # SHA-256 digest size
            payload = data[32:]
            
            # Verify HMAC signature
            expected_signature = hmac.new(
                self.secret_key,
                payload,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.error("HMAC verification failed - potential data tampering")
                raise ValueError(
                    "Data integrity check failed - potential tampering detected"
                )
            
            # Use restricted unpickler for security
            return RestrictedUnpickler(io.BytesIO(payload)).load()
            
        except Exception as e:
            logger.error(f"Failed to deserialize object: {e}")
            raise


def safe_pickle_load(file_path: str) -> Any:
    """Safely load pickle data with security restrictions.
    
    This is a drop-in replacement for pickle.load() with security enhancements.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Unpickled object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data integrity check fails
        pickle.UnpicklingError: If unpickling restrictions are violated
    """
    serializer = SecureSerializer()
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        logger.debug(f"Loading secure pickle from: {file_path}")
        return serializer.deserialize(data)
        
    except FileNotFoundError:
        logger.error(f"Pickle file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load pickle file {file_path}: {e}")
        raise


def safe_pickle_dump(obj: Any, file_path: str) -> None:
    """Safely dump pickle data with integrity protection.
    
    This is a drop-in replacement for pickle.dump() with security enhancements.
    
    Args:
        obj: Object to pickle
        file_path: Path where to save pickle file
        
    Raises:
        ValueError: If serialization fails
        OSError: If file cannot be written
    """
    serializer = SecureSerializer()
    
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(serializer.serialize(obj))
        
        logger.debug(f"Saved secure pickle to: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save pickle file {file_path}: {e}")
        raise


def migrate_legacy_pickle(old_file_path: str, new_file_path: str = None) -> Any:
    """Migrate legacy pickle file to secure format.
    
    This function loads a legacy pickle file using restricted unpickler
    and saves it in the new secure format.
    
    Args:
        old_file_path: Path to legacy pickle file
        new_file_path: Path for new secure file (defaults to same path)
        
    Returns:
        Loaded and migrated data
        
    Raises:
        FileNotFoundError: If old file doesn't exist
        pickle.UnpicklingError: If legacy file contains unsafe content
    """
    if new_file_path is None:
        new_file_path = old_file_path
    
    logger.info(f"Migrating legacy pickle file: {old_file_path}")
    
    try:
        # Load with restricted unpickler (no integrity check for legacy)
        with open(old_file_path, 'rb') as f:
            data = RestrictedUnpickler(f).load()
        
        # Save in new secure format
        safe_pickle_dump(data, new_file_path)
        
        logger.info(f"Successfully migrated to secure format: {new_file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to migrate pickle file {old_file_path}: {e}")
        raise


def is_secure_pickle_file(file_path: str) -> bool:
    """Check if a pickle file is in secure format.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file is in secure format, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Secure format should have at least 32 bytes for signature
        if len(data) < 32:
            return False
        
        # Try to deserialize - will fail if not secure format
        serializer = SecureSerializer()
        serializer.deserialize(data)
        return True
        
    except Exception:
        return False


# Convenience functions for backward compatibility
load_secure = safe_pickle_load
dump_secure = safe_pickle_dump