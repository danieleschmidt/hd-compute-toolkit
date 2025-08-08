"""Security configuration and hardening utilities for HD-Compute-Toolkit.

This module provides:
- Security configuration management
- Input validation and sanitization
- Cryptographic utilities with modern algorithms
- Security monitoring and audit logging
"""

import os
import re
import hashlib
import secrets
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings for HD-Compute-Toolkit.
    
    This class centralizes all security-related configuration options
    and provides methods to load settings from environment variables.
    """
    
    # Core security features
    enable_input_validation: bool = True
    enable_secure_serialization: bool = True
    enable_audit_logging: bool = True
    enable_integrity_checks: bool = True
    
    # Import security
    restrict_dynamic_imports: bool = True
    allowed_modules: Set[str] = field(default_factory=lambda: {
        'numpy', 'torch', 'jax', 'librosa', 'matplotlib',
        'pandas', 'sklearn', 'scipy', 'psutil'
    })
    
    # File operation security
    restrict_file_access: bool = True
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.json', '.csv', '.txt', '.hdc', '.pkl', '.npy', '.pt', '.wav', '.mp3'
    })
    max_file_size_mb: int = 100
    
    # Cryptographic settings
    use_secure_hashing: bool = True
    hash_algorithm: str = 'sha256'  # Modern default
    allow_legacy_md5: bool = True   # For non-security cache keys only
    
    # Network security (if applicable)
    require_https: bool = True
    verify_ssl: bool = True
    
    # Development vs production modes
    development_mode: bool = False
    strict_mode: bool = False
    
    # Audit and monitoring
    audit_log_path: Optional[str] = None
    max_audit_log_size_mb: int = 10
    audit_retention_days: int = 30
    
    # Cache security
    enable_cache_encryption: bool = False
    cache_integrity_checks: bool = True
    
    @classmethod
    def from_environment(cls) -> 'SecurityConfig':
        """Load security configuration from environment variables.
        
        Environment variables:
        - HDC_SECURITY_INPUT_VALIDATION: Enable input validation (default: true)
        - HDC_SECURITY_SECURE_SERIALIZATION: Enable secure serialization (default: true)
        - HDC_SECURITY_AUDIT_LOGGING: Enable audit logging (default: true)
        - HDC_SECURITY_RESTRICT_IMPORTS: Restrict dynamic imports (default: true)
        - HDC_SECURITY_RESTRICT_FILES: Restrict file access (default: true)
        - HDC_SECURITY_DEVELOPMENT_MODE: Enable development mode (default: false)
        - HDC_SECURITY_STRICT_MODE: Enable strict mode (default: false)
        
        Returns:
            SecurityConfig instance with environment-based settings
        """
        def env_bool(key: str, default: bool) -> bool:
            return os.getenv(key, str(default).lower()).lower() in ('true', '1', 'yes', 'on')
        
        def env_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        return cls(
            enable_input_validation=env_bool('HDC_SECURITY_INPUT_VALIDATION', True),
            enable_secure_serialization=env_bool('HDC_SECURITY_SECURE_SERIALIZATION', True),
            enable_audit_logging=env_bool('HDC_SECURITY_AUDIT_LOGGING', True),
            enable_integrity_checks=env_bool('HDC_SECURITY_INTEGRITY_CHECKS', True),
            restrict_dynamic_imports=env_bool('HDC_SECURITY_RESTRICT_IMPORTS', True),
            restrict_file_access=env_bool('HDC_SECURITY_RESTRICT_FILES', True),
            use_secure_hashing=env_bool('HDC_SECURITY_SECURE_HASHING', True),
            require_https=env_bool('HDC_SECURITY_REQUIRE_HTTPS', True),
            verify_ssl=env_bool('HDC_SECURITY_VERIFY_SSL', True),
            development_mode=env_bool('HDC_SECURITY_DEVELOPMENT_MODE', False),
            strict_mode=env_bool('HDC_SECURITY_STRICT_MODE', False),
            audit_log_path=os.getenv('HDC_SECURITY_AUDIT_LOG_PATH'),
            max_audit_log_size_mb=env_int('HDC_SECURITY_MAX_AUDIT_LOG_SIZE_MB', 10),
            audit_retention_days=env_int('HDC_SECURITY_AUDIT_RETENTION_DAYS', 30),
            enable_cache_encryption=env_bool('HDC_SECURITY_CACHE_ENCRYPTION', False),
            cache_integrity_checks=env_bool('HDC_SECURITY_CACHE_INTEGRITY', True),
        )
    
    def validate_config(self) -> List[str]:
        """Validate security configuration and return any issues.
        
        Returns:
            List of configuration issues/warnings
        """
        issues = []
        
        if self.development_mode and self.strict_mode:
            issues.append("Development mode and strict mode are both enabled")
        
        if not self.enable_secure_serialization and self.cache_integrity_checks:
            issues.append("Cache integrity checks require secure serialization")
        
        if self.audit_log_path and not Path(self.audit_log_path).parent.exists():
            issues.append(f"Audit log directory does not exist: {Path(self.audit_log_path).parent}")
        
        if self.max_file_size_mb <= 0:
            issues.append("Max file size must be positive")
        
        return issues


class SecurityValidator:
    """Input validation and security checking utilities."""
    
    # Patterns for detecting potentially malicious input
    MALICIOUS_PATTERNS = [
        # Code execution patterns
        r'\beval\s*\(',
        r'\bexec\s*\(',  
        r'__import__\s*\(',
        r'\bcompile\s*\(',
        
        # System command patterns
        r'\bsubprocess\.',
        r'\bos\.system\s*\(',
        r'\bos\.popen\s*\(',
        r'\bos\.execv?\s*\(',
        
        # Path traversal patterns
        r'\.\.\/\.\.\/',  # Basic path traversal
        r'\.\.\\\.\.\\',  # Windows path traversal
        r'\/etc\/passwd', # Unix sensitive file
        r'\/proc\/',      # Linux proc filesystem
        
        # Script injection patterns
        r'<script[^>]*>',  # HTML script tags
        r'javascript:',    # JavaScript protocol
        r'vbscript:',      # VBScript protocol
        r'data:text\/html', # HTML data URLs
        
        # SQL injection patterns (basic)
        r'(DROP|DELETE|UPDATE|INSERT)\s+',
        r'UNION\s+SELECT',
        r';\s*--',        # SQL comment
        r'\'\s*OR\s*\'\w*\'\s*=\s*\'\w*\'', # Basic OR injection
        
        # Shell command injection
        r'[;&|`$\(\)]',   # Shell metacharacters
        r'>\s*\/dev\/null', # Output redirection
        
        # Pickle/serialization attacks
        r'pickle\.loads?\s*\(',
        r'cPickle\.loads?\s*\(',
        r'__reduce__',    # Pickle exploit method
        r'__setstate__',  # Pickle exploit method
    ]
    
    # File patterns that should be restricted
    SENSITIVE_FILE_PATTERNS = [
        r'^\/etc\/',      # Unix system files
        r'^\/proc\/',     # Linux process info
        r'^\/sys\/',      # Linux system info
        r'^C:\\Windows\\', # Windows system files
        r'^C:\\System32\\', # Windows system files
        r'\.\.\/\.\.\/',  # Path traversal
        r'id_rsa$',       # SSH private keys
        r'\.pem$',        # Certificate files
        r'\.key$',        # Key files
        r'password',      # Files containing passwords
        r'secret',        # Files containing secrets
    ]
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security validator.
        
        Args:
            config: Security configuration to use
        """
        self.config = config or SecurityConfig.from_environment()
        self.crypto_utils = CryptographicUtils(self.config)
    
    def validate_user_input(self, input_data: str, context: str = "general") -> bool:
        """Validate user input for malicious patterns.
        
        Args:
            input_data: User input to validate
            context: Context of the input (e.g., 'filename', 'module_name', 'data')
            
        Returns:
            True if input appears safe, False if potentially malicious
        """
        if not self.config.enable_input_validation:
            return True  # Validation disabled
        
        if not input_data or not isinstance(input_data, str):
            return False
        
        # Check for malicious patterns
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, input_data, re.IGNORECASE | re.MULTILINE):
                logger.warning(f"Malicious pattern detected in {context}: {pattern}")
                self._log_security_event('MALICIOUS_INPUT_DETECTED', {
                    'context': context,
                    'pattern': pattern,
                    'input_preview': input_data[:100]  # Log first 100 chars only
                })
                return False
        
        # Context-specific validation
        if context == 'filename':
            return self._validate_filename(input_data)
        elif context == 'module_name':
            return self._validate_module_name(input_data)
        elif context == 'file_path':
            return self._validate_file_path(input_data)
        
        return True
    
    def _validate_filename(self, filename: str) -> bool:
        """Validate filename for security issues."""
        # Check for dangerous characters
        dangerous_chars = '<>:"/\\|?*\x00'
        if any(char in filename for char in dangerous_chars):
            return False
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        if filename.upper().split('.')[0] in reserved_names:
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        # Check for hidden files starting with dot (context-dependent)
        if filename.startswith('.') and self.config.strict_mode:
            return False
        
        return True
    
    def _validate_module_name(self, module_name: str) -> bool:
        """Validate module name format."""
        if not module_name:
            return False
        
        # Must be valid Python identifier pattern with dots
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', module_name):
            return False
        
        # Each part must be valid identifier
        parts = module_name.replace('-', '_').split('.')
        return all(part.isidentifier() for part in parts)
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        if not self.config.restrict_file_access:
            return True
        
        # Check for sensitive file patterns
        for pattern in self.SENSITIVE_FILE_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                logger.warning(f"Sensitive file path blocked: {file_path}")
                return False
        
        # Check file extension
        path = Path(file_path)
        if path.suffix.lower() not in self.config.allowed_file_extensions:
            if self.config.strict_mode:
                logger.warning(f"File extension not allowed: {path.suffix}")
                return False
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe use.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        safe_chars = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Handle reserved names by adding suffix
        base_name = safe_chars.split('.')[0].upper()
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL'}
        if base_name in reserved_names:
            safe_chars = f"safe_{safe_chars}"
        
        # Limit length
        if len(safe_chars) > 255:
            name, ext = os.path.splitext(safe_chars)
            max_name_len = 255 - len(ext)
            safe_chars = name[:max_name_len] + ext
        
        return safe_chars
    
    def validate_hypervector_data(self, data: Any) -> bool:
        """Validate hypervector data for safety.
        
        Args:
            data: Hypervector data to validate
            
        Returns:
            True if data appears safe
        """
        try:
            # Check for numeric data types
            if hasattr(data, 'dtype'):
                # NumPy array or similar
                if not data.dtype.kind in 'biufc':  # Numeric types only
                    return False
                
                # Check for infinite or NaN values
                if hasattr(data, 'isfinite'):
                    import numpy as np
                    if not np.all(np.isfinite(data)):
                        logger.warning("Hypervector contains infinite or NaN values")
                        return self.config.development_mode  # Only allow in dev mode
            
            # Check for reasonable size
            if hasattr(data, '__len__'):
                if len(data) > 100000:  # Arbitrary large size limit
                    logger.warning(f"Hypervector unusually large: {len(data)} elements")
                    return not self.config.strict_mode
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating hypervector data: {e}")
            return False
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-relevant events."""
        if self.config.enable_audit_logging:
            event_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'details': details
            }
            logger.warning(f"SECURITY_EVENT: {event_data}")


class CryptographicUtils:
    """Modern cryptographic utilities."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize crypto utilities.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig.from_environment()
    
    def secure_hash(self, data: Union[str, bytes]) -> str:
        """Generate secure hash using modern algorithm.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.config.use_secure_hashing:
            if self.config.hash_algorithm == 'sha256':
                return hashlib.sha256(data).hexdigest()
            elif self.config.hash_algorithm == 'sha512':
                return hashlib.sha512(data).hexdigest()
            elif self.config.hash_algorithm == 'blake2b':
                return hashlib.blake2b(data).hexdigest()
            else:
                # Default to SHA-256
                return hashlib.sha256(data).hexdigest()
        else:
            # Legacy mode - use MD5 (not recommended for security)
            logger.warning("Using legacy MD5 hashing - not recommended for security purposes")
            return hashlib.md5(data).hexdigest()
    
    def fast_hash(self, data: Union[str, bytes]) -> str:
        """Generate fast hash for non-security purposes (cache keys, etc).
        
        Note: This uses MD5 for performance in non-security contexts.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # MD5 is acceptable for cache keys and other non-security uses
        return hashlib.md5(data).hexdigest()
    
    def secure_random(self) -> float:
        """Generate cryptographically secure random float.
        
        Returns:
            Secure random float in range [0.0, 1.0)
        """
        return secrets.SystemRandom().random()
    
    def secure_random_int(self, min_val: int, max_val: int) -> int:
        """Generate cryptographically secure random integer.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            Secure random integer
        """
        return secrets.randbelow(max_val - min_val + 1) + min_val
    
    def generate_salt(self, length: int = 32) -> bytes:
        """Generate cryptographically secure salt.
        
        Args:
            length: Salt length in bytes
            
        Returns:
            Random salt bytes
        """
        return secrets.token_bytes(length)
    
    def generate_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure key.
        
        Args:
            length: Key length in bytes (32 for AES-256)
            
        Returns:
            Random key bytes
        """
        return secrets.token_bytes(length)


class SecurityAuditLogger:
    """Security event audit logging."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize audit logger.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig.from_environment()
        self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Setup audit logging configuration."""
        if not self.config.enable_audit_logging:
            return
        
        # Create audit logger
        self.audit_logger = logging.getLogger('hdc.security.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Setup file handler if path specified
        if self.config.audit_log_path:
            try:
                from logging.handlers import RotatingFileHandler
                
                # Ensure directory exists
                Path(self.config.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Setup rotating file handler
                max_bytes = self.config.max_audit_log_size_mb * 1024 * 1024
                handler = RotatingFileHandler(
                    self.config.audit_log_path,
                    maxBytes=max_bytes,
                    backupCount=5
                )
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                
                self.audit_logger.addHandler(handler)
                
            except Exception as e:
                logger.error(f"Failed to setup audit log file: {e}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = 'INFO'):
        """Log security event.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Log severity level
        """
        if not self.config.enable_audit_logging:
            return
        
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.audit_logger.log(log_level, f"SECURITY_EVENT: {event_data}")


# Global instances for convenience
_default_config = None
_default_validator = None
_default_crypto = None
_default_audit_logger = None


def get_security_config() -> SecurityConfig:
    """Get default security configuration."""
    global _default_config
    if _default_config is None:
        _default_config = SecurityConfig.from_environment()
    return _default_config


def get_security_validator() -> SecurityValidator:
    """Get default security validator."""
    global _default_validator
    if _default_validator is None:
        _default_validator = SecurityValidator(get_security_config())
    return _default_validator


def get_crypto_utils() -> CryptographicUtils:
    """Get default cryptographic utilities."""
    global _default_crypto
    if _default_crypto is None:
        _default_crypto = CryptographicUtils(get_security_config())
    return _default_crypto


def get_audit_logger() -> SecurityAuditLogger:
    """Get default audit logger."""
    global _default_audit_logger
    if _default_audit_logger is None:
        _default_audit_logger = SecurityAuditLogger(get_security_config())
    return _default_audit_logger