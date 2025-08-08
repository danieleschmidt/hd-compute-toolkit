"""Secure dynamic import utilities for HD-Compute-Toolkit.

This module provides secure alternatives to __import__() and dynamic imports with:
- Module allowlist validation
- Input sanitization and validation
- Safe dependency checking
- Comprehensive logging and monitoring
"""

import importlib
import sys
from typing import Any, Optional, Set, Dict, List
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SecureImporter:
    """Secure dynamic import utilities with allowlist validation.
    
    This class replaces unsafe __import__() calls with secure alternatives that:
    - Validate module names against an allowlist
    - Sanitize input to prevent injection attacks
    - Provide comprehensive logging
    - Handle errors gracefully
    """
    
    # Allowlist of safe modules for dynamic import
    ALLOWED_MODULES: Set[str] = {
        # Core scientific computing
        'numpy', 'scipy',
        'torch', 'torchvision', 'torchaudio',
        'jax', 'jaxlib', 'flax', 'optax',
        'tensorflow', 'keras',
        
        # Data processing
        'pandas', 'polars', 'dask',
        'sklearn', 'scikit-learn',
        'matplotlib', 'seaborn', 'plotly',
        'PIL', 'pillow',
        
        # Audio/signal processing for HDC research
        'librosa', 'soundfile', 'audioread',
        'scipy.signal', 'scipy.fft',
        
        # System utilities
        'psutil', 'tqdm', 'joblib',
        'requests', 'urllib3',
        
        # Machine learning experiment tracking
        'wandb', 'tensorboard', 'mlflow',
        'neptune', 'comet_ml',
        
        # Development tools
        'pytest', 'unittest', 'nose2',
        'black', 'flake8', 'mypy',
        'pre-commit', 'bandit',
        'coverage', 'pytest-cov',
        
        # Serialization (allowed with restrictions)
        'json', 'yaml', 'toml',
        'h5py', 'zarr',
        
        # Standard library modules (commonly used)
        'datetime', 'time', 'calendar',
        'collections', 'itertools', 'functools',
        'pathlib', 'glob', 'shutil',
        'hashlib', 'hmac', 'secrets',
        'logging', 'warnings',
        'multiprocessing', 'threading', 'concurrent.futures',
        'queue', 'heapq', 'bisect',
        'statistics', 'math', 'random',
        'typing', 'dataclasses', 'enum',
        'contextlib', 'weakref', 'gc',
        
        # HDC-specific modules
        'hd_compute', 'hd_compute.core', 'hd_compute.memory',
        'hd_compute.applications', 'hd_compute.research',
        'hd_compute.utils', 'hd_compute.security',
    }
    
    # Modules that are explicitly forbidden
    FORBIDDEN_MODULES: Set[str] = {
        'os', 'subprocess', 'sys',  # System access
        'eval', 'exec', 'compile',  # Code execution
        'imp', 'importlib.util',    # Import manipulation
        'ctypes', 'cffi',          # Low-level system access
        'socket', 'http', 'urllib.request',  # Network access (use requests)
        'pickle', 'cPickle', 'dill',  # Unsafe serialization (use secure alternatives)
    }
    
    @classmethod
    def safe_import(cls, module_name: str, 
                   fromlist: Optional[List[str]] = None,
                   level: int = 0) -> Optional[Any]:
        """Safely import a module with validation.
        
        This is a secure replacement for __import__() that validates
        module names and prevents dangerous imports.
        
        Args:
            module_name: Name of module to import
            fromlist: List of attributes to import from module
            level: Relative import level (0 for absolute imports)
            
        Returns:
            Imported module if successful, None if failed or blocked
            
        Example:
            >>> numpy = SecureImporter.safe_import('numpy')
            >>> if numpy:
            ...     arr = numpy.array([1, 2, 3])
        """
        # Validate module name format
        if not cls._validate_module_name(module_name):
            logger.warning(f"Invalid module name format: {module_name}")
            cls._log_security_event('INVALID_MODULE_NAME', module_name)
            return None
        
        # Check against forbidden modules
        if cls._is_forbidden_module(module_name):
            logger.warning(f"Module explicitly forbidden: {module_name}")
            cls._log_security_event('FORBIDDEN_MODULE', module_name)
            return None
        
        # Check allowlist
        if not cls._is_allowed_module(module_name):
            logger.warning(f"Module not in allowlist: {module_name}")
            cls._log_security_event('MODULE_NOT_ALLOWLISTED', module_name)
            return None
        
        try:
            if fromlist:
                # Handle "from module import ..." style imports
                module = importlib.import_module(module_name)
                if len(fromlist) == 1 and fromlist[0] != '*':
                    # Single attribute import
                    return getattr(module, fromlist[0], None)
                else:
                    # Multiple attributes or wildcard - return module
                    return module
            else:
                # Simple import
                return importlib.import_module(module_name)
                
        except ImportError as e:
            logger.debug(f"Module {module_name} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error importing module {module_name}: {e}")
            cls._log_security_event('IMPORT_ERROR', module_name, str(e))
            return None
    
    @staticmethod
    def _validate_module_name(module_name: str) -> bool:
        """Validate module name format for security.
        
        Args:
            module_name: Module name to validate
            
        Returns:
            True if module name is valid and safe
        """
        if not module_name or not isinstance(module_name, str):
            return False
        
        # Check for dangerous characters that could indicate injection
        dangerous_chars = {'/', '\\', '..', ';', '|', '&', '$', '`', 
                          '(', ')', '[', ']', '{', '}', '<', '>',
                          '"', "'", ' ', '\n', '\r', '\t'}
        if any(char in module_name for char in dangerous_chars):
            return False
        
        # Must be valid Python module identifier pattern
        # Allow dots for submodules, underscores, and hyphens (converted to underscores)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', module_name):
            return False
        
        # Each part must be a valid identifier
        parts = module_name.replace('-', '_').split('.')
        return all(part.isidentifier() for part in parts)
    
    @classmethod
    def _is_forbidden_module(cls, module_name: str) -> bool:
        """Check if module is explicitly forbidden.
        
        Args:
            module_name: Module name to check
            
        Returns:
            True if module is forbidden
        """
        # Check exact match
        if module_name in cls.FORBIDDEN_MODULES:
            return True
        
        # Check if it's a submodule of a forbidden module
        module_root = module_name.split('.')[0]
        return module_root in cls.FORBIDDEN_MODULES
    
    @classmethod
    def _is_allowed_module(cls, module_name: str) -> bool:
        """Check if module is in allowlist.
        
        Args:
            module_name: Module name to check
            
        Returns:
            True if module is allowed
        """
        # Check exact match
        if module_name in cls.ALLOWED_MODULES:
            return True
        
        # Check if it's a submodule of an allowed module
        module_parts = module_name.split('.')
        for i in range(len(module_parts)):
            parent_module = '.'.join(module_parts[:i+1])
            if parent_module in cls.ALLOWED_MODULES:
                return True
        
        return False
    
    @classmethod 
    def check_dependency(cls, dep_name: str, version_spec: str = None) -> Dict[str, Any]:
        """Check if dependency is available with secure import.
        
        This replaces unsafe __import__() usage in dependency checking.
        
        Args:
            dep_name: Dependency name to check
            version_spec: Optional version specification (e.g., '>=1.0.0')
            
        Returns:
            Dictionary with availability information
            
        Example:
            >>> result = SecureImporter.check_dependency('numpy', '>=1.20.0')
            >>> if result['available']:
            ...     print(f"NumPy {result['version']} is available")
        """
        result = {
            'name': dep_name,
            'available': False,
            'version': None,
            'module': None,
            'error': None,
            'version_compatible': None
        }
        
        module = cls.safe_import(dep_name)
        
        if module is None:
            result['error'] = 'Import failed or not allowed'
            return result
        
        # Get version information
        version = getattr(module, '__version__', None)
        if version is None:
            # Try alternative version attributes
            for version_attr in ['VERSION', 'version', '_version']:
                version = getattr(module, version_attr, None)
                if version is not None:
                    if hasattr(version, '__str__'):
                        version = str(version)
                    break
        
        result.update({
            'available': True,
            'version': version or 'unknown',
            'module': module
        })
        
        # Check version compatibility if specified
        if version_spec and version and version != 'unknown':
            result['version_compatible'] = cls._check_version_compatibility(
                version, version_spec
            )
        
        return result
    
    @staticmethod
    def _check_version_compatibility(version: str, version_spec: str) -> bool:
        """Check if version meets specification.
        
        This is a simplified version checker. For production use,
        consider using packaging.specifiers.
        
        Args:
            version: Actual version string
            version_spec: Version specification (e.g., '>=1.0.0')
            
        Returns:
            True if version is compatible
        """
        try:
            import packaging.version
            import packaging.specifiers
            
            spec = packaging.specifiers.SpecifierSet(version_spec)
            return packaging.version.Version(version) in spec
        except ImportError:
            # Fallback to simple string comparison
            logger.warning("packaging library not available, using simple version check")
            
            if version_spec.startswith('>='):
                min_version = version_spec[2:].strip()
                return version >= min_version
            elif version_spec.startswith('>'):
                min_version = version_spec[1:].strip()
                return version > min_version
            elif version_spec.startswith('=='):
                exact_version = version_spec[2:].strip()
                return version == exact_version
            else:
                return True  # Can't check, assume compatible
    
    @staticmethod
    def _log_security_event(event_type: str, module_name: str, details: str = None):
        """Log security-relevant import events.
        
        Args:
            event_type: Type of security event
            module_name: Module name involved
            details: Additional details about the event
        """
        event_data = {
            'event_type': event_type,
            'module_name': module_name,
            'details': details
        }
        
        # This will be picked up by security monitoring
        logger.warning(f"SECURITY_EVENT: {event_data}")


class EnvironmentChecker:
    """Secure environment and dependency checking utilities."""
    
    def __init__(self):
        self.importer = SecureImporter()
    
    def check_all_dependencies(self, requirements: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Check multiple dependencies safely.
        
        Args:
            requirements: Dictionary of {module_name: version_spec}
            
        Returns:
            Dictionary of dependency check results
        """
        results = {}
        
        for dep_name, version_spec in requirements.items():
            results[dep_name] = self.importer.check_dependency(dep_name, version_spec)
        
        return results
    
    def validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment for HD-Compute-Toolkit.
        
        Returns:
            Environment validation results
        """
        results = {
            'python_version': {
                'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'compatible': True
            },
            'required_dependencies': {},
            'optional_dependencies': {},
            'warnings': [],
            'errors': []
        }
        
        # Define required and optional dependencies
        required_deps = {
            'numpy': '>=1.21.0',
            'torch': '>=1.12.0',
        }
        
        optional_deps = {
            'jax': '>=0.3.0',
            'librosa': '>=0.9.0',
            'matplotlib': '>=3.5.0',
            'psutil': '>=5.8.0',
        }
        
        # Check required dependencies
        for dep_name, version_spec in required_deps.items():
            result = self.importer.check_dependency(dep_name, version_spec)
            results['required_dependencies'][dep_name] = result
            
            if not result['available']:
                results['errors'].append(f"Required dependency '{dep_name}' not available")
        
        # Check optional dependencies
        for dep_name, version_spec in optional_deps.items():
            result = self.importer.check_dependency(dep_name, version_spec)
            results['optional_dependencies'][dep_name] = result
            
            if not result['available']:
                results['warnings'].append(f"Optional dependency '{dep_name}' not available")
        
        return results


# Convenience functions for backward compatibility
def secure_import(module_name: str) -> Optional[Any]:
    """Convenience function for secure module import.
    
    Args:
        module_name: Name of module to import
        
    Returns:
        Imported module or None if failed/blocked
    """
    return SecureImporter.safe_import(module_name)


def check_dependency(dep_name: str, version_spec: str = None) -> Dict[str, Any]:
    """Convenience function for dependency checking.
    
    Args:
        dep_name: Dependency name
        version_spec: Optional version specification
        
    Returns:
        Dependency check results
    """
    return SecureImporter.check_dependency(dep_name, version_spec)