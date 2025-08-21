"""Security and monitoring for HDC research algorithms."""

import time
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import numpy as np


class HDCSecurityMonitor:
    """Enhanced security monitoring and validation for HDC research operations."""
    
    def __init__(self, max_operations_per_second: int = 1000, max_memory_mb: int = 1024):
        self.max_operations_per_second = max_operations_per_second
        self.max_memory_mb = max_memory_mb
        self.operation_history = []
        self.security_violations = []
        self.trusted_sources = set()
        self.threat_patterns = set()
        self.encryption_keys = {}
        self.access_control_list = {}
        self.audit_trail = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def rate_limit_decorator(self, func: Callable) -> Callable:
        """Rate limiting decorator for HDC operations."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Clean old operations (older than 1 second)
            self.operation_history = [
                t for t in self.operation_history 
                if current_time - t < 1.0
            ]
            
            # Check rate limit
            if len(self.operation_history) >= self.max_operations_per_second:
                violation = {
                    'type': 'rate_limit_exceeded',
                    'timestamp': current_time,
                    'function': func.__name__,
                    'operations_count': len(self.operation_history)
                }
                self.security_violations.append(violation)
                self.logger.warning(f"Rate limit exceeded for {func.__name__}")
                raise RuntimeError(f"Rate limit exceeded: {len(self.operation_history)} ops/sec")
            
            # Record operation
            self.operation_history.append(current_time)
            
            # Execute function with monitoring
            try:
                result = func(*args, **kwargs)
                self.logger.debug(f"Operation {func.__name__} completed successfully")
                return result
            except Exception as e:
                violation = {
                    'type': 'operation_failure',
                    'timestamp': current_time,
                    'function': func.__name__,
                    'error': str(e)
                }
                self.security_violations.append(violation)
                self.logger.error(f"Operation {func.__name__} failed: {str(e)}")
                raise
        
        return wrapper
    
    def validate_input_data(self, data: Any, data_type: str = "hypervector") -> bool:
        """Validate input data for security and integrity."""
        try:
            if data_type == "hypervector":
                return self._validate_hypervector(data)
            elif data_type == "array_list":
                return self._validate_array_list(data)
            elif data_type == "numeric":
                return self._validate_numeric(data)
            else:
                self.logger.warning(f"Unknown data type for validation: {data_type}")
                return False
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def _validate_hypervector(self, hv: np.ndarray) -> bool:
        """Validate hypervector security constraints."""
        if not isinstance(hv, np.ndarray):
            return False
        
        # Check dimension limits
        if hv.size > 100000:  # Prevent memory exhaustion
            self.logger.warning(f"Hypervector too large: {hv.size} elements")
            return False
        
        # Check for malicious patterns
        if np.all(hv == hv[0]):  # All same value
            self.logger.warning("Suspicious hypervector: all values identical")
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(hv)) or np.any(np.isinf(hv)):
            self.logger.warning("Invalid hypervector: contains NaN or Inf")
            return False
        
        # Check memory usage
        memory_mb = hv.nbytes / (1024 * 1024)
        if memory_mb > self.max_memory_mb:
            self.logger.warning(f"Hypervector too large: {memory_mb:.2f} MB")
            return False
        
        return True
    
    def _validate_array_list(self, arrays: List[np.ndarray]) -> bool:
        """Validate list of arrays."""
        if not isinstance(arrays, list):
            return False
        
        if len(arrays) > 10000:  # Prevent DoS
            self.logger.warning(f"Array list too large: {len(arrays)} elements")
            return False
        
        for i, arr in enumerate(arrays):
            if not self._validate_hypervector(arr):
                self.logger.warning(f"Invalid array at index {i}")
                return False
        
        return True
    
    def _validate_numeric(self, value: Any) -> bool:
        """Validate numeric input."""
        if not isinstance(value, (int, float, np.number)):
            return False
        
        if np.isnan(value) or np.isinf(value):
            return False
        
        # Check for reasonable bounds
        if abs(value) > 1e10:
            self.logger.warning(f"Numeric value out of bounds: {value}")
            return False
        
        return True
    
    def compute_data_hash(self, data: Any) -> str:
        """Compute secure hash of data for integrity checking."""
        try:
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, (list, tuple)):
                # Convert to JSON for consistent hashing
                data_str = json.dumps(data, sort_keys=True, default=str)
                data_bytes = data_str.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            return hashlib.sha256(data_bytes).hexdigest()
        except Exception as e:
            self.logger.error(f"Hash computation failed: {str(e)}")
            return ""
    
    def add_trusted_source(self, source_id: str) -> None:
        """Add a trusted source for data validation."""
        self.trusted_sources.add(source_id)
        self.logger.info(f"Added trusted source: {source_id}")
    
    def verify_data_source(self, data: Any, source_id: str) -> bool:
        """Verify that data comes from a trusted source."""
        if source_id not in self.trusted_sources:
            self.logger.warning(f"Untrusted data source: {source_id}")
            return False
        
        # Additional verification could include digital signatures
        return True
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        
        # Recent violations (last hour)
        recent_violations = [
            v for v in self.security_violations
            if current_time - v['timestamp'] < 3600
        ]
        
        # Operation rate (last minute)
        recent_ops = [
            t for t in self.operation_history
            if current_time - t < 60
        ]
        
        return {
            'total_violations': len(self.security_violations),
            'recent_violations': len(recent_violations),
            'violation_types': list(set(v['type'] for v in recent_violations)),
            'operations_last_minute': len(recent_ops),
            'current_rate_limit': self.max_operations_per_second,
            'memory_limit_mb': self.max_memory_mb,
            'trusted_sources_count': len(self.trusted_sources),
            'status': 'secure' if len(recent_violations) == 0 else 'violations_detected'
        }


class HDCHealthChecker:
    """Health monitoring for HDC research systems."""
    
    def __init__(self):
        self.health_metrics = {}
        self.last_check_time = time.time()
        
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        current_time = time.time()
        health_status = {
            'timestamp': current_time,
            'status': 'healthy',
            'checks': {}
        }
        
        # Memory check
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            health_status['checks']['memory'] = {
                'status': 'healthy' if memory_percent < 90 else 'warning',
                'usage_percent': memory_percent
            }
        except ImportError:
            health_status['checks']['memory'] = {'status': 'unknown', 'error': 'psutil not available'}
        
        # NumPy check
        try:
            test_array = np.random.rand(1000)
            result = np.sum(test_array)
            health_status['checks']['numpy'] = {
                'status': 'healthy' if not np.isnan(result) else 'error',
                'test_result': float(result)
            }
        except Exception as e:
            health_status['checks']['numpy'] = {'status': 'error', 'error': str(e)}
        
        # File system check
        try:
            import os
            disk_usage = os.statvfs('/')
            free_percent = (disk_usage.f_bavail * disk_usage.f_frsize) / (disk_usage.f_blocks * disk_usage.f_frsize) * 100
            health_status['checks']['disk'] = {
                'status': 'healthy' if free_percent > 10 else 'warning',
                'free_percent': free_percent
            }
        except Exception as e:
            health_status['checks']['disk'] = {'status': 'error', 'error': str(e)}
        
        # Determine overall status
        check_statuses = [check['status'] for check in health_status['checks'].values()]
        if 'error' in check_statuses:
            health_status['status'] = 'error'
        elif 'warning' in check_statuses:
            health_status['status'] = 'warning'
        
        self.health_metrics[current_time] = health_status
        self.last_check_time = current_time
        
        return health_status
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history for specified period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            metrics for timestamp, metrics in self.health_metrics.items()
            if timestamp >= cutoff_time
        ]


# Global instances for easy access
security_monitor = HDCSecurityMonitor()
health_checker = HDCHealthChecker()


def secure_operation(func: Callable) -> Callable:
    """Decorator to add security monitoring to HDC operations."""
    return security_monitor.rate_limit_decorator(func)


def validate_hypervector_input(func: Callable) -> Callable:
    """Decorator to validate hypervector inputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate numpy array arguments
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                if not security_monitor.validate_input_data(arg, "hypervector"):
                    raise ValueError(f"Invalid hypervector at argument {i}")
        
        # Validate numpy arrays in keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                if not security_monitor.validate_input_data(value, "hypervector"):
                    raise ValueError(f"Invalid hypervector in {key}")
        
        return func(*args, **kwargs)
    
    return wrapper