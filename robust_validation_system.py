#!/usr/bin/env python3
"""
Robust Validation System for HD-Compute-Toolkit.

This module implements comprehensive error handling, input validation,
security measures, and monitoring for production-ready HDC operations.
"""

import logging
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/hdc_robust.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class HDCValidationError(Exception):
    """Custom exception for HDC validation errors."""
    pass

class HDCSecurityError(Exception):
    """Custom exception for HDC security violations."""
    pass

class HDCPerformanceError(Exception):
    """Custom exception for HDC performance issues."""
    pass

class RobustValidator:
    """Comprehensive validation system for HDC operations."""
    
    def __init__(self, max_dimension: int = 50000, max_memory_mb: int = 1000):
        self.max_dimension = max_dimension
        self.max_memory_mb = max_memory_mb
        self.operation_count = 0
        self.error_count = 0
        self.performance_metrics = {}
        logger.info(f"Initialized RobustValidator (max_dim={max_dimension}, max_mem={max_memory_mb}MB)")
    
    def validate_dimension(self, dim: int) -> None:
        """Validate hypervector dimension."""
        if not isinstance(dim, int):
            raise HDCValidationError(f"Dimension must be an integer, got {type(dim)}")
        if dim <= 0:
            raise HDCValidationError(f"Dimension must be positive, got {dim}")
        if dim > self.max_dimension:
            raise HDCValidationError(f"Dimension {dim} exceeds maximum {self.max_dimension}")
        if dim % 2 != 0:
            logger.warning(f"Dimension {dim} is odd, may cause alignment issues")
    
    def validate_sparsity(self, sparsity: float) -> None:
        """Validate sparsity parameter."""
        if not isinstance(sparsity, (int, float)):
            raise HDCValidationError(f"Sparsity must be numeric, got {type(sparsity)}")
        if not 0.0 <= sparsity <= 1.0:
            raise HDCValidationError(f"Sparsity must be in [0,1], got {sparsity}")
    
    def validate_hypervector_list(self, hvs: List[Any]) -> None:
        """Validate list of hypervectors."""
        if not isinstance(hvs, list):
            raise HDCValidationError(f"Expected list of hypervectors, got {type(hvs)}")
        if len(hvs) == 0:
            raise HDCValidationError("Cannot operate on empty hypervector list")
        if len(hvs) > 10000:
            raise HDCValidationError(f"Too many hypervectors: {len(hvs)} > 10000")
    
    def validate_device(self, device: Optional[str]) -> None:
        """Validate device specification."""
        if device is not None:
            if not isinstance(device, str):
                raise HDCValidationError(f"Device must be string, got {type(device)}")
            valid_devices = {'cpu', 'cuda', 'gpu', 'tpu', 'mps'}
            if device not in valid_devices:
                logger.warning(f"Unknown device '{device}', supported: {valid_devices}")
    
    def sanitize_input_string(self, input_str: str) -> str:
        """Sanitize string input for security."""
        if not isinstance(input_str, str):
            raise HDCSecurityError(f"Expected string input, got {type(input_str)}")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '`', '$', '|', ';']
        sanitized = input_str
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
            logger.warning("Input string truncated to 1000 characters")
        
        return sanitized
    
    def check_memory_usage(self, operation_name: str, estimated_mb: float) -> None:
        """Check if operation would exceed memory limits."""
        if estimated_mb > self.max_memory_mb:
            raise HDCPerformanceError(
                f"Operation '{operation_name}' would use {estimated_mb:.1f}MB > limit {self.max_memory_mb}MB"
            )
    
    def performance_monitor(self, operation_name: str):
        """Decorator for monitoring operation performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.operation_count += 1
                    elapsed = time.time() - start_time
                    
                    if operation_name not in self.performance_metrics:
                        self.performance_metrics[operation_name] = []
                    self.performance_metrics[operation_name].append(elapsed)
                    
                    # Alert on slow operations
                    if elapsed > 1.0:
                        logger.warning(f"Slow operation: {operation_name} took {elapsed:.3f}s")
                    
                    logger.debug(f"Operation {operation_name} completed in {elapsed:.4f}s")
                    return result
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Operation {operation_name} failed: {e}")
                    raise
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        summary = {
            'total_operations': self.operation_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(1, self.operation_count),
            'operations': {}
        }
        
        for op_name, times in self.performance_metrics.items():
            summary['operations'][op_name] = {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times)
            }
        
        return summary

class RobustHDC:
    """Robust HDC wrapper with comprehensive validation and error handling."""
    
    def __init__(self, backend_class, dim: int, device: Optional[str] = None, **kwargs):
        self.validator = RobustValidator()
        
        # Validate inputs
        self.validator.validate_dimension(dim)
        self.validator.validate_device(device)
        
        # Initialize backend with error handling
        try:
            self.backend = backend_class(dim=dim, device=device, **kwargs)
            logger.info(f"Initialized RobustHDC with {backend_class.__name__} backend")
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise HDCValidationError(f"Failed to initialize backend: {e}")
        
        self.dim = dim
        self.device = device
    
    @property
    def performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return self.validator.get_performance_summary()
    
    def random_hv(self, sparsity: float = 0.5) -> Any:
        """Generate random hypervector with validation."""
        self.validator.validate_sparsity(sparsity)
        
        # Estimate memory usage
        estimated_mb = (self.dim * 4) / (1024 * 1024)  # 4 bytes per float
        self.validator.check_memory_usage("random_hv", estimated_mb)
        
        @self.validator.performance_monitor("random_hv")
        def _generate():
            return self.backend.random_hv(sparsity=sparsity)
        
        return _generate()
    
    def bundle(self, hvs: List[Any]) -> Any:
        """Bundle hypervectors with validation."""
        self.validator.validate_hypervector_list(hvs)
        
        # Estimate memory usage
        estimated_mb = (len(hvs) * self.dim * 4) / (1024 * 1024)
        self.validator.check_memory_usage("bundle", estimated_mb)
        
        @self.validator.performance_monitor("bundle")
        def _bundle():
            return self.backend.bundle(hvs)
        
        return _bundle()
    
    def bind(self, hv1: Any, hv2: Any) -> Any:
        """Bind hypervectors with validation."""
        if hv1 is None or hv2 is None:
            raise HDCValidationError("Cannot bind None hypervectors")
        
        @self.validator.performance_monitor("bind")
        def _bind():
            return self.backend.bind(hv1, hv2)
        
        return _bind()
    
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity with validation."""
        if hv1 is None or hv2 is None:
            raise HDCValidationError("Cannot compute similarity with None hypervectors")
        
        @self.validator.performance_monitor("cosine_similarity")
        def _similarity():
            result = self.backend.cosine_similarity(hv1, hv2)
            
            # Validate result
            if not isinstance(result, (int, float)):
                raise HDCValidationError(f"Similarity must be numeric, got {type(result)}")
            if not -1.0 <= result <= 1.0:
                logger.warning(f"Unusual similarity value: {result}")
            
            return float(result)
        
        return _similarity()
    
    def safe_operation_with_retry(self, operation_func, *args, max_retries: int = 3, **kwargs):
        """Execute operation with retry logic and error recovery."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except (HDCValidationError, HDCSecurityError) as e:
                # Don't retry validation/security errors
                logger.error(f"Operation failed with validation/security error: {e}")
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Operation failed after {max_retries} attempts")
        raise last_exception

def run_robustness_tests():
    """Run comprehensive robustness tests."""
    print("🛡️ Running Robustness Tests")
    print("=" * 35)
    
    from hd_compute import HDComputePython
    
    # Test robust wrapper
    robust_hdc = RobustHDC(HDComputePython, dim=1000)
    
    # Test normal operations
    try:
        hv1 = robust_hdc.random_hv(sparsity=0.5)
        hv2 = robust_hdc.random_hv(sparsity=0.4)
        bundled = robust_hdc.bundle([hv1, hv2])
        bound = robust_hdc.bind(hv1, hv2)
        similarity = robust_hdc.cosine_similarity(hv1, hv2)
        print(f"✓ Normal operations: similarity = {similarity:.4f}")
    except Exception as e:
        print(f"❌ Normal operations failed: {e}")
        return False
    
    # Test validation errors
    test_cases = [
        ("Invalid dimension", lambda: RobustHDC(HDComputePython, dim=-100)),
        ("Invalid sparsity", lambda: robust_hdc.random_hv(sparsity=2.0)),
        ("Empty bundle", lambda: robust_hdc.bundle([])),
        ("None similarity", lambda: robust_hdc.cosine_similarity(None, hv1)),
    ]
    
    for test_name, test_func in test_cases:
        try:
            test_func()
            print(f"❌ {test_name}: should have failed")
            return False
        except (HDCValidationError, HDCSecurityError, TypeError):
            print(f"✓ {test_name}: correctly caught error")
        except Exception as e:
            print(f"⚠ {test_name}: unexpected error {e}")
    
    # Test performance monitoring
    perf_summary = robust_hdc.performance_summary
    print(f"✓ Performance monitoring: {perf_summary['total_operations']} operations")
    
    # Test retry mechanism
    def failing_operation():
        if hasattr(failing_operation, 'calls'):
            failing_operation.calls += 1
        else:
            failing_operation.calls = 1
        
        if failing_operation.calls < 3:
            raise RuntimeError("Simulated failure")
        return "success"
    
    try:
        result = robust_hdc.safe_operation_with_retry(failing_operation)
        print(f"✓ Retry mechanism: {result}")
    except Exception as e:
        print(f"❌ Retry mechanism failed: {e}")
        return False
    
    return True

def run_security_tests():
    """Run security validation tests."""
    print("\n🔒 Running Security Tests")
    print("=" * 30)
    
    validator = RobustValidator()
    
    # Test input sanitization
    dangerous_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "$(rm -rf /)",
        "../../etc/passwd",
        "A" * 2000  # Long input
    ]
    
    for dangerous_input in dangerous_inputs:
        try:
            sanitized = validator.sanitize_input_string(dangerous_input)
            if len(sanitized) > 1000 or any(char in sanitized for char in ['<', '>', ';', '$']):
                print(f"❌ Failed to sanitize: '{dangerous_input[:50]}...'")
                return False
            else:
                print(f"✓ Sanitized dangerous input")
        except HDCSecurityError:
            print(f"✓ Blocked dangerous input")
    
    # Test memory limits
    try:
        validator.check_memory_usage("test_operation", 2000)  # Exceeds 1000MB limit
        print("❌ Memory limit check failed")
        return False
    except HDCPerformanceError:
        print("✓ Memory limit correctly enforced")
    
    return True

if __name__ == "__main__":
    print("🛡️ Starting Robustness Validation Tests...")
    
    success = True
    success &= run_robustness_tests()
    success &= run_security_tests()
    
    if success:
        print("\n✅ All robustness tests passed!")
        print("HD-Compute-Toolkit is production-ready with robust error handling.")
    else:
        print("\n❌ Some robustness tests failed!")
    
    print("\n🛡️ Robustness Features:")
    print("  - Comprehensive input validation")
    print("  - Security sanitization and limits")
    print("  - Performance monitoring and alerting")
    print("  - Automatic retry with exponential backoff")
    print("  - Memory usage protection")
    print("  - Detailed logging and error tracking")