"""Robust Pure Python HDC implementation with comprehensive error handling."""

import functools
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from ..pure_python.hdc_python import HDComputePython, SimpleArray
from ..utils.validation import (
    validate_dimension, validate_sparsity, validate_hypervector,
    validate_hypervector_list, ParameterValidator, HDCValidationError,
    DimensionMismatchError, InvalidParameterError
)
from ..security.input_sanitization import InputSanitizer
from ..security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


def robust_operation(operation_name: str = "operation"):
    """Decorator for robust HDC operations with error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            try:
                # Pre-operation validation
                if hasattr(self, '_validate_operation_preconditions'):
                    self._validate_operation_preconditions(operation_name, args, kwargs)
                
                # Execute operation
                result = func(self, *args, **kwargs)
                
                # Post-operation validation
                if hasattr(self, '_validate_operation_result'):
                    self._validate_operation_result(operation_name, result)
                
                # Log successful operation
                execution_time = (time.time() - start_time) * 1000
                if self._audit_logger:
                    self._audit_logger.log_event(
                        event_type="HDC_OPERATION_SUCCESS",
                        description=f"Successfully executed {operation_name}",
                        metadata={
                            "operation": operation_name,
                            "execution_time_ms": execution_time,
                            "args_count": len(args),
                            "kwargs_count": len(kwargs)
                        }
                    )
                
                return result
                
            except HDCValidationError as e:
                # Log validation error
                if self._audit_logger:
                    self._audit_logger.log_security_event(
                        severity="MEDIUM",
                        event_type="VALIDATION_ERROR", 
                        description=f"Validation error in {operation_name}: {str(e)}",
                        additional_data={
                            "operation": operation_name,
                            "error_type": type(e).__name__
                        }
                    )
                # Try to recover or gracefully degrade
                return self._attempt_recovery(operation_name, e, args, kwargs)
                
            except Exception as e:
                # Log unexpected error
                execution_time = (time.time() - start_time) * 1000
                if self._audit_logger:
                    self._audit_logger.log_security_event(
                        severity="HIGH",
                        event_type="OPERATION_ERROR",
                        description=f"Unexpected error in {operation_name}: {str(e)}",
                        additional_data={
                            "operation": operation_name,
                            "error_type": type(e).__name__,
                            "execution_time_ms": execution_time
                        }
                    )
                # Attempt recovery
                return self._attempt_recovery(operation_name, e, args, kwargs)
        
        return wrapper
    return decorator


class RobustHDComputePython(HDComputePython):
    """Robust Pure Python HDC implementation with comprehensive error handling."""
    
    def __init__(self, dim: int, device: Optional[str] = None, dtype=float,
                 enable_audit_logging: bool = True, strict_validation: bool = True):
        """Initialize robust HDC context.
        
        Args:
            dim: Dimensionality of hypervectors
            device: Device specification (ignored for Python)
            dtype: Data type for hypervectors
            enable_audit_logging: Whether to enable audit logging
            strict_validation: Whether to use strict validation
        """
        # Initialize security components
        self._sanitizer = InputSanitizer()
        self._audit_logger = None
        if enable_audit_logging:
            try:
                self._audit_logger = AuditLogger()
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger: {e}")
        self.strict_validation = strict_validation
        
        # Validate and sanitize parameters
        try:
            validated_dim = validate_dimension(dim)
            if device is not None:
                device = self._sanitizer.sanitize_string(str(device), 50)
        except Exception as e:
            if self._audit_logger:
                self._audit_logger.log_security_event(
                    severity="HIGH",
                    event_type="INITIALIZATION_ERROR",
                    description=f"Failed to initialize RobustHDComputePython: {str(e)}"
                )
            raise InvalidParameterError(f"Initialization failed: {str(e)}") from e
        
        # Initialize parent class
        super().__init__(validated_dim, device, dtype)
        
        # Track operation statistics
        self._operation_stats = {
            'total_operations': 0,
            'failed_operations': 0,
            'recovered_operations': 0,
            'validation_errors': 0
        }
        
        # Recovery strategies
        self._recovery_strategies = {
            'random_hv': self._recover_random_hv,
            'bundle': self._recover_bundle,
            'bind': self._recover_bind,
            'cosine_similarity': self._recover_cosine_similarity
        }
        
        if self._audit_logger:
            self._audit_logger.log_event(
                event_type="HDC_BACKEND_INITIALIZED",
                description="RobustHDComputePython initialized successfully",
                metadata={
                    "dimension": validated_dim,
                    "device": device,
                    "strict_validation": strict_validation
                }
            )
    
    def _validate_operation_preconditions(self, operation_name: str, args: tuple, kwargs: dict):
        """Validate preconditions for operations."""
        self._operation_stats['total_operations'] += 1
        
        # Validate common preconditions based on operation
        if operation_name in ['bundle', 'hierarchical_bind'] and args:
            hvs = args[0]
            if isinstance(hvs, (list, tuple)):
                try:
                    validate_hypervector_list(hvs, self.dim)
                except Exception as e:
                    self._operation_stats['validation_errors'] += 1
                    raise HDCValidationError(f"Invalid hypervector list in {operation_name}: {e}")
        
        elif operation_name in ['bind', 'cosine_similarity', 'jensen_shannon_divergence', 
                               'wasserstein_distance', 'fractional_bind', 'entanglement_measure'] and len(args) >= 2:
            hv1, hv2 = args[0], args[1]
            try:
                validate_hypervector(hv1, self.dim)
                validate_hypervector(hv2, self.dim)
            except Exception as e:
                self._operation_stats['validation_errors'] += 1
                raise HDCValidationError(f"Invalid hypervectors in {operation_name}: {e}")
        
        elif operation_name == 'random_hv' and 'sparsity' in kwargs:
            try:
                validate_sparsity(kwargs['sparsity'])
            except Exception as e:
                self._operation_stats['validation_errors'] += 1
                raise HDCValidationError(f"Invalid sparsity in {operation_name}: {e}")
    
    def _validate_operation_result(self, operation_name: str, result: Any):
        """Validate operation results."""
        if result is None:
            raise HDCValidationError(f"Operation {operation_name} returned None")
        
        # Validate hypervector results
        if operation_name in ['random_hv', 'bundle', 'bind', 'fractional_bind', 
                             'quantum_superposition', 'coherence_decay', 'adaptive_threshold',
                             'hierarchical_bind']:
            if isinstance(result, SimpleArray):
                if len(result.data) != self.dim:
                    raise HDCValidationError(
                        f"Result dimension mismatch in {operation_name}: "
                        f"expected {self.dim}, got {len(result.data)}"
                    )
                
                # Check for suspicious patterns
                if not self._sanitizer.validate_hypervector_data(result):
                    if self._audit_logger:
                        self._audit_logger.log_security_event(
                            severity="MEDIUM",
                            event_type="SUSPICIOUS_DATA",
                            description=f"Suspicious hypervector data detected in {operation_name}"
                        )
        
        # Validate numeric results
        elif operation_name in ['cosine_similarity', 'jensen_shannon_divergence', 
                               'wasserstein_distance', 'entanglement_measure']:
            if not isinstance(result, (int, float)):
                raise HDCValidationError(f"Invalid numeric result type in {operation_name}")
            
            # Check for invalid float values
            if isinstance(result, float):
                if not (-1e10 <= result <= 1e10):  # Reasonable bounds
                    raise HDCValidationError(f"Extreme numeric result in {operation_name}: {result}")
                
                # Check for NaN or infinity
                import math
                if math.isnan(result) or math.isinf(result):
                    raise HDCValidationError(f"Invalid float result in {operation_name}: {result}")
    
    def _attempt_recovery(self, operation_name: str, error: Exception, args: tuple, kwargs: dict):
        """Attempt to recover from operation errors."""
        self._operation_stats['failed_operations'] += 1
        
        if self.strict_validation:
            # In strict mode, don't attempt recovery
            raise error
        
        # Try operation-specific recovery
        if operation_name in self._recovery_strategies:
            try:
                recovery_result = self._recovery_strategies[operation_name](error, args, kwargs)
                self._operation_stats['recovered_operations'] += 1
                
                if self._audit_logger:
                    self._audit_logger.log_event(
                        event_type="OPERATION_RECOVERED",
                        description=f"Successfully recovered from error in {operation_name}",
                        metadata={
                            "operation": operation_name,
                            "error_type": type(error).__name__,
                            "recovery_strategy": f"_recover_{operation_name}"
                        }
                    )
                
                return recovery_result
                
            except Exception as recovery_error:
                if self._audit_logger:
                    self._audit_logger.log_security_event(
                        severity="HIGH",
                        event_type="RECOVERY_FAILED",
                        description=f"Recovery failed for {operation_name}",
                        additional_data={
                            "original_error": str(error),
                            "recovery_error": str(recovery_error)
                        }
                    )
        
        # If no recovery possible, raise original error
        raise error
    
    def _recover_random_hv(self, error: Exception, args: tuple, kwargs: dict) -> SimpleArray:
        """Recovery strategy for random_hv operations."""
        # Use default parameters
        logger.warning(f"Recovering from random_hv error: {error}")
        try:
            return super().random_hv(sparsity=0.5)  # Safe default
        except:
            # Last resort: create minimal valid hypervector
            data = [0.5 if i % 2 == 0 else 0.0 for i in range(self.dim)]
            return SimpleArray(data, (self.dim,))
    
    def _recover_bundle(self, error: Exception, args: tuple, kwargs: dict) -> SimpleArray:
        """Recovery strategy for bundle operations."""
        logger.warning(f"Recovering from bundle error: {error}")
        
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
            hvs = args[0]
            # Try to use only valid hypervectors
            valid_hvs = []
            for hv in hvs:
                try:
                    validate_hypervector(hv, self.dim)
                    valid_hvs.append(hv)
                except:
                    continue
            
            if valid_hvs:
                try:
                    return super().bundle(valid_hvs)
                except:
                    pass
        
        # Fallback: return random hypervector
        return self.random_hv()
    
    def _recover_bind(self, error: Exception, args: tuple, kwargs: dict) -> SimpleArray:
        """Recovery strategy for bind operations."""
        logger.warning(f"Recovering from bind error: {error}")
        
        if len(args) >= 2:
            hv1, hv2 = args[0], args[1]
            
            # Try with validated inputs
            try:
                validate_hypervector(hv1, self.dim)
                validate_hypervector(hv2, self.dim)
                return super().bind(hv1, hv2)
            except:
                pass
        
        # Fallback: return random hypervector
        return self.random_hv()
    
    def _recover_cosine_similarity(self, error: Exception, args: tuple, kwargs: dict) -> float:
        """Recovery strategy for cosine_similarity operations."""
        logger.warning(f"Recovering from cosine_similarity error: {error}")
        
        # Return neutral similarity value
        return 0.0
    
    # Wrap all public methods with robust error handling
    @robust_operation("random_hv")
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> SimpleArray:
        return super().random_hv(sparsity, batch_size)
    
    @robust_operation("bundle")
    def bundle(self, hvs: List[SimpleArray], threshold: Optional[float] = None) -> SimpleArray:
        return super().bundle(hvs, threshold)
    
    @robust_operation("bind")
    def bind(self, hv1: SimpleArray, hv2: SimpleArray) -> SimpleArray:
        return super().bind(hv1, hv2)
    
    @robust_operation("cosine_similarity")
    def cosine_similarity(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        return super().cosine_similarity(hv1, hv2)
    
    @robust_operation("jensen_shannon_divergence")
    def jensen_shannon_divergence(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        return super().jensen_shannon_divergence(hv1, hv2)
    
    @robust_operation("wasserstein_distance")
    def wasserstein_distance(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        return super().wasserstein_distance(hv1, hv2)
    
    @robust_operation("fractional_bind")
    def fractional_bind(self, hv1: SimpleArray, hv2: SimpleArray, power: float = 0.5) -> SimpleArray:
        return super().fractional_bind(hv1, hv2, power)
    
    @robust_operation("quantum_superposition")
    def quantum_superposition(self, hvs: List[SimpleArray], amplitudes: Optional[List[float]] = None) -> SimpleArray:
        return super().quantum_superposition(hvs, amplitudes)
    
    @robust_operation("entanglement_measure")
    def entanglement_measure(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        return super().entanglement_measure(hv1, hv2)
    
    @robust_operation("coherence_decay")
    def coherence_decay(self, hv: SimpleArray, decay_rate: float = 0.1) -> SimpleArray:
        return super().coherence_decay(hv, decay_rate)
    
    @robust_operation("adaptive_threshold")
    def adaptive_threshold(self, hv: SimpleArray, target_sparsity: float = 0.5) -> SimpleArray:
        return super().adaptive_threshold(hv, target_sparsity)
    
    @robust_operation("hierarchical_bind")
    def hierarchical_bind(self, structure: dict) -> SimpleArray:
        return super().hierarchical_bind(structure)
    
    @robust_operation("semantic_projection")
    def semantic_projection(self, hv: SimpleArray, basis_hvs: List[SimpleArray]) -> List[float]:
        return super().semantic_projection(hv, basis_hvs)
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics for monitoring."""
        stats = self._operation_stats.copy()
        stats['success_rate'] = (
            (stats['total_operations'] - stats['failed_operations']) / 
            max(stats['total_operations'], 1)
        )
        stats['recovery_rate'] = (
            stats['recovered_operations'] / max(stats['failed_operations'], 1)
        )
        return stats
    
    def reset_statistics(self):
        """Reset operation statistics."""
        self._operation_stats = {
            'total_operations': 0,
            'failed_operations': 0,
            'recovered_operations': 0,
            'validation_errors': 0
        }
        
        if self._audit_logger:
            self._audit_logger.log_event(
                event_type="STATS_RESET",
                description="Operation statistics reset"
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            'status': 'healthy',
            'checks_passed': 0,
            'total_checks': 0,
            'issues': []
        }
        
        checks = [
            ('dimension_valid', lambda: self.dim > 0),
            ('can_generate_hv', lambda: self.random_hv() is not None),
            ('can_compute_similarity', lambda: isinstance(
                self.cosine_similarity(self.random_hv(), self.random_hv()), float
            )),
            ('operation_stats_valid', lambda: isinstance(self.get_operation_statistics(), dict)),
        ]
        
        for check_name, check_func in checks:
            health['total_checks'] += 1
            try:
                if check_func():
                    health['checks_passed'] += 1
                else:
                    health['issues'].append(f"{check_name}: check returned False")
            except Exception as e:
                health['issues'].append(f"{check_name}: {str(e)}")
        
        if health['checks_passed'] < health['total_checks']:
            health['status'] = 'degraded' if health['checks_passed'] > 0 else 'unhealthy'
        
        if self._audit_logger:
            self._audit_logger.log_event(
                event_type="HEALTH_CHECK",
                description=f"Health check completed: {health['status']}",
                metadata=health
            )
        
        return health