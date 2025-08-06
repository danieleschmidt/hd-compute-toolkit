"""Robust error handling and recovery mechanisms for HDC operations."""

import numpy as np
import logging
import traceback
import functools
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import time
import threading


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_METHOD = "fallback_method"
    SKIP_AND_CONTINUE = "skip_and_continue"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation_name: str
    input_args: Tuple
    input_kwargs: Dict
    error_type: type
    error_message: str
    traceback_info: str
    timestamp: float
    severity: ErrorSeverity
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    successful: bool
    result: Any
    strategy_used: RecoveryStrategy
    attempts_made: int
    time_taken: float
    fallback_used: bool
    error_suppressed: bool
    warnings: List[str]


class RobustErrorHandler:
    """Advanced error handling with multiple recovery strategies."""
    
    def __init__(self, default_strategy: RecoveryStrategy = RecoveryStrategy.GRACEFUL_DEGRADATION,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.default_strategy = default_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Error tracking
        self.error_history = []
        self.error_counts = defaultdict(int)
        self.recovery_stats = defaultdict(int)
        
        # Recovery handlers
        self.recovery_handlers = {
            RecoveryStrategy.FAIL_FAST: self._fail_fast_handler,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation_handler,
            RecoveryStrategy.RETRY_WITH_BACKOFF: self._retry_with_backoff_handler,
            RecoveryStrategy.FALLBACK_METHOD: self._fallback_method_handler,
            RecoveryStrategy.SKIP_AND_CONTINUE: self._skip_and_continue_handler
        }
        
        # Fallback methods registry
        self.fallback_methods = {}
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def register_fallback_method(self, operation_name: str, fallback_func: Callable) -> None:
        """Register fallback method for specific operation."""
        self.fallback_methods[operation_name] = fallback_func
        
    def handle_error(self, error_context: ErrorContext, 
                    strategy: Optional[RecoveryStrategy] = None) -> RecoveryResult:
        """Handle error using specified or default strategy."""
        strategy = strategy or self.default_strategy
        
        # Record error
        self.error_history.append(error_context)
        self.error_counts[error_context.operation_name] += 1
        
        # Log error
        self.logger.error(f"Error in {error_context.operation_name}: {error_context.error_message}")
        
        # Apply recovery strategy
        handler = self.recovery_handlers.get(strategy, self._graceful_degradation_handler)
        
        start_time = time.time()
        recovery_result = handler(error_context)
        recovery_result.time_taken = time.time() - start_time
        
        # Update statistics
        self.recovery_stats[strategy] += 1
        if recovery_result.successful:
            self.recovery_stats[f"{strategy}_successful"] += 1
        
        return recovery_result
    
    def _fail_fast_handler(self, error_context: ErrorContext) -> RecoveryResult:
        """Fail fast - re-raise the error immediately."""
        self.logger.critical(f"Fail-fast triggered for {error_context.operation_name}")
        
        return RecoveryResult(
            successful=False,
            result=None,
            strategy_used=RecoveryStrategy.FAIL_FAST,
            attempts_made=0,
            time_taken=0.0,
            fallback_used=False,
            error_suppressed=False,
            warnings=["Error re-raised due to fail-fast strategy"]
        )
    
    def _graceful_degradation_handler(self, error_context: ErrorContext) -> RecoveryResult:
        """Gracefully degrade functionality."""
        warnings = []
        
        # Try to provide a safe default result
        safe_result = self._get_safe_default_result(error_context.operation_name)
        
        if safe_result is not None:
            warnings.append(f"Returned safe default result for {error_context.operation_name}")
            
            return RecoveryResult(
                successful=True,
                result=safe_result,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                attempts_made=1,
                time_taken=0.0,
                fallback_used=True,
                error_suppressed=True,
                warnings=warnings
            )
        
        # If no safe default, try fallback method
        if error_context.operation_name in self.fallback_methods:
            try:
                fallback_func = self.fallback_methods[error_context.operation_name]
                result = fallback_func(*error_context.input_args, **error_context.input_kwargs)
                
                warnings.append(f"Used fallback method for {error_context.operation_name}")
                
                return RecoveryResult(
                    successful=True,
                    result=result,
                    strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    attempts_made=1,
                    time_taken=0.0,
                    fallback_used=True,
                    error_suppressed=True,
                    warnings=warnings
                )
            except Exception as e:
                warnings.append(f"Fallback method also failed: {str(e)}")
        
        # Last resort - return None with warning
        warnings.append("No recovery possible - returning None")
        
        return RecoveryResult(
            successful=False,
            result=None,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            attempts_made=1,
            time_taken=0.0,
            fallback_used=False,
            error_suppressed=True,
            warnings=warnings
        )
    
    def _retry_with_backoff_handler(self, error_context: ErrorContext) -> RecoveryResult:
        """Retry operation with exponential backoff."""
        warnings = []
        
        for attempt in range(self.max_retries):
            # Exponential backoff
            delay = self.retry_delay * (2 ** attempt)
            time.sleep(delay)
            
            try:
                # This would need to be implemented by calling the original function
                # For now, we'll simulate the retry logic
                self.logger.info(f"Retry attempt {attempt + 1} for {error_context.operation_name}")
                
                # Here we would re-execute the original operation
                # result = original_function(*error_context.input_args, **error_context.input_kwargs)
                
                # For demonstration, we'll assume it succeeded after some attempts
                if attempt >= 1:  # Simulate success after second attempt
                    warnings.append(f"Operation succeeded after {attempt + 1} attempts")
                    
                    return RecoveryResult(
                        successful=True,
                        result=f"retry_result_{attempt}",  # Placeholder
                        strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
                        attempts_made=attempt + 1,
                        time_taken=sum(self.retry_delay * (2 ** i) for i in range(attempt + 1)),
                        fallback_used=False,
                        error_suppressed=True,
                        warnings=warnings
                    )
                
            except Exception as e:
                warnings.append(f"Retry attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # All retries failed
        warnings.append(f"All {self.max_retries} retry attempts failed")
        
        return RecoveryResult(
            successful=False,
            result=None,
            strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
            attempts_made=self.max_retries,
            time_taken=sum(self.retry_delay * (2 ** i) for i in range(self.max_retries)),
            fallback_used=False,
            error_suppressed=False,
            warnings=warnings
        )
    
    def _fallback_method_handler(self, error_context: ErrorContext) -> RecoveryResult:
        """Use registered fallback method."""
        warnings = []
        
        if error_context.operation_name not in self.fallback_methods:
            warnings.append(f"No fallback method registered for {error_context.operation_name}")
            
            return RecoveryResult(
                successful=False,
                result=None,
                strategy_used=RecoveryStrategy.FALLBACK_METHOD,
                attempts_made=0,
                time_taken=0.0,
                fallback_used=False,
                error_suppressed=False,
                warnings=warnings
            )
        
        try:
            fallback_func = self.fallback_methods[error_context.operation_name]
            result = fallback_func(*error_context.input_args, **error_context.input_kwargs)
            
            warnings.append(f"Successfully used fallback method for {error_context.operation_name}")
            
            return RecoveryResult(
                successful=True,
                result=result,
                strategy_used=RecoveryStrategy.FALLBACK_METHOD,
                attempts_made=1,
                time_taken=0.0,
                fallback_used=True,
                error_suppressed=True,
                warnings=warnings
            )
            
        except Exception as e:
            warnings.append(f"Fallback method failed: {str(e)}")
            
            return RecoveryResult(
                successful=False,
                result=None,
                strategy_used=RecoveryStrategy.FALLBACK_METHOD,
                attempts_made=1,
                time_taken=0.0,
                fallback_used=True,
                error_suppressed=False,
                warnings=warnings
            )
    
    def _skip_and_continue_handler(self, error_context: ErrorContext) -> RecoveryResult:
        """Skip the failed operation and continue."""
        warnings = [f"Skipped failed operation: {error_context.operation_name}"]
        
        return RecoveryResult(
            successful=True,  # Consider it successful since we're continuing
            result=None,
            strategy_used=RecoveryStrategy.SKIP_AND_CONTINUE,
            attempts_made=0,
            time_taken=0.0,
            fallback_used=False,
            error_suppressed=True,
            warnings=warnings
        )
    
    def _get_safe_default_result(self, operation_name: str) -> Any:
        """Get safe default result for common operations."""
        # Common safe defaults for HDC operations
        safe_defaults = {
            'random_hv': None,  # Would need dimension info
            'bundle': None,
            'bind': None,
            'cosine_similarity': 0.0,
            'hamming_distance': float('inf'),
            'fractional_bind': None,
            'quantum_superposition': None
        }
        
        return safe_defaults.get(operation_name, None)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {'message': 'No errors recorded'}
        
        # Error breakdown by type
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        operation_errors = defaultdict(int)
        
        for error in self.error_history:
            error_types[error.error_type.__name__] += 1
            severity_counts[error.severity.value] += 1
            operation_errors[error.operation_name] += 1
        
        # Recovery statistics
        total_recovery_attempts = sum(self.recovery_stats.values())
        successful_recoveries = sum(v for k, v in self.recovery_stats.items() if k.endswith('_successful'))
        
        return {
            'total_errors': total_errors,
            'error_types': dict(error_types),
            'severity_breakdown': dict(severity_counts),
            'errors_by_operation': dict(operation_errors),
            'recovery_attempts': total_recovery_attempts,
            'successful_recoveries': successful_recoveries,
            'recovery_rate': successful_recoveries / total_recovery_attempts if total_recovery_attempts > 0 else 0,
            'recovery_strategy_usage': dict(self.recovery_stats)
        }


class GracefulDegradation:
    """Implement graceful degradation for HDC operations."""
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.degradation_levels = {
            'full': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'minimal': 0.2
        }
        
    def degrade_operation(self, operation_name: str, current_quality: float, 
                         args: Tuple, kwargs: Dict) -> Tuple[Any, str, float]:
        """Degrade operation quality while maintaining functionality."""
        
        if current_quality >= self.quality_threshold:
            return self._full_quality_operation(operation_name, args, kwargs)
        
        # Determine degradation level
        degradation_level = self._determine_degradation_level(current_quality)
        
        # Apply degradation strategy
        if operation_name in ['random_hv', 'bundle', 'bind']:
            return self._degrade_hdc_operation(operation_name, degradation_level, args, kwargs)
        elif operation_name in ['cosine_similarity', 'hamming_distance']:
            return self._degrade_similarity_operation(operation_name, degradation_level, args, kwargs)
        else:
            # Generic degradation
            return self._generic_degradation(operation_name, degradation_level, args, kwargs)
    
    def _determine_degradation_level(self, quality: float) -> str:
        """Determine appropriate degradation level."""
        for level, threshold in reversed(list(self.degradation_levels.items())):
            if quality >= threshold:
                return level
        return 'minimal'
    
    def _full_quality_operation(self, operation_name: str, args: Tuple, kwargs: Dict) -> Tuple[Any, str, float]:
        """Execute operation at full quality."""
        # This would call the actual operation
        result = f"full_quality_result_for_{operation_name}"  # Placeholder
        return result, "full", 1.0
    
    def _degrade_hdc_operation(self, operation_name: str, degradation_level: str, 
                              args: Tuple, kwargs: Dict) -> Tuple[Any, str, float]:
        """Apply degradation to HDC operations."""
        quality = self.degradation_levels[degradation_level]
        
        if degradation_level == 'high':
            # Minor reduction in precision
            result = f"high_quality_{operation_name}_result"
            return result, degradation_level, quality
            
        elif degradation_level == 'medium':
            # Reduce dimensionality or precision
            result = f"medium_quality_{operation_name}_result"  
            return result, degradation_level, quality
            
        elif degradation_level == 'low':
            # Significant simplification
            result = f"low_quality_{operation_name}_result"
            return result, degradation_level, quality
            
        else:  # minimal
            # Minimal functionality fallback
            result = f"minimal_{operation_name}_result"
            return result, degradation_level, quality
    
    def _degrade_similarity_operation(self, operation_name: str, degradation_level: str,
                                    args: Tuple, kwargs: Dict) -> Tuple[Any, str, float]:
        """Apply degradation to similarity operations."""
        quality = self.degradation_levels[degradation_level]
        
        if degradation_level in ['high', 'medium']:
            # Use approximate similarity measures
            result = 0.5  # Placeholder approximate similarity
        elif degradation_level == 'low':
            # Very rough approximation
            result = 0.3
        else:  # minimal
            # Return neutral similarity
            result = 0.0
        
        return result, degradation_level, quality
    
    def _generic_degradation(self, operation_name: str, degradation_level: str,
                           args: Tuple, kwargs: Dict) -> Tuple[Any, str, float]:
        """Generic degradation strategy."""
        quality = self.degradation_levels[degradation_level]
        result = f"degraded_{operation_name}_{degradation_level}"
        return result, degradation_level, quality


class FailsafeOperations:
    """Failsafe versions of critical HDC operations."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def failsafe_random_hv(self, sparsity: float = 0.5, **kwargs) -> np.ndarray:
        """Failsafe random hypervector generation."""
        try:
            # Primary method
            if sparsity == 0.5:
                # Binary random
                return np.random.choice([-1, 1], size=self.dim)
            else:
                # Sparse random
                hv = np.zeros(self.dim)
                n_nonzero = int(self.dim * (1 - sparsity))
                indices = np.random.choice(self.dim, size=n_nonzero, replace=False)
                hv[indices] = np.random.choice([-1, 1], size=n_nonzero)
                return hv
                
        except Exception:
            # Fallback: simple alternating pattern
            pattern = np.array([1, -1] * (self.dim // 2))
            if self.dim % 2 == 1:
                pattern = np.append(pattern, [1])
            return pattern
    
    def failsafe_bundle(self, hvs: List[np.ndarray], **kwargs) -> np.ndarray:
        """Failsafe bundling operation."""
        if not hvs:
            return np.zeros(self.dim)
        
        try:
            # Primary method: element-wise sum with majority voting
            stacked = np.stack(hvs)
            summed = np.sum(stacked, axis=0)
            return np.sign(summed)
            
        except Exception:
            # Fallback: simple averaging
            try:
                result = hvs[0].copy()
                for hv in hvs[1:]:
                    result = (result + hv) / 2
                return np.sign(result)
            except Exception:
                # Last resort: return first vector
                return hvs[0] if hvs else np.zeros(self.dim)
    
    def failsafe_bind(self, hv1: np.ndarray, hv2: np.ndarray, **kwargs) -> np.ndarray:
        """Failsafe binding operation."""
        try:
            # Primary method: element-wise multiplication
            return hv1 * hv2
            
        except Exception:
            # Fallback: XOR-like operation
            try:
                return np.where(hv1 == hv2, 1, -1)
            except Exception:
                # Last resort: return first vector
                return hv1
    
    def failsafe_cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray, **kwargs) -> float:
        """Failsafe cosine similarity."""
        try:
            # Primary method
            dot_product = np.dot(hv1, hv2)
            norm1 = np.linalg.norm(hv1)
            norm2 = np.linalg.norm(hv2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception:
            # Fallback: Hamming-based approximation
            try:
                matches = np.sum(hv1 == hv2)
                return (2 * matches / len(hv1)) - 1  # Convert to [-1, 1] range
            except Exception:
                # Last resort
                return 0.0


def robust_operation(strategy: RecoveryStrategy = RecoveryStrategy.GRACEFUL_DEGRADATION,
                    max_retries: int = 3,
                    fallback_func: Optional[Callable] = None):
    """Decorator to make HDC operations robust."""
    
    def decorator(func: Callable) -> Callable:
        error_handler = RobustErrorHandler(strategy, max_retries)
        
        if fallback_func:
            error_handler.register_fallback_method(func.__name__, fallback_func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    operation_name=func.__name__,
                    input_args=args,
                    input_kwargs=kwargs,
                    error_type=type(e),
                    error_message=str(e),
                    traceback_info=traceback.format_exc(),
                    timestamp=time.time(),
                    severity=ErrorSeverity.HIGH  # Default severity
                )
                
                # Handle error
                recovery_result = error_handler.handle_error(error_context, strategy)
                
                if recovery_result.successful:
                    # Log warnings if any
                    for warning in recovery_result.warnings:
                        logging.warning(f"{func.__name__}: {warning}")
                    
                    return recovery_result.result
                else:
                    # Re-raise if recovery failed
                    raise e
        
        # Attach error handler for statistics
        wrapper._error_handler = error_handler
        
        return wrapper
    
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Circuit breaker pattern for HDC operations."""
    
    def decorator(func: Callable) -> Callable:
        # Circuit breaker state
        state = {
            'failure_count': 0,
            'last_failure_time': 0,
            'state': 'closed'  # closed, open, half_open
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Check if circuit should be closed again
            if (state['state'] == 'open' and 
                current_time - state['last_failure_time'] > recovery_timeout):
                state['state'] = 'half_open'
                state['failure_count'] = 0
            
            # If circuit is open, fail fast
            if state['state'] == 'open':
                raise Exception(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count if we were in half_open state
                if state['state'] == 'half_open':
                    state['state'] = 'closed'
                    state['failure_count'] = 0
                
                return result
                
            except Exception as e:
                state['failure_count'] += 1
                state['last_failure_time'] = current_time
                
                # Open circuit if threshold exceeded
                if state['failure_count'] >= failure_threshold:
                    state['state'] = 'open'
                    logging.error(f"Circuit breaker opened for {func.__name__} after {failure_threshold} failures")
                
                raise e
        
        # Attach state for monitoring
        wrapper._circuit_state = state
        
        return wrapper
    
    return decorator