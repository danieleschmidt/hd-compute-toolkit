"""
Advanced Error Recovery and Fault Tolerance System
=================================================

Comprehensive error recovery, circuit breakers, and fault tolerance mechanisms
for hyperdimensional computing research operations.
"""

import time
import logging
import traceback
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from enum import Enum
from collections import deque, defaultdict
import numpy as np


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAK = "circuit_break"


class CircuitBreaker:
    """Circuit breaker for protecting HDC operations."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.RLock()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.state_history = deque(maxlen=1000)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self._log_state_change("OPEN -> HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Success - reset if half-open or record if closed
                if self.state == CircuitState.HALF_OPEN:
                    self._reset()
                
                return result
                
            except self.expected_exception as e:
                # Handle expected failures
                self._record_failure()
                
                if self.state == CircuitState.HALF_OPEN:
                    # Failed during recovery attempt
                    self.state = CircuitState.OPEN
                    self._log_state_change("HALF_OPEN -> OPEN (recovery failed)")
                elif self.failure_count >= self.failure_threshold:
                    # Trip the circuit
                    self.state = CircuitState.OPEN
                    self._log_state_change("CLOSED -> OPEN")
                
                raise CircuitBreakerError(f"Circuit breaker tripped: {str(e)}") from e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        return (self.last_failure_time is not None and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _reset(self) -> None:
        """Reset circuit breaker to normal operation."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._log_state_change("RESET -> CLOSED")
    
    def _record_failure(self) -> None:
        """Record a failure occurrence."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
    
    def _log_state_change(self, change: str) -> None:
        """Log circuit breaker state changes."""
        timestamp = time.time()
        self.state_history.append({
            'timestamp': timestamp,
            'change': change,
            'failure_count': self.failure_count
        })
        logging.info(f"Circuit breaker state change: {change}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                'state': self.state.value,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'current_failure_count': self.failure_count,
                'failure_rate': self.total_failures / max(1, self.total_calls),
                'last_failure_time': self.last_failure_time,
                'state_history_length': len(self.state_history)
            }


class RetryManager:
    """Advanced retry mechanism with exponential backoff and jitter."""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        # Statistics
        self.retry_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0, 'failures': 0})
    
    def __call__(self, 
                 exceptions: Union[type, tuple] = Exception,
                 on_retry: Optional[Callable] = None) -> Callable:
        """Decorator for retry logic."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(func, exceptions, on_retry, *args, **kwargs)
            return wrapper
        return decorator
    
    def execute_with_retry(self,
                          func: Callable,
                          exceptions: Union[type, tuple] = Exception,
                          on_retry: Optional[Callable] = None,
                          *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        func_name = func.__name__
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            self.retry_stats[func_name]['attempts'] += 1
            
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logging.info(f"Function {func_name} succeeded on attempt {attempt + 1}")
                
                self.retry_stats[func_name]['successes'] += 1
                return result
                
            except exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    
                    logging.warning(f"Attempt {attempt + 1} failed for {func_name}: {str(e)}. "
                                  f"Retrying in {delay:.2f} seconds...")
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logging.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(delay)
                else:
                    logging.error(f"All {self.max_retries + 1} attempts failed for {func_name}")
                    self.retry_stats[func_name]['failures'] += 1
        
        # All retries exhausted
        raise RetryExhaustedError(f"Failed after {self.max_retries + 1} attempts") from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25% of delay)
            import random
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.1, delay)  # Minimum 0.1 second delay
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get retry statistics."""
        return dict(self.retry_stats)


class FallbackManager:
    """Manages fallback strategies for failed operations."""
    
    def __init__(self):
        self.fallback_strategies = {}
        self.fallback_stats = defaultdict(lambda: {'used': 0, 'succeeded': 0, 'failed': 0})
    
    def register_fallback(self, operation_name: str, fallback_func: Callable) -> None:
        """Register a fallback function for an operation."""
        self.fallback_strategies[operation_name] = fallback_func
        logging.info(f"Registered fallback for operation: {operation_name}")
    
    def execute_with_fallback(self, operation_name: str, primary_func: Callable, 
                            *args, **kwargs) -> Any:
        """Execute operation with fallback on failure."""
        try:
            # Try primary function
            return primary_func(*args, **kwargs)
            
        except Exception as e:
            logging.warning(f"Primary operation {operation_name} failed: {str(e)}")
            
            # Try fallback if available
            if operation_name in self.fallback_strategies:
                self.fallback_stats[operation_name]['used'] += 1
                
                try:
                    fallback_func = self.fallback_strategies[operation_name]
                    result = fallback_func(*args, **kwargs)
                    
                    self.fallback_stats[operation_name]['succeeded'] += 1
                    logging.info(f"Fallback succeeded for {operation_name}")
                    
                    return result
                    
                except Exception as fallback_error:
                    self.fallback_stats[operation_name]['failed'] += 1
                    logging.error(f"Fallback also failed for {operation_name}: {fallback_error}")
                    raise FallbackError(f"Both primary and fallback failed") from fallback_error
            else:
                # No fallback available
                raise NoFallbackError(f"No fallback registered for {operation_name}") from e
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get fallback usage statistics."""
        return dict(self.fallback_stats)


class HealthChecker:
    """Health monitoring and self-healing capabilities."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_status = {}
        self.health_history = deque(maxlen=1000)
        self.running = False
        self.thread = None
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.health_status[name] = {'status': 'unknown', 'last_check': None, 'details': {}}
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            raise ValueError(f"Health check '{name}' not registered")
        
        check_func = self.health_checks[name]
        
        try:
            start_time = time.time()
            result = check_func()
            duration = time.time() - start_time
            
            if isinstance(result, bool):
                status = 'healthy' if result else 'unhealthy'
                details = {'duration': duration}
            elif isinstance(result, dict):
                status = result.get('status', 'unknown')
                details = result.get('details', {})
                details['duration'] = duration
            else:
                status = 'unknown'
                details = {'result': str(result), 'duration': duration}
            
            self.health_status[name] = {
                'status': status,
                'last_check': time.time(),
                'details': details
            }
            
            return self.health_status[name]
            
        except Exception as e:
            self.health_status[name] = {
                'status': 'error',
                'last_check': time.time(),
                'details': {'error': str(e)}
            }
            return self.health_status[name]
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks:
            results[name] = self.run_health_check(name)
        
        # Record overall health
        overall_healthy = all(
            status['status'] == 'healthy' 
            for status in results.values()
        )
        
        self.health_history.append({
            'timestamp': time.time(),
            'overall_healthy': overall_healthy,
            'individual_results': results.copy()
        })
        
        return results
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    self.run_all_health_checks()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logging.error(f"Health monitoring error: {e}")
                    time.sleep(self.check_interval)
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
        logging.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logging.info("Health monitoring stopped")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.health_status:
            return {'status': 'no_checks', 'message': 'No health checks registered'}
        
        healthy_count = sum(1 for status in self.health_status.values() 
                          if status['status'] == 'healthy')
        total_count = len(self.health_status)
        
        overall_status = 'healthy' if healthy_count == total_count else 'degraded'
        if healthy_count == 0:
            overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'healthy_checks': healthy_count,
            'total_checks': total_count,
            'health_percentage': (healthy_count / total_count) * 100,
            'individual_status': self.health_status.copy()
        }


class ErrorRecoverySystem:
    """Comprehensive error recovery and fault tolerance system."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_manager = RetryManager()
        self.fallback_manager = FallbackManager()
        self.health_checker = HealthChecker()
        self.error_patterns = {}
        self.recovery_strategies = {}
        
        # Error tracking
        self.error_history = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default error recovery strategies."""
        
        # Default health checks
        def memory_health_check():
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'status': 'healthy' if memory.percent < 90 else 'unhealthy',
                    'details': {
                        'memory_percent': memory.percent,
                        'available_gb': memory.available / (1024**3)
                    }
                }
            except ImportError:
                return {'status': 'unknown', 'details': {'error': 'psutil not available'}}
        
        def disk_health_check():
            """Check disk space."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return {
                    'status': 'healthy' if disk.percent < 90 else 'unhealthy',
                    'details': {
                        'disk_percent': disk.percent,
                        'free_gb': disk.free / (1024**3)
                    }
                }
            except ImportError:
                return {'status': 'unknown', 'details': {'error': 'psutil not available'}}
        
        self.health_checker.register_health_check('memory', memory_health_check)
        self.health_checker.register_health_check('disk', disk_health_check)
        
        # Default fallback strategies
        def safe_mean_fallback(*args, **kwargs):
            """Safe fallback for mean calculation."""
            if args and hasattr(args[0], '__len__') and len(args[0]) > 0:
                return 0.0  # Safe default
            return 0.0
        
        self.fallback_manager.register_fallback('numpy_mean', safe_mean_fallback)
        
        # Common error patterns and strategies
        self.error_patterns.update({
            'MemoryError': RecoveryStrategy.DEGRADE,
            'RuntimeError': RecoveryStrategy.RETRY,
            'ValueError': RecoveryStrategy.FALLBACK,
            'TimeoutError': RecoveryStrategy.CIRCUIT_BREAK,
            'ConnectionError': RecoveryStrategy.RETRY
        })
    
    def protect_operation(self, 
                         operation_name: str,
                         max_failures: int = 5,
                         recovery_timeout: float = 60.0,
                         max_retries: int = 3,
                         fallback_func: Optional[Callable] = None) -> Callable:
        """Comprehensive protection decorator for operations."""
        
        # Create circuit breaker if not exists
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                failure_threshold=max_failures,
                recovery_timeout=recovery_timeout
            )
        
        # Register fallback if provided
        if fallback_func:
            self.fallback_manager.register_fallback(operation_name, fallback_func)
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_protected_operation(
                    operation_name, func, max_retries, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def _execute_protected_operation(self, operation_name: str, func: Callable,
                                   max_retries: int, *args, **kwargs) -> Any:
        """Execute operation with full protection."""
        circuit_breaker = self.circuit_breakers.get(operation_name)
        
        def protected_execution():
            try:
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                # Record error
                self._record_error(operation_name, e)
                
                # Determine recovery strategy
                strategy = self._get_recovery_strategy(e)
                
                if strategy == RecoveryStrategy.FALLBACK:
                    return self.fallback_manager.execute_with_fallback(
                        operation_name, func, *args, **kwargs
                    )
                else:
                    raise
        
        # Apply retry logic
        retry_manager = RetryManager(max_retries=max_retries)
        
        try:
            return retry_manager.execute_with_retry(
                protected_execution,
                exceptions=Exception,
                on_retry=lambda attempt, error, delay: self._on_retry_callback(
                    operation_name, attempt, error, delay
                )
            )
        except RetryExhaustedError:
            # Last resort: try fallback
            if operation_name in self.fallback_manager.fallback_strategies:
                logging.warning(f"Using fallback as last resort for {operation_name}")
                return self.fallback_manager.execute_with_fallback(
                    operation_name, func, *args, **kwargs
                )
            else:
                raise
    
    def _record_error(self, operation_name: str, error: Exception) -> None:
        """Record error for analysis and monitoring."""
        error_info = {
            'timestamp': time.time(),
            'operation': operation_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        self.error_counts[f"{operation_name}:{type(error).__name__}"] += 1
        
        logging.error(f"Error in {operation_name}: {error}")
    
    def _get_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Determine recovery strategy based on error type."""
        error_type = type(error).__name__
        return self.error_patterns.get(error_type, RecoveryStrategy.FAIL_FAST)
    
    def _on_retry_callback(self, operation_name: str, attempt: int, 
                          error: Exception, delay: float) -> None:
        """Callback for retry attempts."""
        logging.info(f"Retrying {operation_name} (attempt {attempt + 1}) "
                    f"after {error.__class__.__name__}: {str(error)}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        self.health_checker.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_checker.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_summary = self.health_checker.get_health_summary()
        
        # Circuit breaker stats
        cb_stats = {}
        for name, cb in self.circuit_breakers.items():
            cb_stats[name] = cb.get_stats()
        
        # Recent errors (last hour)
        recent_errors = [
            error for error in self.error_history
            if time.time() - error['timestamp'] < 3600
        ]
        
        return {
            'overall_health': health_summary,
            'circuit_breakers': cb_stats,
            'retry_stats': self.retry_manager.get_stats(),
            'fallback_stats': self.fallback_manager.get_stats(),
            'recent_errors': len(recent_errors),
            'error_patterns': dict(self.error_counts),
            'system_uptime': 'monitoring' if self.health_checker.running else 'stopped'
        }


# Custom exceptions
class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class FallbackError(Exception):
    """Raised when fallback operation fails."""
    pass


class NoFallbackError(Exception):
    """Raised when no fallback is available."""
    pass


# Global error recovery system
global_error_recovery = ErrorRecoverySystem()


# Convenient decorators
def robust_operation(operation_name: str, max_failures: int = 5, max_retries: int = 3):
    """Decorator for making operations robust with error recovery."""
    return global_error_recovery.protect_operation(
        operation_name, max_failures=max_failures, max_retries=max_retries
    )


def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator for adding circuit breaker protection."""
    def decorator(func):
        cb = CircuitBreaker(failure_threshold, recovery_timeout)
        return cb(func)
    return decorator


def with_retry(max_retries: int = 3, exceptions: Union[type, tuple] = Exception):
    """Decorator for adding retry logic."""
    retry_manager = RetryManager(max_retries=max_retries)
    return retry_manager(exceptions=exceptions)


# Example usage
if __name__ == "__main__":
    # Initialize error recovery system
    recovery_system = ErrorRecoverySystem()
    recovery_system.start_monitoring()
    
    # Example protected operation
    @robust_operation('test_operation', max_failures=3, max_retries=2)
    def unreliable_function(should_fail: bool = False):
        if should_fail:
            raise RuntimeError("Simulated failure")
        return "Success!"
    
    # Test the protection
    try:
        result = unreliable_function(should_fail=False)
        print(f"Operation succeeded: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")
    
    # Get system status
    status = recovery_system.get_system_status()
    print(f"System health: {status['overall_health']['status']}")
    
    recovery_system.stop_monitoring()