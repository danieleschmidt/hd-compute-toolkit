"""Logging configuration for HD-Compute-Toolkit."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    log_format: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup logging configuration for HD-Compute-Toolkit.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        log_format: Custom log format string
        max_file_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Get root logger for the package
    logger = logging.getLogger('hd_compute')
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default log format
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file_logging and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module or component name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'hd_compute.{name}')


def configure_external_loggers(log_level: str = "WARNING"):
    """Configure logging for external libraries to reduce noise.
    
    Args:
        log_level: Log level for external libraries
    """
    external_loggers = [
        'urllib3',
        'requests',
        'matplotlib',
        'PIL',
        'numba',
        'jax',
        'torch',
        'tensorboard',
        'wandb'
    ]
    
    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))


def setup_performance_logging(logger: logging.Logger) -> Dict[str, Any]:
    """Setup performance logging utilities.
    
    Args:
        logger: Logger instance to use for performance logs
        
    Returns:
        Dictionary with performance logging utilities
    """
    import time
    from functools import wraps
    
    def log_execution_time(func_name: Optional[str] = None):
        """Decorator to log function execution time."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    name = func_name or func.__name__
                    logger.debug(f"Function '{name}' executed in {execution_time:.4f}s")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    name = func_name or func.__name__
                    logger.error(f"Function '{name}' failed after {execution_time:.4f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    class PerformanceContext:
        """Context manager for logging block execution time."""
        
        def __init__(self, operation_name: str):
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            logger.debug(f"Starting operation: {self.operation_name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_time = time.time() - self.start_time
            if exc_type is None:
                logger.debug(f"Operation '{self.operation_name}' completed in {execution_time:.4f}s")
            else:
                logger.error(f"Operation '{self.operation_name}' failed after {execution_time:.4f}s")
    
    return {
        'log_execution_time': log_execution_time,
        'PerformanceContext': PerformanceContext
    }


def setup_structured_logging(logger: logging.Logger, enable_json: bool = False):
    """Setup structured logging with JSON output.
    
    Args:
        logger: Logger instance to configure
        enable_json: Whether to enable JSON structured logging
    """
    if enable_json:
        try:
            import json
            
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        'timestamp': self.formatTime(record),
                        'level': record.levelname,
                        'logger': record.name,
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno,
                        'message': record.getMessage(),
                    }
                    
                    # Add exception info if present
                    if record.exc_info:
                        log_entry['exception'] = self.formatException(record.exc_info)
                    
                    # Add extra fields if present
                    if hasattr(record, 'extra_fields'):
                        log_entry.update(record.extra_fields)
                    
                    return json.dumps(log_entry)
            
            # Replace formatters with JSON formatter
            json_formatter = JSONFormatter()
            for handler in logger.handlers:
                handler.setFormatter(json_formatter)
            
            logger.info("Structured JSON logging enabled")
            
        except ImportError:
            logger.warning("JSON logging requested but json module not available")


def create_audit_logger(audit_log_file: str = "logs/audit.log") -> logging.Logger:
    """Create a separate logger for audit events.
    
    Args:
        audit_log_file: Path to audit log file
        
    Returns:
        Audit logger instance
    """
    audit_logger = logging.getLogger('hd_compute.audit')
    audit_logger.setLevel(logging.INFO)
    
    # Ensure audit log directory exists
    log_path = Path(audit_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler for audit logs (append only, no rotation)
    audit_handler = logging.FileHandler(audit_log_file, mode='a', encoding='utf-8')
    audit_format = '%(asctime)s - AUDIT - %(message)s'
    audit_handler.setFormatter(logging.Formatter(audit_format))
    
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False
    
    return audit_logger


def log_system_info(logger: logging.Logger):
    """Log system information for debugging purposes.
    
    Args:
        logger: Logger instance to use
    """
    import platform
    import psutil
    
    try:
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage_gb': psutil.disk_usage('/').total / (1024**3)
        }
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                system_info['cuda_available'] = True
                system_info['cuda_device_count'] = torch.cuda.device_count()
                system_info['cuda_device_name'] = torch.cuda.get_device_name(0)
            else:
                system_info['cuda_available'] = False
        except ImportError:
            system_info['cuda_available'] = 'torch_not_available'
        
        logger.info(f"System information: {system_info}")
        
    except Exception as e:
        logger.warning(f"Failed to log system information: {e}")