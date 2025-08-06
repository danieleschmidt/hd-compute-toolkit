"""Input validation and error handling utilities."""

import functools
import logging
from typing import Any, Callable, List, Optional, Union, Dict, Tuple
import warnings

logger = logging.getLogger(__name__)


class HDCValidationError(Exception):
    """Base exception for HDC validation errors."""
    pass


class DimensionMismatchError(HDCValidationError):
    """Raised when hypervector dimensions don't match."""
    pass


class InvalidParameterError(HDCValidationError):
    """Raised when parameters are invalid."""
    pass


class MemoryError(HDCValidationError):
    """Raised when memory operations fail."""
    pass


def validate_dimension(dimension: int) -> int:
    """Validate hypervector dimension.
    
    Args:
        dimension: Dimension to validate
        
    Returns:
        Validated dimension
        
    Raises:
        InvalidParameterError: If dimension is invalid
    """
    if not isinstance(dimension, int):
        raise InvalidParameterError(f"Dimension must be an integer, got {type(dimension)}")
    
    if dimension <= 0:
        raise InvalidParameterError(f"Dimension must be positive, got {dimension}")
    
    if dimension < 100:
        warnings.warn(f"Very small dimension ({dimension}) may not be effective for HDC")
    
    if dimension > 100000:
        warnings.warn(f"Very large dimension ({dimension}) may cause memory issues")
    
    return dimension


def validate_sparsity(sparsity: float) -> float:
    """Validate sparsity parameter.
    
    Args:
        sparsity: Sparsity level to validate
        
    Returns:
        Validated sparsity
        
    Raises:
        InvalidParameterError: If sparsity is invalid
    """
    if not isinstance(sparsity, (int, float)):
        raise InvalidParameterError(f"Sparsity must be a number, got {type(sparsity)}")
    
    if not 0.0 <= sparsity <= 1.0:
        raise InvalidParameterError(f"Sparsity must be between 0 and 1, got {sparsity}")
    
    if sparsity < 0.1 or sparsity > 0.9:
        warnings.warn(f"Extreme sparsity ({sparsity}) may reduce effectiveness")
    
    return float(sparsity)


def validate_hypervector(hv: Any, expected_dim: Optional[int] = None) -> Any:
    """Validate hypervector structure.
    
    Args:
        hv: Hypervector to validate
        expected_dim: Expected dimension (optional)
        
    Returns:
        Validated hypervector
        
    Raises:
        InvalidParameterError: If hypervector is invalid
        DimensionMismatchError: If dimension doesn't match expected
    """
    if hv is None:
        raise InvalidParameterError("Hypervector cannot be None")
    
    # Check for SimpleArray (pure Python)
    if hasattr(hv, 'data') and hasattr(hv, 'shape'):
        actual_dim = len(hv.data)
    # Check for numpy array
    elif hasattr(hv, 'shape') and hasattr(hv, 'ndim'):
        if hv.ndim != 1:
            raise InvalidParameterError(f"Hypervector must be 1-dimensional, got {hv.ndim}D")
        actual_dim = hv.shape[0]
    # Check for torch tensor
    elif hasattr(hv, 'dim') and hasattr(hv, 'size'):
        if hv.dim() != 1:
            raise InvalidParameterError(f"Hypervector must be 1-dimensional, got {hv.dim()}D")
        actual_dim = hv.size(0)
    # Check for list/tuple
    elif isinstance(hv, (list, tuple)):
        actual_dim = len(hv)
    else:
        # Try to get length as fallback
        try:
            actual_dim = len(hv)
        except:
            raise InvalidParameterError(f"Invalid hypervector type: {type(hv)}")
    
    if actual_dim == 0:
        raise InvalidParameterError("Hypervector cannot be empty")
    
    if expected_dim is not None and actual_dim != expected_dim:
        raise DimensionMismatchError(
            f"Hypervector dimension {actual_dim} doesn't match expected {expected_dim}"
        )
    
    return hv


def validate_hypervector_list(hvs: List[Any], expected_dim: Optional[int] = None) -> List[Any]:
    """Validate list of hypervectors.
    
    Args:
        hvs: List of hypervectors to validate
        expected_dim: Expected dimension for all hypervectors
        
    Returns:
        Validated list of hypervectors
        
    Raises:
        InvalidParameterError: If list is invalid
        DimensionMismatchError: If dimensions don't match
    """
    if not isinstance(hvs, (list, tuple)):
        raise InvalidParameterError(f"Expected list of hypervectors, got {type(hvs)}")
    
    if len(hvs) == 0:
        raise InvalidParameterError("Cannot process empty list of hypervectors")
    
    validated_hvs = []
    first_dim = None
    
    for i, hv in enumerate(hvs):
        try:
            validated_hv = validate_hypervector(hv, expected_dim)
            
            # Get dimension of first hypervector
            if first_dim is None:
                if hasattr(validated_hv, 'data'):
                    first_dim = len(validated_hv.data)
                elif hasattr(validated_hv, 'shape'):
                    first_dim = validated_hv.shape[0]
                elif hasattr(validated_hv, 'size'):
                    first_dim = validated_hv.size(0)
                else:
                    first_dim = len(validated_hv)
            
            # Validate consistency with first hypervector
            validate_hypervector(validated_hv, first_dim)
            validated_hvs.append(validated_hv)
            
        except (InvalidParameterError, DimensionMismatchError) as e:
            raise type(e)(f"Error in hypervector {i}: {e}")
    
    return validated_hvs


def validate_device(device: str) -> str:
    """Validate device specification.
    
    Args:
        device: Device string to validate
        
    Returns:
        Validated device string
        
    Raises:
        InvalidParameterError: If device is invalid
    """
    if not isinstance(device, str):
        raise InvalidParameterError(f"Device must be a string, got {type(device)}")
    
    valid_devices = ['cpu', 'cuda', 'mps', 'auto', 'jax', 'tpu']
    device_lower = device.lower()
    
    if device_lower not in valid_devices:
        raise InvalidParameterError(
            f"Invalid device '{device}'. Valid options: {valid_devices}"
        )
    
    return device_lower


def validate_positive_int(value: int, name: str, min_value: int = 1) -> int:
    """Validate positive integer parameter.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        
    Returns:
        Validated value
        
    Raises:
        InvalidParameterError: If value is invalid
    """
    if not isinstance(value, int):
        raise InvalidParameterError(f"{name} must be an integer, got {type(value)}")
    
    if value < min_value:
        raise InvalidParameterError(f"{name} must be >= {min_value}, got {value}")
    
    return value


def validate_probability(value: float, name: str) -> float:
    """Validate probability parameter (0-1).
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        Validated value
        
    Raises:
        InvalidParameterError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise InvalidParameterError(f"{name} must be a number, got {type(value)}")
    
    if not 0.0 <= value <= 1.0:
        raise InvalidParameterError(f"{name} must be between 0 and 1, got {value}")
    
    return float(value)


def validate_string_list(values: List[str], name: str, allow_empty: bool = False) -> List[str]:
    """Validate list of strings.
    
    Args:
        values: List to validate
        name: Parameter name for error messages
        allow_empty: Whether to allow empty strings
        
    Returns:
        Validated list
        
    Raises:
        InvalidParameterError: If list is invalid
    """
    if not isinstance(values, (list, tuple)):
        raise InvalidParameterError(f"{name} must be a list, got {type(values)}")
    
    if len(values) == 0:
        raise InvalidParameterError(f"{name} cannot be empty")
    
    validated_values = []
    for i, value in enumerate(values):
        if not isinstance(value, str):
            raise InvalidParameterError(f"{name}[{i}] must be a string, got {type(value)}")
        
        if not allow_empty and len(value.strip()) == 0:
            raise InvalidParameterError(f"{name}[{i}] cannot be empty")
        
        validated_values.append(value)
    
    return validated_values


def safe_operation(operation_name: str = "operation"):
    """Decorator to safely execute operations with error handling.
    
    Args:
        operation_name: Name of the operation for error messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HDCValidationError:
                # Re-raise validation errors as-is
                raise
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}")
                raise HDCValidationError(f"Failed to execute {operation_name}: {e}") from e
        
        return wrapper
    return decorator


def validate_memory_capacity(capacity: int) -> int:
    """Validate memory capacity parameter.
    
    Args:
        capacity: Capacity to validate
        
    Returns:
        Validated capacity
        
    Raises:
        InvalidParameterError: If capacity is invalid
    """
    if not isinstance(capacity, int):
        raise InvalidParameterError(f"Capacity must be an integer, got {type(capacity)}")
    
    if capacity <= 0:
        raise InvalidParameterError(f"Capacity must be positive, got {capacity}")
    
    if capacity > 100000:
        warnings.warn(f"Large capacity ({capacity}) may use significant memory")
    
    return capacity


def validate_similarity_threshold(threshold: float) -> float:
    """Validate similarity threshold parameter.
    
    Args:
        threshold: Threshold to validate
        
    Returns:
        Validated threshold
        
    Raises:
        InvalidParameterError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise InvalidParameterError(f"Threshold must be a number, got {type(threshold)}")
    
    # Allow slightly outside [-1, 1] range due to floating point precision
    if not -1.1 <= threshold <= 1.1:
        raise InvalidParameterError(f"Similarity threshold must be between -1 and 1, got {threshold}")
    
    return float(threshold)


class ParameterValidator:
    """Centralized parameter validation class."""
    
    @staticmethod
    def validate_hdc_init_params(
        dim: int, 
        device: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Optional[str]]:
        """Validate HDC initialization parameters.
        
        Args:
            dim: Hypervector dimension
            device: Computing device
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (validated_dim, validated_device)
            
        Raises:
            InvalidParameterError: If parameters are invalid
        """
        validated_dim = validate_dimension(dim)
        validated_device = validate_device(device) if device else None
        
        return validated_dim, validated_device
    
    @staticmethod
    def validate_random_hv_params(
        sparsity: float = 0.5,
        batch_size: Optional[int] = None
    ) -> Tuple[float, Optional[int]]:
        """Validate random hypervector generation parameters.
        
        Args:
            sparsity: Sparsity level
            batch_size: Batch size for generation
            
        Returns:
            Tuple of (validated_sparsity, validated_batch_size)
            
        Raises:
            InvalidParameterError: If parameters are invalid
        """
        validated_sparsity = validate_sparsity(sparsity)
        validated_batch_size = None
        
        if batch_size is not None:
            validated_batch_size = validate_positive_int(batch_size, "batch_size")
        
        return validated_sparsity, validated_batch_size
    
    @staticmethod
    def validate_memory_init_params(
        capacity: int,
        items: Optional[List[str]] = None
    ) -> Tuple[int, Optional[List[str]]]:
        """Validate memory initialization parameters.
        
        Args:
            capacity: Memory capacity
            items: Initial items list
            
        Returns:
            Tuple of (validated_capacity, validated_items)
            
        Raises:
            InvalidParameterError: If parameters are invalid
        """
        validated_capacity = validate_memory_capacity(capacity)
        validated_items = None
        
        if items is not None:
            validated_items = validate_string_list(items, "items")
        
        return validated_capacity, validated_items