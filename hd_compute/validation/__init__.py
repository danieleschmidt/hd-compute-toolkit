"""Enhanced validation and error handling for research-grade HDC."""

from .research_validation import (
    ResearchValidator,
    ExperimentalValidation,
    StatisticalValidation,
    HypervectorIntegrityChecker
)
from .error_recovery import (
    RobustErrorHandler,
    GracefulDegradation,
    FailsafeOperations
)
from .quality_assurance import (
    QualityMetrics,
    ReproducibilityChecker,
    PerformanceMonitor
)

__all__ = [
    'ResearchValidator',
    'ExperimentalValidation', 
    'StatisticalValidation',
    'HypervectorIntegrityChecker',
    'RobustErrorHandler',
    'GracefulDegradation',
    'FailsafeOperations',
    'QualityMetrics',
    'ReproducibilityChecker',
    'PerformanceMonitor'
]