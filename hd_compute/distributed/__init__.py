"""Distributed and high-performance computing for HDC at scale."""

from .distributed_hdc import (
    DistributedHDC,
    ClusterManager,
    LoadBalancer
)
from .parallel_processing import (
    ParallelHDC,
    GPUAccelerator,
    MultiProcessingEngine
)
from .optimization import (
    PerformanceOptimizer,
    CacheManager,
    VectorizedOperations
)
from .scaling import (
    AutoScaler,
    ResourceManager,
    WorkloadDistributor
)

__all__ = [
    'DistributedHDC',
    'ClusterManager', 
    'LoadBalancer',
    'ParallelHDC',
    'GPUAccelerator',
    'MultiProcessingEngine',
    'PerformanceOptimizer',
    'CacheManager',
    'VectorizedOperations',
    'AutoScaler',
    'ResourceManager',
    'WorkloadDistributor'
]