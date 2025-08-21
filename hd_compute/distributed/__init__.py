"""Distributed and high-performance computing for HDC at scale."""

# Import quantum distributed computing
from .quantum_distributed_computing import (
    DistributedComputeEngine,
    TaskPriority,
    NodeType,
    ComputeResource,
    DistributedTask,
    QuantumTaskScheduler,
    ClusterManager as QuantumClusterManager,
    LoadBalancer as QuantumLoadBalancer,
    global_compute_engine,
    distributed_task,
    start_distributed_computing,
    stop_distributed_computing
)

# Try to import existing modules, but don't fail if dependencies are missing
try:
    from .distributed_hdc import (
        DistributedHDC,
        ClusterManager,
        LoadBalancer
    )
except ImportError:
    DistributedHDC = None
    ClusterManager = None
    LoadBalancer = None

try:
    from .parallel_processing import (
        ParallelHDC,
        GPUAccelerator,
        MultiProcessingEngine
    )
except ImportError:
    ParallelHDC = None
    GPUAccelerator = None
    MultiProcessingEngine = None

# Always available: quantum distributed computing
__all__ = [
    'DistributedComputeEngine',
    'TaskPriority',
    'NodeType',
    'ComputeResource',
    'DistributedTask',
    'QuantumTaskScheduler',
    'QuantumClusterManager',
    'QuantumLoadBalancer',
    'global_compute_engine',
    'distributed_task',
    'start_distributed_computing',
    'stop_distributed_computing'
]

# Add legacy imports if available
if DistributedHDC is not None:
    __all__.extend(['DistributedHDC', 'ClusterManager', 'LoadBalancer'])
if ParallelHDC is not None:
    __all__.extend(['ParallelHDC', 'GPUAccelerator', 'MultiProcessingEngine'])