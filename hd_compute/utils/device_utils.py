"""Device detection and management utilities."""

import logging
import platform
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        'platform': platform.platform(),
        'cpu_count': platform.machine(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0],
    }
    
    # Get CPU information
    try:
        import psutil
        device_info['cpu_physical_cores'] = psutil.cpu_count(logical=False)
        device_info['cpu_logical_cores'] = psutil.cpu_count(logical=True)
        device_info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        device_info['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        logger.warning("psutil not available, limited system information")
    
    # Check CUDA availability
    device_info['cuda_available'] = False
    device_info['cuda_devices'] = []
    try:
        import torch
        if torch.cuda.is_available():
            device_info['cuda_available'] = True
            device_info['cuda_device_count'] = torch.cuda.device_count()
            device_info['cuda_version'] = torch.version.cuda
            device_info['pytorch_version'] = torch.__version__
            
            # Get info for each CUDA device
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info['cuda_devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'total_memory_gb': device_props.total_memory / (1024**3),
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                })
    except ImportError:
        logger.debug("PyTorch not available")
    
    # Check JAX availability and devices
    device_info['jax_available'] = False
    device_info['jax_devices'] = []
    try:
        import jax
        device_info['jax_available'] = True
        device_info['jax_version'] = jax.__version__
        
        # Get JAX devices
        jax_devices = jax.devices()
        for device in jax_devices:
            device_info['jax_devices'].append({
                'platform': device.platform,
                'device_kind': device.device_kind,
                'id': device.id,
            })
    except ImportError:
        logger.debug("JAX not available")
    
    # Check Apple Metal Performance Shaders (MPS)
    device_info['mps_available'] = False
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['mps_available'] = True
    except:
        pass
    
    # Check for FPGA support (basic detection)
    device_info['fpga_available'] = False
    device_info['fpga_devices'] = []
    try:
        # Check for common FPGA device files
        import glob
        fpga_patterns = [
            '/dev/xdma*',
            '/dev/fpga*',
            '/dev/xilinx*'
        ]
        
        for pattern in fpga_patterns:
            devices = glob.glob(pattern)
            if devices:
                device_info['fpga_available'] = True
                device_info['fpga_devices'].extend(devices)
    except:
        pass
    
    # Check for Vulkan support
    device_info['vulkan_available'] = False
    try:
        # This is a basic check - in practice you'd need vulkan-tools
        import subprocess
        result = subprocess.run(['vulkaninfo', '--summary'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            device_info['vulkan_available'] = True
    except:
        pass
    
    return device_info


def select_optimal_device(
    preference: Optional[str] = None,
    require_gpu: bool = False,
    min_memory_gb: float = 1.0
) -> Tuple[str, Dict[str, Any]]:
    """Select optimal device for computation.
    
    Args:
        preference: Preferred device type ('cpu', 'cuda', 'mps', 'auto')
        require_gpu: Whether GPU is required
        min_memory_gb: Minimum required memory in GB
        
    Returns:
        Tuple of (selected_device, device_info)
    """
    device_info = get_device_info()
    
    if preference == 'cpu':
        return 'cpu', device_info
    
    # Check available options
    available_devices = ['cpu']
    
    if device_info['cuda_available']:
        # Check if any CUDA device has enough memory
        for cuda_device in device_info['cuda_devices']:
            if cuda_device['total_memory_gb'] >= min_memory_gb:
                available_devices.append('cuda')
                break
    
    if device_info['mps_available']:
        available_devices.append('mps')
    
    # Handle preferences
    if preference in available_devices:
        return preference, device_info
    
    if preference and preference != 'auto':
        logger.warning(f"Preferred device '{preference}' not available, selecting automatically")
    
    # Auto-select best available device
    if require_gpu and 'cuda' not in available_devices and 'mps' not in available_devices:
        raise RuntimeError("GPU required but no compatible GPU found")
    
    # Priority order: CUDA > MPS > CPU
    for device in ['cuda', 'mps', 'cpu']:
        if device in available_devices:
            logger.info(f"Auto-selected device: {device}")
            return device, device_info
    
    return 'cpu', device_info


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary with memory usage in MB
    """
    memory_info = {}
    
    # System memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory_info['system_total_mb'] = vm.total / (1024**2)
        memory_info['system_used_mb'] = vm.used / (1024**2)
        memory_info['system_available_mb'] = vm.available / (1024**2)
        memory_info['system_percent'] = vm.percent
    except ImportError:
        pass
    
    # CUDA memory
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                reserved = torch.cuda.memory_reserved(i) / (1024**2)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                
                memory_info[f'cuda_{i}_allocated_mb'] = allocated
                memory_info[f'cuda_{i}_reserved_mb'] = reserved
                memory_info[f'cuda_{i}_total_mb'] = total
                memory_info[f'cuda_{i}_free_mb'] = total - reserved
    except ImportError:
        pass
    
    # Process memory
    try:
        import psutil
        process = psutil.Process()
        memory_info['process_rss_mb'] = process.memory_info().rss / (1024**2)
        memory_info['process_vms_mb'] = process.memory_info().vms / (1024**2)
        memory_info['process_percent'] = process.memory_percent()
    except ImportError:
        pass
    
    return memory_info


def cleanup_gpu_memory():
    """Clean up GPU memory caches."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA memory cache")
    except ImportError:
        pass
    
    try:
        import jax
        # JAX doesn't have an explicit cache clearing function
        # but we can try to force garbage collection
        import gc
        gc.collect()
        logger.debug("Forced garbage collection for JAX")
    except ImportError:
        pass


def set_device_memory_fraction(device: str, fraction: float = 0.8):
    """Set memory fraction for GPU devices.
    
    Args:
        device: Device type ('cuda')
        fraction: Fraction of memory to use (0.0 to 1.0)
    """
    if device == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                # PyTorch doesn't have direct memory fraction setting
                # but we can set memory allocation limit
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    memory_limit = int(total_memory * fraction)
                    torch.cuda.set_per_process_memory_fraction(fraction, device=i)
                    logger.info(f"Set CUDA device {i} memory fraction to {fraction}")
        except ImportError:
            logger.warning("PyTorch not available for CUDA memory management")


def benchmark_device_performance(device: str, dimension: int = 10000) -> Dict[str, float]:
    """Benchmark basic operations on a device.
    
    Args:
        device: Device to benchmark
        dimension: Size of test arrays
        
    Returns:
        Dictionary with benchmark results (times in milliseconds)
    """
    import time
    import numpy as np
    
    results = {}
    
    if device == 'cpu':
        # NumPy operations
        start = time.time()
        a = np.random.rand(dimension)
        b = np.random.rand(dimension)
        results['array_creation_ms'] = (time.time() - start) * 1000
        
        start = time.time()
        c = a + b
        results['addition_ms'] = (time.time() - start) * 1000
        
        start = time.time()
        d = np.dot(a, b)
        results['dot_product_ms'] = (time.time() - start) * 1000
    
    elif device == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
                start = time.time()
                a = torch.rand(dimension, device='cuda')
                b = torch.rand(dimension, device='cuda')
                torch.cuda.synchronize()
                results['array_creation_ms'] = (time.time() - start) * 1000
                
                start = time.time()
                c = a + b
                torch.cuda.synchronize()
                results['addition_ms'] = (time.time() - start) * 1000
                
                start = time.time()
                d = torch.dot(a, b)
                torch.cuda.synchronize()
                results['dot_product_ms'] = (time.time() - start) * 1000
        except ImportError:
            results['error'] = 'PyTorch not available'
    
    elif device == 'jax':
        try:
            import jax.numpy as jnp
            from jax import random
            
            key = random.PRNGKey(0)
            
            start = time.time()
            a = random.uniform(key, (dimension,))
            b = random.uniform(key, (dimension,))
            results['array_creation_ms'] = (time.time() - start) * 1000
            
            start = time.time()
            c = a + b
            c.block_until_ready()  # JAX is async by default
            results['addition_ms'] = (time.time() - start) * 1000
            
            start = time.time()
            d = jnp.dot(a, b)
            d.block_until_ready()
            results['dot_product_ms'] = (time.time() - start) * 1000
        except ImportError:
            results['error'] = 'JAX not available'
    
    return results