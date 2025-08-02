"""Tests for GPU acceleration functionality."""

import pytest
import torch
import numpy as np


@pytest.mark.gpu
class TestGPUAcceleration:
    """Test GPU-specific functionality."""
    
    def test_cuda_availability(self, skip_if_no_gpu):
        """Test CUDA availability and basic functionality."""
        assert torch.cuda.is_available(), "CUDA should be available"
        assert torch.cuda.device_count() > 0, "At least one CUDA device should be available"
        
        # Test basic tensor operations on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        result = x + y
        
        assert result.device.type == 'cuda'
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result.cpu(), expected)
    
    def test_gpu_memory_management(self, skip_if_no_gpu):
        """Test GPU memory management for large tensors."""
        # Test allocation and deallocation
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000, device='cuda')
        
        allocated_memory = torch.cuda.memory_allocated()
        assert allocated_memory > initial_memory
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= allocated_memory
    
    def test_gpu_hypervector_operations(self, skip_if_no_gpu, large_dimension):
        """Test hypervector operations on GPU."""
        # Create binary tensors on GPU
        hv1 = torch.randint(0, 2, (large_dimension,), dtype=torch.bool, device='cuda')
        hv2 = torch.randint(0, 2, (large_dimension,), dtype=torch.bool, device='cuda')
        
        # Test bundling (majority vote)
        bundled = (hv1.int() + hv2.int()) > 1
        assert bundled.device.type == 'cuda'
        assert bundled.dtype == torch.bool
        
        # Test binding (XOR)
        bound = torch.logical_xor(hv1, hv2)
        assert bound.device.type == 'cuda'
        assert bound.dtype == torch.bool
        
        # Test similarity computation
        intersection = torch.logical_and(hv1, hv2).sum().float()
        total = torch.tensor(large_dimension, dtype=torch.float, device='cuda')
        similarity = (2 * intersection - total) / total
        
        assert similarity.device.type == 'cuda'
        assert -1.0 <= similarity.item() <= 1.0
    
    @pytest.mark.benchmark
    def test_gpu_performance_vs_cpu(self, skip_if_no_gpu, large_dimension, benchmark_timeout):
        """Compare GPU vs CPU performance for HDC operations."""
        import time
        
        # Create test data
        np.random.seed(42)
        n_vectors = 100
        cpu_hvs = [torch.randint(0, 2, (large_dimension,), dtype=torch.bool) for _ in range(n_vectors)]
        gpu_hvs = [hv.cuda() for hv in cpu_hvs]
        
        # CPU timing
        start_time = time.time()
        cpu_result = torch.stack(cpu_hvs).sum(dim=0) > (n_vectors // 2)
        cpu_time = time.time() - start_time
        
        # GPU timing
        torch.cuda.synchronize()
        start_time = time.time()
        gpu_result = torch.stack(gpu_hvs).sum(dim=0) > (n_vectors // 2)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Results should be equivalent
        assert torch.equal(cpu_result, gpu_result.cpu())
        
        # GPU should be faster for large operations (not always true for small data)
        speedup = cpu_time / gpu_time
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s, Speedup: {speedup:.2f}x")
        
        # At minimum, GPU should not be significantly slower
        assert gpu_time < cpu_time * 2.0, "GPU should not be much slower than CPU"
    
    def test_multiple_gpu_detection(self, skip_if_no_gpu):
        """Test detection and basic usage of multiple GPUs if available."""
        device_count = torch.cuda.device_count()
        
        if device_count > 1:
            # Test operations on different GPUs
            hv1 = torch.randint(0, 2, (1000,), dtype=torch.bool, device='cuda:0')
            hv2 = torch.randint(0, 2, (1000,), dtype=torch.bool, device='cuda:1')
            
            # Move to same device for operations
            hv2_on_gpu0 = hv2.to('cuda:0')
            result = torch.logical_xor(hv1, hv2_on_gpu0)
            
            assert result.device.index == 0
        else:
            pytest.skip("Multiple GPUs not available")
    
    def test_gpu_memory_efficiency(self, skip_if_no_gpu, performance_config):
        """Test memory efficiency of GPU operations."""
        memory_limit_bytes = performance_config["memory_limit_mb"] * 1024 * 1024
        
        # Calculate maximum tensor size that fits in memory limit
        bytes_per_element = 1  # bool tensors use 1 byte per element
        max_elements = memory_limit_bytes // bytes_per_element
        
        # Test with tensor that should fit
        safe_size = max_elements // 4  # Use 25% of limit for safety
        
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            large_hv = torch.randint(0, 2, (safe_size,), dtype=torch.bool, device='cuda')
            allocated_memory = torch.cuda.memory_allocated()
            
            memory_used = allocated_memory - initial_memory
            assert memory_used <= memory_limit_bytes, f"Used {memory_used} bytes, limit {memory_limit_bytes}"
            
            # Test that we can still do operations
            result = torch.logical_not(large_hv)
            assert result.shape == large_hv.shape
            
        finally:
            torch.cuda.empty_cache()


@pytest.mark.hardware
class TestHardwareAcceleration:
    """Test hardware acceleration interfaces."""
    
    def test_device_detection(self):
        """Test detection of available acceleration devices."""
        devices = {
            'cpu': True,  # Always available
            'cuda': torch.cuda.is_available(),
        }
        
        # At minimum, CPU should be available
        assert devices['cpu'], "CPU should always be available"
        
        # Log available devices for debugging
        available_devices = [name for name, available in devices.items() if available]
        print(f"Available devices: {available_devices}")
    
    def test_device_switching(self):
        """Test switching between different compute devices."""
        # Test CPU
        cpu_tensor = torch.randint(0, 2, (1000,), dtype=torch.bool, device='cpu')
        assert cpu_tensor.device.type == 'cpu'
        
        if torch.cuda.is_available():
            # Test GPU
            gpu_tensor = cpu_tensor.cuda()
            assert gpu_tensor.device.type == 'cuda'
            
            # Test moving back to CPU
            cpu_tensor2 = gpu_tensor.cpu()
            assert cpu_tensor2.device.type == 'cpu'
            assert torch.equal(cpu_tensor, cpu_tensor2)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_agnostic_operations(self, device):
        """Test that operations work on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create tensors on specified device
        hv1 = torch.randint(0, 2, (1000,), dtype=torch.bool, device=device)
        hv2 = torch.randint(0, 2, (1000,), dtype=torch.bool, device=device)
        
        # Test operations
        bundled = (hv1.int() + hv2.int()) > 1
        bound = torch.logical_xor(hv1, hv2)
        similarity = (2 * torch.logical_and(hv1, hv2).sum().float() - 1000) / 1000
        
        # All results should be on the same device
        assert bundled.device.type == device
        assert bound.device.type == device
        assert similarity.device.type == device


@pytest.mark.hardware
class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def test_operation_timing(self, device, medium_dimension):
        """Test timing of HDC operations."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import time
        
        # Create test data
        hv1 = torch.randint(0, 2, (medium_dimension,), dtype=torch.bool, device=device)
        hv2 = torch.randint(0, 2, (medium_dimension,), dtype=torch.bool, device=device)
        
        # Time binding operation
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        result = torch.logical_xor(hv1, hv2)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly
        assert elapsed_time < 1.0, f"Operation took too long: {elapsed_time:.4f}s"
        assert result.shape == hv1.shape
    
    def test_memory_profiling(self, device, medium_dimension):
        """Test memory usage profiling."""
        if device == "cuda":
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate tensor
            hv = torch.randint(0, 2, (medium_dimension,), dtype=torch.bool, device=device)
            
            allocated_memory = torch.cuda.memory_allocated()
            memory_used = allocated_memory - initial_memory
            
            # Should use approximately the expected amount
            expected_bytes = medium_dimension  # 1 byte per bool
            assert memory_used >= expected_bytes, f"Used {memory_used} bytes, expected at least {expected_bytes}"
            
            # Clean up
            del hv
            torch.cuda.empty_cache()
        else:
            # For CPU, just test that tensors are created correctly
            hv = torch.randint(0, 2, (medium_dimension,), dtype=torch.bool, device=device)
            assert hv.numel() == medium_dimension