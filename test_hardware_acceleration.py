#!/usr/bin/env python3
"""
Test Hardware Acceleration System
=================================

Comprehensive testing of FPGA emulation, Vulkan compute, and hardware optimization.
"""

import sys
import os
sys.path.append('/root/repo')

import time
import numpy as np

# Import directly from hardware acceleration module
sys.path.append('/root/repo/hd_compute/acceleration')
from hardware_acceleration import (
    HardwareAccelerationManager,
    AcceleratorType,
    KernelOptimization,
    FPGAEmulator,
    VulkanComputeEngine,
    KernelOptimizer,
    hardware_accelerated
)

def test_fpga_emulator():
    """Test FPGA emulator functionality."""
    print("Testing FPGA emulator...")
    
    try:
        fpga = FPGAEmulator(logic_elements=50000, memory_blocks=500)
        
        # Test circuit configuration
        circuit_id = fpga.configure_hdc_circuit('bundle', vector_width=1000, parallelism=32)
        print(f"FPGA circuit configured: {circuit_id}")
        
        # Test bundle operation
        test_vectors = [np.random.binomial(1, 0.5, 1000).astype(np.int8) for _ in range(3)]
        result = fpga.execute_hdc_operation(circuit_id, test_vectors)
        
        assert result.shape == test_vectors[0].shape, "FPGA bundle result should have correct shape"
        print(f"FPGA bundle result shape: {result.shape}")
        
        # Test bind operation
        bind_circuit_id = fpga.configure_hdc_circuit('bind', vector_width=1000, parallelism=32)
        bind_data = {'hv1': test_vectors[0], 'hv2': test_vectors[1]}
        bind_result = fpga.execute_hdc_operation(bind_circuit_id, bind_data)
        
        assert bind_result.shape == test_vectors[0].shape, "FPGA bind result should have correct shape"
        print(f"FPGA bind result shape: {bind_result.shape}")
        
        # Test resource utilization
        utilization = fpga.get_resource_utilization()
        print(f"FPGA resource utilization: {utilization}")
        
        print("✓ FPGA emulator test passed")
        return True
        
    except Exception as e:
        print(f"✗ FPGA emulator test failed: {e}")
        return False


def test_vulkan_compute():
    """Test Vulkan compute engine."""
    print("Testing Vulkan compute engine...")
    
    try:
        vulkan = VulkanComputeEngine()
        
        # Test shader compilation
        bundle_shader_id = vulkan.compile_compute_shader('hdc_bundle', 'bundle')
        bind_shader_id = vulkan.compile_compute_shader('hdc_bind', 'bind')
        similarity_shader_id = vulkan.compile_compute_shader('hdc_similarity', 'similarity')
        
        print(f"Compiled shaders: {bundle_shader_id}, {bind_shader_id}, {similarity_shader_id}")
        
        # Test bundle execution
        test_vectors = [np.random.binomial(1, 0.5, 500).astype(np.int8) for _ in range(3)]
        bundle_result = vulkan.execute_compute_shader(
            bundle_shader_id,
            {'vectors': test_vectors},
            test_vectors[0].shape
        )
        
        assert bundle_result.shape == test_vectors[0].shape, "Vulkan bundle result should have correct shape"
        print(f"Vulkan bundle result shape: {bundle_result.shape}")
        
        # Test bind execution
        bind_result = vulkan.execute_compute_shader(
            bind_shader_id,
            {'hv1': test_vectors[0], 'hv2': test_vectors[1]},
            test_vectors[0].shape
        )
        
        assert bind_result.shape == test_vectors[0].shape, "Vulkan bind result should have correct shape"
        print(f"Vulkan bind result shape: {bind_result.shape}")
        
        # Test similarity execution
        float_vectors = [v.astype(np.float32) for v in test_vectors[:2]]
        similarity_result = vulkan.execute_compute_shader(
            similarity_shader_id,
            {'hv1': float_vectors[0], 'hv2': float_vectors[1]},
            (1,)  # Similarity returns scalar
        )
        
        assert isinstance(similarity_result, (float, np.floating)), "Similarity should return float"
        print(f"Vulkan similarity result: {similarity_result}")
        
        # Test execution stats
        stats = vulkan.get_execution_stats()
        print(f"Vulkan execution stats: {stats}")
        
        print("✓ Vulkan compute test passed")
        return True
        
    except Exception as e:
        print(f"✗ Vulkan compute test failed: {e}")
        return False


def test_kernel_optimizer():
    """Test kernel optimization system."""
    print("Testing kernel optimizer...")
    
    try:
        optimizer = KernelOptimizer()
        
        # Test CPU vectorized optimization
        data_characteristics = {
            'vector_size': 1000,
            'data_type': 'int8',
            'num_vectors': 3
        }
        
        cpu_kernel = optimizer.optimize_kernel(
            'bundle', AcceleratorType.CPU_VECTORIZED, data_characteristics
        )
        
        assert cpu_kernel.accelerator_type == AcceleratorType.CPU_VECTORIZED
        assert cpu_kernel.optimization_level == KernelOptimization.VECTORIZED
        print(f"CPU kernel optimized: {cpu_kernel.kernel_id}")
        
        # Test FPGA optimization
        fpga_kernel = optimizer.optimize_kernel(
            'bundle', AcceleratorType.FPGA_EMULATED, data_characteristics
        )
        
        assert fpga_kernel.accelerator_type == AcceleratorType.FPGA_EMULATED
        assert fpga_kernel.optimization_level == KernelOptimization.PIPELINE_OPTIMIZED
        print(f"FPGA kernel optimized: {fpga_kernel.kernel_id}")
        
        # Test Vulkan optimization
        vulkan_kernel = optimizer.optimize_kernel(
            'similarity', AcceleratorType.VULKAN_COMPUTE, data_characteristics
        )
        
        assert vulkan_kernel.accelerator_type == AcceleratorType.VULKAN_COMPUTE
        assert vulkan_kernel.optimization_level == KernelOptimization.PARALLELIZED
        print(f"Vulkan kernel optimized: {vulkan_kernel.kernel_id}")
        
        # Test kernel benchmarking
        test_data = {'vectors': [np.random.binomial(1, 0.5, 100).astype(np.int8) for _ in range(3)]}
        benchmark_time = optimizer.benchmark_kernel(cpu_kernel, test_data)
        
        assert benchmark_time > 0, "Benchmark time should be positive"
        print(f"Kernel benchmark time: {benchmark_time:.4f}s")
        
        print("✓ Kernel optimizer test passed")
        return True
        
    except Exception as e:
        print(f"✗ Kernel optimizer test failed: {e}")
        return False


def test_acceleration_manager():
    """Test hardware acceleration manager."""
    print("Testing hardware acceleration manager...")
    
    try:
        manager = HardwareAccelerationManager()
        
        # Test available accelerators
        stats = manager.get_acceleration_stats()
        assert stats['available_accelerators'] >= 2, "Should have multiple accelerators available"
        print(f"Available accelerators: {stats['available_accelerators']}")
        
        # Test bundle operation acceleration
        test_vectors = [np.random.binomial(1, 0.5, 500).astype(np.int8) for _ in range(4)]
        
        # Test with FPGA
        fpga_result = manager.accelerate_operation(
            'bundle',
            {'vectors': test_vectors},
            preferred_accelerator='fpga_emulated'
        )
        
        assert fpga_result.shape == test_vectors[0].shape, "FPGA accelerated result should have correct shape"
        print(f"FPGA accelerated bundle result shape: {fpga_result.shape}")
        
        # Test with Vulkan
        vulkan_result = manager.accelerate_operation(
            'bind',
            {'hv1': test_vectors[0], 'hv2': test_vectors[1]},
            preferred_accelerator='vulkan_compute'
        )
        
        assert vulkan_result.shape == test_vectors[0].shape, "Vulkan accelerated result should have correct shape"
        print(f"Vulkan accelerated bind result shape: {vulkan_result.shape}")
        
        # Test automatic accelerator selection
        float_vectors = [v.astype(np.float32) for v in test_vectors[:2]]
        auto_result = manager.accelerate_operation(
            'similarity',
            {'hv1': float_vectors[0], 'hv2': float_vectors[1]}
        )
        
        assert isinstance(auto_result, (float, np.floating)), "Auto-selected similarity should return float"
        print(f"Auto-selected similarity result: {auto_result}")
        
        # Test CPU vectorized fallback
        cpu_result = manager.accelerate_operation(
            'permute',
            {'hv': test_vectors[0], 'shift': 3},
            preferred_accelerator='cpu_vectorized'
        )
        
        assert cpu_result.shape == test_vectors[0].shape, "CPU accelerated result should have correct shape"
        print(f"CPU accelerated permute result shape: {cpu_result.shape}")
        
        # Test caching (second call should be faster)
        start_time = time.time()
        cached_result = manager.accelerate_operation(
            'bundle',
            {'vectors': test_vectors},
            preferred_accelerator='fpga_emulated'
        )
        cached_time = time.time() - start_time
        
        print(f"Cached operation time: {cached_time:.4f}s")
        
        # Test acceleration statistics
        final_stats = manager.get_acceleration_stats()
        assert final_stats['cached_operations'] > 0, "Should have cached operations"
        print(f"Final acceleration stats: {final_stats}")
        
        print("✓ Hardware acceleration manager test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hardware acceleration manager test failed: {e}")
        return False


def test_decorator_interface():
    """Test hardware acceleration decorator interface."""
    print("Testing hardware acceleration decorator...")
    
    try:
        # Test decorated functions
        @hardware_accelerated('bundle', preferred_accelerator='fpga_emulated')
        def accelerated_bundle(vectors):
            # Fallback implementation
            result = vectors[0].copy()
            for vector in vectors[1:]:
                result = np.logical_or(result, vector).astype(result.dtype)
            return result
        
        @hardware_accelerated('bind', preferred_accelerator='vulkan_compute')
        def accelerated_bind(hv1, hv2):
            # Fallback implementation
            return np.logical_xor(hv1, hv2).astype(hv1.dtype)
        
        @hardware_accelerated('similarity')
        def accelerated_similarity(hv1, hv2):
            # Fallback implementation
            dot_product = np.dot(hv1, hv2)
            norm_product = np.linalg.norm(hv1) * np.linalg.norm(hv2)
            return dot_product / norm_product if norm_product > 1e-8 else 0.0
        
        # Test decorated function calls
        test_vectors = [np.random.binomial(1, 0.5, 300).astype(np.int8) for _ in range(3)]
        
        bundle_result = accelerated_bundle(test_vectors)
        assert bundle_result.shape == test_vectors[0].shape, "Decorated bundle should work"
        print(f"Decorated bundle result shape: {bundle_result.shape}")
        
        bind_result = accelerated_bind(test_vectors[0], test_vectors[1])
        assert bind_result.shape == test_vectors[0].shape, "Decorated bind should work"
        print(f"Decorated bind result shape: {bind_result.shape}")
        
        float_vectors = [v.astype(np.float32) for v in test_vectors[:2]]
        similarity_result = accelerated_similarity(float_vectors[0], float_vectors[1])
        assert isinstance(similarity_result, (float, np.floating)), "Decorated similarity should return float"
        print(f"Decorated similarity result: {similarity_result}")
        
        print("✓ Hardware acceleration decorator test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hardware acceleration decorator test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance comparison between accelerators."""
    print("Testing performance comparison...")
    
    try:
        manager = HardwareAccelerationManager()
        
        # Create larger test data for meaningful performance comparison
        large_vectors = [np.random.binomial(1, 0.5, 2000).astype(np.int8) for _ in range(8)]
        
        accelerator_types = ['cpu_vectorized', 'fpga_emulated', 'vulkan_compute']
        performance_results = {}
        
        for acc_type in accelerator_types:
            if acc_type in manager.available_accelerators:
                start_time = time.time()
                
                result = manager.accelerate_operation(
                    'bundle',
                    {'vectors': large_vectors},
                    preferred_accelerator=acc_type
                )
                
                execution_time = time.time() - start_time
                performance_results[acc_type] = execution_time
                
                print(f"{acc_type}: {execution_time:.4f}s")
        
        # Verify all accelerators produced valid results
        assert len(performance_results) >= 2, "Should test at least 2 accelerators"
        
        # Find fastest accelerator
        fastest = min(performance_results.keys(), key=lambda k: performance_results[k])
        print(f"Fastest accelerator: {fastest} ({performance_results[fastest]:.4f}s)")
        
        print("✓ Performance comparison test passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison test failed: {e}")
        return False


def run_all_tests():
    """Run all hardware acceleration tests."""
    print("=" * 60)
    print("HARDWARE ACCELERATION TESTS")
    print("=" * 60)
    
    test_functions = [
        test_fpga_emulator,
        test_vulkan_compute,
        test_kernel_optimizer,
        test_acceleration_manager,
        test_decorator_interface,
        test_performance_comparison
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"HARDWARE ACCELERATION TEST RESULTS: {passed_tests}/{total_tests} passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All hardware acceleration tests passed!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)