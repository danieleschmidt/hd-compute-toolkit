"""
Hardware Acceleration Module for HDC
====================================

Advanced hardware acceleration capabilities including FPGA emulation,
Vulkan compute shaders, and optimized kernel implementations.
"""

from .hardware_acceleration import (
    HardwareAccelerationManager,
    AcceleratorType,
    KernelOptimization,
    AcceleratorProfile,
    KernelImplementation,
    FPGAEmulator,
    VulkanComputeEngine,
    KernelOptimizer,
    global_acceleration_manager,
    hardware_accelerated
)

__all__ = [
    'HardwareAccelerationManager',
    'AcceleratorType',
    'KernelOptimization',
    'AcceleratorProfile',
    'KernelImplementation',
    'FPGAEmulator',
    'VulkanComputeEngine',
    'KernelOptimizer',
    'global_acceleration_manager',
    'hardware_accelerated'
]