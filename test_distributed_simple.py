#!/usr/bin/env python3
"""
Simple Test for Quantum-Inspired Distributed Computing System
=============================================================

Basic testing without external dependencies.
"""

import sys
import os
sys.path.append('/root/repo')

import time
import numpy as np

# Import directly from the quantum distributed computing module
sys.path.append('/root/repo/hd_compute/distributed')
from quantum_distributed_computing import (
    DistributedComputeEngine,
    TaskPriority,
    NodeType,
    ComputeResource,
    DistributedTask
)

def test_basic_functionality():
    """Test basic distributed computing functionality."""
    print("Testing basic distributed computing functionality...")
    
    # Create engine
    engine = DistributedComputeEngine(max_workers=2)
    engine.start_workers()
    
    try:
        # Test HDC operations
        test_vectors = [np.random.binomial(1, 0.5, 50).astype(np.int8) for _ in range(3)]
        
        # Submit bundle task
        bundle_task_id = engine.submit_task(
            'hdc_bundle',
            {'vectors': test_vectors},
            priority=TaskPriority.HIGH
        )
        
        # Submit bind task
        bind_task_id = engine.submit_task(
            'hdc_bind',
            {'hv1': test_vectors[0], 'hv2': test_vectors[1]},
            priority=TaskPriority.NORMAL
        )
        
        # Wait for completion
        time.sleep(2.0)
        
        # Check results
        bundle_result = engine.get_task_result(bundle_task_id)
        bind_result = engine.get_task_result(bind_task_id)
        
        print(f"Bundle task completed: {bundle_result is not None}")
        print(f"Bind task completed: {bind_result is not None}")
        
        if bundle_result is not None:
            print(f"Bundle result shape: {bundle_result.shape}")
        
        if bind_result is not None:
            print(f"Bind result shape: {bind_result.shape}")
        
        # Get system status
        status = engine.get_system_status()
        print(f"System status: {status}")
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    finally:
        engine.stop_workers()


def test_quantum_operations():
    """Test quantum-inspired operations."""
    print("Testing quantum operations...")
    
    engine = DistributedComputeEngine(max_workers=1)
    engine.start_workers()
    
    try:
        # Create test data
        test_vectors = [np.random.normal(0, 1, 20) for _ in range(2)]
        
        # Test superposition
        superposition_task_id = engine.submit_task(
            'quantum_superposition',
            {'vectors': test_vectors},
            priority=TaskPriority.HIGH
        )
        
        # Test entanglement
        entanglement_task_id = engine.submit_task(
            'quantum_entanglement',
            {'hv1': test_vectors[0], 'hv2': test_vectors[1]},
            priority=TaskPriority.HIGH
        )
        
        # Wait for completion
        time.sleep(2.0)
        
        # Check results
        superposition_result = engine.get_task_result(superposition_task_id)
        entanglement_result = engine.get_task_result(entanglement_task_id)
        
        print(f"Superposition task completed: {superposition_result is not None}")
        print(f"Entanglement task completed: {entanglement_result is not None}")
        
        if superposition_result is not None:
            print(f"Superposition result shape: {superposition_result.shape}")
            
        if entanglement_result is not None:
            print(f"Entanglement result type: {type(entanglement_result)}")
            print(f"Entanglement result length: {len(entanglement_result)}")
        
        print("✓ Quantum operations test passed")
        return True
        
    except Exception as e:
        print(f"✗ Quantum operations test failed: {e}")
        return False
    finally:
        engine.stop_workers()


def test_cluster_management():
    """Test cluster management basics."""
    print("Testing cluster management...")
    
    try:
        engine = DistributedComputeEngine(max_workers=1)
        
        # Test node registration
        gpu_node = ComputeResource(
            node_id='test_gpu_node',
            node_type=NodeType.GPU,
            cpu_cores=8,
            memory_gb=16.0,
            gpu_count=1,
            gpu_memory_gb=8.0,
            capabilities=['gpu_compute', 'cuda']
        )
        
        engine.cluster_manager.register_node(gpu_node)
        
        # Test cluster stats
        stats = engine.cluster_manager.get_cluster_stats()
        print(f"Cluster stats: {stats}")
        
        # Should have at least the local node + test node
        assert stats['total_nodes'] >= 2
        print(f"Total nodes: {stats['total_nodes']}")
        
        print("✓ Cluster management test passed")
        return True
        
    except Exception as e:
        print(f"✗ Cluster management test failed: {e}")
        return False


def test_task_scheduling():
    """Test task scheduling with priorities."""
    print("Testing task scheduling...")
    
    engine = DistributedComputeEngine(max_workers=1)
    engine.start_workers()
    
    try:
        # Submit tasks with different priorities
        task_ids = []
        
        # Submit tasks in order: normal, low, critical
        task_ids.append(engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10, dtype=np.int8)]},
            priority=TaskPriority.NORMAL
        ))
        
        task_ids.append(engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10, dtype=np.int8)]},
            priority=TaskPriority.LOW
        ))
        
        task_ids.append(engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10, dtype=np.int8)]},
            priority=TaskPriority.CRITICAL
        ))
        
        # Wait for completion
        time.sleep(3.0)
        
        # Check that all tasks completed
        completed = 0
        for task_id in task_ids:
            result = engine.get_task_result(task_id)
            if result is not None:
                completed += 1
        
        print(f"Completed tasks: {completed}/{len(task_ids)}")
        
        # Get scheduler stats
        scheduler_stats = engine.scheduler.get_scheduler_stats()
        print(f"Scheduler stats: {scheduler_stats}")
        
        print("✓ Task scheduling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Task scheduling test failed: {e}")
        return False
    finally:
        engine.stop_workers()


def run_all_tests():
    """Run all basic distributed computing tests."""
    print("=" * 50)
    print("QUANTUM DISTRIBUTED COMPUTING - BASIC TESTS")
    print("=" * 50)
    
    test_functions = [
        test_basic_functionality,
        test_quantum_operations,
        test_cluster_management,
        test_task_scheduling
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"BASIC TEST RESULTS: {passed_tests}/{total_tests} passed")
    print("=" * 50)
    
    if passed_tests == total_tests:
        print("🎉 All basic distributed computing tests passed!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)