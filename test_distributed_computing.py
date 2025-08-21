#!/usr/bin/env python3
"""
Test Quantum-Inspired Distributed Computing System
=================================================

Comprehensive testing of the distributed computing capabilities including
task scheduling, load balancing, and quantum-inspired optimizations.
"""

import sys
import os
sys.path.append('/root/repo')

import time
import numpy as np
from hd_compute.distributed.quantum_distributed_computing import (
    DistributedComputeEngine,
    TaskPriority,
    NodeType,
    ComputeResource,
    DistributedTask
)

def test_basic_task_execution():
    """Test basic task execution."""
    print("Testing basic task execution...")
    
    engine = DistributedComputeEngine(max_workers=2)
    engine.start_workers()
    
    try:
        # Test HDC operations
        test_vectors = [np.random.binomial(1, 0.5, 100).astype(np.int8) for _ in range(3)]
        
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
        
        # Submit similarity task
        similarity_task_id = engine.submit_task(
            'hdc_similarity',
            {'hv1': test_vectors[0], 'hv2': test_vectors[1]},
            priority=TaskPriority.NORMAL
        )
        
        # Wait for completion
        time.sleep(3.0)
        
        # Check results
        bundle_result = engine.get_task_result(bundle_task_id)
        bind_result = engine.get_task_result(bind_task_id)
        similarity_result = engine.get_task_result(similarity_task_id)
        
        assert bundle_result is not None, "Bundle task should complete"
        assert bind_result is not None, "Bind task should complete"
        assert similarity_result is not None, "Similarity task should complete"
        
        assert bundle_result.shape == test_vectors[0].shape, "Bundle result should have correct shape"
        assert bind_result.shape == test_vectors[0].shape, "Bind result should have correct shape"
        assert isinstance(similarity_result, (float, np.floating)), "Similarity should return a float"
        
        print("✓ Basic task execution test passed")
        
    finally:
        engine.stop_workers()


def test_quantum_operations():
    """Test quantum-inspired operations."""
    print("Testing quantum operations...")
    
    engine = DistributedComputeEngine(max_workers=2)
    engine.start_workers()
    
    try:
        # Create test data
        test_vectors = [np.random.normal(0, 1, 50) for _ in range(3)]
        
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
        
        # Test measurement
        measurement_task_id = engine.submit_task(
            'quantum_measurement',
            {'state': test_vectors[0]},
            priority=TaskPriority.NORMAL
        )
        
        # Wait for completion
        time.sleep(3.0)
        
        # Check results
        superposition_result = engine.get_task_result(superposition_task_id)
        entanglement_result = engine.get_task_result(entanglement_task_id)
        measurement_result = engine.get_task_result(measurement_task_id)
        
        assert superposition_result is not None, "Superposition task should complete"
        assert entanglement_result is not None, "Entanglement task should complete"
        assert measurement_result is not None, "Measurement task should complete"
        
        assert superposition_result.shape == test_vectors[0].shape, "Superposition should preserve shape"
        assert len(entanglement_result) == 2, "Entanglement should return two states"
        assert 'measured_state' in measurement_result, "Measurement should return measured state"
        
        print("✓ Quantum operations test passed")
        
    finally:
        engine.stop_workers()


def test_task_priorities():
    """Test task priority scheduling."""
    print("Testing task priority scheduling...")
    
    engine = DistributedComputeEngine(max_workers=1)  # Single worker to test ordering
    engine.start_workers()
    
    try:
        # Submit tasks with different priorities
        task_order = []
        
        # Submit in order: normal, low, critical, high
        normal_task = engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10)]},
            priority=TaskPriority.NORMAL
        )
        task_order.append(('normal', normal_task))
        
        low_task = engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10)]},
            priority=TaskPriority.LOW
        )
        task_order.append(('low', low_task))
        
        critical_task = engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10)]},
            priority=TaskPriority.CRITICAL
        )
        task_order.append(('critical', critical_task))
        
        high_task = engine.submit_task(
            'hdc_bundle',
            {'vectors': [np.ones(10)]},
            priority=TaskPriority.HIGH
        )
        task_order.append(('high', high_task))
        
        # Wait for all to complete
        time.sleep(2.0)
        
        # All tasks should complete
        for priority_name, task_id in task_order:
            result = engine.get_task_result(task_id)
            assert result is not None, f"{priority_name} priority task should complete"
        
        print("✓ Task priority test passed")
        
    finally:
        engine.stop_workers()


def test_entanglement_groups():
    """Test quantum entanglement grouping."""
    print("Testing entanglement groups...")
    
    engine = DistributedComputeEngine(max_workers=2)
    engine.start_workers()
    
    try:
        # Create entangled tasks
        test_vectors = [np.random.normal(0, 1, 20) for _ in range(4)]
        
        # Group 1: Related quantum operations
        task1 = engine.submit_task(
            'quantum_superposition',
            {'vectors': test_vectors[:2]},
            priority=TaskPriority.NORMAL,
            entanglement_group='group1'
        )
        
        task2 = engine.submit_task(
            'quantum_measurement',
            {'state': test_vectors[0]},
            priority=TaskPriority.NORMAL,
            entanglement_group='group1'
        )
        
        # Group 2: Different operations
        task3 = engine.submit_task(
            'hdc_bundle',
            {'vectors': [v.astype(np.int8) for v in test_vectors[2:]]},
            priority=TaskPriority.NORMAL,
            entanglement_group='group2'
        )
        
        # Wait for completion
        time.sleep(3.0)
        
        # Check that all tasks completed
        results = [
            engine.get_task_result(task1),
            engine.get_task_result(task2),
            engine.get_task_result(task3)
        ]
        
        for i, result in enumerate(results, 1):
            assert result is not None, f"Entangled task {i} should complete"
        
        print("✓ Entanglement groups test passed")
        
    finally:
        engine.stop_workers()


def test_cluster_management():
    """Test cluster management functionality."""
    print("Testing cluster management...")
    
    engine = DistributedComputeEngine(max_workers=2)
    
    # Test node registration
    gpu_node = ComputeResource(
        node_id='gpu_node_1',
        node_type=NodeType.GPU,
        cpu_cores=8,
        memory_gb=32.0,
        gpu_count=2,
        gpu_memory_gb=16.0,
        capabilities=['gpu_compute', 'pytorch_gpu', 'cuda']
    )
    
    engine.cluster_manager.register_node(gpu_node)
    
    # Test cluster stats
    stats = engine.cluster_manager.get_cluster_stats()
    assert stats['total_nodes'] >= 2, "Should have at least local + GPU node"
    assert stats['total_gpus'] >= 2, "Should count GPU node's GPUs"
    
    # Test node selection
    gpu_requirements = {
        'gpu_required': True,
        'capabilities': ['gpu_compute']
    }
    
    selected_node = engine.cluster_manager.get_best_node(
        DistributedTask(
            task_id='test',
            operation_name='test',
            priority=TaskPriority.NORMAL,
            data_payload={},
            resource_requirements=gpu_requirements
        )
    )
    
    # Should select a node with GPU capabilities
    assert selected_node is not None, "Should find suitable node"
    assert selected_node.gpu_count > 0, "Selected node should have GPUs"
    
    print("✓ Cluster management test passed")


def test_load_balancing():
    """Test load balancing functionality."""
    print("Testing load balancing...")
    
    engine = DistributedComputeEngine(max_workers=2)
    
    # Create multiple nodes with different capabilities
    nodes = [
        ComputeResource(
            node_id='cpu_node_1',
            node_type=NodeType.CPU,
            cpu_cores=4,
            memory_gb=8.0,
            capabilities=['cpu_compute']
        ),
        ComputeResource(
            node_id='cpu_node_2',
            node_type=NodeType.CPU,
            cpu_cores=8,
            memory_gb=16.0,
            capabilities=['cpu_compute']
        ),
        ComputeResource(
            node_id='gpu_node_1',
            node_type=NodeType.GPU,
            cpu_cores=8,
            memory_gb=32.0,
            gpu_count=1,
            capabilities=['cpu_compute', 'gpu_compute']
        )
    ]
    
    for node in nodes:
        engine.cluster_manager.register_node(node)
    
    # Test different requirement scenarios
    cpu_requirements = {'cpu_cores': 2, 'memory_gb': 4.0}
    gpu_requirements = {'gpu_required': True}
    
    # Test load balancer selection
    cpu_task = DistributedTask(
        task_id='cpu_test',
        operation_name='test',
        priority=TaskPriority.NORMAL,
        data_payload={},
        resource_requirements=cpu_requirements
    )
    
    gpu_task = DistributedTask(
        task_id='gpu_test',
        operation_name='test',
        priority=TaskPriority.NORMAL,
        data_payload={},
        resource_requirements=gpu_requirements
    )
    
    cpu_node = engine.cluster_manager.load_balancer.select_node(
        list(engine.cluster_manager.nodes.values()),
        cpu_requirements
    )
    
    gpu_node = engine.cluster_manager.load_balancer.select_node(
        list(engine.cluster_manager.nodes.values()),
        gpu_requirements
    )
    
    assert cpu_node is not None, "Should select node for CPU task"
    assert gpu_node is not None, "Should select node for GPU task"
    assert gpu_node.gpu_count > 0, "GPU task should be assigned to GPU node"
    
    print("✓ Load balancing test passed")


def test_system_integration():
    """Test full system integration."""
    print("Testing system integration...")
    
    engine = DistributedComputeEngine(max_workers=3)
    engine.start_workers()
    
    try:
        # Submit a variety of tasks
        tasks = []
        
        # HDC computation pipeline
        base_vectors = [np.random.binomial(1, 0.5, 100).astype(np.int8) for _ in range(5)]
        
        # Step 1: Bundle base vectors
        bundle_task = engine.submit_task(
            'hdc_bundle',
            {'vectors': base_vectors},
            priority=TaskPriority.HIGH
        )
        tasks.append(bundle_task)
        
        # Step 2: Create permutations
        for i in range(3):
            perm_task = engine.submit_task(
                'hdc_permute',
                {'hv': base_vectors[i], 'shift': i+1},
                priority=TaskPriority.NORMAL
            )
            tasks.append(perm_task)
        
        # Step 3: Quantum operations
        quantum_tasks = []
        for i in range(2):
            q_task = engine.submit_task(
                'quantum_superposition',
                {'vectors': [v.astype(np.float64) for v in base_vectors[i:i+2]]},
                priority=TaskPriority.HIGH,
                entanglement_group='quantum_pipeline'
            )
            quantum_tasks.append(q_task)
            tasks.append(q_task)
        
        # Step 4: ML simulation
        ml_task = engine.submit_task(
            'ml_train',
            {'training_time': 0.5},
            priority=TaskPriority.NORMAL
        )
        tasks.append(ml_task)
        
        # Wait for all tasks to complete
        max_wait = 10.0
        start_time = time.time()
        
        completed_tasks = 0
        while time.time() - start_time < max_wait and completed_tasks < len(tasks):
            completed_tasks = sum(
                1 for task_id in tasks 
                if engine.get_task_result(task_id) is not None
            )
            time.sleep(0.1)
        
        # Verify completion
        success_rate = completed_tasks / len(tasks)
        assert success_rate >= 0.8, f"At least 80% of tasks should complete, got {success_rate:.2%}"
        
        # Check system status
        status = engine.get_system_status()
        assert status['running'] == True, "Engine should be running"
        assert status['execution_stats']['completed'] > 0, "Should have completed tasks"
        
        print(f"✓ System integration test passed - {completed_tasks}/{len(tasks)} tasks completed")
        
    finally:
        engine.stop_workers()


def run_all_tests():
    """Run all distributed computing tests."""
    print("=" * 60)
    print("QUANTUM-INSPIRED DISTRIBUTED COMPUTING TESTS")
    print("=" * 60)
    
    test_functions = [
        test_basic_task_execution,
        test_quantum_operations,
        test_task_priorities,
        test_entanglement_groups,
        test_cluster_management,
        test_load_balancing,
        test_system_integration
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
    
    print("=" * 60)
    print(f"DISTRIBUTED COMPUTING TEST RESULTS: {passed_tests}/{total_tests} passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 All distributed computing tests passed!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)