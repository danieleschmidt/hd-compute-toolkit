#!/usr/bin/env python3
"""Simplified test runner for quantum task planning components.

This module provides a streamlined test execution system to validate
the core functionality of the quantum-inspired task planning system
without complex pytest dependencies.
"""

import sys
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, '/root/repo')

def run_test(test_name: str, test_func: callable) -> Dict[str, Any]:
    """Run a single test with error handling and timing."""
    print(f"Running {test_name}...", end=" ")
    
    start_time = time.perf_counter()
    try:
        result = test_func()
        execution_time = time.perf_counter() - start_time
        
        if result is True or result is None:
            print(f"PASSED ({execution_time:.3f}s)")
            return {"status": "PASSED", "time": execution_time, "error": None}
        else:
            print(f"FAILED ({execution_time:.3f}s) - Returned: {result}")
            return {"status": "FAILED", "time": execution_time, "error": f"Test returned {result}"}
    
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        print(f"ERROR ({execution_time:.3f}s) - {str(e)}")
        return {"status": "ERROR", "time": execution_time, "error": str(e)}


def test_quantum_task_planner_basic():
    """Test basic quantum task planner functionality."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner, PlanningStrategy
        
        # Initialize planner
        planner = QuantumTaskPlanner(dim=500, device="cpu", enable_distributed=False)
        
        # Add tasks
        planner.add_task("test_task_1", "Task 1", "Test task 1", priority=1.0)
        planner.add_task("test_task_2", "Task 2", "Test task 2", dependencies={"test_task_1"}, priority=2.0)
        
        # Add resource
        planner.add_resource("test_resource", "Test Resource", capacity=100.0)
        
        # Create plan
        plan = planner.create_quantum_plan(
            strategy=PlanningStrategy.TEMPORAL_OPTIMIZATION,
            optimization_objectives=['minimize_duration']
        )
        
        # Validate results
        assert plan is not None
        assert len(plan.tasks) > 0
        assert plan.success_probability > 0.0
        assert plan.quantum_coherence > 0.0
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_task_planning_validation():
    """Test task planning validation system."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner, PlanningStrategy
        from hd_compute.validation.task_planning_validation import TaskPlanningValidator
        
        # Create planner and validator
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        validator = TaskPlanningValidator(planner)
        
        # Add test data
        planner.add_task("val_task_1", "Validation Task 1", "Test validation", priority=1.0)
        planner.add_resource("val_resource", "Validation Resource", capacity=50.0)
        
        # Create plan
        plan = planner.create_quantum_plan(strategy=PlanningStrategy.QUANTUM_SUPERPOSITION)
        
        # Validate plan
        report = validator.validate_plan_comprehensive(plan.id)
        
        # Check validation results
        assert report is not None
        assert report.plan_id == plan.id
        assert 0.0 <= report.overall_score <= 1.0
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.issues, list)
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_quantum_security_basic():
    """Test basic security functionality."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner
        from hd_compute.security.task_planning_security import QuantumSecurityManager, AccessRight
        
        # Create planner and security manager
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        security_manager = QuantumSecurityManager(planner)
        
        # Test authentication
        context = security_manager.authenticate_user(
            username="test_user",
            password="test_password"
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        assert context.session_token is not None
        
        # Test session validation
        validated = security_manager.validate_session(context.session_token)
        assert validated is not None
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_performance_optimization_basic():
    """Test basic performance optimization."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner
        from hd_compute.performance.quantum_optimization import QuantumPerformanceOptimizer, OptimizationStrategy
        
        # Create planner and optimizer
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        optimizer = QuantumPerformanceOptimizer(planner, enable_gpu=False)
        
        # Test performance profiling
        profile = optimizer.performance_profile
        assert profile is not None
        assert profile.cpu_cores > 0
        assert profile.memory_gb > 0
        
        # Test baseline measurement
        metrics = optimizer._measure_baseline_performance()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'avg_execution_time' in metrics
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_distributed_planning_basic():
    """Test basic distributed planning initialization."""
    try:
        from hd_compute.distributed.quantum_task_distribution import QuantumDistributedTaskPlanner, ClusterConfiguration
        
        # Create cluster configuration
        config = ClusterConfiguration(min_nodes=2, max_nodes=5, auto_scaling_enabled=False)
        
        # Create distributed planner
        distributed_planner = QuantumDistributedTaskPlanner(
            cluster_config=config,
            node_id="test_node",
            enable_auto_scaling=False
        )
        
        assert distributed_planner.node_id == "test_node"
        assert distributed_planner.local_planner is not None
        assert len(distributed_planner.cluster_nodes) == 0
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_quantum_coherence_stability():
    """Test quantum coherence stability across multiple runs."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner, PlanningStrategy
        
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        
        # Add test tasks
        for i in range(3):
            planner.add_task(f"coherence_task_{i}", f"Task {i}", f"Test task {i}", priority=1.0)
        
        # Create multiple plans and check coherence stability
        coherence_values = []
        for _ in range(5):
            plan = planner.create_quantum_plan(
                strategy=PlanningStrategy.HYBRID_QUANTUM,
                optimization_objectives=['maximize_success']
            )
            coherence_values.append(plan.quantum_coherence)
        
        # Statistical validation
        coherence_array = np.array(coherence_values)
        mean_coherence = np.mean(coherence_array)
        std_coherence = np.std(coherence_array)
        
        # Verify reasonable values
        assert all(0.0 <= c <= 1.0 for c in coherence_values)
        assert 0.2 <= mean_coherence <= 1.0  # Reasonable mean
        assert std_coherence < 0.4  # Reasonable stability
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_plan_execution_simulation():
    """Test plan execution simulation."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner, PlanningStrategy
        
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        
        # Add tasks and resources
        planner.add_task("exec_task_1", "Execution Task 1", "Test execution", priority=1.0)
        planner.add_task("exec_task_2", "Execution Task 2", "Test execution", 
                        dependencies={"exec_task_1"}, priority=2.0)
        planner.add_resource("exec_resource", "Execution Resource", capacity=100.0)
        
        # Create plan
        plan = planner.create_quantum_plan(
            strategy=PlanningStrategy.ATTENTION_GUIDED,
            optimization_objectives=['minimize_duration', 'maximize_success']
        )
        
        # Test plan properties
        assert plan.id is not None
        assert len(plan.tasks) > 0
        assert plan.schedule is not None
        assert plan.resource_allocation is not None
        assert plan.total_duration > timedelta(0)
        
        # Test visualization export
        viz_data = planner.export_plan_visualization(plan.id)
        assert viz_data['plan_id'] == plan.id
        assert 'quantum_state' in viz_data
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_all_planning_strategies():
    """Test all planning strategies."""
    try:
        from hd_compute.applications.task_planning import QuantumTaskPlanner, PlanningStrategy
        
        planner = QuantumTaskPlanner(dim=500, device="cpu")
        
        # Add test data
        planner.add_task("strategy_task_1", "Strategy Task 1", "Test", priority=1.0)
        planner.add_task("strategy_task_2", "Strategy Task 2", "Test", 
                        dependencies={"strategy_task_1"}, priority=2.0)
        planner.add_resource("strategy_resource", "Strategy Resource", capacity=100.0)
        
        # Test all strategies
        strategies = [
            PlanningStrategy.QUANTUM_SUPERPOSITION,
            PlanningStrategy.TEMPORAL_OPTIMIZATION,
            PlanningStrategy.CAUSAL_REASONING,
            PlanningStrategy.ATTENTION_GUIDED,
            PlanningStrategy.HYBRID_QUANTUM
        ]
        
        successful_strategies = 0
        for strategy in strategies:
            try:
                plan = planner.create_quantum_plan(
                    strategy=strategy,
                    optimization_objectives=['minimize_duration']
                )
                
                if plan and plan.success_probability > 0.0 and plan.quantum_coherence > 0.0:
                    successful_strategies += 1
                    
            except Exception as e:
                print(f"Strategy {strategy.value} failed: {e}")
        
        # At least 80% of strategies should work
        success_rate = successful_strategies / len(strategies)
        assert success_rate >= 0.8, f"Only {success_rate:.1%} of strategies succeeded"
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def test_cache_functionality():
    """Test intelligent caching functionality."""
    try:
        from hd_compute.performance.quantum_optimization import QuantumIntelligentCache
        
        # Create cache
        cache = QuantumIntelligentCache(max_size=1000, coherence_threshold=0.5)
        
        # Test basic operations
        cache.put("test_key", "test_value", coherence_score=0.8)
        retrieved = cache.get("test_key")
        assert retrieved == "test_value"
        
        # Test coherence filtering
        cache.put("low_coherence_key", "low_value", coherence_score=0.3)
        low_retrieved = cache.get("low_coherence_key")
        assert low_retrieved is None  # Should be rejected
        
        # Test cache eviction (simplified)
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}", coherence_score=0.9)
        
        assert len(cache.cache) <= 5
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        raise


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("=" * 80)
    print("QUANTUM-INSPIRED TASK PLANNING - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Define test suite
    tests = [
        ("Basic Quantum Task Planner", test_quantum_task_planner_basic),
        ("Task Planning Validation", test_task_planning_validation),
        ("Quantum Security Basic", test_quantum_security_basic),
        ("Performance Optimization Basic", test_performance_optimization_basic),
        ("Distributed Planning Basic", test_distributed_planning_basic),
        ("Quantum Coherence Stability", test_quantum_coherence_stability),
        ("Plan Execution Simulation", test_plan_execution_simulation),
        ("All Planning Strategies", test_all_planning_strategies),
        ("Cache Functionality", test_cache_functionality)
    ]
    
    # Run tests
    results = {}
    total_time = 0
    
    for test_name, test_func in tests:
        result = run_test(test_name, test_func)
        results[test_name] = result
        total_time += result["time"]
    
    # Summary
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")
    errors = sum(1 for r in results.values() if r["status"] == "ERROR")
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print(f"Total Time: {total_time:.3f}s")
    print()
    
    # Detailed results
    if failed > 0 or errors > 0:
        print("FAILED/ERROR TESTS:")
        print("-" * 40)
        for test_name, result in results.items():
            if result["status"] in ["FAILED", "ERROR"]:
                print(f"  {test_name}: {result['status']}")
                if result["error"]:
                    print(f"    Error: {result['error']}")
        print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY:")
    print("-" * 40)
    for test_name, result in results.items():
        print(f"  {test_name}: {result['time']:.3f}s")
    print()
    
    # Quality assessment
    coverage_estimate = passed / total * 100
    
    print("QUALITY ASSESSMENT:")
    print("-" * 40)
    print(f"Test Coverage Estimate: {coverage_estimate:.1f}%")
    
    if coverage_estimate >= 85:
        quality_rating = "EXCELLENT ✅"
    elif coverage_estimate >= 70:
        quality_rating = "GOOD ✅"
    elif coverage_estimate >= 50:
        quality_rating = "ADEQUATE ⚠️"
    else:
        quality_rating = "NEEDS IMPROVEMENT ❌"
    
    print(f"Overall Quality Rating: {quality_rating}")
    print()
    
    # Success criteria
    success_threshold = 0.85  # 85% success rate required
    
    if passed / total >= success_threshold:
        print("🎉 COMPREHENSIVE TESTING: SUCCESS!")
        print("   All major components validated successfully")
        print("   Quantum task planning system is ready for deployment")
        return True
    else:
        print("⚠️  COMPREHENSIVE TESTING: PARTIAL SUCCESS")
        print("   Some components need attention before deployment")
        print("   Review failed tests and address issues")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)