"""Comprehensive test suite for quantum-inspired task planning.

This module provides extensive testing coverage for all quantum task planning
components including validation, security, distributed computing, and performance
optimization with statistical rigor and reproducibility verification.
"""

import pytest
import numpy as np
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock
import logging

# Import modules under test
from hd_compute.applications.task_planning import (
    QuantumTaskPlanner, Task, Resource, ExecutionPlan, TaskStatus, PlanningStrategy
)
from hd_compute.validation.task_planning_validation import (
    TaskPlanningValidator, ValidationSeverity, PlanValidationReport
)
from hd_compute.security.task_planning_security import (
    QuantumSecurityManager, SecurityContext, SecurityLevel, AccessRight, SecurityPolicy
)
from hd_compute.distributed.quantum_task_distribution import (
    QuantumDistributedTaskPlanner, ClusterConfiguration, NodeRole
)
from hd_compute.performance.quantum_optimization import (
    QuantumPerformanceOptimizer, OptimizationStrategy
)


class TestQuantumTaskPlanner:
    """Test suite for the quantum task planner core functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create a quantum task planner instance for testing."""
        return QuantumTaskPlanner(
            dim=1000,  # Smaller dimension for faster tests
            device="cpu",
            max_superposition_states=10,
            enable_distributed=False
        )
    
    @pytest.fixture
    def sample_tasks(self, planner):
        """Create sample tasks for testing."""
        tasks = []
        
        # Task with no dependencies
        planner.add_task(
            task_id="task_1",
            name="Initial Setup",
            description="Setup phase with no dependencies",
            priority=2.0,
            estimated_duration=timedelta(hours=1)
        )
        tasks.append("task_1")
        
        # Task with dependency
        planner.add_task(
            task_id="task_2", 
            name="Data Processing",
            description="Process data after setup",
            dependencies={"task_1"},
            priority=3.0,
            estimated_duration=timedelta(hours=2)
        )
        tasks.append("task_2")
        
        # Task with multiple dependencies
        planner.add_task(
            task_id="task_3",
            name="Analysis",
            description="Analyze processed data",
            dependencies={"task_2"},
            priority=1.5,
            estimated_duration=timedelta(hours=3)
        )
        tasks.append("task_3")
        
        return tasks
    
    @pytest.fixture
    def sample_resources(self, planner):
        """Create sample resources for testing."""
        resources = []
        
        planner.add_resource(
            resource_id="cpu_cluster",
            name="CPU Cluster",
            capacity=100.0,
            cost_per_unit=0.5
        )
        resources.append("cpu_cluster")
        
        planner.add_resource(
            resource_id="gpu_nodes",
            name="GPU Nodes", 
            capacity=50.0,
            cost_per_unit=2.0
        )
        resources.append("gpu_nodes")
        
        planner.add_resource(
            resource_id="storage",
            name="Storage System",
            capacity=1000.0,
            cost_per_unit=0.1
        )
        resources.append("storage")
        
        return resources
    
    def test_planner_initialization(self):
        """Test quantum task planner initialization."""
        planner = QuantumTaskPlanner(dim=2000, device="cpu")
        
        assert planner.dim == 2000
        assert planner.device == "cpu"
        assert planner.max_superposition_states == 100
        assert len(planner.tasks) == 0
        assert len(planner.resources) == 0
        assert len(planner.plans) == 0
    
    def test_task_addition(self, planner):
        """Test task addition and encoding."""
        planner.add_task(
            task_id="test_task",
            name="Test Task",
            description="A task for testing",
            priority=2.5,
            estimated_duration=timedelta(hours=1.5)
        )
        
        assert "test_task" in planner.tasks
        task = planner.tasks["test_task"]
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.description == "A task for testing"
        assert task.priority == 2.5
        assert task.estimated_duration == timedelta(hours=1.5)
        assert task.status == TaskStatus.PENDING
        assert task.hypervector is not None
    
    def test_resource_addition(self, planner):
        """Test resource addition and encoding."""
        planner.add_resource(
            resource_id="test_resource",
            name="Test Resource",
            capacity=75.0,
            cost_per_unit=1.5
        )
        
        assert "test_resource" in planner.resources
        resource = planner.resources["test_resource"]
        
        assert resource.id == "test_resource"
        assert resource.name == "Test Resource"
        assert resource.capacity == 75.0
        assert resource.available == 75.0
        assert resource.cost_per_unit == 1.5
        assert resource.hypervector is not None
    
    def test_quantum_plan_creation(self, planner, sample_tasks, sample_resources):
        """Test quantum plan creation with different strategies."""
        strategies = [
            PlanningStrategy.QUANTUM_SUPERPOSITION,
            PlanningStrategy.TEMPORAL_OPTIMIZATION,
            PlanningStrategy.CAUSAL_REASONING,
            PlanningStrategy.ATTENTION_GUIDED,
            PlanningStrategy.HYBRID_QUANTUM
        ]
        
        for strategy in strategies:
            plan = planner.create_quantum_plan(
                strategy=strategy,
                optimization_objectives=['minimize_duration', 'maximize_success'],
                constraints={'max_parallel_tasks': 2}
            )
            
            assert plan is not None
            assert plan.id is not None
            assert len(plan.tasks) > 0
            assert plan.success_probability > 0.0
            assert plan.quantum_coherence > 0.0
            assert plan.total_duration > timedelta(0)
            assert plan.plan_hypervector is not None
            assert plan.id in planner.plans
    
    def test_dependency_resolution(self, planner, sample_tasks):
        """Test task dependency resolution in planning."""
        plan = planner.create_quantum_plan(
            strategy=PlanningStrategy.CAUSAL_REASONING,
            optimization_objectives=['minimize_duration']
        )
        
        # Verify task ordering respects dependencies
        task_positions = {task_id: pos for pos, task_id in enumerate(plan.tasks)}
        
        # task_1 should come before task_2
        if "task_1" in task_positions and "task_2" in task_positions:
            assert task_positions["task_1"] < task_positions["task_2"]
        
        # task_2 should come before task_3
        if "task_2" in task_positions and "task_3" in task_positions:
            assert task_positions["task_2"] < task_positions["task_3"]
    
    def test_quantum_coherence_stability(self, planner, sample_tasks):
        """Test quantum coherence stability across multiple planning runs."""
        coherence_values = []
        
        for _ in range(10):
            plan = planner.create_quantum_plan(
                strategy=PlanningStrategy.HYBRID_QUANTUM,
                optimization_objectives=['maximize_success']
            )
            coherence_values.append(plan.quantum_coherence)
        
        # Coherence values should be within reasonable range
        assert all(0.0 <= c <= 1.0 for c in coherence_values)
        
        # Coherence should have reasonable stability (not too much variation)
        coherence_std = np.std(coherence_values)
        assert coherence_std < 0.3  # Less than 30% variation
    
    @pytest.mark.asyncio
    async def test_async_plan_execution(self, planner, sample_tasks, sample_resources):
        """Test asynchronous plan execution."""
        plan = planner.create_quantum_plan(
            strategy=PlanningStrategy.TEMPORAL_OPTIMIZATION
        )
        
        # Execute plan asynchronously
        execution_task = planner.execute_plan_async(plan.id)
        assert execution_task is not None
        
        # Wait for execution with timeout
        execution_results = await asyncio.wait_for(execution_task, timeout=10.0)
        
        assert execution_results is not None
        assert execution_results['plan_id'] == plan.id
        assert 'start_time' in execution_results
        assert 'end_time' in execution_results
        assert 'task_results' in execution_results
        assert 'quantum_decoherence' in execution_results
    
    def test_planning_analytics(self, planner, sample_tasks):
        """Test planning analytics functionality."""
        # Create a few plans
        for _ in range(3):
            planner.create_quantum_plan(
                strategy=PlanningStrategy.HYBRID_QUANTUM,
                optimization_objectives=['minimize_duration', 'maximize_success']
            )
        
        analytics = planner.get_planning_analytics()
        
        assert 'total_plans_created' in analytics
        assert analytics['total_plans_created'] >= 3
        assert 'average_success_probability' in analytics
        assert 'average_quantum_coherence' in analytics
        assert 'strategy_distribution' in analytics
        assert 'performance_metrics' in analytics
    
    def test_plan_visualization_export(self, planner, sample_tasks):
        """Test plan visualization data export."""
        plan = planner.create_quantum_plan(
            strategy=PlanningStrategy.ATTENTION_GUIDED
        )
        
        viz_data = planner.export_plan_visualization(plan.id)
        
        assert viz_data['plan_id'] == plan.id
        assert 'tasks' in viz_data
        assert 'dependencies' in viz_data
        assert 'timeline' in viz_data
        assert 'quantum_state' in viz_data
        
        # Verify quantum state information
        quantum_state = viz_data['quantum_state']
        assert 'coherence' in quantum_state
        assert 'success_probability' in quantum_state
        assert 'superposition_dimension' in quantum_state
        assert quantum_state['superposition_dimension'] == planner.dim


class TestTaskPlanningValidation:
    """Test suite for task planning validation system."""
    
    @pytest.fixture
    def planner_with_validator(self):
        """Create planner with validator for testing."""
        planner = QuantumTaskPlanner(dim=1000, device="cpu")
        validator = TaskPlanningValidator(planner)
        return planner, validator
    
    @pytest.fixture
    def valid_plan(self, planner_with_validator):
        """Create a valid plan for testing."""
        planner, validator = planner_with_validator
        
        # Add tasks and resources
        planner.add_task("task_1", "Task 1", "Description 1", priority=1.0)
        planner.add_task("task_2", "Task 2", "Description 2", dependencies={"task_1"}, priority=2.0)
        planner.add_resource("resource_1", "Resource 1", capacity=100.0)
        
        # Create plan
        plan = planner.create_quantum_plan(strategy=PlanningStrategy.TEMPORAL_OPTIMIZATION)
        return planner, validator, plan
    
    def test_validator_initialization(self, planner_with_validator):
        """Test validator initialization."""
        planner, validator = planner_with_validator
        
        assert validator.planner is planner
        assert validator.min_quantum_coherence == 0.3
        assert validator.min_success_probability == 0.5
        assert validator.qa_system is not None
        assert validator.validation_circuit_breaker is not None
    
    def test_comprehensive_plan_validation(self, valid_plan):
        """Test comprehensive plan validation."""
        planner, validator, plan = valid_plan
        
        report = validator.validate_plan_comprehensive(plan.id)
        
        assert isinstance(report, PlanValidationReport)
        assert report.plan_id == plan.id
        assert report.validation_timestamp is not None
        assert 0.0 <= report.overall_score <= 1.0
        assert isinstance(report.issues, list)
        assert isinstance(report.metrics, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.is_valid, bool)
    
    def test_quantum_state_validation(self, valid_plan):
        """Test quantum state validation."""
        planner, validator, plan = valid_plan
        
        # Test low coherence plan
        plan.quantum_coherence = 0.1  # Below threshold
        
        report = validator.validate_plan_comprehensive(plan.id)
        
        # Should have coherence-related issues
        coherence_issues = [issue for issue in report.issues 
                          if issue.category == "quantum_coherence"]
        assert len(coherence_issues) > 0
        
        # Should have lower overall score
        assert report.overall_score < 0.8
    
    def test_temporal_consistency_validation(self, valid_plan):
        """Test temporal consistency validation."""
        planner, validator, plan = valid_plan
        
        # Test plan with temporal issues
        plan.total_duration = timedelta(days=10)  # Exceeds recommended maximum
        
        report = validator.validate_plan_comprehensive(plan.id)
        
        # Should have temporal issues
        temporal_issues = [issue for issue in report.issues 
                         if issue.category == "plan_duration"]
        assert len(temporal_issues) > 0 or report.temporal_consistency_score < 1.0
    
    def test_resource_feasibility_validation(self, valid_plan):
        """Test resource feasibility validation."""
        planner, validator, plan = valid_plan
        
        # Create resource overallocation
        for task_id in plan.tasks:
            plan.resource_allocation[task_id] = {"resource_1": 150.0}  # Exceeds capacity
        
        report = validator.validate_plan_comprehensive(plan.id)
        
        # Should have resource issues
        resource_issues = [issue for issue in report.issues 
                         if issue.category == "resource_overallocation"]
        assert len(resource_issues) > 0
        assert report.resource_feasibility_score < 1.0
    
    def test_validation_analytics(self, planner_with_validator):
        """Test validation analytics."""
        planner, validator = planner_with_validator
        
        # Create and validate multiple plans
        for i in range(5):
            planner.add_task(f"task_{i}", f"Task {i}", f"Description {i}")
            plan = planner.create_quantum_plan(strategy=PlanningStrategy.QUANTUM_SUPERPOSITION)
            validator.validate_plan_comprehensive(plan.id)
        
        analytics = validator.get_validation_analytics()
        
        assert 'total_validations' in analytics
        assert analytics['total_validations'] >= 5
        assert 'recent_validations' in analytics
        assert 'average_score' in analytics
        assert 'validation_success_rate' in analytics
        assert 'common_issue_categories' in analytics


class TestQuantumSecurity:
    """Test suite for quantum task planning security."""
    
    @pytest.fixture
    def planner_with_security(self):
        """Create planner with security manager for testing."""
        planner = QuantumTaskPlanner(dim=1000, device="cpu")
        security_manager = QuantumSecurityManager(planner)
        return planner, security_manager
    
    @pytest.fixture
    def authenticated_context(self, planner_with_security):
        """Create authenticated security context for testing."""
        planner, security_manager = planner_with_security
        
        context = security_manager.authenticate_user(
            username="test_user",
            password="test_password",
            ip_address="127.0.0.1"
        )
        
        return planner, security_manager, context
    
    def test_security_manager_initialization(self, planner_with_security):
        """Test security manager initialization."""
        planner, security_manager = planner_with_security
        
        assert security_manager.planner is planner
        assert security_manager.policy is not None
        assert security_manager.audit_logger is not None
        assert security_manager.input_sanitizer is not None
        assert len(security_manager.active_sessions) == 0
    
    def test_user_authentication(self, planner_with_security):
        """Test user authentication functionality."""
        planner, security_manager = planner_with_security
        
        # Test successful authentication
        context = security_manager.authenticate_user(
            username="test_user",
            password="valid_password",
            ip_address="127.0.0.1"
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        assert context.session_token is not None
        assert context.ip_address == "127.0.0.1"
        assert context.session_token in security_manager.active_sessions
        
        # Test failed authentication
        failed_context = security_manager.authenticate_user(
            username="invalid_user",
            password="",  # Empty password should fail
            ip_address="127.0.0.1"
        )
        
        assert failed_context is None
    
    def test_session_validation(self, authenticated_context):
        """Test session validation and timeout."""
        planner, security_manager, context = authenticated_context
        
        # Test valid session
        validated_context = security_manager.validate_session(context.session_token)
        assert validated_context is not None
        assert validated_context.user_id == context.user_id
        
        # Test invalid session token
        invalid_validation = security_manager.validate_session("invalid_token")
        assert invalid_validation is None
        
        # Test session invalidation
        security_manager.invalidate_session(context.session_token)
        invalidated_validation = security_manager.validate_session(context.session_token)
        assert invalidated_validation is None
    
    def test_secure_task_creation(self, authenticated_context):
        """Test secure task creation with access control."""
        planner, security_manager, context = authenticated_context
        
        # Grant necessary permissions
        security_manager.user_permissions[context.user_id] = {AccessRight.WRITE}
        
        # Test successful task creation
        success = security_manager.secure_task_creation(
            context=context,
            task_id="secure_task_1",
            name="Secure Task",
            description="A securely created task",
            priority=1.0
        )
        
        assert success is True
        assert "secure_task_1" in planner.tasks
        
        # Test task creation without permissions
        security_manager.user_permissions[context.user_id] = set()  # Remove permissions
        
        failure = security_manager.secure_task_creation(
            context=context,
            task_id="unauthorized_task",
            name="Unauthorized Task",
            description="This should fail",
            priority=1.0
        )
        
        assert failure is False
        assert "unauthorized_task" not in planner.tasks
    
    def test_secure_plan_creation(self, authenticated_context):
        """Test secure plan creation with risk assessment."""
        planner, security_manager, context = authenticated_context
        
        # Grant necessary permissions
        security_manager.user_permissions[context.user_id] = {AccessRight.EXECUTE}
        
        # Add some tasks first
        planner.add_task("task_1", "Task 1", "Description 1")
        
        # Test successful plan creation
        plan_id = security_manager.secure_plan_creation(
            context=context,
            strategy="temporal_optimization",
            objectives=["minimize_duration"],
            constraints={}
        )
        
        assert plan_id is not None
        assert plan_id in planner.plans
    
    def test_security_risk_assessment(self, planner_with_security):
        """Test security risk assessment for tasks and plans."""
        planner, security_manager = planner_with_security
        
        # Test low-risk task
        low_risk_score = security_manager._assess_task_security_risk(
            name="simple_task",
            description="A simple data processing task",
            kwargs={"priority": 1.0}
        )
        assert 0.0 <= low_risk_score <= 0.5
        
        # Test high-risk task
        high_risk_score = security_manager._assess_task_security_risk(
            name="admin_delete_database",
            description="Delete all security logs and admin passwords",
            kwargs={
                "priority": 10.0,
                "estimated_duration": timedelta(days=2),
                "resources_required": {"database": 100.0}
            }
        )
        assert high_risk_score > 0.5
    
    def test_security_analytics(self, authenticated_context):
        """Test security analytics and monitoring."""
        planner, security_manager, context = authenticated_context
        
        # Perform some secured operations
        security_manager.user_permissions[context.user_id] = {AccessRight.WRITE, AccessRight.EXECUTE}
        
        for i in range(3):
            security_manager.secure_task_creation(
                context=context,
                task_id=f"analytics_task_{i}",
                name=f"Analytics Task {i}",
                description="Task for analytics testing",
                priority=1.0
            )
        
        analytics = security_manager.get_security_analytics()
        
        assert 'total_security_events' in analytics
        assert 'recent_events' in analytics
        assert 'access_granted_rate' in analytics
        assert 'average_risk_score' in analytics
        assert 'operations_by_type' in analytics
    
    def test_security_report_export(self, authenticated_context):
        """Test comprehensive security report export."""
        planner, security_manager, context = authenticated_context
        
        # Generate some security events
        security_manager.user_permissions[context.user_id] = {AccessRight.WRITE}
        
        security_manager.secure_task_creation(
            context=context,
            task_id="report_task",
            name="Report Task",
            description="Task for report testing",
            priority=1.0
        )
        
        report = security_manager.export_security_report(time_range_hours=1)
        
        assert 'report_timestamp' in report
        assert 'time_range_hours' in report
        assert 'summary' in report
        assert 'security_incidents' in report
        assert 'recommendations' in report
        
        # Verify report structure
        summary = report['summary']
        assert 'total_events' in summary
        assert 'successful_operations' in summary
        assert 'failed_operations' in summary


class TestDistributedPlanning:
    """Test suite for distributed quantum task planning."""
    
    @pytest.fixture
    def cluster_config(self):
        """Create cluster configuration for testing."""
        return ClusterConfiguration(
            min_nodes=2,
            max_nodes=5,
            auto_scaling_enabled=True,
            heartbeat_interval=10,
            health_check_timeout=5
        )
    
    @pytest.fixture
    def distributed_planner(self, cluster_config):
        """Create distributed planner for testing."""
        return QuantumDistributedTaskPlanner(
            cluster_config=cluster_config,
            node_id="test_coordinator",
            enable_auto_scaling=False  # Disable for tests
        )
    
    def test_distributed_planner_initialization(self, distributed_planner):
        """Test distributed planner initialization."""
        assert distributed_planner.node_id == "test_coordinator"
        assert distributed_planner.node_role == NodeRole.COORDINATOR
        assert distributed_planner.local_planner is not None
        assert distributed_planner.cluster_config is not None
        assert len(distributed_planner.cluster_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_cluster_node_lifecycle(self, distributed_planner):
        """Test cluster node start/stop lifecycle."""
        # Start node
        start_task = asyncio.create_task(
            distributed_planner.start_cluster_node(port=18080)
        )
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        # Check if started properly
        assert distributed_planner.session is not None
        assert distributed_planner.heartbeat_task is not None
        
        # Stop node
        await distributed_planner.stop_cluster_node()
        
        # Cancel the start task to clean up
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    def test_cluster_analytics(self, distributed_planner):
        """Test cluster analytics functionality."""
        # Add some mock nodes
        from hd_compute.distributed.quantum_task_distribution import DistributedNode, NodeStatus
        
        test_node = DistributedNode(
            node_id="test_node_1",
            endpoint="http://localhost:8081",
            role=NodeRole.PLANNER,
            status=NodeStatus.HEALTHY,
            capabilities={NodeRole.PLANNER},
            max_concurrent_tasks=10
        )
        
        distributed_planner.cluster_nodes["test_node_1"] = test_node
        
        analytics = distributed_planner.get_cluster_analytics()
        
        assert 'cluster_overview' in analytics
        assert 'performance_metrics' in analytics
        assert 'resource_utilization' in analytics
        assert 'load_balancing_stats' in analytics
        
        cluster_overview = analytics['cluster_overview']
        assert cluster_overview['total_nodes'] == 1
        assert cluster_overview['healthy_nodes'] == 1


class TestPerformanceOptimization:
    """Test suite for performance optimization."""
    
    @pytest.fixture
    def planner_with_optimizer(self):
        """Create planner with performance optimizer for testing."""
        planner = QuantumTaskPlanner(dim=1000, device="cpu")
        optimizer = QuantumPerformanceOptimizer(
            planner=planner,
            enable_gpu=False,  # Disable GPU for tests
            cache_size_mb=64
        )
        return planner, optimizer
    
    def test_optimizer_initialization(self, planner_with_optimizer):
        """Test performance optimizer initialization."""
        planner, optimizer = planner_with_optimizer
        
        assert optimizer.planner is planner
        assert optimizer.enable_gpu is False
        assert optimizer.performance_profile is not None
        assert optimizer.memory_manager is not None
        assert optimizer.quantum_cache is not None
    
    def test_performance_profiling(self, planner_with_optimizer):
        """Test system performance profiling."""
        planner, optimizer = planner_with_optimizer
        
        profile = optimizer.performance_profile
        
        assert profile.cpu_cores > 0
        assert profile.memory_gb > 0
        assert profile.gpu_available == optimizer.enable_gpu
        assert isinstance(profile.workload_characteristics, dict)
        assert 'quantum_operation_ratio' in profile.workload_characteristics
    
    def test_baseline_performance_measurement(self, planner_with_optimizer):
        """Test baseline performance measurement."""
        planner, optimizer = planner_with_optimizer
        
        metrics = optimizer._measure_baseline_performance()
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'avg_execution_time' in metrics
        assert 'quantum_coherence' in metrics
        assert 'throughput' in metrics
        
        # Verify reasonable values
        assert 0.0 <= metrics['cpu_usage'] <= 100.0
        assert 0.0 <= metrics['memory_usage'] <= 100.0
        assert metrics['avg_execution_time'] > 0.0
        assert 0.0 <= metrics['quantum_coherence'] <= 1.0
        assert metrics['throughput'] > 0.0
    
    def test_memory_optimization(self, planner_with_optimizer):
        """Test memory-focused optimization."""
        planner, optimizer = planner_with_optimizer
        
        result = optimizer.optimize_planning_performance(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZED,
            objectives=['minimize_memory_usage'],
            target_metrics={'memory_usage': 50.0}
        )
        
        assert isinstance(result, optimizer.__class__.__module__.split('.')[-1] == 'quantum_optimization' and hasattr(optimizer, 'OptimizationResult'))
        assert result.strategy == OptimizationStrategy.MEMORY_OPTIMIZED
        assert result.performance_improvement is not None
        assert isinstance(result.recommendations, list)
    
    def test_cache_optimization(self, planner_with_optimizer):
        """Test cache-focused optimization."""
        planner, optimizer = planner_with_optimizer
        
        result = optimizer.optimize_planning_performance(
            strategy=OptimizationStrategy.CACHE_OPTIMIZED,
            objectives=['maximize_cache_hits'],
            target_metrics={'cache_hit_rate': 0.8}
        )
        
        assert result.strategy == OptimizationStrategy.CACHE_OPTIMIZED
        assert len(result.recommendations) > 0
        
        # Test cache functionality
        cache = optimizer.quantum_cache
        
        # Test cache operations
        cache.put("test_key", "test_value", coherence_score=0.8)
        retrieved = cache.get("test_key")
        assert retrieved == "test_value"
        
        # Test low coherence rejection
        cache.put("low_coherence_key", "low_value", coherence_score=0.3)
        low_retrieved = cache.get("low_coherence_key")
        assert low_retrieved is None  # Should be rejected due to low coherence
    
    def test_optimization_analytics(self, planner_with_optimizer):
        """Test optimization analytics."""
        planner, optimizer = planner_with_optimizer
        
        # Perform several optimizations
        strategies = [
            OptimizationStrategy.CPU_INTENSIVE,
            OptimizationStrategy.MEMORY_OPTIMIZED,
            OptimizationStrategy.CACHE_OPTIMIZED
        ]
        
        for strategy in strategies:
            optimizer.optimize_planning_performance(
                strategy=strategy,
                objectives=['minimize_duration']
            )
        
        analytics = optimizer.get_optimization_analytics()
        
        assert 'optimization_summary' in analytics
        assert 'strategy_effectiveness' in analytics
        assert 'system_profile' in analytics
        assert 'current_resource_utilization' in analytics
        
        summary = analytics['optimization_summary']
        assert summary['total_optimizations'] >= 3
        assert 'average_performance_improvement' in summary


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    @pytest.fixture
    def full_system(self):
        """Create full system with all components for integration testing."""
        # Create planner
        planner = QuantumTaskPlanner(dim=1000, device="cpu", enable_distributed=False)
        
        # Create validator
        validator = TaskPlanningValidator(planner)
        
        # Create security manager
        security_manager = QuantumSecurityManager(planner)
        
        # Create performance optimizer
        optimizer = QuantumPerformanceOptimizer(
            planner=planner,
            enable_gpu=False,
            cache_size_mb=64
        )
        
        return planner, validator, security_manager, optimizer
    
    def test_secure_validated_planning_workflow(self, full_system):
        """Test complete secure and validated planning workflow."""
        planner, validator, security_manager, optimizer = full_system
        
        # Step 1: Authentication
        context = security_manager.authenticate_user(
            username="integration_user",
            password="secure_password",
            ip_address="127.0.0.1"
        )
        assert context is not None
        
        # Step 2: Grant permissions
        security_manager.user_permissions[context.user_id] = {
            AccessRight.READ, AccessRight.WRITE, AccessRight.EXECUTE
        }
        
        # Step 3: Secure task creation
        task_ids = []
        for i in range(3):
            success = security_manager.secure_task_creation(
                context=context,
                task_id=f"integration_task_{i}",
                name=f"Integration Task {i}",
                description=f"Integration testing task {i}",
                priority=float(i + 1),
                estimated_duration=timedelta(hours=i + 1)
            )
            assert success is True
            task_ids.append(f"integration_task_{i}")
        
        # Step 4: Add resources
        planner.add_resource("integration_resource", "Integration Resource", capacity=200.0)
        
        # Step 5: Optimize performance before planning
        optimizer.optimize_planning_performance(
            strategy=OptimizationStrategy.AUTO_ADAPTIVE,
            objectives=['minimize_duration', 'maximize_success']
        )
        
        # Step 6: Secure plan creation
        plan_id = security_manager.secure_plan_creation(
            context=context,
            strategy="hybrid_quantum",
            objectives=['minimize_duration', 'maximize_success'],
            constraints={'max_parallel_tasks': 2}
        )
        assert plan_id is not None
        
        # Step 7: Validate the plan
        validation_report = validator.validate_plan_comprehensive(plan_id)
        assert validation_report.is_valid or validation_report.overall_score > 0.5
        
        # Step 8: Execute if valid
        if validation_report.is_valid:
            success = security_manager.secure_plan_execution(context, plan_id)
            assert success is True
    
    def test_multi_strategy_performance_comparison(self, full_system):
        """Test performance comparison across multiple strategies."""
        planner, validator, security_manager, optimizer = full_system
        
        # Add test tasks and resources
        for i in range(5):
            planner.add_task(
                task_id=f"perf_task_{i}",
                name=f"Performance Task {i}",
                description=f"Task for performance testing {i}",
                priority=float(i + 1)
            )
        
        planner.add_resource("perf_resource", "Performance Resource", capacity=500.0)
        
        # Test different strategies
        strategies = [
            PlanningStrategy.QUANTUM_SUPERPOSITION,
            PlanningStrategy.TEMPORAL_OPTIMIZATION,
            PlanningStrategy.CAUSAL_REASONING,
            PlanningStrategy.HYBRID_QUANTUM
        ]
        
        results = []
        for strategy in strategies:
            plan = planner.create_quantum_plan(
                strategy=strategy,
                optimization_objectives=['minimize_duration', 'maximize_success']
            )
            
            # Validate plan
            validation_report = validator.validate_plan_comprehensive(plan.id)
            
            results.append({
                'strategy': strategy.value,
                'plan_id': plan.id,
                'success_probability': plan.success_probability,
                'quantum_coherence': plan.quantum_coherence,
                'total_duration': plan.total_duration.total_seconds(),
                'validation_score': validation_report.overall_score,
                'is_valid': validation_report.is_valid
            })
        
        # Verify all strategies produced results
        assert len(results) == len(strategies)
        
        # Verify reasonable performance across strategies
        success_probs = [r['success_probability'] for r in results]
        coherence_values = [r['quantum_coherence'] for r in results]
        
        assert all(p > 0.0 for p in success_probs)
        assert all(c > 0.0 for c in coherence_values)
        assert np.mean(success_probs) > 0.5  # Average success probability > 50%
        assert np.mean(coherence_values) > 0.3  # Average coherence > 30%
    
    def test_error_recovery_and_fallback(self, full_system):
        """Test error recovery and fallback mechanisms."""
        planner, validator, security_manager, optimizer = full_system
        
        # Test invalid plan creation scenario
        with pytest.raises(Exception):
            # This should trigger error recovery
            planner.create_quantum_plan(
                strategy=PlanningStrategy.QUANTUM_SUPERPOSITION,
                optimization_objectives=[],  # Empty objectives
                constraints={'invalid_constraint': 'invalid_value'}
            )
        
        # Test validator circuit breaker
        # Simulate multiple validation failures to trigger circuit breaker
        for _ in range(6):  # More than failure threshold
            try:
                validator.validate_plan_comprehensive("non_existent_plan")
            except Exception:
                pass  # Expected failures
        
        # Circuit breaker should now be open
        assert validator.validation_circuit_breaker.state in ['open', 'half-open']
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, full_system):
        """Test concurrent operations handling."""
        planner, validator, security_manager, optimizer = full_system
        
        # Add tasks for concurrent testing
        for i in range(10):
            planner.add_task(
                task_id=f"concurrent_task_{i}",
                name=f"Concurrent Task {i}",
                description=f"Task for concurrent testing {i}",
                priority=1.0
            )
        
        planner.add_resource("concurrent_resource", "Concurrent Resource", capacity=1000.0)
        
        # Create multiple plans concurrently
        async def create_plan_async(strategy):
            return planner.create_quantum_plan(
                strategy=strategy,
                optimization_objectives=['minimize_duration']
            )
        
        strategies = [
            PlanningStrategy.QUANTUM_SUPERPOSITION,
            PlanningStrategy.TEMPORAL_OPTIMIZATION,
            PlanningStrategy.CAUSAL_REASONING
        ]
        
        # Run concurrent plan creation
        tasks = [create_plan_async(strategy) for strategy in strategies]
        plans = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify successful concurrent execution
        successful_plans = [p for p in plans if not isinstance(p, Exception)]
        assert len(successful_plans) >= 2  # At least 2 out of 3 should succeed
        
        # Verify plan uniqueness
        plan_ids = [p.id for p in successful_plans]
        assert len(set(plan_ids)) == len(plan_ids)  # All IDs should be unique


class TestStatisticalValidation:
    """Statistical validation and reproducibility tests."""
    
    @pytest.fixture
    def statistical_planner(self):
        """Create planner for statistical testing."""
        return QuantumTaskPlanner(
            dim=2000,
            device="cpu",
            max_superposition_states=50
        )
    
    def test_planning_reproducibility(self, statistical_planner):
        """Test planning reproducibility with fixed random seeds."""
        planner = statistical_planner
        
        # Add consistent tasks
        for i in range(5):
            planner.add_task(
                task_id=f"repro_task_{i}",
                name=f"Reproducible Task {i}",
                description="Task for reproducibility testing",
                priority=1.0
            )
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create multiple plans with same configuration
        plans = []
        for _ in range(5):
            plan = planner.create_quantum_plan(
                strategy=PlanningStrategy.QUANTUM_SUPERPOSITION,
                optimization_objectives=['minimize_duration']
            )
            plans.append(plan)
        
        # Verify some level of consistency (not exact due to quantum uncertainty)
        success_probs = [p.success_probability for p in plans]
        coherence_values = [p.quantum_coherence for p in plans]
        
        # Statistical consistency checks
        success_std = np.std(success_probs)
        coherence_std = np.std(coherence_values)
        
        assert success_std < 0.3  # Reasonable variation
        assert coherence_std < 0.3  # Reasonable variation
    
    def test_performance_statistical_significance(self, statistical_planner):
        """Test statistical significance of performance improvements."""
        planner = statistical_planner
        
        # Add tasks for performance testing
        for i in range(10):
            planner.add_task(
                task_id=f"stat_task_{i}",
                name=f"Statistical Task {i}",
                description="Task for statistical testing",
                priority=1.0
            )
        
        # Compare different strategies statistically
        strategy_a_times = []
        strategy_b_times = []
        
        for _ in range(10):
            # Strategy A: Quantum Superposition
            start_time = time.perf_counter()
            plan_a = planner.create_quantum_plan(
                strategy=PlanningStrategy.QUANTUM_SUPERPOSITION,
                optimization_objectives=['minimize_duration']
            )
            strategy_a_times.append(time.perf_counter() - start_time)
            
            # Strategy B: Temporal Optimization
            start_time = time.perf_counter()
            plan_b = planner.create_quantum_plan(
                strategy=PlanningStrategy.TEMPORAL_OPTIMIZATION,
                optimization_objectives=['minimize_duration']
            )
            strategy_b_times.append(time.perf_counter() - start_time)
        
        # Statistical analysis
        from scipy import stats
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(strategy_a_times, strategy_b_times)
        
        # Log results for analysis
        logging.info(f"Strategy A mean time: {np.mean(strategy_a_times):.4f}s")
        logging.info(f"Strategy B mean time: {np.mean(strategy_b_times):.4f}s")
        logging.info(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
        
        # Verify statistical validity
        assert all(t > 0 for t in strategy_a_times)  # All times positive
        assert all(t > 0 for t in strategy_b_times)  # All times positive
        assert np.std(strategy_a_times) < np.mean(strategy_a_times)  # Reasonable variance
        assert np.std(strategy_b_times) < np.mean(strategy_b_times)  # Reasonable variance
    
    def test_quantum_coherence_distribution(self, statistical_planner):
        """Test quantum coherence statistical distribution."""
        planner = statistical_planner
        
        # Add tasks
        for i in range(3):
            planner.add_task(
                task_id=f"coherence_task_{i}",
                name=f"Coherence Task {i}",
                description="Task for coherence testing",
                priority=1.0
            )
        
        # Collect coherence values
        coherence_values = []
        for _ in range(50):
            plan = planner.create_quantum_plan(
                strategy=PlanningStrategy.HYBRID_QUANTUM,
                optimization_objectives=['maximize_success']
            )
            coherence_values.append(plan.quantum_coherence)
        
        # Statistical analysis of coherence distribution
        coherence_array = np.array(coherence_values)
        
        # Descriptive statistics
        mean_coherence = np.mean(coherence_array)
        std_coherence = np.std(coherence_array)
        min_coherence = np.min(coherence_array)
        max_coherence = np.max(coherence_array)
        
        # Verify reasonable distribution
        assert 0.0 <= min_coherence <= 1.0
        assert 0.0 <= max_coherence <= 1.0
        assert min_coherence < max_coherence
        assert 0.2 <= mean_coherence <= 1.0  # Reasonable mean
        assert std_coherence < 0.5  # Not too much variation
        
        # Test for normality (Shapiro-Wilk test)
        from scipy.stats import shapiro
        stat, p_value = shapiro(coherence_array)
        
        logging.info(f"Coherence distribution - Mean: {mean_coherence:.3f}, Std: {std_coherence:.3f}")
        logging.info(f"Normality test - Statistic: {stat:.3f}, P-value: {p_value:.3f}")
        
        # Log distribution characteristics
        assert len(coherence_values) == 50  # All values collected


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration for the entire test session."""
    return {
        'test_dimension': 1000,
        'test_device': 'cpu',
        'enable_gpu_tests': False,  # Set to True if GPU testing needed
        'test_timeout': 30.0,
        'statistical_samples': 20
    }


@pytest.mark.parametrize("strategy", [
    PlanningStrategy.QUANTUM_SUPERPOSITION,
    PlanningStrategy.TEMPORAL_OPTIMIZATION,
    PlanningStrategy.CAUSAL_REASONING,
    PlanningStrategy.ATTENTION_GUIDED,
    PlanningStrategy.HYBRID_QUANTUM
])
def test_all_planning_strategies(strategy):
    """Parametrized test for all planning strategies."""
    planner = QuantumTaskPlanner(dim=500, device="cpu")
    
    # Add minimal tasks
    planner.add_task("param_task_1", "Task 1", "Description 1")
    planner.add_task("param_task_2", "Task 2", "Description 2", dependencies={"param_task_1"})
    
    # Test strategy
    plan = planner.create_quantum_plan(
        strategy=strategy,
        optimization_objectives=['minimize_duration']
    )
    
    assert plan is not None
    assert plan.success_probability > 0.0
    assert plan.quantum_coherence > 0.0
    assert len(plan.tasks) > 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "--verbose",
        "--tb=short",
        "--cov=hd_compute.applications.task_planning",
        "--cov=hd_compute.validation.task_planning_validation", 
        "--cov=hd_compute.security.task_planning_security",
        "--cov=hd_compute.distributed.quantum_task_distribution",
        "--cov=hd_compute.performance.quantum_optimization",
        "--cov-report=html",
        "--cov-report=term-missing",
        __file__
    ])