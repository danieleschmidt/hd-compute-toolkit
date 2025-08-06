"""Comprehensive validation framework for quantum-inspired task planning.

This module provides rigorous validation, error handling, and quality assurance
for the task planning system, ensuring research-grade reliability and robustness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from collections import defaultdict
import traceback
from enum import Enum

from ..applications.task_planning import QuantumTaskPlanner, Task, Resource, ExecutionPlan, TaskStatus, PlanningStrategy
from ..validation.error_recovery import CircuitBreaker, RetryStrategy, ErrorRecovery
from ..validation.quality_assurance import QualityAssurance, QualityMetrics


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    task_id: Optional[str] = None
    plan_id: Optional[str] = None
    suggestion: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class PlanValidationReport:
    """Comprehensive plan validation report."""
    plan_id: str
    validation_timestamp: datetime
    overall_score: float
    issues: List[ValidationIssue]
    metrics: Dict[str, float]
    recommendations: List[str]
    is_valid: bool
    quantum_coherence_analysis: Dict[str, float]
    resource_feasibility_score: float
    temporal_consistency_score: float


class TaskPlanningValidator:
    """Comprehensive validator for task planning system with quantum-aware validation."""
    
    def __init__(self, planner: QuantumTaskPlanner):
        """Initialize the validator.
        
        Args:
            planner: The quantum task planner to validate
        """
        self.planner = planner
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.min_quantum_coherence = 0.3
        self.min_success_probability = 0.5
        self.max_plan_duration_hours = 168  # 1 week
        self.max_resource_overallocation = 1.2  # 20% over capacity
        
        # Quality assurance system
        self.qa_system = QualityAssurance(
            metrics_config={
                'coherence_stability': {'min': 0.3, 'target': 0.8},
                'resource_efficiency': {'min': 0.5, 'target': 0.9},
                'temporal_consistency': {'min': 0.7, 'target': 0.95},
                'success_probability': {'min': 0.5, 'target': 0.85}
            }
        )
        
        # Circuit breakers for error handling
        self.validation_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300,
            expected_exception=Exception
        )
        
        self.planning_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=180,
            expected_exception=(ValueError, RuntimeError)
        )
        
        # Error recovery system
        self.error_recovery = ErrorRecovery()
        
        # Validation history
        self.validation_history: List[PlanValidationReport] = []
        
    def validate_plan_comprehensive(self, plan_id: str) -> PlanValidationReport:
        """Perform comprehensive validation of an execution plan.
        
        Args:
            plan_id: ID of the plan to validate
            
        Returns:
            Comprehensive validation report
        """
        try:
            return self._execute_with_circuit_breaker(
                self.validation_circuit_breaker,
                self._validate_plan_internal,
                plan_id
            )
        except Exception as e:
            self.logger.error(f"Critical validation failure for plan {plan_id}: {str(e)}")
            return self._create_failure_report(plan_id, str(e))
    
    def _validate_plan_internal(self, plan_id: str) -> PlanValidationReport:
        """Internal comprehensive plan validation."""
        if plan_id not in self.planner.plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.planner.plans[plan_id]
        issues = []
        metrics = {}
        
        self.logger.info(f"Starting comprehensive validation for plan {plan_id}")
        
        # 1. Quantum State Validation
        quantum_metrics = self._validate_quantum_state(plan, issues)
        metrics.update(quantum_metrics)
        
        # 2. Temporal Consistency Validation
        temporal_score = self._validate_temporal_consistency(plan, issues)
        metrics['temporal_consistency'] = temporal_score
        
        # 3. Resource Feasibility Validation
        resource_score = self._validate_resource_feasibility(plan, issues)
        metrics['resource_feasibility'] = resource_score
        
        # 4. Dependency Validation
        dependency_score = self._validate_dependencies(plan, issues)
        metrics['dependency_consistency'] = dependency_score
        
        # 5. Logical Consistency Validation
        logical_score = self._validate_logical_consistency(plan, issues)
        metrics['logical_consistency'] = logical_score
        
        # 6. Performance Validation
        performance_metrics = self._validate_performance_characteristics(plan, issues)
        metrics.update(performance_metrics)
        
        # 7. Security and Safety Validation
        security_score = self._validate_security_aspects(plan, issues)
        metrics['security_score'] = security_score
        
        # Calculate overall validation score
        overall_score = self._calculate_overall_score(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(plan, issues, metrics)
        
        # Determine if plan is valid
        is_valid = self._determine_plan_validity(issues, metrics)
        
        # Create validation report
        report = PlanValidationReport(
            plan_id=plan_id,
            validation_timestamp=datetime.now(),
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            is_valid=is_valid,
            quantum_coherence_analysis=quantum_metrics,
            resource_feasibility_score=resource_score,
            temporal_consistency_score=temporal_score
        )
        
        # Store in history
        self.validation_history.append(report)
        
        # Update QA metrics
        self.qa_system.update_metrics({
            'coherence_stability': quantum_metrics.get('coherence_stability', 0.0),
            'resource_efficiency': resource_score,
            'temporal_consistency': temporal_score,
            'success_probability': plan.success_probability
        })
        
        self.logger.info(f"Validation completed for plan {plan_id}. Score: {overall_score:.3f}, Valid: {is_valid}")
        
        return report
    
    def _validate_quantum_state(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> Dict[str, float]:
        """Validate quantum aspects of the plan."""
        metrics = {}
        
        # Check quantum coherence
        coherence = plan.quantum_coherence
        metrics['quantum_coherence'] = coherence
        
        if coherence < self.min_quantum_coherence:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="quantum_coherence",
                message=f"Quantum coherence {coherence:.3f} below minimum threshold {self.min_quantum_coherence}",
                plan_id=plan.id,
                suggestion="Consider regenerating plan with higher initial coherence or apply coherence restoration"
            ))
        elif coherence < 0.6:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="quantum_coherence",
                message=f"Quantum coherence {coherence:.3f} is low, may affect plan stability",
                plan_id=plan.id,
                suggestion="Monitor coherence during execution and prepare for adaptive replanning"
            ))
        
        # Validate success probability consistency
        success_prob = plan.success_probability
        metrics['success_probability'] = success_prob
        
        if success_prob < self.min_success_probability:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="success_probability",
                message=f"Success probability {success_prob:.3f} below minimum threshold {self.min_success_probability}",
                plan_id=plan.id,
                suggestion="Revise plan to improve task success rates or reduce risk factors"
            ))
        
        # Check for quantum decoherence risk
        coherence_decay_risk = self._estimate_coherence_decay_risk(plan)
        metrics['coherence_decay_risk'] = coherence_decay_risk
        
        if coherence_decay_risk > 0.7:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="coherence_decay",
                message=f"High risk of coherence decay during execution ({coherence_decay_risk:.3f})",
                plan_id=plan.id,
                suggestion="Implement coherence monitoring and restoration mechanisms"
            ))
        
        # Validate hypervector plan representation
        if plan.plan_hypervector is None:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="plan_representation",
                message="Plan hypervector is missing",
                plan_id=plan.id,
                suggestion="Regenerate plan with proper hypervector encoding"
            ))
        else:
            # Check hypervector properties
            hv_quality = self._assess_hypervector_quality(plan.plan_hypervector)
            metrics['hypervector_quality'] = hv_quality
            
            if hv_quality < 0.5:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="hypervector_quality",
                    message=f"Plan hypervector quality is low ({hv_quality:.3f})",
                    plan_id=plan.id,
                    suggestion="Consider re-encoding plan with improved bundling strategy"
                ))
        
        # Coherence stability analysis
        coherence_stability = self._analyze_coherence_stability(plan)
        metrics['coherence_stability'] = coherence_stability
        
        return metrics
    
    def _validate_temporal_consistency(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> float:
        """Validate temporal aspects of the plan."""
        consistency_score = 1.0
        
        # Check total duration reasonableness
        total_hours = plan.total_duration.total_seconds() / 3600
        if total_hours > self.max_plan_duration_hours:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="plan_duration",
                message=f"Plan duration {total_hours:.1f}h exceeds recommended maximum {self.max_plan_duration_hours}h",
                plan_id=plan.id,
                suggestion="Consider breaking plan into smaller sub-plans"
            ))
            consistency_score *= 0.8
        
        # Validate task scheduling consistency
        schedule_issues = 0
        for i, task_id in enumerate(plan.tasks):
            if task_id not in plan.schedule:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="scheduling",
                    message=f"Task {task_id} missing from schedule",
                    plan_id=plan.id,
                    task_id=task_id,
                    suggestion="Ensure all tasks have scheduled start times"
                ))
                schedule_issues += 1
                continue
            
            task_start = plan.schedule[task_id]
            
            # Check dependency timing
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in plan.schedule and dep_id in self.planner.tasks:
                        dep_task = self.planner.tasks[dep_id]
                        dep_end = plan.schedule[dep_id] + dep_task.estimated_duration
                        
                        if task_start < dep_end:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="dependency_timing",
                                message=f"Task {task_id} scheduled before dependency {dep_id} completion",
                                plan_id=plan.id,
                                task_id=task_id,
                                suggestion="Adjust scheduling to respect dependency completion times"
                            ))
                            schedule_issues += 1
        
        if schedule_issues > 0:
            consistency_score *= max(0.2, 1.0 - (schedule_issues * 0.1))
        
        # Check for deadline violations
        deadline_violations = 0
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                if task.deadline and task_id in plan.schedule:
                    task_end = plan.schedule[task_id] + task.estimated_duration
                    if task_end > task.deadline:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="deadline_violation",
                            message=f"Task {task_id} scheduled after deadline",
                            plan_id=plan.id,
                            task_id=task_id,
                            suggestion="Reschedule task or negotiate deadline extension"
                        ))
                        deadline_violations += 1
        
        if deadline_violations > 0:
            consistency_score *= max(0.3, 1.0 - (deadline_violations * 0.15))
        
        return consistency_score
    
    def _validate_resource_feasibility(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> float:
        """Validate resource allocation feasibility."""
        feasibility_score = 1.0
        
        # Calculate resource usage over time
        resource_timeline = self._build_resource_timeline(plan)
        
        # Check for resource overallocation
        overallocation_issues = 0
        for resource_id in self.planner.resources:
            resource = self.planner.resources[resource_id]
            max_usage = max(resource_timeline.get(resource_id, {}).values()) if resource_timeline.get(resource_id) else 0
            
            if max_usage > resource.capacity * self.max_resource_overallocation:
                severity = ValidationSeverity.ERROR if max_usage > resource.capacity * 1.5 else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    category="resource_overallocation",
                    message=f"Resource {resource_id} overallocated: {max_usage:.2f} > {resource.capacity:.2f}",
                    plan_id=plan.id,
                    suggestion="Redistribute resource allocation or extend timeline"
                ))
                overallocation_issues += 1
                
                # Penalize feasibility score
                overallocation_factor = max_usage / resource.capacity
                feasibility_score *= max(0.1, 1.0 / overallocation_factor)
        
        # Check for missing resource allocations
        missing_allocations = 0
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                if task_id not in plan.resource_allocation:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="missing_allocation",
                        message=f"Task {task_id} missing resource allocation",
                        plan_id=plan.id,
                        task_id=task_id,
                        suggestion="Define resource requirements for all tasks"
                    ))
                    missing_allocations += 1
        
        if missing_allocations > 0:
            feasibility_score *= max(0.2, 1.0 - (missing_allocations * 0.1))
        
        return feasibility_score
    
    def _validate_dependencies(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> float:
        """Validate task dependencies."""
        dependency_score = 1.0
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(plan)
        if circular_deps:
            for cycle in circular_deps:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="circular_dependency",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    plan_id=plan.id,
                    suggestion="Break circular dependency by redefining task relationships"
                ))
            dependency_score *= max(0.2, 1.0 - len(circular_deps) * 0.2)
        
        # Check for missing dependencies
        missing_deps = 0
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id not in plan.tasks:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="missing_dependency",
                            message=f"Task {task_id} depends on {dep_id} which is not in plan",
                            plan_id=plan.id,
                            task_id=task_id,
                            suggestion="Include dependency task in plan or remove dependency"
                        ))
                        missing_deps += 1
        
        if missing_deps > 0:
            dependency_score *= max(0.3, 1.0 - missing_deps * 0.1)
        
        return dependency_score
    
    def _validate_logical_consistency(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> float:
        """Validate logical consistency of the plan."""
        consistency_score = 1.0
        
        # Check for empty plans
        if not plan.tasks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="empty_plan",
                message="Plan contains no tasks",
                plan_id=plan.id,
                suggestion="Add tasks to create a valid execution plan"
            ))
            return 0.0
        
        # Check for impossible task combinations
        impossible_combinations = self._detect_impossible_combinations(plan)
        if impossible_combinations:
            for combo in impossible_combinations:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="impossible_combination",
                    message=f"Potentially impossible task combination: {combo}",
                    plan_id=plan.id,
                    suggestion="Review task compatibility and resource requirements"
                ))
            consistency_score *= max(0.5, 1.0 - len(impossible_combinations) * 0.1)
        
        # Validate task status consistency
        status_issues = 0
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                if task.status not in [TaskStatus.PENDING, TaskStatus.READY]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="task_status",
                        message=f"Task {task_id} has non-plannable status: {task.status.value}",
                        plan_id=plan.id,
                        task_id=task_id,
                        suggestion="Only include pending or ready tasks in new plans"
                    ))
                    status_issues += 1
        
        if status_issues > 0:
            consistency_score *= max(0.6, 1.0 - status_issues * 0.05)
        
        return consistency_score
    
    def _validate_performance_characteristics(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> Dict[str, float]:
        """Validate performance characteristics of the plan."""
        metrics = {}
        
        # Calculate efficiency metrics
        if plan.tasks:
            avg_task_duration = plan.total_duration.total_seconds() / len(plan.tasks)
            metrics['avg_task_duration'] = avg_task_duration
            
            # Check for unusually long or short tasks
            for task_id in plan.tasks:
                if task_id in self.planner.tasks:
                    task = self.planner.tasks[task_id]
                    task_duration = task.estimated_duration.total_seconds()
                    
                    if task_duration > avg_task_duration * 5:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            category="task_duration",
                            message=f"Task {task_id} has unusually long duration",
                            plan_id=plan.id,
                            task_id=task_id,
                            suggestion="Consider breaking down long tasks into subtasks"
                        ))
                    elif task_duration < avg_task_duration * 0.1:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            category="task_duration",
                            message=f"Task {task_id} has very short duration",
                            plan_id=plan.id,
                            task_id=task_id,
                            suggestion="Consider combining short tasks for efficiency"
                        ))
        
        # Calculate resource utilization
        resource_utilization = self._calculate_resource_utilization(plan)
        metrics['resource_utilization'] = resource_utilization
        
        if resource_utilization < 0.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="resource_utilization",
                message=f"Low resource utilization ({resource_utilization:.3f})",
                plan_id=plan.id,
                suggestion="Consider optimizing resource allocation or adding more tasks"
            ))
        
        # Calculate parallelization potential
        parallelization_score = self._calculate_parallelization_potential(plan)
        metrics['parallelization_potential'] = parallelization_score
        
        return metrics
    
    def _validate_security_aspects(self, plan: ExecutionPlan, issues: List[ValidationIssue]) -> float:
        """Validate security aspects of the plan."""
        security_score = 1.0
        
        # Check for sensitive task combinations
        sensitive_combinations = self._detect_sensitive_task_combinations(plan)
        if sensitive_combinations:
            for combo in sensitive_combinations:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="security_risk",
                    message=f"Potentially sensitive task combination: {combo}",
                    plan_id=plan.id,
                    suggestion="Review security implications of task sequencing"
                ))
            security_score *= max(0.7, 1.0 - len(sensitive_combinations) * 0.1)
        
        # Check for resource access control
        access_violations = self._check_resource_access_control(plan)
        if access_violations > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="access_control",
                message=f"Potential access control issues detected: {access_violations}",
                plan_id=plan.id,
                suggestion="Verify resource access permissions for all tasks"
            ))
            security_score *= max(0.5, 1.0 - access_violations * 0.05)
        
        return security_score
    
    def _execute_with_circuit_breaker(self, circuit_breaker: CircuitBreaker, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        return circuit_breaker.call(func, *args, **kwargs)
    
    def _create_failure_report(self, plan_id: str, error_message: str) -> PlanValidationReport:
        """Create a failure report when validation cannot complete."""
        return PlanValidationReport(
            plan_id=plan_id,
            validation_timestamp=datetime.now(),
            overall_score=0.0,
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="validation_failure",
                message=f"Validation failed: {error_message}",
                plan_id=plan_id,
                suggestion="Check system logs and retry validation"
            )],
            metrics={},
            recommendations=["System validation failure - investigate logs"],
            is_valid=False,
            quantum_coherence_analysis={},
            resource_feasibility_score=0.0,
            temporal_consistency_score=0.0
        )
    
    # Helper methods for specific validation aspects
    
    def _estimate_coherence_decay_risk(self, plan: ExecutionPlan) -> float:
        """Estimate risk of quantum coherence decay during execution."""
        # Simplified model based on plan duration and complexity
        duration_factor = min(1.0, plan.total_duration.total_seconds() / 86400)  # Days
        complexity_factor = min(1.0, len(plan.tasks) / 100)  # Task count
        return (duration_factor * 0.6 + complexity_factor * 0.4)
    
    def _assess_hypervector_quality(self, hypervector: Any) -> float:
        """Assess quality of a hypervector representation."""
        try:
            # Use the planner's HDC to assess quality
            if hasattr(hypervector, 'shape'):
                # Check for reasonable sparsity and distribution
                if hasattr(hypervector, 'mean') and hasattr(hypervector, 'std'):
                    mean_val = float(hypervector.mean())
                    std_val = float(hypervector.std())
                    
                    # Good hypervectors should be roughly centered with reasonable variance
                    mean_quality = 1.0 - min(1.0, abs(mean_val) * 2)
                    std_quality = min(1.0, std_val * 2)
                    
                    return (mean_quality + std_quality) / 2
            
            return 0.7  # Default reasonable quality
        except Exception:
            return 0.0  # Unable to assess
    
    def _analyze_coherence_stability(self, plan: ExecutionPlan) -> float:
        """Analyze quantum coherence stability over time."""
        # Simplified stability analysis
        base_coherence = plan.quantum_coherence
        task_count_factor = min(1.0, len(plan.tasks) / 50)  # More tasks = less stable
        duration_factor = min(1.0, plan.total_duration.total_seconds() / 172800)  # 2 days
        
        stability = base_coherence * (1.0 - task_count_factor * 0.2 - duration_factor * 0.3)
        return max(0.0, stability)
    
    def _build_resource_timeline(self, plan: ExecutionPlan) -> Dict[str, Dict[datetime, float]]:
        """Build resource usage timeline."""
        timeline = defaultdict(lambda: defaultdict(float))
        
        for task_id in plan.tasks:
            if task_id in plan.schedule and task_id in plan.resource_allocation:
                start_time = plan.schedule[task_id]
                if task_id in self.planner.tasks:
                    duration = self.planner.tasks[task_id].estimated_duration
                    resources = plan.resource_allocation[task_id]
                    
                    # Simplified: assume constant resource usage during task
                    for resource_id, amount in resources.items():
                        timeline[resource_id][start_time] += amount
        
        return dict(timeline)
    
    def _detect_circular_dependencies(self, plan: ExecutionPlan) -> List[List[str]]:
        """Detect circular dependencies in the plan."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str, path: List[str]):
            if task_id in rec_stack:
                # Found cycle
                cycle_start = path.index(task_id)
                cycles.append(path[cycle_start:] + [task_id])
                return
            
            if task_id in visited:
                return
            
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)
            
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in plan.tasks:
                        dfs(dep_id, path[:])
            
            rec_stack.remove(task_id)
        
        for task_id in plan.tasks:
            if task_id not in visited:
                dfs(task_id, [])
        
        return cycles
    
    def _detect_impossible_combinations(self, plan: ExecutionPlan) -> List[str]:
        """Detect potentially impossible task combinations."""
        impossible = []
        
        # Check for mutually exclusive tasks (simplified)
        exclusive_pairs = [
            ('backup', 'restore'),
            ('deploy', 'rollback'),
            ('start', 'stop')
        ]
        
        task_names = []
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task_names.append(self.planner.tasks[task_id].name.lower())
        
        for name1, name2 in exclusive_pairs:
            if any(name1 in name for name in task_names) and any(name2 in name for name in task_names):
                impossible.append(f"{name1} and {name2} tasks")
        
        return impossible
    
    def _calculate_resource_utilization(self, plan: ExecutionPlan) -> float:
        """Calculate overall resource utilization."""
        if not self.planner.resources:
            return 1.0
        
        total_capacity = sum(res.capacity for res in self.planner.resources.values())
        total_used = 0
        
        for task_id in plan.tasks:
            if task_id in plan.resource_allocation:
                resources = plan.resource_allocation[task_id]
                total_used += sum(resources.values())
        
        if total_capacity == 0:
            return 0.0
        
        return min(1.0, total_used / total_capacity)
    
    def _calculate_parallelization_potential(self, plan: ExecutionPlan) -> float:
        """Calculate potential for task parallelization."""
        if len(plan.tasks) <= 1:
            return 0.0
        
        # Count tasks that could potentially run in parallel
        parallel_tasks = 0
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task = self.planner.tasks[task_id]
                # Task can run in parallel if it has no dependencies or limited dependencies
                if len(task.dependencies) <= 1:
                    parallel_tasks += 1
        
        return parallel_tasks / len(plan.tasks)
    
    def _detect_sensitive_task_combinations(self, plan: ExecutionPlan) -> List[str]:
        """Detect potentially sensitive task combinations."""
        sensitive = []
        
        # Look for security-sensitive patterns (simplified)
        sensitive_keywords = ['delete', 'remove', 'backup', 'security', 'admin', 'root']
        
        task_names = []
        for task_id in plan.tasks:
            if task_id in self.planner.tasks:
                task_names.append(self.planner.tasks[task_id].name.lower())
        
        sensitive_task_count = sum(1 for name in task_names 
                                 if any(keyword in name for keyword in sensitive_keywords))
        
        if sensitive_task_count > len(plan.tasks) * 0.3:  # >30% sensitive tasks
            sensitive.append(f"{sensitive_task_count} sensitive tasks in sequence")
        
        return sensitive
    
    def _check_resource_access_control(self, plan: ExecutionPlan) -> int:
        """Check for resource access control issues."""
        # Simplified access control check
        violations = 0
        
        for task_id in plan.tasks:
            if task_id in plan.resource_allocation:
                resources = plan.resource_allocation[task_id]
                # Check if any resource allocation seems excessive
                for resource_id, amount in resources.items():
                    if resource_id in self.planner.resources:
                        resource = self.planner.resources[resource_id]
                        if amount > resource.capacity * 0.8:  # Using >80% of resource
                            violations += 1
        
        return violations
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall validation score."""
        # Weighted average of key metrics
        weights = {
            'quantum_coherence': 0.2,
            'temporal_consistency': 0.25,
            'resource_feasibility': 0.25,
            'dependency_consistency': 0.15,
            'logical_consistency': 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(
        self,
        plan: ExecutionPlan,
        issues: List[ValidationIssue],
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # High-priority recommendations based on critical issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("CRITICAL: Address all critical validation failures before proceeding")
        
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        if error_issues:
            recommendations.append(f"Fix {len(error_issues)} error-level issues to ensure plan viability")
        
        # Performance recommendations
        if metrics.get('resource_utilization', 1.0) < 0.5:
            recommendations.append("Consider optimizing resource allocation to improve efficiency")
        
        if metrics.get('quantum_coherence', 1.0) < 0.6:
            recommendations.append("Implement coherence monitoring and restoration during execution")
        
        if metrics.get('parallelization_potential', 0.0) > 0.7:
            recommendations.append("Consider parallel execution to reduce total duration")
        
        # Add specific suggestions from issues
        unique_suggestions = set()
        for issue in issues:
            if issue.suggestion and issue.suggestion not in unique_suggestions:
                unique_suggestions.add(issue.suggestion)
                recommendations.append(issue.suggestion)
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _determine_plan_validity(self, issues: List[ValidationIssue], metrics: Dict[str, float]) -> bool:
        """Determine if plan is valid for execution."""
        # Plan is invalid if there are critical or too many error issues
        critical_count = sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        
        if critical_count > 0:
            return False
        
        if error_count > 3:  # Too many errors
            return False
        
        # Check minimum thresholds
        if metrics.get('quantum_coherence', 0.0) < self.min_quantum_coherence:
            return False
        
        if metrics.get('temporal_consistency', 0.0) < 0.5:
            return False
        
        if metrics.get('resource_feasibility', 0.0) < 0.3:
            return False
        
        return True
    
    def get_validation_analytics(self) -> Dict[str, Any]:
        """Get analytics about validation performance."""
        if not self.validation_history:
            return {'message': 'No validation history available'}
        
        recent_validations = self.validation_history[-50:]  # Last 50 validations
        
        analytics = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'average_score': np.mean([v.overall_score for v in recent_validations]),
            'validation_success_rate': np.mean([v.is_valid for v in recent_validations]),
            'common_issue_categories': {},
            'metrics_trends': {},
            'qa_status': self.qa_system.get_quality_report()
        }
        
        # Analyze common issue categories
        all_categories = []
        for validation in recent_validations:
            for issue in validation.issues:
                all_categories.append(issue.category)
        
        category_counts = defaultdict(int)
        for category in all_categories:
            category_counts[category] += 1
        
        analytics['common_issue_categories'] = dict(sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return analytics