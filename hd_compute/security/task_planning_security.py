"""Security framework for quantum-inspired task planning.

This module provides comprehensive security measures for the task planning system,
including input sanitization, access control, audit logging, and quantum-safe security.
"""

import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from functools import wraps
import asyncio
from collections import defaultdict

from ..applications.task_planning import Task, Resource, ExecutionPlan, QuantumTaskPlanner
from ..security.audit_logger import AuditLogger
from ..security.input_sanitization import InputSanitizer


class SecurityLevel(Enum):
    """Security levels for tasks and resources."""
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    RESTRICTED = 4
    TOP_SECRET = 5


class AccessRight(Enum):
    """Access rights for resources and operations."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute" 
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    role: str
    security_level: SecurityLevel
    permissions: Set[AccessRight]
    session_token: str
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    source_system: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy for task planning."""
    max_plan_duration: timedelta = timedelta(days=7)
    max_concurrent_executions: int = 10
    require_approval_for_sensitive_tasks: bool = True
    enable_quantum_encryption: bool = True
    audit_all_operations: bool = True
    allowed_resource_usage_percent: float = 0.8
    session_timeout: timedelta = timedelta(hours=8)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    timestamp: datetime
    user_id: str
    operation: str
    resource_type: str
    resource_id: str
    security_level: SecurityLevel
    access_granted: bool
    risk_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class QuantumSecurityManager:
    """Quantum-safe security manager for task planning system."""
    
    def __init__(
        self,
        planner: QuantumTaskPlanner,
        security_policy: Optional[SecurityPolicy] = None
    ):
        """Initialize the security manager.
        
        Args:
            planner: The quantum task planner to secure
            security_policy: Security policy configuration
        """
        self.planner = planner
        self.policy = security_policy or SecurityPolicy()
        
        # Security components
        self.audit_logger = AuditLogger()
        self.input_sanitizer = InputSanitizer()
        
        # Session management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_tokens: Dict[str, str] = {}  # token -> user_id
        
        # Access control
        self.user_permissions: Dict[str, Set[AccessRight]] = {}
        self.resource_security_levels: Dict[str, SecurityLevel] = {}
        self.task_security_levels: Dict[str, SecurityLevel] = {}
        
        # Security monitoring
        self.security_events: List[SecurityAuditEntry] = []
        self.threat_indicators: Dict[str, float] = defaultdict(float)
        self.quantum_encryption_keys: Dict[str, bytes] = {}
        
        # Quantum-safe cryptography
        self._initialize_quantum_safe_crypto()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_quantum_safe_crypto(self):
        """Initialize quantum-safe cryptographic components."""
        # Generate quantum-safe encryption keys
        if self.policy.enable_quantum_encryption:
            self.master_key = secrets.token_bytes(64)  # 512-bit key
            self.quantum_salt = secrets.token_bytes(32)  # 256-bit salt
            
            # Initialize quantum-resistant key derivation
            self._derive_quantum_keys()
    
    def _derive_quantum_keys(self):
        """Derive quantum-resistant encryption keys."""
        # Use PBKDF2 with high iteration count for quantum resistance
        for purpose in ['plan_encryption', 'session_encryption', 'audit_encryption']:
            key = hashlib.pbkdf2_hmac(
                'sha512',
                self.master_key,
                self.quantum_salt + purpose.encode(),
                iterations=1000000  # High iteration count for quantum resistance
            )
            self.quantum_encryption_keys[purpose] = key
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        source_system: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate user and create security context.
        
        Args:
            username: User identifier
            password: User password (would integrate with actual auth system)
            ip_address: Client IP address
            source_system: Source system identifier
            
        Returns:
            Security context if authentication successful
        """
        # Sanitize inputs
        username = self.input_sanitizer.sanitize_string(username)
        
        # In production, this would integrate with real authentication
        # For demo, we'll simulate authentication
        if self._simulate_authentication(username, password):
            # Generate secure session token
            session_token = secrets.token_urlsafe(64)
            
            # Create security context
            security_context = SecurityContext(
                user_id=username,
                role=self._get_user_role(username),
                security_level=self._get_user_security_level(username),
                permissions=self.user_permissions.get(username, set()),
                session_token=session_token,
                ip_address=ip_address,
                source_system=source_system
            )
            
            # Store session
            self.active_sessions[session_token] = security_context
            self.session_tokens[session_token] = username
            
            # Audit log
            self._audit_security_event(
                security_context,
                "user_authentication",
                "session",
                session_token,
                True,
                {"ip_address": ip_address, "source_system": source_system}
            )
            
            self.logger.info(f"User {username} authenticated successfully")
            return security_context
        else:
            # Log failed authentication
            self._audit_security_event(
                SecurityContext(
                    user_id=username,
                    role="unknown",
                    security_level=SecurityLevel.PUBLIC,
                    permissions=set(),
                    session_token="",
                    ip_address=ip_address
                ),
                "user_authentication",
                "session",
                "",
                False,
                {"failure_reason": "invalid_credentials", "ip_address": ip_address}
            )
            
            self.logger.warning(f"Authentication failed for user {username}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate session token and return security context."""
        if session_token not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_token]
        
        # Check session timeout
        if datetime.now() - context.timestamp > self.policy.session_timeout:
            self.invalidate_session(session_token)
            return None
        
        # Update last access time
        context.timestamp = datetime.now()
        
        return context
    
    def invalidate_session(self, session_token: str) -> None:
        """Invalidate user session."""
        if session_token in self.active_sessions:
            context = self.active_sessions[session_token]
            del self.active_sessions[session_token]
            del self.session_tokens[session_token]
            
            self._audit_security_event(
                context,
                "session_invalidation",
                "session",
                session_token,
                True,
                {"reason": "explicit_logout"}
            )
            
            self.logger.info(f"Session invalidated for user {context.user_id}")
    
    def secure_task_creation(
        self,
        context: SecurityContext,
        task_id: str,
        name: str,
        description: str,
        **kwargs
    ) -> bool:
        """Securely create a task with access control and validation.
        
        Args:
            context: Security context of the requesting user
            task_id: Task identifier
            name: Task name
            description: Task description
            **kwargs: Additional task parameters
            
        Returns:
            True if task created successfully
        """
        # Validate session
        if not self._validate_session_context(context):
            return False
        
        # Check permissions
        if not self._check_permission(context, AccessRight.WRITE, "task", task_id):
            self._audit_security_event(
                context, "task_creation", "task", task_id, False,
                {"reason": "insufficient_permissions"}
            )
            return False
        
        # Sanitize inputs
        task_id = self.input_sanitizer.sanitize_string(task_id)
        name = self.input_sanitizer.sanitize_string(name)
        description = self.input_sanitizer.sanitize_string(description)
        
        # Validate task security
        security_risk = self._assess_task_security_risk(name, description, kwargs)
        if security_risk > 0.8:  # High risk threshold
            self._audit_security_event(
                context, "task_creation", "task", task_id, False,
                {"reason": "high_security_risk", "risk_score": security_risk}
            )
            self.logger.warning(f"Task creation blocked: high security risk ({security_risk:.3f})")
            return False
        
        # Determine task security level
        task_security_level = self._classify_task_security_level(name, description, kwargs)
        
        # Check if user has sufficient security level
        if task_security_level.value > context.security_level.value:
            self._audit_security_event(
                context, "task_creation", "task", task_id, False,
                {"reason": "insufficient_security_level", "required": task_security_level.value}
            )
            return False
        
        try:
            # Create task with security metadata
            self.planner.add_task(
                task_id=task_id,
                name=name,
                description=description,
                metadata=kwargs.get('metadata', {}).update({
                    'creator': context.user_id,
                    'security_level': task_security_level.value,
                    'creation_timestamp': datetime.now().isoformat(),
                    'security_classification': task_security_level.name
                }),
                **{k: v for k, v in kwargs.items() if k != 'metadata'}
            )
            
            # Store security classification
            self.task_security_levels[task_id] = task_security_level
            
            # Audit successful creation
            self._audit_security_event(
                context, "task_creation", "task", task_id, True,
                {"security_level": task_security_level.name, "risk_score": security_risk}
            )
            
            self.logger.info(f"Task {task_id} created by {context.user_id} with security level {task_security_level.name}")
            return True
            
        except Exception as e:
            self._audit_security_event(
                context, "task_creation", "task", task_id, False,
                {"reason": "creation_error", "error": str(e)}
            )
            self.logger.error(f"Task creation failed: {str(e)}")
            return False
    
    def secure_plan_creation(
        self,
        context: SecurityContext,
        strategy: str,
        objectives: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Securely create an execution plan.
        
        Args:
            context: Security context
            strategy: Planning strategy
            objectives: Optimization objectives
            constraints: Planning constraints
            
        Returns:
            Plan ID if successful, None otherwise
        """
        # Validate session
        if not self._validate_session_context(context):
            return None
        
        # Check permissions
        if not self._check_permission(context, AccessRight.EXECUTE, "plan", "new"):
            self._audit_security_event(
                context, "plan_creation", "plan", "new", False,
                {"reason": "insufficient_permissions"}
            )
            return None
        
        # Validate maximum concurrent executions
        active_plans = len([p for p in self.planner.plans.values() 
                          if any(self.planner.tasks[tid].status.value in ['in_progress', 'ready'] 
                               for tid in p.tasks if tid in self.planner.tasks)])
        
        if active_plans >= self.policy.max_concurrent_executions:
            self._audit_security_event(
                context, "plan_creation", "plan", "new", False,
                {"reason": "max_concurrent_limit", "active_plans": active_plans}
            )
            return None
        
        # Sanitize inputs
        if objectives:
            objectives = [self.input_sanitizer.sanitize_string(obj) for obj in objectives]
        if constraints:
            constraints = {
                self.input_sanitizer.sanitize_string(k): v 
                for k, v in constraints.items()
            }
        
        # Assess security risk of plan
        plan_risk = self._assess_plan_security_risk(context, strategy, objectives, constraints)
        if plan_risk > 0.7:  # High risk threshold
            self._audit_security_event(
                context, "plan_creation", "plan", "new", False,
                {"reason": "high_security_risk", "risk_score": plan_risk}
            )
            return None
        
        # Check if plan requires approval
        if self.policy.require_approval_for_sensitive_tasks and plan_risk > 0.4:
            # In production, this would trigger approval workflow
            self.logger.info(f"Plan creation by {context.user_id} requires approval (risk: {plan_risk:.3f})")
        
        try:
            from ..applications.task_planning import PlanningStrategy
            strategy_enum = PlanningStrategy(strategy.lower().replace('_', '_'))
            
            # Create plan
            plan = self.planner.create_quantum_plan(
                strategy=strategy_enum,
                optimization_objectives=objectives,
                constraints=constraints
            )
            
            # Encrypt plan if quantum encryption enabled
            if self.policy.enable_quantum_encryption:
                self._encrypt_plan_quantum(plan)
            
            # Audit successful creation
            self._audit_security_event(
                context, "plan_creation", "plan", plan.id, True,
                {"strategy": strategy, "risk_score": plan_risk, "task_count": len(plan.tasks)}
            )
            
            self.logger.info(f"Plan {plan.id} created by {context.user_id} with strategy {strategy}")
            return plan.id
            
        except Exception as e:
            self._audit_security_event(
                context, "plan_creation", "plan", "new", False,
                {"reason": "creation_error", "error": str(e)}
            )
            self.logger.error(f"Plan creation failed: {str(e)}")
            return None
    
    def secure_plan_execution(
        self,
        context: SecurityContext,
        plan_id: str
    ) -> bool:
        """Securely execute a plan with monitoring.
        
        Args:
            context: Security context
            plan_id: Plan to execute
            
        Returns:
            True if execution started successfully
        """
        # Validate session
        if not self._validate_session_context(context):
            return False
        
        # Check permissions
        if not self._check_permission(context, AccessRight.EXECUTE, "plan", plan_id):
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, False,
                {"reason": "insufficient_permissions"}
            )
            return False
        
        # Validate plan exists
        if plan_id not in self.planner.plans:
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, False,
                {"reason": "plan_not_found"}
            )
            return False
        
        plan = self.planner.plans[plan_id]
        
        # Security validation of plan execution
        execution_risk = self._assess_execution_security_risk(context, plan)
        if execution_risk > 0.8:
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, False,
                {"reason": "high_execution_risk", "risk_score": execution_risk}
            )
            return False
        
        # Check plan duration against policy
        if plan.total_duration > self.policy.max_plan_duration:
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, False,
                {"reason": "duration_exceeds_policy", "duration_hours": plan.total_duration.total_seconds() / 3600}
            )
            return False
        
        try:
            # Start secure execution monitoring
            self._start_execution_monitoring(context, plan_id)
            
            # Execute plan asynchronously
            execution_task = self.planner.execute_plan_async(plan_id)
            
            # Audit successful execution start
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, True,
                {"execution_risk": execution_risk, "duration_hours": plan.total_duration.total_seconds() / 3600}
            )
            
            self.logger.info(f"Plan {plan_id} execution started by {context.user_id}")
            return True
            
        except Exception as e:
            self._audit_security_event(
                context, "plan_execution", "plan", plan_id, False,
                {"reason": "execution_error", "error": str(e)}
            )
            self.logger.error(f"Plan execution failed: {str(e)}")
            return False
    
    def _simulate_authentication(self, username: str, password: str) -> bool:
        """Simulate user authentication (replace with real auth system)."""
        # In production, integrate with proper authentication system
        # For demo purposes, allow any non-empty username/password
        return len(username) > 0 and len(password) > 0
    
    def _get_user_role(self, username: str) -> str:
        """Get user role (integrate with role management system)."""
        # Simplified role assignment
        if 'admin' in username.lower():
            return 'administrator'
        elif 'manager' in username.lower():
            return 'manager'
        else:
            return 'user'
    
    def _get_user_security_level(self, username: str) -> SecurityLevel:
        """Get user security clearance level."""
        role = self._get_user_role(username)
        security_mapping = {
            'administrator': SecurityLevel.RESTRICTED,
            'manager': SecurityLevel.CONFIDENTIAL,
            'user': SecurityLevel.INTERNAL
        }
        return security_mapping.get(role, SecurityLevel.PUBLIC)
    
    def _validate_session_context(self, context: SecurityContext) -> bool:
        """Validate security context."""
        # Check if session is still active
        if context.session_token not in self.active_sessions:
            return False
        
        # Check session timeout
        if datetime.now() - context.timestamp > self.policy.session_timeout:
            self.invalidate_session(context.session_token)
            return False
        
        return True
    
    def _check_permission(
        self,
        context: SecurityContext,
        required_right: AccessRight,
        resource_type: str,
        resource_id: str
    ) -> bool:
        """Check if user has required permission."""
        # Admin users have all permissions
        if context.role == 'administrator':
            return True
        
        # Check explicit permissions
        if required_right in context.permissions:
            return True
        
        # Check resource-specific permissions (simplified)
        resource_key = f"{resource_type}:{resource_id}"
        if resource_key in self.user_permissions.get(context.user_id, set()):
            return True
        
        return False
    
    def _assess_task_security_risk(
        self,
        name: str,
        description: str,
        kwargs: Dict[str, Any]
    ) -> float:
        """Assess security risk of a task."""
        risk_score = 0.0
        
        # Check for risky keywords in name/description
        risky_keywords = [
            'delete', 'remove', 'drop', 'truncate', 'destroy',
            'admin', 'root', 'sudo', 'chmod', 'chown',
            'backup', 'restore', 'migrate', 'deploy',
            'security', 'password', 'key', 'token',
            'network', 'firewall', 'port', 'access'
        ]
        
        text_to_check = (name + ' ' + description).lower()
        risky_word_count = sum(1 for keyword in risky_keywords if keyword in text_to_check)
        risk_score += min(0.5, risky_word_count * 0.1)
        
        # Check for sensitive resource requirements
        resources_required = kwargs.get('resources_required', {})
        if isinstance(resources_required, dict):
            sensitive_resources = ['database', 'network', 'storage', 'compute']
            for resource in sensitive_resources:
                if any(resource in res_name.lower() for res_name in resources_required.keys()):
                    risk_score += 0.1
        
        # Check for long duration (potential for more damage)
        estimated_duration = kwargs.get('estimated_duration')
        if estimated_duration and hasattr(estimated_duration, 'total_seconds'):
            if estimated_duration.total_seconds() > 86400:  # > 1 day
                risk_score += 0.1
        
        # Check for high priority (could indicate privileged operation)
        priority = kwargs.get('priority', 1.0)
        if priority > 5.0:
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _classify_task_security_level(
        self,
        name: str,
        description: str,
        kwargs: Dict[str, Any]
    ) -> SecurityLevel:
        """Classify task security level."""
        risk_score = self._assess_task_security_risk(name, description, kwargs)
        
        if risk_score >= 0.8:
            return SecurityLevel.RESTRICTED
        elif risk_score >= 0.6:
            return SecurityLevel.CONFIDENTIAL
        elif risk_score >= 0.3:
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def _assess_plan_security_risk(
        self,
        context: SecurityContext,
        strategy: str,
        objectives: Optional[List[str]],
        constraints: Optional[Dict[str, Any]]
    ) -> float:
        """Assess security risk of plan creation."""
        risk_score = 0.0
        
        # Strategy-based risk
        high_risk_strategies = ['quantum_superposition', 'hybrid_quantum']
        if strategy in high_risk_strategies:
            risk_score += 0.2
        
        # Objectives-based risk
        if objectives:
            risky_objectives = ['minimize_cost', 'maximize_resource_usage']
            for obj in objectives:
                if any(risky in obj.lower() for risky in risky_objectives):
                    risk_score += 0.1
        
        # Task-based risk (sum of individual task risks)
        total_task_risk = 0.0
        task_count = 0
        for task in self.planner.tasks.values():
            if task.status.value in ['pending', 'ready']:
                task_risk = self._assess_task_security_risk(
                    task.name,
                    task.description,
                    {'resources_required': task.resources_required, 'priority': task.priority}
                )
                total_task_risk += task_risk
                task_count += 1
        
        if task_count > 0:
            avg_task_risk = total_task_risk / task_count
            risk_score += avg_task_risk * 0.5
        
        return min(1.0, risk_score)
    
    def _assess_execution_security_risk(
        self,
        context: SecurityContext,
        plan: ExecutionPlan
    ) -> float:
        """Assess security risk of plan execution."""
        risk_score = 0.0
        
        # Quantum coherence risk (low coherence = higher risk)
        coherence_risk = 1.0 - plan.quantum_coherence
        risk_score += coherence_risk * 0.3
        
        # Success probability risk (low success = higher risk)
        success_risk = 1.0 - plan.success_probability
        risk_score += success_risk * 0.2
        
        # Duration risk (longer execution = higher risk)
        duration_hours = plan.total_duration.total_seconds() / 3600
        if duration_hours > 24:  # > 1 day
            risk_score += min(0.3, (duration_hours - 24) * 0.01)
        
        # Task count risk (more tasks = higher complexity risk)
        if len(plan.tasks) > 20:
            risk_score += min(0.2, (len(plan.tasks) - 20) * 0.005)
        
        return min(1.0, risk_score)
    
    def _encrypt_plan_quantum(self, plan: ExecutionPlan) -> None:
        """Apply quantum-safe encryption to plan data."""
        if not self.policy.enable_quantum_encryption:
            return
        
        try:
            # Create HMAC for plan integrity
            plan_data = {
                'tasks': plan.tasks,
                'schedule': {k: v.isoformat() for k, v in plan.schedule.items()},
                'resource_allocation': plan.resource_allocation
            }
            
            plan_json = json.dumps(plan_data, sort_keys=True).encode()
            
            # Generate HMAC using quantum-safe key
            hmac_key = self.quantum_encryption_keys['plan_encryption']
            plan_hmac = hmac.new(hmac_key, plan_json, hashlib.sha512).hexdigest()
            
            # Store HMAC in plan metadata
            if not hasattr(plan, 'metadata'):
                plan.metadata = {}
            plan.metadata['quantum_signature'] = plan_hmac
            
            self.logger.debug(f"Applied quantum encryption to plan {plan.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt plan {plan.id}: {str(e)}")
    
    def _start_execution_monitoring(self, context: SecurityContext, plan_id: str) -> None:
        """Start security monitoring for plan execution."""
        # Initialize monitoring state
        monitoring_state = {
            'plan_id': plan_id,
            'executor': context.user_id,
            'start_time': datetime.now(),
            'security_alerts': [],
            'anomaly_count': 0
        }
        
        # Store monitoring state (would integrate with monitoring system)
        self.logger.info(f"Started security monitoring for plan {plan_id}")
    
    def _audit_security_event(
        self,
        context: SecurityContext,
        operation: str,
        resource_type: str,
        resource_id: str,
        access_granted: bool,
        details: Dict[str, Any] = None
    ) -> None:
        """Log security audit event."""
        audit_entry = SecurityAuditEntry(
            timestamp=datetime.now(),
            user_id=context.user_id,
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            security_level=context.security_level,
            access_granted=access_granted,
            risk_score=details.get('risk_score', 0.0) if details else 0.0,
            details=details or {}
        )
        
        self.security_events.append(audit_entry)
        
        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Log to audit system
        self.audit_logger.log_security_event(
            user_id=context.user_id,
            action=operation,
            resource=f"{resource_type}:{resource_id}",
            granted=access_granted,
            details=details or {}
        )
    
    def get_security_analytics(self) -> Dict[str, Any]:
        """Get security analytics and monitoring data."""
        if not self.security_events:
            return {'message': 'No security events recorded'}
        
        recent_events = self.security_events[-1000:]  # Last 1000 events
        
        analytics = {
            'total_security_events': len(self.security_events),
            'recent_events': len(recent_events),
            'active_sessions': len(self.active_sessions),
            'access_granted_rate': np.mean([e.access_granted for e in recent_events]),
            'average_risk_score': np.mean([e.risk_score for e in recent_events]),
            'operations_by_type': {},
            'risk_distribution': {},
            'security_alerts': []
        }
        
        # Operations distribution
        operations = [e.operation for e in recent_events]
        operation_counts = defaultdict(int)
        for op in operations:
            operation_counts[op] += 1
        analytics['operations_by_type'] = dict(operation_counts)
        
        # Risk distribution
        risk_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        risk_distribution = {f"{r[0]}-{r[1]}": 0 for r in risk_ranges}
        
        for event in recent_events:
            for range_low, range_high in risk_ranges:
                if range_low <= event.risk_score < range_high:
                    risk_distribution[f"{range_low}-{range_high}"] += 1
                    break
        
        analytics['risk_distribution'] = risk_distribution
        
        # Recent high-risk events
        high_risk_events = [e for e in recent_events if e.risk_score > 0.7]
        analytics['high_risk_events_count'] = len(high_risk_events)
        
        # Failed access attempts
        failed_access = [e for e in recent_events if not e.access_granted]
        analytics['failed_access_attempts'] = len(failed_access)
        
        return analytics
    
    def export_security_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Export comprehensive security report."""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        relevant_events = [
            e for e in self.security_events
            if e.timestamp > cutoff_time
        ]
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'time_range_hours': time_range_hours,
            'summary': {
                'total_events': len(relevant_events),
                'successful_operations': len([e for e in relevant_events if e.access_granted]),
                'failed_operations': len([e for e in relevant_events if not e.access_granted]),
                'unique_users': len(set(e.user_id for e in relevant_events)),
                'average_risk_score': np.mean([e.risk_score for e in relevant_events]) if relevant_events else 0.0
            },
            'security_incidents': [],
            'recommendations': []
        }
        
        # Identify security incidents
        high_risk_events = [e for e in relevant_events if e.risk_score > 0.8]
        failed_access_events = [e for e in relevant_events if not e.access_granted]
        
        report['security_incidents'] = [
            {
                'type': 'high_risk_operation',
                'count': len(high_risk_events),
                'events': [
                    {
                        'user': e.user_id,
                        'operation': e.operation,
                        'resource': f"{e.resource_type}:{e.resource_id}",
                        'risk_score': e.risk_score,
                        'timestamp': e.timestamp.isoformat()
                    }
                    for e in high_risk_events[:10]  # Top 10
                ]
            },
            {
                'type': 'failed_access_attempts',
                'count': len(failed_access_events),
                'events': [
                    {
                        'user': e.user_id,
                        'operation': e.operation,
                        'resource': f"{e.resource_type}:{e.resource_id}",
                        'timestamp': e.timestamp.isoformat(),
                        'details': e.details
                    }
                    for e in failed_access_events[:10]  # Top 10
                ]
            }
        ]
        
        # Generate recommendations
        if len(high_risk_events) > len(relevant_events) * 0.1:  # > 10% high risk
            report['recommendations'].append(
                "High percentage of high-risk operations detected. Review security policies."
            )
        
        if len(failed_access_events) > 20:
            report['recommendations'].append(
                "Multiple failed access attempts detected. Investigate potential security threats."
            )
        
        unique_failed_users = set(e.user_id for e in failed_access_events)
        if len(unique_failed_users) > 5:
            report['recommendations'].append(
                f"Failed access attempts from {len(unique_failed_users)} different users. Monitor for coordinated attacks."
            )
        
        return report


def require_security_context(permission: AccessRight = AccessRight.READ):
    """Decorator to require security context for function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, context: SecurityContext, *args, **kwargs):
            if not hasattr(self, 'security_manager'):
                raise RuntimeError("Security manager not initialized")
            
            security_manager = self.security_manager
            
            # Validate session
            if not security_manager._validate_session_context(context):
                raise PermissionError("Invalid or expired security context")
            
            # Check permission
            resource_type = getattr(self, '__class__', type(self)).__name__.lower()
            resource_id = kwargs.get('resource_id', 'default')
            
            if not security_manager._check_permission(context, permission, resource_type, resource_id):
                security_manager._audit_security_event(
                    context, func.__name__, resource_type, resource_id, False,
                    {"reason": "insufficient_permissions", "required_permission": permission.value}
                )
                raise PermissionError(f"Insufficient permissions for {func.__name__}")
            
            # Execute function
            try:
                result = func(self, context, *args, **kwargs)
                
                # Audit successful access
                security_manager._audit_security_event(
                    context, func.__name__, resource_type, resource_id, True,
                    {"function": func.__name__}
                )
                
                return result
                
            except Exception as e:
                # Audit failed execution
                security_manager._audit_security_event(
                    context, func.__name__, resource_type, resource_id, False,
                    {"reason": "execution_error", "error": str(e)}
                )
                raise
        
        return wrapper
    return decorator