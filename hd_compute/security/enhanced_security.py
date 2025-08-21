"""
Enhanced Security System for HDC Research
=========================================

Advanced security features including encryption, access control, threat detection,
and comprehensive audit trails for hyperdimensional computing research.
"""

import time
import logging
import hashlib
import json
import os
from typing import Dict, Any, List, Optional, Callable, Set
from functools import wraps
import numpy as np
from collections import defaultdict, deque


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class ThreatDetector:
    """Advanced threat detection for HDC operations."""
    
    def __init__(self):
        self.threat_signatures = set()
        self.anomaly_baselines = {}
        self.detection_history = deque(maxlen=10000)
        self.threat_score_threshold = 0.7
        
    def add_threat_signature(self, signature: str, threat_type: str) -> None:
        """Add a known threat signature."""
        self.threat_signatures.add((signature, threat_type))
        
    def calculate_threat_score(self, data: np.ndarray) -> float:
        """Calculate threat score for hypervector data."""
        threat_score = 0.0
        
        # Check for statistical anomalies
        if len(data) > 0:
            # Extremely high variance might indicate injection
            variance = np.var(data)
            if variance > 1000:
                threat_score += 0.3
            
            # Check for suspicious patterns
            unique_ratio = len(np.unique(data)) / len(data)
            if unique_ratio < 0.01:  # Too uniform
                threat_score += 0.4
            
            # Check for extreme values
            if np.any(np.abs(data) > 1e6):
                threat_score += 0.5
            
            # Check for NaN/Inf values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                threat_score += 0.8
        
        return min(1.0, threat_score)
    
    def is_threat_detected(self, data: np.ndarray) -> bool:
        """Determine if data represents a security threat."""
        threat_score = self.calculate_threat_score(data)
        
        detection_result = {
            'timestamp': time.time(),
            'threat_score': threat_score,
            'is_threat': threat_score >= self.threat_score_threshold,
            'data_shape': data.shape if hasattr(data, 'shape') else 'unknown'
        }
        
        self.detection_history.append(detection_result)
        
        return detection_result['is_threat']


class AccessController:
    """Role-based access control for HDC operations."""
    
    def __init__(self):
        self.user_roles = {}  # user_id -> set of roles
        self.role_permissions = {}  # role -> set of permissions
        self.session_tokens = {}  # token -> user_id
        self.access_log = []
        
    def create_role(self, role_name: str, permissions: Set[str]) -> None:
        """Create a new role with specified permissions."""
        self.role_permissions[role_name] = permissions.copy()
        
    def assign_role(self, user_id: str, role: str) -> None:
        """Assign a role to a user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
        
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            if permission in role_perms or '*' in role_perms:
                self._log_access(user_id, permission, 'granted')
                return True
        
        self._log_access(user_id, permission, 'denied')
        return False
        
    def create_session_token(self, user_id: str) -> str:
        """Create a session token for authenticated user."""
        token = hashlib.sha256(f"{user_id}:{time.time()}:{os.urandom(16).hex()}".encode()).hexdigest()
        self.session_tokens[token] = user_id
        return token
        
    def validate_token(self, token: str) -> Optional[str]:
        """Validate session token and return user_id."""
        return self.session_tokens.get(token)
        
    def _log_access(self, user_id: str, permission: str, result: str) -> None:
        """Log access attempts."""
        log_entry = {
            'timestamp': time.time(),
            'user_id': user_id,
            'permission': permission,
            'result': result
        }
        self.access_log.append(log_entry)
        
        # Keep only recent logs
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]


class DataEncryption:
    """Data encryption and secure serialization for HDC."""
    
    def __init__(self, default_key: Optional[str] = None):
        self.default_key = default_key or self._generate_key()
        self.key_rotation_interval = 3600  # 1 hour
        self.last_key_rotation = time.time()
        
    def _generate_key(self) -> str:
        """Generate a cryptographic key."""
        return hashlib.sha256(os.urandom(32)).hexdigest()
        
    def encrypt_hypervector(self, hv: np.ndarray, key: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt hypervector with metadata."""
        encryption_key = key or self.default_key
        
        # Create metadata
        metadata = {
            'shape': hv.shape,
            'dtype': str(hv.dtype),
            'checksum': hashlib.sha256(hv.tobytes()).hexdigest(),
            'timestamp': time.time()
        }
        
        # Simple XOR encryption (use proper crypto libraries in production)
        key_bytes = hashlib.sha256(encryption_key.encode()).digest()
        encrypted_data = bytearray()
        
        hv_bytes = hv.tobytes()
        for i, byte in enumerate(hv_bytes):
            encrypted_data.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return {
            'encrypted_data': encrypted_data.hex(),
            'metadata': metadata,
            'encryption_version': '1.0'
        }
        
    def decrypt_hypervector(self, encrypted_package: Dict[str, Any], key: Optional[str] = None) -> np.ndarray:
        """Decrypt hypervector and verify integrity."""
        encryption_key = key or self.default_key
        
        # Extract metadata
        metadata = encrypted_package['metadata']
        encrypted_hex = encrypted_package['encrypted_data']
        
        # Decrypt data
        key_bytes = hashlib.sha256(encryption_key.encode()).digest()
        encrypted_bytes = bytes.fromhex(encrypted_hex)
        decrypted_data = bytearray()
        
        for i, byte in enumerate(encrypted_bytes):
            decrypted_data.append(byte ^ key_bytes[i % len(key_bytes)])
        
        # Reconstruct array
        shape = tuple(metadata['shape'])
        dtype = metadata['dtype']
        hv = np.frombuffer(decrypted_data, dtype=dtype).reshape(shape)
        
        # Verify integrity
        actual_checksum = hashlib.sha256(hv.tobytes()).hexdigest()
        expected_checksum = metadata['checksum']
        
        if actual_checksum != expected_checksum:
            raise SecurityError("Data integrity check failed - possible tampering detected")
        
        return hv
        
    def rotate_keys(self) -> str:
        """Rotate encryption keys for security."""
        current_time = time.time()
        if current_time - self.last_key_rotation > self.key_rotation_interval:
            old_key = self.default_key
            self.default_key = self._generate_key()
            self.last_key_rotation = current_time
            return old_key
        return self.default_key


class SecurityAuditor:
    """Comprehensive security auditing and compliance."""
    
    def __init__(self):
        self.audit_events = []
        self.compliance_rules = {}
        self.audit_file = None
        
    def log_event(self, event_type: str, details: Dict[str, Any], user_id: Optional[str] = None) -> None:
        """Log security-relevant events."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details,
            'event_id': hashlib.sha256(f"{time.time()}:{event_type}:{user_id}".encode()).hexdigest()[:16]
        }
        
        self.audit_events.append(event)
        
        # Write to audit file if configured
        if self.audit_file:
            try:
                with open(self.audit_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
            except Exception as e:
                logging.error(f"Failed to write audit log: {e}")
        
        # Rotate logs if too large
        if len(self.audit_events) > 50000:
            self.audit_events = self.audit_events[-25000:]
            
    def set_audit_file(self, file_path: str) -> None:
        """Set file for persistent audit logging."""
        self.audit_file = file_path
        
    def add_compliance_rule(self, rule_name: str, rule_function: Callable) -> None:
        """Add compliance rule checker."""
        self.compliance_rules[rule_name] = rule_function
        
    def check_compliance(self) -> Dict[str, bool]:
        """Check all compliance rules."""
        results = {}
        
        for rule_name, rule_function in self.compliance_rules.items():
            try:
                results[rule_name] = rule_function(self.audit_events)
            except Exception as e:
                logging.error(f"Compliance rule {rule_name} failed: {e}")
                results[rule_name] = False
                
        return results
        
    def generate_audit_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        current_time = time.time()
        cutoff_time = current_time - (time_range_hours * 3600)
        
        recent_events = [e for e in self.audit_events if e['timestamp'] >= cutoff_time]
        
        # Event statistics
        event_counts = defaultdict(int)
        user_activity = defaultdict(int)
        
        for event in recent_events:
            event_counts[event['event_type']] += 1
            if event['user_id']:
                user_activity[event['user_id']] += 1
        
        # Security incidents
        security_events = [e for e in recent_events if 'security' in e['event_type'].lower()]
        
        return {
            'report_period_hours': time_range_hours,
            'total_events': len(recent_events),
            'event_types': dict(event_counts),
            'user_activity': dict(user_activity),
            'security_incidents': len(security_events),
            'compliance_status': self.check_compliance(),
            'most_active_user': max(user_activity.items(), key=lambda x: x[1])[0] if user_activity else None,
            'generated_at': current_time
        }


class EnhancedSecurityManager:
    """Comprehensive security manager combining all security components."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.access_controller = AccessController()
        self.data_encryption = DataEncryption()
        self.auditor = SecurityAuditor()
        self.security_policies = {}
        self.incident_response_enabled = True
        
        # Initialize default security policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self) -> None:
        """Initialize default security policies."""
        # Create default roles
        self.access_controller.create_role('researcher', {
            'read_data', 'run_experiments', 'view_results'
        })
        
        self.access_controller.create_role('admin', {
            '*'  # All permissions
        })
        
        self.access_controller.create_role('guest', {
            'view_results'
        })
        
        # Default compliance rules
        def data_retention_rule(events):
            # Check if data older than 30 days exists
            cutoff = time.time() - (30 * 24 * 3600)
            old_events = [e for e in events if e['timestamp'] < cutoff]
            return len(old_events) == 0
            
        def access_control_rule(events):
            # Check for unauthorized access attempts
            security_events = [e for e in events if e['event_type'] == 'access_denied']
            return len(security_events) < 100  # Less than 100 denials
            
        self.auditor.add_compliance_rule('data_retention', data_retention_rule)
        self.auditor.add_compliance_rule('access_control', access_control_rule)
        
    def secure_operation(self, operation_name: str, user_id: str, permission: str):
        """Decorator for securing HDC operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check authentication and authorization
                if not self.access_controller.check_permission(user_id, permission):
                    self.auditor.log_event('access_denied', {
                        'operation': operation_name,
                        'permission': permission
                    }, user_id)
                    raise SecurityError(f"Access denied for operation: {operation_name}")
                
                # Log operation start
                self.auditor.log_event('operation_start', {
                    'operation': operation_name,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }, user_id)
                
                try:
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    # Log successful completion
                    self.auditor.log_event('operation_success', {
                        'operation': operation_name
                    }, user_id)
                    
                    return result
                    
                except Exception as e:
                    # Log operation failure
                    self.auditor.log_event('operation_failure', {
                        'operation': operation_name,
                        'error': str(e)
                    }, user_id)
                    
                    # Trigger incident response if enabled
                    if self.incident_response_enabled:
                        self._trigger_incident_response(operation_name, str(e), user_id)
                    
                    raise
                    
            return wrapper
        return decorator
        
    def validate_data_security(self, data: np.ndarray, user_id: str) -> bool:
        """Comprehensive data security validation."""
        # Threat detection
        if self.threat_detector.is_threat_detected(data):
            self.auditor.log_event('threat_detected', {
                'data_shape': data.shape,
                'threat_score': self.threat_detector.calculate_threat_score(data)
            }, user_id)
            return False
        
        # Basic validation
        if data.size > 1000000:  # 1M elements max
            self.auditor.log_event('data_size_violation', {
                'size': data.size
            }, user_id)
            return False
            
        # Check for malicious patterns
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            self.auditor.log_event('invalid_data_detected', {
                'has_nan': bool(np.any(np.isnan(data))),
                'has_inf': bool(np.any(np.isinf(data)))
            }, user_id)
            return False
        
        return True
        
    def _trigger_incident_response(self, operation: str, error: str, user_id: str) -> None:
        """Trigger automated incident response."""
        self.auditor.log_event('security_incident', {
            'operation': operation,
            'error': error,
            'response_actions': ['log_incident', 'notify_admin']
        }, user_id)
        
        # In a real system, this would trigger notifications, alerts, etc.
        logging.warning(f"Security incident: {operation} failed for user {user_id}: {error}")
        
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        audit_report = self.auditor.generate_audit_report()
        
        # Recent threats
        recent_threats = [
            detection for detection in self.threat_detector.detection_history
            if detection['is_threat'] and time.time() - detection['timestamp'] < 3600
        ]
        
        # Access statistics
        access_attempts = len([
            event for event in self.auditor.audit_events
            if event['event_type'] in ['access_granted', 'access_denied']
            and time.time() - event['timestamp'] < 3600
        ])
        
        return {
            'audit_summary': audit_report,
            'threats_detected_last_hour': len(recent_threats),
            'access_attempts_last_hour': access_attempts,
            'active_sessions': len(self.access_controller.session_tokens),
            'security_status': 'secure' if len(recent_threats) == 0 else 'alert',
            'last_key_rotation': self.data_encryption.last_key_rotation,
            'compliance_status': audit_report['compliance_status']
        }


# Global security manager instance
global_security_manager = EnhancedSecurityManager()


# Convenient decorators for common security operations
def secure_research_operation(operation_name: str, user_id: str = 'default', permission: str = 'research'):
    """Decorator for securing research operations."""
    return global_security_manager.secure_operation(operation_name, user_id, permission)


def validate_hypervector_input(func):
    """Decorator to validate hypervector inputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find numpy arrays in arguments
        for arg in args:
            if isinstance(arg, np.ndarray):
                if not global_security_manager.validate_data_security(arg, 'system'):
                    raise SecurityError("Input validation failed")
        
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                if not global_security_manager.validate_data_security(value, 'system'):
                    raise SecurityError(f"Input validation failed for parameter: {key}")
        
        return func(*args, **kwargs)
    return wrapper


# Example usage
if __name__ == "__main__":
    # Initialize security manager
    security = EnhancedSecurityManager()
    
    # Create user and assign role
    security.access_controller.assign_role('researcher_1', 'researcher')
    
    # Example secured operation
    @secure_research_operation('test_operation', 'researcher_1', 'read_data')
    def test_secure_function(data):
        return np.mean(data)
    
    # Test with valid data
    test_data = np.random.normal(0, 1, 1000)
    try:
        result = test_secure_function(test_data)
        print(f"Operation successful: {result}")
    except SecurityError as e:
        print(f"Security error: {e}")
    
    # Get security dashboard
    dashboard = security.get_security_dashboard()
    print(f"Security status: {dashboard['security_status']}")