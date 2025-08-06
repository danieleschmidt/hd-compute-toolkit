"""Audit logging utilities for security monitoring."""

import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import threading


class AuditLogger:
    """Audit logger for security-relevant events."""
    
    def __init__(self, audit_file: str = "logs/audit.log", max_file_size: int = 100 * 1024 * 1024):
        """Initialize audit logger.
        
        Args:
            audit_file: Path to audit log file
            max_file_size: Maximum file size before rotation (bytes)
        """
        self.audit_file = Path(audit_file)
        self.max_file_size = max_file_size
        self.lock = threading.Lock()
        
        # Ensure log directory exists
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.logger = logging.getLogger('hdc_audit')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add file handler
        handler = logging.FileHandler(self.audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
        
        self._log_startup()
    
    def _log_startup(self):
        """Log audit system startup."""
        self.log_event(
            event_type="SYSTEM_START",
            description="Audit logging system initialized",
            metadata={"audit_file": str(self.audit_file)}
        )
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds maximum size."""
        try:
            if self.audit_file.exists() and self.audit_file.stat().st_size > self.max_file_size:
                # Create backup filename with timestamp
                timestamp = int(time.time())
                backup_file = self.audit_file.with_suffix(f'.{timestamp}.log')
                
                # Rotate the file
                self.audit_file.rename(backup_file)
                
                # Log the rotation in new file
                self.log_event(
                    event_type="LOG_ROTATION",
                    description="Audit log rotated due to size limit",
                    metadata={
                        "old_file": str(backup_file),
                        "new_file": str(self.audit_file),
                        "max_size_mb": self.max_file_size / (1024 * 1024)
                    }
                )
        except Exception as e:
            # If rotation fails, log to stderr
            print(f"Audit log rotation failed: {e}")
    
    def log_event(self, event_type: str, description: str, 
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  ip_address: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log a security audit event.
        
        Args:
            event_type: Type of event (e.g., "LOGIN", "DATA_ACCESS", "CONFIG_CHANGE")
            description: Human-readable description of the event
            user_id: User identifier (if applicable)
            session_id: Session identifier (if applicable)
            ip_address: Source IP address (if applicable)
            metadata: Additional event metadata
        """
        with self.lock:
            try:
                # Rotate log if needed
                self._rotate_log_if_needed()
                
                # Create audit event
                audit_event = {
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "description": description,
                    "user_id": user_id,
                    "session_id": session_id,
                    "ip_address": ip_address,
                    "metadata": metadata or {}
                }
                
                # Add event hash for integrity
                event_hash = self._calculate_event_hash(audit_event)
                audit_event["event_hash"] = event_hash
                
                # Log as JSON for structured parsing
                self.logger.info(json.dumps(audit_event, default=str))
                
            except Exception as e:
                # Critical: audit logging must not fail
                print(f"Audit logging failed: {e}")
    
    def _calculate_event_hash(self, event: Dict[str, Any]) -> str:
        """Calculate hash of event for integrity verification."""
        # Create deterministic string representation
        event_str = json.dumps(event, sort_keys=True, default=str)
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def log_authentication_attempt(self, success: bool, user_id: str = None, 
                                 ip_address: str = None, failure_reason: str = None):
        """Log authentication attempts.
        
        Args:
            success: Whether authentication succeeded
            user_id: User identifier
            ip_address: Source IP address
            failure_reason: Reason for failure (if applicable)
        """
        event_type = "AUTH_SUCCESS" if success else "AUTH_FAILURE"
        description = f"Authentication {'succeeded' if success else 'failed'}"
        if not success and failure_reason:
            description += f": {failure_reason}"
        
        metadata = {}
        if not success and failure_reason:
            metadata["failure_reason"] = failure_reason
        
        self.log_event(
            event_type=event_type,
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            metadata=metadata
        )
    
    def log_data_access(self, operation: str, resource: str, user_id: str = None,
                       session_id: str = None, ip_address: str = None, 
                       success: bool = True):
        """Log data access events.
        
        Args:
            operation: Type of operation (READ, WRITE, DELETE, etc.)
            resource: Resource being accessed
            user_id: User identifier
            session_id: Session identifier
            ip_address: Source IP address
            success: Whether operation succeeded
        """
        event_type = f"DATA_{operation.upper()}"
        description = f"Data {operation.lower()} operation on {resource}"
        
        metadata = {
            "operation": operation,
            "resource": resource,
            "success": success
        }
        
        self.log_event(
            event_type=event_type,
            description=description,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            metadata=metadata
        )
    
    def log_configuration_change(self, setting: str, old_value: Any, new_value: Any,
                                user_id: str = None, ip_address: str = None):
        """Log configuration changes.
        
        Args:
            setting: Configuration setting name
            old_value: Previous value
            new_value: New value
            user_id: User who made the change
            ip_address: Source IP address
        """
        # Don't log sensitive values in plain text
        if any(sensitive in setting.lower() for sensitive in ['password', 'secret', 'key', 'token']):
            old_value = "[REDACTED]"
            new_value = "[REDACTED]"
        
        metadata = {
            "setting": setting,
            "old_value": str(old_value),
            "new_value": str(new_value)
        }
        
        self.log_event(
            event_type="CONFIG_CHANGE",
            description=f"Configuration setting '{setting}' changed",
            user_id=user_id,
            ip_address=ip_address,
            metadata=metadata
        )
    
    def log_security_event(self, severity: str, event_type: str, description: str,
                          source_ip: str = None, user_id: str = None, 
                          additional_data: Dict[str, Any] = None):
        """Log security-related events.
        
        Args:
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            event_type: Type of security event
            description: Event description
            source_ip: Source IP address
            user_id: User identifier (if applicable)
            additional_data: Additional event data
        """
        metadata = {
            "severity": severity,
            **(additional_data or {})
        }
        
        self.log_event(
            event_type=f"SECURITY_{event_type.upper()}",
            description=f"[{severity}] {description}",
            user_id=user_id,
            ip_address=source_ip,
            metadata=metadata
        )
    
    def log_api_request(self, method: str, endpoint: str, status_code: int,
                       user_id: str = None, session_id: str = None, 
                       ip_address: str = None, response_time_ms: float = None):
        """Log API requests for monitoring.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            user_id: User identifier
            session_id: Session identifier
            ip_address: Source IP address
            response_time_ms: Response time in milliseconds
        """
        metadata = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code
        }
        
        if response_time_ms is not None:
            metadata["response_time_ms"] = response_time_ms
        
        # Determine if this was a suspicious request
        suspicious = (
            status_code in [401, 403, 404, 429, 500] or
            response_time_ms and response_time_ms > 10000  # >10s response time
        )
        
        event_type = "API_REQUEST_SUSPICIOUS" if suspicious else "API_REQUEST"
        
        self.log_event(
            event_type=event_type,
            description=f"{method} {endpoint} - {status_code}",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            metadata=metadata
        )
    
    def log_performance_anomaly(self, operation: str, expected_time_ms: float, 
                               actual_time_ms: float, threshold_multiplier: float = 2.0):
        """Log performance anomalies that might indicate security issues.
        
        Args:
            operation: Name of the operation
            expected_time_ms: Expected execution time
            actual_time_ms: Actual execution time
            threshold_multiplier: Multiplier for anomaly detection
        """
        if actual_time_ms > expected_time_ms * threshold_multiplier:
            metadata = {
                "operation": operation,
                "expected_time_ms": expected_time_ms,
                "actual_time_ms": actual_time_ms,
                "slowdown_factor": actual_time_ms / expected_time_ms
            }
            
            self.log_event(
                event_type="PERFORMANCE_ANOMALY",
                description=f"Operation '{operation}' took {actual_time_ms:.2f}ms (expected ~{expected_time_ms:.2f}ms)",
                metadata=metadata
            )
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Audit summary dictionary
        """
        summary = {
            "period_hours": hours,
            "total_events": 0,
            "event_types": {},
            "security_events": 0,
            "authentication_failures": 0,
            "suspicious_activities": []
        }
        
        try:
            if not self.audit_file.exists():
                return summary
            
            cutoff_time = time.time() - (hours * 3600)
            
            with open(self.audit_file, 'r') as f:
                for line in f:
                    try:
                        # Parse audit log line
                        if ' - AUDIT - INFO - ' in line:
                            json_part = line.split(' - AUDIT - INFO - ', 1)[1]
                            event = json.loads(json_part)
                            
                            # Check if event is within time window
                            if event.get('timestamp', 0) < cutoff_time:
                                continue
                            
                            summary["total_events"] += 1
                            
                            # Count event types
                            event_type = event.get('event_type', 'UNKNOWN')
                            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
                            
                            # Count security events
                            if event_type.startswith('SECURITY_'):
                                summary["security_events"] += 1
                            
                            # Count authentication failures
                            if event_type == 'AUTH_FAILURE':
                                summary["authentication_failures"] += 1
                            
                            # Identify suspicious activities
                            if event_type in ['SECURITY_INTRUSION', 'API_REQUEST_SUSPICIOUS', 'PERFORMANCE_ANOMALY']:
                                summary["suspicious_activities"].append({
                                    "timestamp": event.get('timestamp'),
                                    "type": event_type,
                                    "description": event.get('description'),
                                    "ip_address": event.get('ip_address')
                                })
                    
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed log lines
                        continue
        
        except Exception as e:
            print(f"Error generating audit summary: {e}")
        
        return summary
    
    def close(self):
        """Close audit logger and cleanup resources."""
        self.log_event(
            event_type="SYSTEM_SHUTDOWN",
            description="Audit logging system shutting down"
        )
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)