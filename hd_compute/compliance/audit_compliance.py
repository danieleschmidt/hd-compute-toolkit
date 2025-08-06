"""
Audit compliance module for tracking and managing compliance audits.
Provides audit trail functionality and compliance monitoring.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import hashlib

# Setup logging
logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events that can be logged."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    CONSENT_CHANGE = "consent_change"
    USER_AUTHENTICATION = "user_authentication"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_INCIDENT = "security_incident"
    BREACH_NOTIFICATION = "breach_notification"

class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"

@dataclass
class AuditEvent:
    """Represents an audit event in the compliance system."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    risk_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "risk_level": self.risk_level
        }

@dataclass
class ComplianceAudit:
    """Represents a compliance audit session."""
    audit_id: str
    start_date: datetime
    end_date: Optional[datetime]
    auditor: str
    scope: List[str]  # Areas being audited
    status: ComplianceStatus
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # File paths to evidence
    
class AuditComplianceManager:
    """Manages audit trails and compliance monitoring."""
    
    def __init__(self, retention_days: int = 2555):  # 7 years default
        self.audit_events: List[AuditEvent] = []
        self.compliance_audits: Dict[str, ComplianceAudit] = {}
        self.retention_days = retention_days
        self.event_index: Dict[str, List[int]] = {}  # Index by user_id for fast lookup
        
        # Compliance monitoring thresholds
        self.risk_thresholds = {
            "failed_logins_per_hour": 5,
            "data_access_per_hour": 100,
            "bulk_operations_per_day": 10,
            "configuration_changes_per_day": 5
        }
        
        logger.info("Audit compliance manager initialized")
    
    def log_audit_event(self, 
                       event_type: AuditEventType,
                       resource: str,
                       action: str,
                       details: Dict[str, Any],
                       user_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       success: bool = True,
                       risk_level: str = "low") -> str:
        """Log an audit event."""
        
        # Generate unique event ID
        event_data = f"{datetime.now().isoformat()}{event_type.value}{resource}{action}"
        event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]
        
        audit_event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            risk_level=risk_level
        )
        
        self.audit_events.append(audit_event)
        
        # Index by user_id for fast lookup
        if user_id:
            if user_id not in self.event_index:
                self.event_index[user_id] = []
            self.event_index[user_id].append(len(self.audit_events) - 1)
        
        # Check for suspicious patterns
        self._check_suspicious_activity(audit_event)
        
        logger.info(f"Audit event logged: {event_id} ({event_type.value})")
        return event_id
    
    def _check_suspicious_activity(self, event: AuditEvent) -> None:
        """Check for suspicious activity patterns."""
        if not event.user_id:
            return
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Get recent events for this user
        user_events = self.get_user_events(event.user_id, since=day_ago)
        
        # Check failed login attempts
        if event.event_type == AuditEventType.USER_AUTHENTICATION and not event.success:
            recent_failures = [e for e in user_events 
                             if e.event_type == AuditEventType.USER_AUTHENTICATION 
                             and not e.success 
                             and e.timestamp >= hour_ago]
            
            if len(recent_failures) >= self.risk_thresholds["failed_logins_per_hour"]:
                self._trigger_security_alert("Multiple failed login attempts", event.user_id, event.ip_address)
        
        # Check excessive data access
        if event.event_type == AuditEventType.DATA_ACCESS:
            recent_access = [e for e in user_events 
                           if e.event_type == AuditEventType.DATA_ACCESS 
                           and e.timestamp >= hour_ago]
            
            if len(recent_access) >= self.risk_thresholds["data_access_per_hour"]:
                self._trigger_security_alert("Excessive data access", event.user_id, event.ip_address)
    
    def _trigger_security_alert(self, alert_type: str, user_id: str, ip_address: Optional[str]) -> None:
        """Trigger a security alert for suspicious activity."""
        self.log_audit_event(
            event_type=AuditEventType.SECURITY_INCIDENT,
            resource="security_monitoring",
            action="alert_triggered",
            details={
                "alert_type": alert_type,
                "affected_user": user_id,
                "source_ip": ip_address,
                "timestamp": datetime.now().isoformat()
            },
            risk_level="high"
        )
        
        logger.warning(f"Security alert triggered: {alert_type} for user {user_id}")
    
    def get_user_events(self, user_id: str, since: Optional[datetime] = None, 
                       event_types: Optional[List[AuditEventType]] = None) -> List[AuditEvent]:
        """Get audit events for a specific user."""
        if user_id not in self.event_index:
            return []
        
        events = []
        for idx in self.event_index[user_id]:
            if idx < len(self.audit_events):
                event = self.audit_events[idx]
                
                # Filter by timestamp
                if since and event.timestamp < since:
                    continue
                
                # Filter by event types
                if event_types and event.event_type not in event_types:
                    continue
                
                events.append(event)
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def create_compliance_audit(self, auditor: str, scope: List[str]) -> str:
        """Create a new compliance audit session."""
        audit_id = hashlib.sha256(f"{auditor}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        audit = ComplianceAudit(
            audit_id=audit_id,
            start_date=datetime.now(),
            end_date=None,
            auditor=auditor,
            scope=scope,
            status=ComplianceStatus.PENDING_REVIEW
        )
        
        self.compliance_audits[audit_id] = audit
        
        self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            resource="compliance_audit",
            action="audit_started",
            details={
                "audit_id": audit_id,
                "auditor": auditor,
                "scope": scope
            }
        )
        
        logger.info(f"Compliance audit created: {audit_id}")
        return audit_id
    
    def add_audit_finding(self, audit_id: str, finding: Dict[str, Any]) -> None:
        """Add a finding to a compliance audit."""
        if audit_id not in self.compliance_audits:
            raise ValueError(f"Audit {audit_id} not found")
        
        finding["timestamp"] = datetime.now().isoformat()
        finding["finding_id"] = hashlib.sha256(
            f"{audit_id}{finding.get('description', '')}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        self.compliance_audits[audit_id].findings.append(finding)
        
        self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            resource="compliance_audit",
            action="finding_added",
            details={
                "audit_id": audit_id,
                "finding_id": finding["finding_id"],
                "severity": finding.get("severity", "medium")
            },
            risk_level=finding.get("severity", "medium")
        )
    
    def complete_compliance_audit(self, audit_id: str, status: ComplianceStatus) -> None:
        """Complete a compliance audit."""
        if audit_id not in self.compliance_audits:
            raise ValueError(f"Audit {audit_id} not found")
        
        audit = self.compliance_audits[audit_id]
        audit.end_date = datetime.now()
        audit.status = status
        
        self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            resource="compliance_audit",
            action="audit_completed",
            details={
                "audit_id": audit_id,
                "status": status.value,
                "duration_hours": (audit.end_date - audit.start_date).total_seconds() / 3600,
                "findings_count": len(audit.findings)
            }
        )
        
        logger.info(f"Compliance audit completed: {audit_id} with status {status.value}")
    
    def generate_audit_report(self, audit_id: str) -> Dict[str, Any]:
        """Generate a comprehensive audit report."""
        if audit_id not in self.compliance_audits:
            raise ValueError(f"Audit {audit_id} not found")
        
        audit = self.compliance_audits[audit_id]
        
        # Categorize findings by severity
        findings_by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for finding in audit.findings:
            severity = finding.get("severity", "medium")
            findings_by_severity[severity].append(finding)
        
        # Calculate compliance score
        total_findings = len(audit.findings)
        critical_findings = len(findings_by_severity["critical"])
        high_findings = len(findings_by_severity["high"])
        
        if total_findings == 0:
            compliance_score = 100
        else:
            # Score calculation: critical = -20, high = -10, medium = -5, low = -2
            penalty = (critical_findings * 20 + high_findings * 10 + 
                      len(findings_by_severity["medium"]) * 5 + 
                      len(findings_by_severity["low"]) * 2)
            compliance_score = max(0, 100 - penalty)
        
        report = {
            "audit_id": audit_id,
            "auditor": audit.auditor,
            "scope": audit.scope,
            "start_date": audit.start_date.isoformat(),
            "end_date": audit.end_date.isoformat() if audit.end_date else None,
            "status": audit.status.value,
            "duration_hours": (audit.end_date - audit.start_date).total_seconds() / 3600 if audit.end_date else None,
            "compliance_score": compliance_score,
            "total_findings": total_findings,
            "findings_by_severity": {k: len(v) for k, v in findings_by_severity.items()},
            "critical_findings": findings_by_severity["critical"],
            "high_findings": findings_by_severity["high"],
            "recommendations": audit.recommendations,
            "evidence_files": audit.evidence
        }
        
        return report
    
    def cleanup_old_events(self) -> int:
        """Remove audit events older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        initial_count = len(self.audit_events)
        self.audit_events = [event for event in self.audit_events if event.timestamp >= cutoff_date]
        
        # Rebuild index
        self.event_index = {}
        for idx, event in enumerate(self.audit_events):
            if event.user_id:
                if event.user_id not in self.event_index:
                    self.event_index[event.user_id] = []
                self.event_index[event.user_id].append(idx)
        
        removed_count = initial_count - len(self.audit_events)
        
        if removed_count > 0:
            self.log_audit_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                resource="audit_system",
                action="event_cleanup",
                details={
                    "removed_events": removed_count,
                    "retention_days": self.retention_days,
                    "cutoff_date": cutoff_date.isoformat()
                }
            )
        
        logger.info(f"Cleaned up {removed_count} old audit events")
        return removed_count
    
    def export_events_json(self, since: Optional[datetime] = None, 
                          user_id: Optional[str] = None) -> str:
        """Export audit events to JSON format."""
        events = self.audit_events
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        events_data = [event.to_dict() for event in events]
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_events": len(events_data),
            "filters": {
                "since": since.isoformat() if since else None,
                "user_id": user_id
            },
            "events": events_data
        }
        
        return json.dumps(export_data, indent=2, default=str)

def main():
    """Example usage of audit compliance manager."""
    manager = AuditComplianceManager()
    
    # Log some example events
    manager.log_audit_event(
        event_type=AuditEventType.USER_AUTHENTICATION,
        resource="authentication_system",
        action="login",
        details={"method": "password"},
        user_id="user123",
        ip_address="192.168.1.100",
        success=True
    )
    
    manager.log_audit_event(
        event_type=AuditEventType.DATA_ACCESS,
        resource="user_data",
        action="read",
        details={"records_accessed": 50},
        user_id="user123"
    )
    
    # Create a compliance audit
    audit_id = manager.create_compliance_audit("auditor@example.com", ["data_protection", "security"])
    
    # Add a finding
    manager.add_audit_finding(audit_id, {
        "description": "Missing encryption for data at rest",
        "severity": "high",
        "affected_systems": ["database_server"],
        "remediation": "Implement database encryption"
    })
    
    # Complete the audit
    manager.complete_compliance_audit(audit_id, ComplianceStatus.PARTIALLY_COMPLIANT)
    
    # Generate report
    report = manager.generate_audit_report(audit_id)
    print(f"Audit completed with compliance score: {report['compliance_score']}%")
    
    return manager

if __name__ == "__main__":
    main()