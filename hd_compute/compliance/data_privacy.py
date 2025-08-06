"""Data privacy management for global compliance."""

import hashlib
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Data category classification for privacy compliance."""
    PERSONAL_IDENTIFIABLE = "personal_identifiable"
    SENSITIVE_PERSONAL = "sensitive_personal"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"


class ProcessingPurpose(Enum):
    """Data processing purposes for compliance."""
    SCIENTIFIC_RESEARCH = "scientific_research"
    MODEL_TRAINING = "model_training"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SYSTEM_OPERATION = "system_operation"
    SECURITY_MONITORING = "security_monitoring"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    data_type: str
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    legal_basis: LegalBasis
    retention_period_days: int
    data_subjects: List[str]
    third_party_sharing: bool
    cross_border_transfer: bool
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass  
class ConsentRecord:
    """User consent record."""
    user_id: str
    consent_type: str
    granted: bool
    purpose: str
    granted_at: Optional[float]
    withdrawn_at: Optional[float]
    consent_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataPrivacyManager:
    """Manager for data privacy and compliance operations."""
    
    def __init__(self):
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, Dict[str, ConsentRecord]] = {}  # user_id -> consent_type -> record
        self.anonymization_cache: Dict[str, str] = {}
        self.retention_policies: Dict[DataCategory, int] = {
            DataCategory.PERSONAL_IDENTIFIABLE: 30,  # 30 days default
            DataCategory.SENSITIVE_PERSONAL: 30,
            DataCategory.BEHAVIORAL: 365,  # 1 year
            DataCategory.TECHNICAL: 1095,  # 3 years
            DataCategory.ANONYMOUS: -1,  # No limit
        }
    
    def register_data_processing(
        self, 
        data_type: str,
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        retention_days: Optional[int] = None,
        data_subjects: List[str] = None,
        third_party_sharing: bool = False,
        cross_border_transfer: bool = False
    ) -> str:
        """Register a data processing activity.
        
        Args:
            data_type: Type of data being processed
            data_category: Category classification
            processing_purpose: Purpose of processing
            legal_basis: Legal basis for processing
            retention_days: Retention period in days (uses default if None)
            data_subjects: List of data subjects
            third_party_sharing: Whether data is shared with third parties
            cross_border_transfer: Whether data crosses borders
            
        Returns:
            Processing record ID
        """
        if retention_days is None:
            retention_days = self.retention_policies.get(data_category, 365)
        
        record_id = hashlib.md5(f"{data_type}_{time.time()}".encode()).hexdigest()
        
        record = DataProcessingRecord(
            data_type=data_type,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            retention_period_days=retention_days,
            data_subjects=data_subjects or [],
            third_party_sharing=third_party_sharing,
            cross_border_transfer=cross_border_transfer,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.processing_records[record_id] = record
        
        logger.info(f"Registered data processing: {data_type} ({data_category.value})")
        return record_id
    
    def record_consent(
        self,
        user_id: str,
        consent_type: str,
        granted: bool,
        purpose: str,
        consent_version: str = "1.0"
    ) -> bool:
        """Record user consent.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent
            granted: Whether consent was granted
            purpose: Purpose of the consent
            consent_version: Version of consent form
            
        Returns:
            True if recorded successfully
        """
        try:
            if user_id not in self.consent_records:
                self.consent_records[user_id] = {}
            
            # Check if consent already exists and is being withdrawn
            existing_consent = self.consent_records[user_id].get(consent_type)
            
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                purpose=purpose,
                granted_at=time.time() if granted else (existing_consent.granted_at if existing_consent else None),
                withdrawn_at=time.time() if not granted else None,
                consent_version=consent_version
            )
            
            self.consent_records[user_id][consent_type] = consent_record
            
            action = "granted" if granted else "withdrawn"
            logger.info(f"Consent {action} for user {user_id}: {consent_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            return False
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has granted consent for specific purpose.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to check
            
        Returns:
            True if consent is granted and valid
        """
        if user_id not in self.consent_records:
            return False
        
        if consent_type not in self.consent_records[user_id]:
            return False
        
        consent = self.consent_records[user_id][consent_type]
        return consent.granted and consent.withdrawn_at is None
    
    def anonymize_identifier(self, identifier: str, salt: str = "hdc_anon") -> str:
        """Anonymize personal identifier using hashing.
        
        Args:
            identifier: Original identifier
            salt: Salt for hashing
            
        Returns:
            Anonymized identifier
        """
        if identifier in self.anonymization_cache:
            return self.anonymization_cache[identifier]
        
        # Create anonymous hash
        hash_input = f"{identifier}_{salt}".encode('utf-8')
        anonymous_id = hashlib.sha256(hash_input).hexdigest()[:16]  # 16 chars
        
        self.anonymization_cache[identifier] = anonymous_id
        
        return anonymous_id
    
    def pseudonymize_data(self, data: Dict[str, Any], pii_fields: List[str]) -> Dict[str, Any]:
        """Pseudonymize PII fields in data.
        
        Args:
            data: Data dictionary to pseudonymize
            pii_fields: List of PII field names
            
        Returns:
            Pseudonymized data dictionary
        """
        pseudonymized = data.copy()
        
        for field in pii_fields:
            if field in pseudonymized and pseudonymized[field]:
                original_value = str(pseudonymized[field])
                pseudonymized[field] = self.anonymize_identifier(original_value)
        
        return pseudonymized
    
    def apply_retention_policy(self) -> Dict[str, int]:
        """Apply data retention policies and clean up expired data.
        
        Returns:
            Dictionary with cleanup statistics
        """
        current_time = time.time()
        cleanup_stats = {
            "records_checked": 0,
            "records_expired": 0,
            "records_cleaned": 0
        }
        
        # Check processing records
        expired_records = []
        
        for record_id, record in self.processing_records.items():
            cleanup_stats["records_checked"] += 1
            
            if record.retention_period_days == -1:  # No expiry
                continue
            
            expiry_time = record.created_at + (record.retention_period_days * 24 * 3600)
            
            if current_time > expiry_time:
                cleanup_stats["records_expired"] += 1
                expired_records.append(record_id)
        
        # Clean up expired records
        for record_id in expired_records:
            del self.processing_records[record_id]
            cleanup_stats["records_cleaned"] += 1
            logger.info(f"Cleaned up expired processing record: {record_id}")
        
        return cleanup_stats
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report.
        
        Returns:
            Privacy report dictionary
        """
        report = {
            "timestamp": time.time(),
            "processing_activities": {
                "total_records": len(self.processing_records),
                "by_category": {},
                "by_purpose": {},
                "by_legal_basis": {},
                "third_party_sharing": 0,
                "cross_border_transfers": 0
            },
            "consent_management": {
                "total_users": len(self.consent_records),
                "total_consents": 0,
                "granted_consents": 0,
                "withdrawn_consents": 0,
                "consent_types": set()
            },
            "data_retention": {
                "total_policies": len(self.retention_policies),
                "policies": {cat.value: days for cat, days in self.retention_policies.items()}
            },
            "anonymization": {
                "cached_anonymizations": len(self.anonymization_cache)
            }
        }
        
        # Analyze processing records
        for record in self.processing_records.values():
            # By category
            category = record.data_category.value
            report["processing_activities"]["by_category"][category] = \
                report["processing_activities"]["by_category"].get(category, 0) + 1
            
            # By purpose
            purpose = record.processing_purpose.value
            report["processing_activities"]["by_purpose"][purpose] = \
                report["processing_activities"]["by_purpose"].get(purpose, 0) + 1
            
            # By legal basis
            legal_basis = record.legal_basis.value
            report["processing_activities"]["by_legal_basis"][legal_basis] = \
                report["processing_activities"]["by_legal_basis"].get(legal_basis, 0) + 1
            
            # Third party sharing and transfers
            if record.third_party_sharing:
                report["processing_activities"]["third_party_sharing"] += 1
            
            if record.cross_border_transfer:
                report["processing_activities"]["cross_border_transfers"] += 1
        
        # Analyze consent records
        for user_consents in self.consent_records.values():
            for consent_type, consent in user_consents.items():
                report["consent_management"]["total_consents"] += 1
                report["consent_management"]["consent_types"].add(consent_type)
                
                if consent.granted and consent.withdrawn_at is None:
                    report["consent_management"]["granted_consents"] += 1
                elif consent.withdrawn_at is not None:
                    report["consent_management"]["withdrawn_consents"] += 1
        
        # Convert set to list for JSON serialization
        report["consent_management"]["consent_types"] = list(report["consent_management"]["consent_types"])
        
        return report
    
    def export_processing_records(self, filename: str):
        """Export processing records for compliance audit.
        
        Args:
            filename: Output filename
        """
        try:
            export_data = {
                "export_timestamp": time.time(),
                "processing_records": [record.to_dict() for record in self.processing_records.values()],
                "retention_policies": {cat.value: days for cat, days in self.retention_policies.items()}
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Processing records exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting processing records: {e}")
    
    def handle_data_subject_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """Handle data subject rights requests (GDPR Article 15-22).
        
        Args:
            user_id: Data subject identifier
            request_type: Type of request (access, rectification, erasure, etc.)
            
        Returns:
            Response dictionary
        """
        response = {
            "user_id": user_id,
            "request_type": request_type,
            "timestamp": time.time(),
            "status": "processed"
        }
        
        if request_type == "access":
            # Right of access (Article 15)
            response["data"] = {
                "processing_activities": [
                    record.to_dict() for record in self.processing_records.values()
                    if user_id in record.data_subjects
                ],
                "consents": self.consent_records.get(user_id, {})
            }
            
        elif request_type == "erasure":
            # Right to erasure (Article 17)
            deleted_records = []
            records_to_delete = []
            
            for record_id, record in self.processing_records.items():
                if user_id in record.data_subjects:
                    deleted_records.append(record_id)
                    records_to_delete.append(record_id)
            
            # Delete the records
            for record_id in records_to_delete:
                del self.processing_records[record_id]
            
            # Delete consent records
            if user_id in self.consent_records:
                del self.consent_records[user_id]
            
            response["deleted_records"] = deleted_records
            response["consents_deleted"] = True
            
        elif request_type == "portability":
            # Right to data portability (Article 20)
            portable_data = {}
            for record in self.processing_records.values():
                if user_id in record.data_subjects and record.legal_basis == LegalBasis.CONSENT:
                    portable_data[record.data_type] = record.to_dict()
            
            response["portable_data"] = portable_data
        
        logger.info(f"Processed {request_type} request for user {user_id}")
        return response
    
    def validate_cross_border_transfer(self, destination_country: str) -> Dict[str, Any]:
        """Validate if cross-border data transfer is compliant.
        
        Args:
            destination_country: ISO country code of destination
            
        Returns:
            Validation result
        """
        # EU/EEA countries with adequate protection
        adequate_countries = {
            'AD', 'AR', 'CA', 'FO', 'GG', 'IL', 'IM', 'JP', 'JE', 'NZ', 'KR', 'CH', 'UY', 'GB'
        }
        
        # Countries with specific adequacy decisions
        partial_adequacy = {
            'US': 'Privacy Shield framework required'  # Note: Privacy Shield invalidated, need new framework
        }
        
        validation_result = {
            "destination_country": destination_country,
            "compliant": False,
            "adequacy_decision": False,
            "safeguards_required": True,
            "recommendations": []
        }
        
        if destination_country in adequate_countries:
            validation_result.update({
                "compliant": True,
                "adequacy_decision": True,
                "safeguards_required": False,
                "recommendations": ["Transfer permitted under adequacy decision"]
            })
        elif destination_country in partial_adequacy:
            validation_result.update({
                "compliant": False,
                "adequacy_decision": False,
                "safeguards_required": True,
                "recommendations": [partial_adequacy[destination_country]]
            })
        else:
            validation_result["recommendations"] = [
                "Standard Contractual Clauses (SCCs) required",
                "Binding Corporate Rules (BCRs) may be applicable", 
                "Consider data minimization",
                "Implement additional technical safeguards"
            ]
        
        return validation_result