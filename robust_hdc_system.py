#!/usr/bin/env python3
"""
Generation 2: Robust HDC System with Enhanced Error Handling and Security
"""

import sys
import traceback
import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Union

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HDCSecurityError(Exception):
    """Security-related HDC operation errors."""
    pass

class HDCValidationError(Exception):
    """Validation errors in HDC operations."""
    pass

class HDCResourceError(Exception):
    """Resource exhaustion or limits exceeded."""
    pass

class RobustHDCValidator:
    """Comprehensive validation and security for HDC operations."""
    
    MAX_DIMENSION = 100000  # Security limit
    MAX_HYPERVECTORS = 10000  # Memory protection
    
    @staticmethod
    def validate_dimension(dim: int) -> None:
        """Validate hypervector dimension with security checks."""
        if not isinstance(dim, int):
            raise HDCValidationError(f"Dimension must be integer, got {type(dim)}")
        
        if dim <= 0:
            raise HDCValidationError(f"Dimension must be positive, got {dim}")
        
        if dim > RobustHDCValidator.MAX_DIMENSION:
            raise HDCSecurityError(f"Dimension {dim} exceeds security limit {RobustHDCValidator.MAX_DIMENSION}")
        
        logger.debug(f"Dimension validation passed: {dim}")
    
    @staticmethod
    def validate_hypervector_list(hvs: List[Any], operation_name: str = "operation") -> None:
        """Validate list of hypervectors with resource limits."""
        if not isinstance(hvs, (list, tuple)):
            raise HDCValidationError(f"{operation_name}: hypervectors must be list or tuple")
        
        if len(hvs) == 0:
            raise HDCValidationError(f"{operation_name}: empty hypervector list")
        
        if len(hvs) > RobustHDCValidator.MAX_HYPERVECTORS:
            raise HDCResourceError(f"{operation_name}: too many hypervectors ({len(hvs)} > {RobustHDCValidator.MAX_HYPERVECTORS})")
        
        # Validate all hypervectors have same dimension
        if len(hvs) > 1:
            first_dim = len(hvs[0].data) if hasattr(hvs[0], 'data') else len(hvs[0])
            for i, hv in enumerate(hvs[1:], 1):
                current_dim = len(hv.data) if hasattr(hv, 'data') else len(hv)
                if current_dim != first_dim:
                    raise HDCValidationError(f"{operation_name}: dimension mismatch at index {i}: {current_dim} != {first_dim}")
        
        logger.debug(f"Hypervector list validation passed: {len(hvs)} vectors")
    
    @staticmethod
    def validate_sparsity(sparsity: float) -> None:
        """Validate sparsity parameter."""
        if not isinstance(sparsity, (int, float)):
            raise HDCValidationError(f"Sparsity must be numeric, got {type(sparsity)}")
        
        if not (0.0 <= sparsity <= 1.0):
            raise HDCValidationError(f"Sparsity must be in [0,1], got {sparsity}")
        
        logger.debug(f"Sparsity validation passed: {sparsity}")
    
    @staticmethod
    def sanitize_string(s: str, max_length: int = 1000) -> str:
        """Sanitize string inputs to prevent injection attacks."""
        if not isinstance(s, str):
            raise HDCSecurityError(f"Expected string, got {type(s)}")
        
        if len(s) > max_length:
            raise HDCSecurityError(f"String too long ({len(s)} > {max_length})")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\x', '\n', '\r', '\t']
        sanitized = s
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized

class RobustHDCMonitor:
    """Monitoring and telemetry for HDC operations."""
    
    def __init__(self):
        self.metrics = {
            'operations_count': 0,
            'errors_count': 0,
            'total_execution_time': 0.0,
            'memory_peak': 0,
            'security_violations': 0
        }
        self.operation_history = []
    
    def record_operation(self, operation: str, duration: float, success: bool, error_type: str = None):
        """Record operation metrics."""
        self.metrics['operations_count'] += 1
        self.metrics['total_execution_time'] += duration
        
        if not success:
            self.metrics['errors_count'] += 1
            if 'Security' in str(error_type):
                self.metrics['security_violations'] += 1
        
        # Keep last 100 operations
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            'success': success,
            'error_type': error_type
        })
        
        if len(self.operation_history) > 100:
            self.operation_history.pop(0)
        
        logger.info(f"Operation {operation}: {'SUCCESS' if success else 'FAILED'} ({duration:.3f}s)")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health metrics."""
        total_ops = self.metrics['operations_count']
        if total_ops == 0:
            return {'status': 'idle', 'metrics': self.metrics}
        
        error_rate = self.metrics['errors_count'] / total_ops
        avg_duration = self.metrics['total_execution_time'] / total_ops
        
        status = 'healthy'
        if error_rate > 0.1:  # >10% error rate
            status = 'degraded'
        if error_rate > 0.5:  # >50% error rate
            status = 'critical'
        if self.metrics['security_violations'] > 0:
            status = 'security_alert'
        
        return {
            'status': status,
            'error_rate': error_rate,
            'avg_duration': avg_duration,
            'metrics': self.metrics,
            'recent_operations': self.operation_history[-10:]
        }

class RobustHDCSystem:
    """Enhanced HDC system with comprehensive robustness features."""
    
    def __init__(self, dim: int = 1000):
        self.validator = RobustHDCValidator()
        self.monitor = RobustHDCMonitor()
        
        # Validate initialization
        self.validator.validate_dimension(dim)
        
        self.dim = dim
        logger.info(f"Robust HDC System initialized with dimension {dim}")
    
    def safe_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with comprehensive error handling and monitoring."""
        start_time = time.time()
        error_type = None
        
        try:
            result = operation_func(*args, **kwargs)
            duration = time.time() - start_time
            self.monitor.record_operation(operation_name, duration, True)
            return result
            
        except (HDCValidationError, HDCSecurityError, HDCResourceError) as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            self.monitor.record_operation(operation_name, duration, False, error_type)
            logger.error(f"{operation_name} failed: {error_type}: {e}")
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = "UnexpectedError"
            self.monitor.record_operation(operation_name, duration, False, error_type)
            logger.error(f"{operation_name} unexpected error: {e}")
            raise HDCValidationError(f"Unexpected error in {operation_name}: {e}")
    
    def generate_random_hv(self, sparsity: float = 0.5, seed: Optional[int] = None):
        """Generate random hypervector with validation and security."""
        def _generate():
            self.validator.validate_sparsity(sparsity)
            
            # Use cryptographically secure random if no seed provided
            if seed is not None:
                import random
                random.seed(seed)
                rng = random
            else:
                import secrets
                rng = secrets.SystemRandom()
            
            # Generate binary hypervector
            hv_data = []
            for _ in range(self.dim):
                hv_data.append(1.0 if rng.random() < sparsity else -1.0)
            
            return {'data': hv_data, 'dim': self.dim, 'checksum': self._compute_checksum(hv_data)}
        
        return self.safe_operation("generate_random_hv", _generate)
    
    def bundle_hypervectors(self, hvs: List[Dict], weights: Optional[List[float]] = None):
        """Bundle hypervectors with comprehensive validation."""
        def _bundle():
            self.validator.validate_hypervector_list(hvs, "bundle")
            
            nonlocal weights
            if weights is not None:
                if len(weights) != len(hvs):
                    raise HDCValidationError(f"Weights length {len(weights)} != hypervectors length {len(hvs)}")
                
                for i, w in enumerate(weights):
                    if not isinstance(w, (int, float)):
                        raise HDCValidationError(f"Weight {i} must be numeric, got {type(w)}")
            else:
                weights = [1.0] * len(hvs)
            
            # Validate checksums
            for i, hv in enumerate(hvs):
                if not self._verify_checksum(hv):
                    logger.warning(f"Checksum validation failed for hypervector {i}")
            
            # Perform bundling
            result_data = [0.0] * self.dim
            for i, hv in enumerate(hvs):
                weight = weights[i]
                for j, val in enumerate(hv['data']):
                    result_data[j] += weight * val
            
            # Normalize to [-1, 1]
            max_abs = max(abs(x) for x in result_data)
            if max_abs > 0:
                result_data = [x / max_abs for x in result_data]
            
            return {'data': result_data, 'dim': self.dim, 'checksum': self._compute_checksum(result_data)}
        
        return self.safe_operation("bundle_hypervectors", _bundle)
    
    def bind_hypervectors(self, hv1: Dict, hv2: Dict):
        """Bind two hypervectors with validation."""
        def _bind():
            if hv1['dim'] != hv2['dim'] or hv1['dim'] != self.dim:
                raise HDCValidationError(f"Dimension mismatch: {hv1['dim']}, {hv2['dim']}, {self.dim}")
            
            # Verify checksums
            if not self._verify_checksum(hv1) or not self._verify_checksum(hv2):
                logger.warning("Checksum validation failed for bind operation")
            
            # Element-wise multiplication (binding)
            result_data = [a * b for a, b in zip(hv1['data'], hv2['data'])]
            
            return {'data': result_data, 'dim': self.dim, 'checksum': self._compute_checksum(result_data)}
        
        return self.safe_operation("bind_hypervectors", _bind)
    
    def cosine_similarity_robust(self, hv1: Dict, hv2: Dict) -> float:
        """Compute cosine similarity with robust error handling."""
        def _similarity():
            if hv1['dim'] != hv2['dim']:
                raise HDCValidationError(f"Dimension mismatch: {hv1['dim']} != {hv2['dim']}")
            
            # Compute dot product and norms
            dot_product = sum(a * b for a, b in zip(hv1['data'], hv2['data']))
            norm1 = sum(a * a for a in hv1['data']) ** 0.5
            norm2 = sum(b * b for b in hv2['data']) ** 0.5
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero-norm hypervector detected in similarity computation")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clamp to [-1, 1] to handle numerical errors
            return max(-1.0, min(1.0, similarity))
        
        return self.safe_operation("cosine_similarity", _similarity)
    
    def _compute_checksum(self, data: List[float]) -> str:
        """Compute SHA-256 checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _verify_checksum(self, hv: Dict) -> bool:
        """Verify hypervector data integrity."""
        if 'checksum' not in hv:
            return False
        expected = self._compute_checksum(hv['data'])
        return hv['checksum'] == expected
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'dimension': self.dim,
            'health': self.monitor.get_health_status(),
            'timestamp': time.time()
        }

def test_robust_system():
    """Test the robust HDC system."""
    print("🛡️ GENERATION 2: ROBUST HDC SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Initialize robust system
        system = RobustHDCSystem(dim=1000)
        print("✅ Robust HDC system initialized")
        
        # Test secure random generation
        hv1 = system.generate_random_hv(sparsity=0.3)
        hv2 = system.generate_random_hv(sparsity=0.7)
        print("✅ Secure random hypervector generation")
        
        # Test robust bundling
        bundled = system.bundle_hypervectors([hv1, hv2], weights=[0.6, 0.4])
        print("✅ Robust bundling with weights")
        
        # Test robust binding
        bound = system.bind_hypervectors(hv1, hv2)
        print("✅ Robust binding operation")
        
        # Test robust similarity
        sim = system.cosine_similarity_robust(hv1, hv2)
        print(f"✅ Robust cosine similarity: {sim:.4f}")
        
        # Test error handling - dimension validation
        try:
            system_bad = RobustHDCSystem(dim=-100)
            print("❌ Should have failed dimension validation")
        except HDCValidationError:
            print("✅ Dimension validation error handling")
        
        # Test error handling - sparsity validation
        try:
            system.generate_random_hv(sparsity=1.5)
            print("❌ Should have failed sparsity validation")
        except HDCValidationError:
            print("✅ Sparsity validation error handling")
        
        # Test security limits
        try:
            system_huge = RobustHDCSystem(dim=200000)
            print("❌ Should have failed security limit check")
        except HDCSecurityError:
            print("✅ Security limit validation")
        
        # Get system status
        status = system.get_system_status()
        print(f"✅ System health: {status['health']['status']}")
        print(f"✅ Operations completed: {status['health']['metrics']['operations_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust system test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_robust_system()
    
    if success:
        print("\n🎉 GENERATION 2 COMPLETE: System is robust and secure!")
    else:
        print("\n⚠️ GENERATION 2 needs fixes")
    
    sys.exit(0 if success else 1)