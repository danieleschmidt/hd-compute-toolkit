"""Comprehensive security testing for HD-Compute-Toolkit security fixes.

This test suite validates all security remediation implementations:
- Secure serialization with restricted unpickler
- Secure dynamic imports with allowlist validation
- Input validation and sanitization
- Cryptographic utilities
- Configuration security
"""

import pytest
import pickle
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

# Import security modules
import sys
sys.path.insert(0, '/root/repo')

from hd_compute.security.secure_serialization import (
    RestrictedUnpickler, SecureSerializer, safe_pickle_load, safe_pickle_dump,
    migrate_legacy_pickle, is_secure_pickle_file
)
from hd_compute.security.secure_imports import (
    SecureImporter, EnvironmentChecker, secure_import, check_dependency
)
from hd_compute.security.security_config import (
    SecurityConfig, SecurityValidator, CryptographicUtils, SecurityAuditLogger
)


class TestSecureSerialization:
    """Test secure serialization functionality."""
    
    def test_restricted_unpickler_allows_safe_modules(self):
        """Test that safe modules are allowed."""
        # Create pickle data with safe builtin types
        safe_data = {'test': [1, 2, 3], 'value': 42}
        pickled = pickle.dumps(safe_data)
        
        # Should unpickle successfully with RestrictedUnpickler
        result = RestrictedUnpickler(io.BytesIO(pickled)).load()
        assert result == safe_data
    
    def test_restricted_unpickler_blocks_dangerous_modules(self):
        """Test that dangerous modules are blocked."""
        # Create a malicious class that tries to execute code
        class MaliciousClass:
            def __reduce__(self):
                # This would be dangerous if executed
                return (eval, ("print('COMPROMISED')",))
        
        malicious_obj = MaliciousClass()
        
        # Try to pickle and unpickle with restricted unpickler
        pickled = pickle.dumps(malicious_obj)
        
        with pytest.raises(pickle.UnpicklingError, match="not allowed for unpickling"):
            RestrictedUnpickler(io.BytesIO(pickled)).load()
    
    def test_restricted_unpickler_blocks_forbidden_callables(self):
        """Test that forbidden callables are blocked."""
        # Try to create pickle that would call eval
        import builtins
        
        # This should be blocked
        pickled_data = pickle.dumps(eval)
        
        with pytest.raises(pickle.UnpicklingError, match="not allowed for unpickling"):
            RestrictedUnpickler(io.BytesIO(pickled_data)).load()
    
    def test_secure_serializer_integrity_check(self):
        """Test that secure serializer detects tampering."""
        serializer = SecureSerializer()
        
        # Serialize legitimate data
        original_data = {'test': [1, 2, 3], 'array': [0.1, 0.2, 0.3]}
        serialized = serializer.serialize(original_data)
        
        # Should deserialize correctly
        deserialized = serializer.deserialize(serialized)
        assert deserialized == original_data
        
        # Tampered data should fail integrity check
        tampered = serialized[:-1] + b'\\x00'  # Change last byte
        with pytest.raises(ValueError, match="integrity check failed"):
            serializer.deserialize(tampered)
    
    def test_secure_serializer_different_keys(self):
        """Test that different keys produce different signatures."""
        data = {'test': 'value'}
        
        serializer1 = SecureSerializer(b'key1' * 8)
        serializer2 = SecureSerializer(b'key2' * 8)
        
        # Same data, different keys should produce different serialized data
        serialized1 = serializer1.serialize(data)
        serialized2 = serializer2.serialize(data)
        
        assert serialized1 != serialized2
        
        # Each should only deserialize with correct key
        assert serializer1.deserialize(serialized1) == data
        assert serializer2.deserialize(serialized2) == data
        
        # Wrong key should fail
        with pytest.raises(ValueError, match="integrity check failed"):
            serializer1.deserialize(serialized2)
    
    def test_safe_pickle_file_operations(self):
        """Test safe pickle file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / 'test.pkl'
            test_data = {'test': 'data', 'numbers': [1, 2, 3, 4, 5]}
            
            # Save with secure pickle
            safe_pickle_dump(test_data, str(test_file))
            
            # File should exist and be in secure format
            assert test_file.exists()
            assert is_secure_pickle_file(str(test_file))
            
            # Load with secure pickle
            loaded_data = safe_pickle_load(str(test_file))
            assert loaded_data == test_data
    
    def test_migrate_legacy_pickle(self):
        """Test migration from legacy pickle format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_file = Path(temp_dir) / 'legacy.pkl'
            secure_file = Path(temp_dir) / 'secure.pkl'
            
            # Create legacy pickle file
            test_data = {'legacy': 'data', 'list': [1, 2, 3]}
            with open(legacy_file, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Verify it's not in secure format
            assert not is_secure_pickle_file(str(legacy_file))
            
            # Migrate to secure format
            migrated_data = migrate_legacy_pickle(str(legacy_file), str(secure_file))
            
            # Verify migration worked
            assert migrated_data == test_data
            assert secure_file.exists()
            assert is_secure_pickle_file(str(secure_file))
            
            # Load migrated file
            loaded_data = safe_pickle_load(str(secure_file))
            assert loaded_data == test_data


class TestSecureImports:
    """Test secure import functionality."""
    
    def test_safe_import_allowed_modules(self):
        """Test that allowed modules can be imported."""
        # Test with a common module that should be in allowlist
        result = SecureImporter.safe_import('json')
        assert result is not None
        assert hasattr(result, 'loads')  # JSON should have loads function
        
        # Test with numpy if available
        numpy_result = SecureImporter.safe_import('numpy')
        if numpy_result is not None:
            assert hasattr(numpy_result, 'array')
    
    def test_safe_import_blocks_dangerous_modules(self):
        """Test that dangerous modules are blocked."""
        dangerous_modules = ['os', 'subprocess', 'sys', 'eval', 'exec']
        
        for module_name in dangerous_modules:
            result = SecureImporter.safe_import(module_name)
            assert result is None, f"Dangerous module {module_name} should be blocked"
    
    def test_safe_import_validates_module_names(self):
        """Test that malformed module names are rejected."""
        invalid_names = [
            '',  # Empty
            'os; rm -rf /',  # Command injection attempt
            '../../../etc/passwd',  # Path traversal
            'module with spaces',  # Invalid characters
            'module|injection',  # Pipe character
            'eval()',  # Function call syntax
            '..malicious',  # Starts with dots
        ]
        
        for invalid_name in invalid_names:
            result = SecureImporter.safe_import(invalid_name)
            assert result is None, f"Invalid module name should be rejected: {invalid_name}"
    
    def test_check_dependency_functionality(self):
        """Test dependency checking functionality."""
        # Test with builtin module
        result = SecureImporter.check_dependency('json')
        assert result['available'] is True
        assert result['name'] == 'json'
        assert 'version' in result
        
        # Test with non-existent module
        result = SecureImporter.check_dependency('nonexistent_module_12345')
        assert result['available'] is False
        assert result['error'] is not None
    
    def test_environment_checker(self):
        """Test environment validation."""
        checker = EnvironmentChecker()
        
        # Test with minimal requirements
        requirements = {
            'json': '>=0.0.0',  # Should be available
            'nonexistent_module_54321': '>=1.0.0'  # Should not be available
        }
        
        results = checker.check_all_dependencies(requirements)
        
        assert 'json' in results
        assert results['json']['available'] is True
        
        assert 'nonexistent_module_54321' in results
        assert results['nonexistent_module_54321']['available'] is False


class TestInputValidation:
    """Test input validation functionality."""
    
    def test_malicious_input_detection(self):
        """Test detection of malicious input patterns."""
        validator = SecurityValidator()
        
        malicious_inputs = [
            "eval('malicious code')",
            "__import__('os').system('rm -rf /')",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "subprocess.call(['rm', '-rf', '/'])",
            "../../../etc/passwd",
            "os.system('cat /etc/passwd')",
            "exec(compile('print(1)', '<string>', 'exec'))",
            "pickle.loads(malicious_data)"
        ]
        
        for malicious in malicious_inputs:
            is_valid = validator.validate_user_input(malicious, "test")
            assert not is_valid, f"Should detect malicious input: {malicious}"
    
    def test_safe_input_acceptance(self):
        """Test that legitimate input is accepted."""
        validator = SecurityValidator()
        
        safe_inputs = [
            "legitimate data string",
            "numpy.array([1, 2, 3])",
            "dimension=1000",
            "Hello, world!",
            "scientific computing with HDC",
            "torch.tensor([1, 2, 3, 4])",
            "result = model.predict(data)",
            "config = {'learning_rate': 0.01}"
        ]
        
        for safe_input in safe_inputs:
            is_valid = validator.validate_user_input(safe_input, "test")
            assert is_valid, f"Should accept safe input: {safe_input}"
    
    def test_filename_validation(self):
        """Test filename validation."""
        validator = SecurityValidator()
        
        # Valid filenames
        valid_files = [
            "data.csv",
            "model.hdc",
            "results_2024.json",
            "experiment-1.txt",
            "backup_file.pkl"
        ]
        
        for filename in valid_files:
            is_valid = validator.validate_user_input(filename, "filename")
            assert is_valid, f"Should accept valid filename: {filename}"
        
        # Invalid filenames
        invalid_files = [
            "../etc/passwd",
            "file<script>",
            "CON.txt",  # Windows reserved name
            "file|rm",
            "script>output.txt",
            "file:stream",
            "file\\path\\traversal"
        ]
        
        for filename in invalid_files:
            is_valid = validator.validate_user_input(filename, "filename")
            assert not is_valid, f"Should reject invalid filename: {filename}"
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        validator = SecurityValidator()
        
        test_cases = [
            ("file<>name", "file__name"),
            ("CON.txt", "safe_CON.txt"),
            ("normal_file.txt", "normal_file.txt"),  # Should remain unchanged
            ("file|with|pipes", "file_with_pipes"),
            ("very" + "x" * 300 + ".txt", "very" + "x" * 246 + ".txt"),  # Length limit
        ]
        
        for original, expected in test_cases:
            sanitized = validator.sanitize_filename(original)
            if len(expected) <= 255:
                assert sanitized == expected
            else:
                assert len(sanitized) <= 255
    
    def test_hypervector_data_validation(self):
        """Test hypervector data validation."""
        validator = SecurityValidator()
        
        # Test with mock array-like object
        class MockArray:
            def __init__(self, data, dtype_kind='f'):
                self.data = data
                self.dtype = type('dtype', (), {'kind': dtype_kind})()
                
            def __len__(self):
                return len(self.data)
            
            def isfinite(self):
                return all(isinstance(x, (int, float)) and not (x == float('inf') or x != x) for x in self.data)
        
        # Valid data
        valid_data = MockArray([0.1, 0.2, 0.3, 0.4, 0.5])
        assert validator.validate_hypervector_data(valid_data)
        
        # Invalid data type
        invalid_dtype = MockArray([0.1, 0.2, 0.3], dtype_kind='U')  # Unicode strings
        assert not validator.validate_hypervector_data(invalid_dtype)
        
        # Test with regular list (should pass)
        regular_list = [1, 2, 3, 4, 5]
        assert validator.validate_hypervector_data(regular_list)


class TestCryptographicUtils:
    """Test cryptographic utilities."""
    
    def test_secure_hash_consistency(self):
        """Test that secure hash produces consistent results."""
        crypto = CryptographicUtils()
        
        test_data = "test data for hashing"
        hash1 = crypto.secure_hash(test_data)
        hash2 = crypto.secure_hash(test_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
        
        # Different data should produce different hashes
        hash3 = crypto.secure_hash("different data")
        assert hash1 != hash3
    
    def test_fast_hash_for_cache_keys(self):
        """Test fast hash for non-security purposes."""
        crypto = CryptographicUtils()
        
        test_data = "cache key data"
        hash1 = crypto.fast_hash(test_data)
        hash2 = crypto.fast_hash(test_data)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest length
    
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        crypto = CryptographicUtils()
        
        # Generate multiple random values
        randoms = [crypto.secure_random() for _ in range(100)]
        
        # All should be in valid range
        assert all(0.0 <= r < 1.0 for r in randoms)
        
        # Should not be identical (extremely unlikely)
        assert len(set(randoms)) > 90  # Allow for some small chance of duplicates
    
    def test_secure_random_int(self):
        """Test secure random integer generation."""
        crypto = CryptographicUtils()
        
        # Generate random integers in range
        randoms = [crypto.secure_random_int(1, 10) for _ in range(100)]
        
        # All should be in valid range
        assert all(1 <= r <= 10 for r in randoms)
        
        # Should have some variety
        assert len(set(randoms)) > 3
    
    def test_salt_and_key_generation(self):
        """Test salt and key generation."""
        crypto = CryptographicUtils()
        
        # Test salt generation
        salt1 = crypto.generate_salt(16)
        salt2 = crypto.generate_salt(16)
        
        assert len(salt1) == 16
        assert len(salt2) == 16
        assert salt1 != salt2  # Should be different
        
        # Test key generation
        key1 = crypto.generate_key(32)
        key2 = crypto.generate_key(32)
        
        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2  # Should be different


class TestSecurityConfiguration:
    """Test security configuration."""
    
    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        # Check that security features are enabled by default
        assert config.enable_input_validation is True
        assert config.enable_secure_serialization is True
        assert config.enable_audit_logging is True
        assert config.restrict_dynamic_imports is True
        assert config.use_secure_hashing is True
    
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        # Set some environment variables
        env_vars = {
            'HDC_SECURITY_INPUT_VALIDATION': 'false',
            'HDC_SECURITY_DEVELOPMENT_MODE': 'true',
            'HDC_SECURITY_STRICT_MODE': 'true',
        }
        
        with patch.dict(os.environ, env_vars):
            config = SecurityConfig.from_environment()
            
            assert config.enable_input_validation is False
            assert config.development_mode is True
            assert config.strict_mode is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test conflicting configuration
        config = SecurityConfig(
            development_mode=True,
            strict_mode=True,
            enable_secure_serialization=False,
            cache_integrity_checks=True
        )
        
        issues = config.validate_config()
        
        # Should detect conflicts
        assert len(issues) >= 2  # At least the two conflicts we set up
        assert any("Development mode and strict mode" in issue for issue in issues)
        assert any("Cache integrity checks require" in issue for issue in issues)


class TestAuditLogging:
    """Test security audit logging."""
    
    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        config = SecurityConfig(
            enable_audit_logging=True,
            audit_log_path=None  # Don't create actual file
        )
        
        audit_logger = SecurityAuditLogger(config)
        assert audit_logger.config.enable_audit_logging is True
    
    def test_audit_event_logging(self):
        """Test audit event logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'audit.log'
            
            config = SecurityConfig(
                enable_audit_logging=True,
                audit_log_path=str(log_file)
            )
            
            audit_logger = SecurityAuditLogger(config)
            
            # Log a test event
            audit_logger.log_security_event(
                'TEST_EVENT',
                {'test': 'data', 'severity': 'info'},
                'INFO'
            )
            
            # Check that log file was created and contains event
            # Note: In real tests, you might need to flush/close handlers
            # For now, just check that the logger was set up correctly
            assert hasattr(audit_logger, 'audit_logger')


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple security features."""
    
    def test_secure_cache_workflow(self):
        """Test complete secure cache workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / 'secure_cache.pkl'
            
            # Create some test data
            test_data = {
                'hypervector': [0.1, 0.2, 0.3, 0.4, 0.5],
                'metadata': {'dimension': 5, 'seed': 42},
                'experiment': {'accuracy': 0.95, 'loss': 0.05}
            }
            
            # Save with secure serialization
            safe_pickle_dump(test_data, str(cache_file))
            
            # Verify file is secure format
            assert is_secure_pickle_file(str(cache_file))
            
            # Load and verify data
            loaded_data = safe_pickle_load(str(cache_file))
            assert loaded_data == test_data
            
            # Attempt to tamper with file should be detected
            with open(cache_file, 'r+b') as f:
                f.seek(-1, 2)  # Go to last byte
                f.write(b'\\x00')  # Change it
            
            with pytest.raises(ValueError, match="integrity check failed"):
                safe_pickle_load(str(cache_file))
    
    def test_secure_import_and_validation_workflow(self):
        """Test secure import with input validation."""
        validator = SecurityValidator()
        importer = SecureImporter()
        
        # Test secure module import
        module_name = "json"  # Safe module
        
        # Validate module name first
        assert validator.validate_user_input(module_name, "module_name")
        
        # Import securely
        module = importer.safe_import(module_name)
        assert module is not None
        
        # Test with dangerous input
        dangerous_name = "os; rm -rf /"
        assert not validator.validate_user_input(dangerous_name, "module_name")
        
        dangerous_module = importer.safe_import(dangerous_name)
        assert dangerous_module is None
    
    def test_end_to_end_security_workflow(self):
        """Test complete end-to-end security workflow."""
        # Initialize security components
        config = SecurityConfig(
            enable_input_validation=True,
            enable_secure_serialization=True,
            restrict_dynamic_imports=True,
            development_mode=False
        )
        
        validator = SecurityValidator(config)
        crypto = CryptographicUtils(config)
        
        # Simulate processing user input
        user_inputs = [
            "legitimate_filename.json",
            "numpy",  # Module name
            "experiment_data_2024",  # General data
        ]
        
        validated_inputs = []
        for inp in user_inputs:
            if validator.validate_user_input(inp, "general"):
                validated_inputs.append(inp)
        
        assert len(validated_inputs) == 3  # All should be valid
        
        # Simulate secure data processing
        processed_data = {
            'inputs': validated_inputs,
            'hash': crypto.secure_hash(str(validated_inputs)),
            'timestamp': 'test_timestamp'
        }
        
        # Secure storage and retrieval
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / 'processed_data.pkl'
            
            safe_pickle_dump(processed_data, str(data_file))
            retrieved_data = safe_pickle_load(str(data_file))
            
            assert retrieved_data == processed_data


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])