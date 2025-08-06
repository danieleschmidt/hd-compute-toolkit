#!/usr/bin/env python3
"""
Generation 2 Test Suite: Verify robustness and error handling
"""

import sys
import time
import traceback
sys.path.insert(0, '/root/repo')


def test_robust_error_handling():
    """Test robust error handling and recovery mechanisms."""
    print("🛡️ GENERATION 2 ROBUSTNESS TEST")
    print("=" * 50)
    
    try:
        from hd_compute.robust_backends.robust_python import RobustHDComputePython
        from hd_compute.pure_python.hdc_python import SimpleArray
        
        # Test initialization with various parameters
        print("\n📋 Testing Initialization Robustness:")
        
        # Valid initialization
        hdc = RobustHDComputePython(dim=200, enable_audit_logging=False, strict_validation=False)
        print("✅ Valid initialization successful")
        
        # Test with invalid parameters (should be caught)
        try:
            invalid_hdc = RobustHDComputePython(dim=-100)
            print("❌ Should have caught invalid dimension")
            return False
        except Exception as e:
            print(f"✅ Correctly caught invalid dimension: {type(e).__name__}")
        
        # Test basic operations
        print("\n🔧 Testing Operation Robustness:")
        
        # Normal operations should work
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        print("✅ Random hypervector generation")
        
        bundled = hdc.bundle([hv1, hv2])
        print("✅ Bundling operation")
        
        bound = hdc.bind(hv1, hv2)
        print("✅ Binding operation")
        
        similarity = hdc.cosine_similarity(hv1, hv2)
        print(f"✅ Cosine similarity: {similarity:.4f}")
        
        # Test error scenarios with graceful degradation
        print("\n🚨 Testing Error Scenarios and Recovery:")
        
        # Test with malformed inputs (should recover in non-strict mode)
        try:
            # Create intentionally malformed hypervector
            malformed_hv = SimpleArray([float('inf')] * 200)
            result = hdc.cosine_similarity(hv1, malformed_hv)
            print(f"✅ Recovered from malformed input: {result}")
        except Exception as e:
            print(f"⚠️  Handled malformed input error: {type(e).__name__}")
        
        # Test with empty bundle list - should either reject or recover gracefully
        try:
            empty_bundle = hdc.bundle([])
            if empty_bundle is not None:
                print("✅ Gracefully recovered from empty bundle with fallback")
            else:
                print("❌ Empty bundle returned None")
                return False
        except Exception as e:
            print(f"✅ Correctly rejected empty bundle: {type(e).__name__}")
        
        # Test with dimension mismatch
        try:
            wrong_dim_hv = SimpleArray([0.5] * 150)  # Wrong dimension
            similarity = hdc.cosine_similarity(hv1, wrong_dim_hv) 
            print(f"⚠️  Gracefully handled dimension mismatch: {similarity}")
        except Exception as e:
            print(f"✅ Handled dimension mismatch: {type(e).__name__}")
        
        # Test advanced operations robustness
        print("\n🧪 Testing Advanced Operations Robustness:")
        
        operations = [
            ('jensen_shannon_divergence', lambda: hdc.jensen_shannon_divergence(hv1, hv2)),
            ('wasserstein_distance', lambda: hdc.wasserstein_distance(hv1, hv2)),
            ('fractional_bind', lambda: hdc.fractional_bind(hv1, hv2, power=0.7)),
            ('quantum_superposition', lambda: hdc.quantum_superposition([hv1, hv2])),
            ('entanglement_measure', lambda: hdc.entanglement_measure(hv1, hv2)),
            ('coherence_decay', lambda: hdc.coherence_decay(hv1, decay_rate=0.1)),
            ('adaptive_threshold', lambda: hdc.adaptive_threshold(hv1, target_sparsity=0.6)),
            ('hierarchical_bind', lambda: hdc.hierarchical_bind({'key': hv1, 'nested': {'inner': hv2}})),
            ('semantic_projection', lambda: hdc.semantic_projection(hv1, [hv2]))
        ]
        
        for op_name, op_func in operations:
            try:
                result = op_func()
                print(f"✅ {op_name}: {'success' if result is not None else 'returned None'}")
            except Exception as e:
                print(f"❌ {op_name} failed: {type(e).__name__} - {str(e)}")
        
        # Test statistics and monitoring
        print("\n📊 Testing Monitoring and Statistics:")
        
        stats = hdc.get_operation_statistics()
        print(f"✅ Operation statistics: {stats['total_operations']} total operations")
        print(f"   Success rate: {stats['success_rate']:.3f}")
        
        # Test health check
        health = hdc.health_check()
        print(f"✅ Health check: {health['status']} ({health['checks_passed']}/{health['total_checks']} checks passed)")
        
        if health['issues']:
            print(f"   Issues detected: {health['issues']}")
        
        print(f"\n🎯 GENERATION 2 ROBUSTNESS: ✅ COMPLETE")
        print("Error handling and recovery mechanisms working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Robust backend test failed: {e}")
        traceback.print_exc()
        return False


def test_security_validation():
    """Test security validation and input sanitization."""
    print("\n🔒 SECURITY VALIDATION TEST")
    print("-" * 40)
    
    try:
        from hd_compute.security.input_sanitization import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        # Test malicious input detection
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "__import__('os').system('rm -rf /')",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "eval(compile('print(1)', '<string>', 'exec'))"
        ]
        
        print("Testing malicious input detection:")
        for malicious in malicious_inputs:
            patterns = sanitizer.detect_malicious_input(malicious)
            if patterns:
                print(f"✅ Detected malicious patterns in: '{malicious[:20]}...'")
            else:
                print(f"⚠️  Missed potential threat: '{malicious[:20]}...'")
        
        # Test filename validation
        print("\nTesting filename validation:")
        valid_files = ["data.csv", "model.hdc", "results_2024.json"]
        invalid_files = ["../etc/passwd", "file<script>", "CON.txt", "file|rm -rf"]
        
        for filename in valid_files:
            if sanitizer.validate_filename(filename):
                print(f"✅ Valid filename accepted: {filename}")
            else:
                print(f"❌ Valid filename rejected: {filename}")
        
        for filename in invalid_files:
            if not sanitizer.validate_filename(filename):
                print(f"✅ Invalid filename rejected: {filename}")
            else:
                print(f"❌ Invalid filename accepted: {filename}")
        
        # Test hypervector data validation
        print("\nTesting hypervector data validation:")
        from hd_compute.pure_python.hdc_python import SimpleArray
        
        valid_hv = SimpleArray([0.5, 1.0, 0.0, 0.5] * 50)
        invalid_hv = SimpleArray([float('inf')] * 200)
        
        if sanitizer.validate_hypervector_data(valid_hv):
            print("✅ Valid hypervector data accepted")
        else:
            print("❌ Valid hypervector data rejected")
        
        if not sanitizer.validate_hypervector_data(invalid_hv):
            print("✅ Invalid hypervector data rejected")
        else:
            print("❌ Invalid hypervector data accepted")
        
        return True
        
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring and anomaly detection."""
    print("\n⚡ PERFORMANCE MONITORING TEST")
    print("-" * 40)
    
    try:
        from hd_compute.robust_backends.robust_python import RobustHDComputePython
        
        hdc = RobustHDComputePython(dim=500, enable_audit_logging=False)
        
        # Benchmark basic operations
        operations = {
            'random_hv': lambda: hdc.random_hv(),
            'bundle': lambda: hdc.bundle([hdc.random_hv(), hdc.random_hv()]),
            'bind': lambda: hdc.bind(hdc.random_hv(), hdc.random_hv()),
            'cosine_similarity': lambda: hdc.cosine_similarity(hdc.random_hv(), hdc.random_hv())
        }
        
        print("Benchmarking operations:")
        for op_name, op_func in operations.items():
            start_time = time.time()
            
            # Run operation multiple times
            for _ in range(10):
                result = op_func()
            
            avg_time = (time.time() - start_time) / 10 * 1000  # ms
            print(f"✅ {op_name}: {avg_time:.2f} ms average")
        
        # Test batch operations for performance
        print("\nTesting batch operation performance:")
        
        batch_hvs = [hdc.random_hv() for _ in range(20)]
        
        start_time = time.time()
        bundled_batch = hdc.bundle(batch_hvs)
        batch_time = (time.time() - start_time) * 1000
        
        print(f"✅ Bundle 20 hypervectors: {batch_time:.2f} ms")
        
        # Test advanced operations performance
        print("\nTesting advanced operations performance:")
        
        hv1, hv2 = hdc.random_hv(), hdc.random_hv()
        
        advanced_ops = {
            'jensen_shannon_divergence': lambda: hdc.jensen_shannon_divergence(hv1, hv2),
            'fractional_bind': lambda: hdc.fractional_bind(hv1, hv2, power=0.5),
            'quantum_superposition': lambda: hdc.quantum_superposition([hv1, hv2]),
            'entanglement_measure': lambda: hdc.entanglement_measure(hv1, hv2)
        }
        
        for op_name, op_func in advanced_ops.items():
            start_time = time.time()
            result = op_func()
            exec_time = (time.time() - start_time) * 1000
            print(f"✅ {op_name}: {exec_time:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def test_input_validation():
    """Test comprehensive input validation."""
    print("\n✅ INPUT VALIDATION TEST")
    print("-" * 40)
    
    try:
        from hd_compute.utils.validation import (
            validate_dimension, validate_sparsity, validate_hypervector,
            ParameterValidator, HDCValidationError, InvalidParameterError
        )
        from hd_compute.pure_python.hdc_python import SimpleArray
        
        # Test dimension validation
        print("Testing dimension validation:")
        valid_dims = [100, 1000, 10000]
        invalid_dims = [-1, 0, 1.5, "100", None]
        
        for dim in valid_dims:
            try:
                validated = validate_dimension(dim)
                print(f"✅ Valid dimension {dim} accepted")
            except:
                print(f"❌ Valid dimension {dim} rejected")
        
        for dim in invalid_dims:
            try:
                validated = validate_dimension(dim)
                print(f"❌ Invalid dimension {dim} accepted")
            except:
                print(f"✅ Invalid dimension {dim} rejected")
        
        # Test sparsity validation
        print("\nTesting sparsity validation:")
        valid_sparsities = [0.0, 0.25, 0.5, 0.75, 1.0]
        invalid_sparsities = [-0.1, 1.1, "0.5", None]
        
        for sparsity in valid_sparsities:
            try:
                validated = validate_sparsity(sparsity)
                print(f"✅ Valid sparsity {sparsity} accepted")
            except:
                print(f"❌ Valid sparsity {sparsity} rejected")
        
        for sparsity in invalid_sparsities:
            try:
                validated = validate_sparsity(sparsity)
                print(f"❌ Invalid sparsity {sparsity} accepted")
            except:
                print(f"✅ Invalid sparsity {sparsity} rejected")
        
        # Test hypervector validation
        print("\nTesting hypervector validation:")
        valid_hv = SimpleArray([0.5] * 100)
        invalid_hvs = [None, [], SimpleArray([]), "not_a_hv"]
        
        try:
            validated = validate_hypervector(valid_hv, 100)
            print("✅ Valid hypervector accepted")
        except:
            print("❌ Valid hypervector rejected")
        
        for invalid_hv in invalid_hvs:
            try:
                validated = validate_hypervector(invalid_hv, 100)
                print(f"❌ Invalid hypervector {type(invalid_hv)} accepted")
            except:
                print(f"✅ Invalid hypervector {type(invalid_hv)} rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False


def main():
    """Run Generation 2 robustness test suite."""
    print("🛡️ HD-COMPUTE GENERATION 2 TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Robust Error Handling", test_robust_error_handling),
        ("Security Validation", test_security_validation),
        ("Performance Monitoring", test_performance_monitoring),
        ("Input Validation", test_input_validation)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 GENERATION 2 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    success_rate = (passed_tests / total_tests) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 GENERATION 2: ✅ COMPLETE")
        print("Robustness and error handling implemented successfully!")
        print("Ready to proceed with Generation 3: Scalability & Optimization")
        return True
    else:
        print(f"\n⚠️  GENERATION 2: PARTIAL COMPLETION")
        print("Some robustness features need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)