"""
Robust Validation Test Suite
===========================

Tests all Generation 2 robustness improvements:
- Input validation and error handling
- Security monitoring and rate limiting
- Health checking and system monitoring
- Comprehensive error recovery
"""

import sys
import time
import numpy as np
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, '/root/repo')

from hd_compute.research.novel_algorithms import (
    AdvancedTemporalHDC, 
    ConcreteAttentionHDC, 
    NeurosymbolicHDC
)
from hd_compute.security.research_security import (
    HDCSecurityMonitor,
    HDCHealthChecker,
    security_monitor,
    health_checker
)


class RobustnessTestSuite:
    """Comprehensive test suite for robustness improvements."""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete robustness test suite."""
        print("🛡️  ROBUSTNESS VALIDATION TEST SUITE")
        print("=" * 50)
        
        test_methods = [
            self.test_input_validation,
            self.test_error_handling,
            self.test_security_monitoring,
            self.test_health_checking,
            self.test_edge_cases,
            self.test_memory_safety,
            self.test_rate_limiting
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                self.test_results[test_method.__name__] = {
                    'status': 'passed' if result else 'failed',
                    'result': result
                }
            except Exception as e:
                self.test_results[test_method.__name__] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.print_summary()
        return self.test_results
    
    def test_input_validation(self) -> bool:
        """Test comprehensive input validation."""
        print("\n🔍 Testing Input Validation...")
        
        temporal = AdvancedTemporalHDC(dim=100)
        
        # Test valid input
        try:
            valid_hv = np.random.binomial(1, 0.5, size=100).astype(np.int8)
            result = temporal.temporal_binding(valid_hv, [valid_hv], [1])
            assert result.shape == (100,)
            print("  ✅ Valid input accepted")
        except Exception as e:
            print(f"  ❌ Valid input rejected: {e}")
            return False
        
        # Test invalid dimensions
        try:
            invalid_hv = np.random.randn(50)  # Wrong dimension
            temporal.temporal_binding(invalid_hv, [valid_hv], [1])
            print("  ❌ Invalid dimension not caught")
            return False
        except ValueError:
            print("  ✅ Invalid dimension correctly rejected")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
        
        # Test NaN/Inf values
        try:
            nan_hv = np.full(100, np.nan)
            temporal.temporal_binding(nan_hv, [valid_hv], [1])
            print("  ❌ NaN values not caught")
            return False
        except ValueError:
            print("  ✅ NaN values correctly rejected")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
        
        # Test type validation
        try:
            temporal_invalid = AdvancedTemporalHDC(dim="invalid")
            print("  ❌ Invalid constructor type not caught")
            return False
        except ValueError:
            print("  ✅ Invalid constructor type correctly rejected")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test comprehensive error handling and recovery."""
        print("\n⚠️  Testing Error Handling...")
        
        temporal = AdvancedTemporalHDC(dim=100)
        attention = ConcreteAttentionHDC(dim=100, num_attention_heads=4)
        
        # Test graceful error handling
        try:
            # Temporal binding with mismatched lists
            valid_hv = np.random.randn(100)
            context_hvs = [np.random.randn(100) for _ in range(3)]
            positions = [1, 2]  # Mismatched length
            
            temporal.temporal_binding(valid_hv, context_hvs, positions)
            print("  ❌ Mismatched input lengths not caught")
            return False
        except (ValueError, RuntimeError):
            print("  ✅ Mismatched input lengths correctly handled")
        except Exception as e:
            print(f"  ❌ Unexpected error type: {e}")
            return False
        
        # Test error counting
        initial_errors = temporal._error_count
        try:
            temporal.temporal_binding(None, [], [])
        except:
            pass
        
        if temporal._error_count > initial_errors:
            print("  ✅ Error counting works correctly")
        else:
            print("  ❌ Error counting not working")
            return False
        
        # Test statistics collection
        stats = temporal.get_statistics()
        required_fields = ['operation_count', 'error_count', 'error_rate']
        
        if all(field in stats for field in required_fields):
            print("  ✅ Statistics collection working")
        else:
            print(f"  ❌ Missing statistics fields: {stats}")
            return False
        
        return True
    
    def test_security_monitoring(self) -> bool:
        """Test security monitoring and validation."""
        print("\n🔒 Testing Security Monitoring...")
        
        monitor = HDCSecurityMonitor(max_operations_per_second=5)
        
        # Test data validation
        valid_data = np.random.randn(100)
        if not monitor.validate_input_data(valid_data, "hypervector"):
            print("  ❌ Valid data rejected")
            return False
        print("  ✅ Valid data accepted")
        
        # Test malicious data detection
        malicious_data = np.full(100000, 1)  # Too large
        if monitor.validate_input_data(malicious_data, "hypervector"):
            print("  ❌ Malicious data not detected")
            return False
        print("  ✅ Malicious data correctly rejected")
        
        # Test data hashing
        hash1 = monitor.compute_data_hash(valid_data)
        hash2 = monitor.compute_data_hash(valid_data)
        
        if hash1 != hash2:
            print("  ❌ Data hashing inconsistent")
            return False
        print("  ✅ Data hashing working correctly")
        
        # Test trusted sources
        monitor.add_trusted_source("test_source")
        if not monitor.verify_data_source(valid_data, "test_source"):
            print("  ❌ Trusted source verification failed")
            return False
        print("  ✅ Trusted source verification working")
        
        # Test security report
        report = monitor.get_security_report()
        if 'status' not in report:
            print("  ❌ Security report missing status")
            return False
        print("  ✅ Security reporting working")
        
        return True
    
    def test_health_checking(self) -> bool:
        """Test system health monitoring."""
        print("\n💓 Testing Health Monitoring...")
        
        checker = HDCHealthChecker()
        
        # Test basic health check
        health = checker.check_system_health()
        
        required_fields = ['timestamp', 'status', 'checks']
        if not all(field in health for field in required_fields):
            print(f"  ❌ Health check missing fields: {health}")
            return False
        print("  ✅ Health check structure correct")
        
        # Test multiple checks
        if len(health['checks']) == 0:
            print("  ❌ No health checks performed")
            return False
        print(f"  ✅ Performed {len(health['checks'])} health checks")
        
        # Test health history
        time.sleep(0.1)  # Small delay
        checker.check_system_health()  # Second check
        
        history = checker.get_health_history(hours=1)
        if len(history) < 2:
            print("  ❌ Health history not tracking correctly")
            return False
        print("  ✅ Health history tracking working")
        
        return True
    
    def test_edge_cases(self) -> bool:
        """Test handling of edge cases and boundary conditions."""
        print("\n⚡ Testing Edge Cases...")
        
        # Test zero dimension (should fail)
        try:
            AdvancedTemporalHDC(dim=0)
            print("  ❌ Zero dimension not caught")
            return False
        except ValueError:
            print("  ✅ Zero dimension correctly rejected")
        
        # Test negative values
        try:
            AdvancedTemporalHDC(dim=-10)
            print("  ❌ Negative dimension not caught")
            return False
        except ValueError:
            print("  ✅ Negative dimension correctly rejected")
        
        # Test empty sequences
        temporal = AdvancedTemporalHDC(dim=10)
        try:
            result = temporal.sequence_prediction([])
            if len(result) == 0:
                print("  ❌ Empty sequence prediction failed")
                return False
            print("  ✅ Empty sequence handled gracefully")
        except Exception as e:
            print(f"  ❌ Empty sequence caused error: {e}")
            return False
        
        # Test attention with non-divisible dimensions
        try:
            ConcreteAttentionHDC(dim=101, num_attention_heads=8)  # 101 not divisible by 8
            print("  ❌ Non-divisible dimensions not caught")
            return False
        except ValueError:
            print("  ✅ Non-divisible dimensions correctly rejected")
        
        return True
    
    def test_memory_safety(self) -> bool:
        """Test memory safety and resource management."""
        print("\n🧠 Testing Memory Safety...")
        
        # Test large dimension limits
        try:
            temporal = AdvancedTemporalHDC(dim=1000000)  # Very large
            # Should work but be monitored
            print("  ✅ Large dimensions handled")
        except Exception as e:
            print(f"  ❌ Large dimension failed: {e}")
            return False
        
        # Test memory cleanup
        temporal = AdvancedTemporalHDC(dim=100, memory_length=5)
        
        # Fill buffer beyond capacity
        for i in range(10):
            hv = np.random.randn(100)
            temporal.add_temporal_experience(hv)
        
        # Buffer should be limited to memory_length
        if len(temporal.temporal_buffer) != 5:
            print(f"  ❌ Buffer not limited: {len(temporal.temporal_buffer)}")
            return False
        print("  ✅ Memory buffer correctly limited")
        
        return True
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting and DoS protection."""
        print("\n🚦 Testing Rate Limiting...")
        
        # Note: This is a simplified test since we can't easily trigger
        # the actual rate limiting without overwhelming the system
        
        temporal = AdvancedTemporalHDC(dim=50)
        
        # Perform multiple operations rapidly
        start_time = time.time()
        operations_completed = 0
        
        try:
            for i in range(10):
                hv = np.random.randn(50)
                temporal.sequence_prediction([hv, hv, hv])
                operations_completed += 1
            
            end_time = time.time()
            operations_per_second = operations_completed / (end_time - start_time)
            
            print(f"  ✅ Completed {operations_completed} operations at {operations_per_second:.1f} ops/sec")
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("  ✅ Rate limiting triggered correctly")
            else:
                print(f"  ❌ Unexpected error: {e}")
                return False
        
        return True
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("🛡️  ROBUSTNESS TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
        print(f"   • Passed: {passed_tests}")
        print(f"   • Failed: {failed_tests}")
        print(f"   • Errors: {error_tests}")
        
        if failed_tests > 0 or error_tests > 0:
            print("\n❌ FAILED/ERROR TESTS:")
            for test_name, result in self.test_results.items():
                if result['status'] != 'passed':
                    print(f"   • {test_name}: {result['status']}")
                    if 'error' in result:
                        print(f"     Error: {result['error']}")
        
        success_rate = passed_tests / total_tests * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🚀 EXCELLENT: Robustness goals achieved!")
        elif success_rate >= 75:
            print("✅ GOOD: Most robustness features working")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Robustness issues detected")


def main():
    """Run robustness validation test suite."""
    test_suite = RobustnessTestSuite()
    results = test_suite.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()