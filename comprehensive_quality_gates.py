#!/usr/bin/env python3
"""
Comprehensive Quality Gates: Testing, Security, Performance, and Validation
"""

import sys
import time
import traceback
import logging
import subprocess
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import our HDC systems
from robust_hdc_system import RobustHDCSystem, HDCValidationError, HDCSecurityError
from scalable_hdc_system import DistributedHDCProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, score: float = 0.0, 
                 details: Dict[str, Any] = None, recommendations: List[str] = None):
        self.name = name
        self.passed = passed
        self.score = score  # 0.0 - 1.0
        self.details = details or {}
        self.recommendations = recommendations or []
        self.timestamp = time.time()

class ComprehensiveTestSuite:
    """Comprehensive testing suite for HDC functionality."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    def run_functionality_tests(self) -> QualityGateResult:
        """Test core HDC functionality."""
        try:
            logger.info("Running functionality tests...")
            
            test_cases = 0
            passed_cases = 0
            details = {}
            
            # Test 1: Basic operations
            system = RobustHDCSystem(dim=1000)
            hv1 = system.generate_random_hv(sparsity=0.3)
            hv2 = system.generate_random_hv(sparsity=0.7)
            
            # Bundle test
            bundled = system.bundle_hypervectors([hv1, hv2])
            test_cases += 1
            if bundled['dim'] == 1000:
                passed_cases += 1
                details['bundle_test'] = 'PASS'
            else:
                details['bundle_test'] = 'FAIL - dimension mismatch'
            
            # Bind test
            bound = system.bind_hypervectors(hv1, hv2)
            test_cases += 1
            if bound['dim'] == 1000:
                passed_cases += 1
                details['bind_test'] = 'PASS'
            else:
                details['bind_test'] = 'FAIL - dimension mismatch'
            
            # Similarity test
            sim = system.cosine_similarity_robust(hv1, hv2)
            test_cases += 1
            if -1.0 <= sim <= 1.0:
                passed_cases += 1
                details['similarity_test'] = f'PASS - similarity: {sim:.4f}'
            else:
                details['similarity_test'] = f'FAIL - invalid similarity: {sim}'
            
            # Test 2: Error handling
            try:
                system.generate_random_hv(sparsity=2.0)  # Should fail
                details['error_handling'] = 'FAIL - should have raised validation error'
            except HDCValidationError:
                test_cases += 1
                passed_cases += 1
                details['error_handling'] = 'PASS - validation error caught'
            
            # Test 3: Edge cases
            # Zero hypervector
            zero_hv = {'data': [0.0] * 1000, 'dim': 1000, 'checksum': 'test'}
            normal_hv = hv1
            zero_sim = system.cosine_similarity_robust(zero_hv, normal_hv)
            test_cases += 1
            if zero_sim == 0.0:
                passed_cases += 1
                details['zero_vector_test'] = 'PASS'
            else:
                details['zero_vector_test'] = f'FAIL - expected 0.0, got {zero_sim}'
            
            # Test 4: Dimension validation
            try:
                system_bad = RobustHDCSystem(dim=0)  # Should fail
                details['dimension_validation'] = 'FAIL - should reject zero dimension'
            except HDCValidationError:
                test_cases += 1
                passed_cases += 1
                details['dimension_validation'] = 'PASS'
            
            score = passed_cases / test_cases if test_cases > 0 else 0.0
            passed = score >= 0.85  # Require 85% pass rate
            
            logger.info(f"Functionality tests: {passed_cases}/{test_cases} passed ({score:.2%})")
            
            return QualityGateResult(
                name="Functionality Tests",
                passed=passed,
                score=score,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Functionality tests failed: {e}")
            return QualityGateResult(
                name="Functionality Tests",
                passed=False,
                score=0.0,
                details={'error': str(e)}
            )
    
    def run_performance_tests(self) -> QualityGateResult:
        """Test system performance benchmarks."""
        try:
            logger.info("Running performance tests...")
            
            processor = DistributedHDCProcessor(max_workers=4)
            details = {}
            recommendations = []
            
            # Generate test data
            test_hvs = []
            for i in range(1000):
                hv_data = [1.0 if (i + j) % 3 == 0 else -1.0 for j in range(500)]
                hv = {
                    'data': hv_data,
                    'dim': 500,
                    'checksum': f"perf_test_{i}"
                }
                test_hvs.append(hv)
            
            # Performance benchmarks
            benchmarks = {}
            
            # 1. Bundle performance
            start_time = time.time()
            bundled = processor.parallel_bundle(test_hvs[:100])
            bundle_time = time.time() - start_time
            bundle_throughput = 100 / bundle_time
            benchmarks['bundle_throughput_hvs_per_sec'] = bundle_throughput
            details['bundle_time'] = f"{bundle_time:.3f}s"
            
            # 2. Search performance  
            start_time = time.time()
            search_results = processor.batch_search(test_hvs[0], test_hvs[1:500], top_k=10)
            search_time = time.time() - start_time
            search_throughput = 500 / search_time
            benchmarks['search_throughput_hvs_per_sec'] = search_throughput
            details['search_time'] = f"{search_time:.3f}s"
            
            # 3. Similarity matrix performance
            start_time = time.time()
            sim_matrix = processor.parallel_similarity_matrix(test_hvs[:25])
            matrix_time = time.time() - start_time
            comparisons = 25 * 25
            matrix_throughput = comparisons / matrix_time
            benchmarks['similarity_matrix_comparisons_per_sec'] = matrix_throughput
            details['matrix_time'] = f"{matrix_time:.3f}s"
            
            # Performance thresholds (items per second)
            thresholds = {
                'bundle_throughput_hvs_per_sec': 1000,      # 1k HVs/sec
                'search_throughput_hvs_per_sec': 5000,      # 5k HVs/sec  
                'similarity_matrix_comparisons_per_sec': 10000  # 10k comparisons/sec
            }
            
            # Evaluate performance
            passed_benchmarks = 0
            total_benchmarks = len(thresholds)
            
            for metric, value in benchmarks.items():
                threshold = thresholds.get(metric, 0)
                if value >= threshold:
                    passed_benchmarks += 1
                    details[f'{metric}_status'] = f'PASS ({value:.0f} >= {threshold})'
                else:
                    details[f'{metric}_status'] = f'FAIL ({value:.0f} < {threshold})'
                    recommendations.append(f"Improve {metric}: current {value:.0f}, target {threshold}")
            
            score = passed_benchmarks / total_benchmarks
            passed = score >= 0.7  # Require 70% of benchmarks to pass
            
            logger.info(f"Performance tests: {passed_benchmarks}/{total_benchmarks} benchmarks passed")
            
            return QualityGateResult(
                name="Performance Tests",
                passed=passed,
                score=score,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return QualityGateResult(
                name="Performance Tests", 
                passed=False,
                score=0.0,
                details={'error': str(e)}
            )
    
    def run_security_tests(self) -> QualityGateResult:
        """Test security measures and vulnerability resistance."""
        try:
            logger.info("Running security tests...")
            
            system = RobustHDCSystem(dim=1000)
            test_cases = 0
            passed_cases = 0
            details = {}
            recommendations = []
            
            # Test 1: Input validation
            malicious_inputs = [
                -1000,      # Negative dimension
                0,          # Zero dimension  
                1000000,    # Excessive dimension
                "invalid",  # Wrong type
            ]
            
            for i, bad_input in enumerate(malicious_inputs):
                try:
                    test_cases += 1
                    if isinstance(bad_input, str):
                        # This would fail at type check level
                        continue
                    else:
                        system_test = RobustHDCSystem(dim=bad_input)
                    details[f'input_validation_{i}'] = 'FAIL - should have rejected input'
                except (HDCValidationError, HDCSecurityError, TypeError):
                    passed_cases += 1
                    details[f'input_validation_{i}'] = 'PASS - rejected malicious input'
            
            # Test 2: Resource limits
            try:
                test_cases += 1
                system_huge = RobustHDCSystem(dim=150000)  # Beyond security limit
                details['resource_limits'] = 'FAIL - should have enforced security limit'
            except HDCSecurityError:
                passed_cases += 1
                details['resource_limits'] = 'PASS - security limit enforced'
            
            # Test 3: Data integrity (checksums)
            hv1 = system.generate_random_hv()
            original_checksum = hv1['checksum']
            
            test_cases += 1
            if len(original_checksum) >= 8:  # Reasonable checksum length
                passed_cases += 1
                details['checksum_generation'] = f'PASS - checksum length: {len(original_checksum)}'
            else:
                details['checksum_generation'] = f'FAIL - weak checksum: {len(original_checksum)} chars'
                recommendations.append("Strengthen checksum algorithm")
            
            # Test 4: Memory safety checks
            # Try to create many large hypervectors (should be handled gracefully)
            test_cases += 1
            try:
                large_hvs = []
                for i in range(100):
                    hv = system.generate_random_hv()
                    large_hvs.append(hv)
                
                # This should work without crashes
                bundled = system.bundle_hypervectors(large_hvs[:50])
                passed_cases += 1
                details['memory_safety'] = 'PASS - handled large operations'
            except Exception as e:
                details['memory_safety'] = f'FAIL - memory safety issue: {e}'
                recommendations.append("Improve memory management")
            
            # Test 5: Secure random generation
            test_cases += 1
            hv1 = system.generate_random_hv(seed=None)  # Should use secure random
            hv2 = system.generate_random_hv(seed=None)
            
            # They should be different (extremely unlikely to be identical)
            if hv1['data'] != hv2['data']:
                passed_cases += 1
                details['secure_random'] = 'PASS - generates different vectors'
            else:
                details['secure_random'] = 'FAIL - suspicious identical vectors'
                recommendations.append("Verify random number generation")
            
            score = passed_cases / test_cases if test_cases > 0 else 0.0
            passed = score >= 0.8  # Require 80% security tests to pass
            
            logger.info(f"Security tests: {passed_cases}/{test_cases} passed ({score:.2%})")
            
            return QualityGateResult(
                name="Security Tests",
                passed=passed,
                score=score,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Security tests failed: {e}")
            return QualityGateResult(
                name="Security Tests",
                passed=False,
                score=0.0,
                details={'error': str(e)}
            )
    
    def run_integration_tests(self) -> QualityGateResult:
        """Test integration between different system components."""
        try:
            logger.info("Running integration tests...")
            
            # Test integration between robust and scalable systems
            robust_system = RobustHDCSystem(dim=500)
            scalable_processor = DistributedHDCProcessor()
            
            test_cases = 0
            passed_cases = 0
            details = {}
            
            # Test 1: Data compatibility
            test_cases += 1
            robust_hv = robust_system.generate_random_hv(sparsity=0.4)
            
            # Test if scalable system can process robust system's hypervectors
            try:
                compatible_hvs = [robust_hv] * 10
                bundled = scalable_processor.parallel_bundle(compatible_hvs)
                
                if bundled['dim'] == robust_hv['dim']:
                    passed_cases += 1
                    details['data_compatibility'] = 'PASS - systems compatible'
                else:
                    details['data_compatibility'] = 'FAIL - dimension mismatch'
            except Exception as e:
                details['data_compatibility'] = f'FAIL - integration error: {e}'
            
            # Test 2: Performance consistency
            test_cases += 1
            
            # Generate same data with both systems
            test_data = []
            for i in range(50):
                hv = robust_system.generate_random_hv(sparsity=0.5, seed=42 + i)
                test_data.append(hv)
            
            # Bundle with robust system
            start_time = time.time()
            robust_bundle = robust_system.bundle_hypervectors(test_data[:10])
            robust_time = time.time() - start_time
            
            # Bundle with scalable system
            start_time = time.time()
            scalable_bundle = scalable_processor.parallel_bundle(test_data[:10])
            scalable_time = time.time() - start_time
            
            # Check if results are reasonable (don't need to be identical due to different algorithms)
            robust_norm = sum(x*x for x in robust_bundle['data']) ** 0.5
            scalable_norm = sum(x*x for x in scalable_bundle['data']) ** 0.5
            
            if abs(robust_norm - scalable_norm) / max(robust_norm, scalable_norm) < 0.1:
                passed_cases += 1
                details['performance_consistency'] = f'PASS - norms: {robust_norm:.3f} vs {scalable_norm:.3f}'
            else:
                details['performance_consistency'] = f'FAIL - significant difference: {robust_norm:.3f} vs {scalable_norm:.3f}'
            
            details['timing_robust'] = f"{robust_time:.3f}s"
            details['timing_scalable'] = f"{scalable_time:.3f}s"
            
            # Test 3: Error propagation
            test_cases += 1
            try:
                # Create incompatible data
                bad_hv = {'data': [1.0] * 100, 'dim': 100, 'checksum': 'bad'}
                mixed_data = test_data[:5] + [bad_hv]
                
                # Should handle gracefully
                result = scalable_processor.parallel_bundle(mixed_data)
                details['error_propagation'] = 'PASS - handled mixed data'
                passed_cases += 1
            except Exception:
                details['error_propagation'] = 'FAIL - did not handle mixed data gracefully'
            
            score = passed_cases / test_cases if test_cases > 0 else 0.0
            passed = score >= 0.7  # Require 70% integration tests to pass
            
            logger.info(f"Integration tests: {passed_cases}/{test_cases} passed ({score:.2%})")
            
            return QualityGateResult(
                name="Integration Tests",
                passed=passed,
                score=score,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return QualityGateResult(
                name="Integration Tests",
                passed=False,
                score=0.0,
                details={'error': str(e)}
            )

class QualityGateOrchestrator:
    """Orchestrates and evaluates all quality gates."""
    
    def __init__(self):
        self.test_suite = ComprehensiveTestSuite()
        self.results = []
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        logger.info("Starting comprehensive quality gate evaluation...")
        
        start_time = time.time()
        
        # Define quality gates
        gates = [
            ("Functionality", self.test_suite.run_functionality_tests),
            ("Performance", self.test_suite.run_performance_tests), 
            ("Security", self.test_suite.run_security_tests),
            ("Integration", self.test_suite.run_integration_tests),
        ]
        
        # Run gates in parallel where possible
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_gate = {
                executor.submit(gate_func): gate_name 
                for gate_name, gate_func in gates
            }
            
            for future in as_completed(future_to_gate):
                gate_name = future_to_gate[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(f"{gate_name} gate: {'✅ PASS' if result.passed else '❌ FAIL'} (score: {result.score:.2%})")
                except Exception as exc:
                    logger.error(f"{gate_name} gate failed with exception: {exc}")
                    self.results.append(QualityGateResult(
                        name=gate_name,
                        passed=False,
                        score=0.0,
                        details={'exception': str(exc)}
                    ))
        
        total_time = time.time() - start_time
        
        # Calculate overall results
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        average_score = sum(r.score for r in self.results) / total_gates if total_gates > 0 else 0.0
        
        # Overall pass criteria: all critical gates pass + average score >= 75%
        critical_gates = ["Functionality", "Security"]
        critical_passed = all(
            r.passed for r in self.results 
            if r.name in critical_gates
        )
        
        overall_pass = critical_passed and average_score >= 0.75
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        report = {
            'overall_pass': overall_pass,
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'average_score': average_score,
            'execution_time': total_time,
            'critical_gates_passed': critical_passed,
            'gate_results': {r.name: {
                'passed': r.passed,
                'score': r.score,
                'details': r.details
            } for r in self.results},
            'recommendations': all_recommendations,
            'timestamp': time.time()
        }
        
        return report
    
    def generate_report_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable report summary."""
        status = "🎉 ALL GATES PASSED" if report['overall_pass'] else "⚠️ QUALITY ISSUES FOUND"
        
        summary = f"""
🛡️ COMPREHENSIVE QUALITY GATES REPORT
{'='*60}

{status}

📊 OVERALL RESULTS:
   Gates Passed: {report['passed_gates']}/{report['total_gates']}
   Average Score: {report['average_score']:.1%}
   Execution Time: {report['execution_time']:.2f}s
   Critical Gates: {'✅ PASSED' if report['critical_gates_passed'] else '❌ FAILED'}

📋 GATE BREAKDOWN:
"""
        
        for gate_name, result in report['gate_results'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            summary += f"   {status} {gate_name}: {result['score']:.1%}\n"
        
        if report['recommendations']:
            summary += f"\n🔧 RECOMMENDATIONS:\n"
            for i, rec in enumerate(report['recommendations'][:10], 1):  # Top 10 recommendations
                summary += f"   {i}. {rec}\n"
        
        if report['overall_pass']:
            summary += f"\n🚀 PRODUCTION READINESS: APPROVED"
        else:
            summary += f"\n⚠️ PRODUCTION READINESS: REQUIRES FIXES"
        
        return summary

def main():
    """Run comprehensive quality gates evaluation."""
    logger.info("Initializing comprehensive quality gates...")
    
    orchestrator = QualityGateOrchestrator()
    
    try:
        # Run all quality gates
        report = orchestrator.run_all_gates()
        
        # Generate and display report
        summary = orchestrator.generate_report_summary(report)
        print(summary)
        
        # Save detailed report
        with open('/root/repo/quality_gates_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Quality gates report saved to quality_gates_report.json")
        
        return report['overall_pass']
        
    except Exception as e:
        logger.error(f"Quality gates evaluation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)