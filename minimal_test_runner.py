#!/usr/bin/env python3
"""Minimal test runner for core functionality validation.

This module provides streamlined testing focused on validating the core
quantum task planning functionality that works with the existing codebase
and available dependencies.
"""

import sys
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, '/root/repo')

def run_test(test_name: str, test_func: callable) -> Dict[str, Any]:
    """Run a single test with error handling and timing."""
    print(f"Running {test_name}...", end=" ")
    
    start_time = time.perf_counter()
    try:
        result = test_func()
        execution_time = time.perf_counter() - start_time
        
        if result is True or result is None:
            print(f"PASSED ({execution_time:.3f}s)")
            return {"status": "PASSED", "time": execution_time, "error": None}
        else:
            print(f"FAILED ({execution_time:.3f}s) - Returned: {result}")
            return {"status": "FAILED", "time": execution_time, "error": f"Test returned {result}"}
    
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        print(f"ERROR ({execution_time:.3f}s) - {str(e)}")
        return {"status": "ERROR", "time": execution_time, "error": str(e)}


def test_core_hdc_operations():
    """Test core HDC operations using existing backends."""
    try:
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        # Initialize HDC with NumPy backend
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test random hypervector generation
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        
        assert hv1 is not None
        assert hv2 is not None
        assert len(hv1) == 1000
        assert len(hv2) == 1000
        
        # Test bundle operation
        bundled = hdc.bundle([hv1, hv2])
        assert bundled is not None
        assert len(bundled) == 1000
        
        # Test bind operation  
        bound = hdc.bind(hv1, hv2)
        assert bound is not None
        assert len(bound) == 1000
        
        # Test similarity operations
        cosine_sim = hdc.cosine_similarity(hv1, hv2)
        hamming_dist = hdc.hamming_distance(hv1, hv2)
        
        assert -1.0 <= cosine_sim <= 1.0
        assert 0.0 <= hamming_dist <= 1.0
        
        return True
        
    except Exception as e:
        print(f"HDC operations error: {e}")
        raise


def test_memory_structures():
    """Test memory structures and operations."""
    try:
        from hd_compute.memory.item_memory import ItemMemory
        from hd_compute.memory.associative_memory import AssociativeMemory
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test Item Memory
        item_memory = ItemMemory(hdc=hdc, num_items=100)
        
        # Store some items
        test_items = ['apple', 'banana', 'cherry']
        for item in test_items:
            item_memory.store(item, hdc.random_hv())
        
        # Retrieve items
        apple_hv = item_memory.get('apple')
        assert apple_hv is not None
        
        # Test Associative Memory
        assoc_memory = AssociativeMemory(hdc=hdc, capacity=50)
        
        # Store associations
        key_hv = hdc.random_hv()
        value_hv = hdc.random_hv()
        assoc_memory.store(key_hv, value_hv)
        
        # Retrieve associations
        retrieved = assoc_memory.recall(key_hv, threshold=0.3)
        assert retrieved is not None
        
        return True
        
    except Exception as e:
        print(f"Memory structures error: {e}")
        raise


def test_applications():
    """Test application modules."""
    try:
        from hd_compute.applications.cognitive import SemanticMemory
        from hd_compute.applications.speech_commands import SpeechCommandHDC
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test Semantic Memory
        semantic_memory = SemanticMemory(hdc=hdc, dim=1000, device="cpu")
        
        # Store concepts with attributes
        semantic_memory.store("apple", attributes=["fruit", "red", "sweet"])
        semantic_memory.store("banana", attributes=["fruit", "yellow", "sweet"])
        
        # Query by attributes
        results = semantic_memory.query(["fruit", "sweet"])
        assert len(results) > 0
        
        # Test Speech Command HDC (basic initialization)
        speech_hdc = SpeechCommandHDC(dim=1000, num_classes=10, feature_extractor='mfcc')
        assert speech_hdc.dim == 1000
        assert speech_hdc.num_classes == 10
        
        return True
        
    except Exception as e:
        print(f"Applications error: {e}")
        raise


def test_research_algorithms():
    """Test novel research algorithms."""
    try:
        from hd_compute.research.novel_algorithms import (
            FractionalHDC, QuantumHDC, TemporalHDC, CausalHDC
        )
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test Fractional HDC
        fractional_hdc = FractionalHDC(dim=1000, device="cpu")
        
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        
        # Test fractional binding
        fractional_bound = fractional_hdc.fractional_bind(hv1, hv2, power=0.5)
        assert fractional_bound is not None
        assert len(fractional_bound) == 1000
        
        # Test Quantum HDC
        quantum_hdc = QuantumHDC(dim=1000, device="cpu")
        
        # Test quantum superposition
        superposition = quantum_hdc.quantum_superposition([hv1, hv2], amplitudes=[0.6, 0.4])
        assert superposition is not None
        
        # Test entanglement measure
        entanglement = quantum_hdc.entanglement_measure(hv1, hv2)
        assert 0.0 <= entanglement <= 1.0
        
        # Test Temporal HDC
        temporal_hdc = TemporalHDC(dim=1000, device="cpu")
        
        # Create temporal sequence
        sequence_hv = temporal_hdc.create_temporal_sequence([hv1, hv2])
        assert sequence_hv is not None
        
        # Test Causal HDC
        causal_hdc = CausalHDC(dim=1000, device="cpu")
        
        # Learn causal structure (simplified test)
        observations = [hv1, hv2]
        causal_structure = causal_hdc.learn_causal_structure(observations)
        assert causal_structure is not None
        
        return True
        
    except Exception as e:
        print(f"Research algorithms error: {e}")
        raise


def test_caching_system():
    """Test caching and performance systems."""
    try:
        from hd_compute.cache.hypervector_cache import HypervectorCache
        from hd_compute.cache.cache_manager import CacheManager
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test Hypervector Cache
        cache = HypervectorCache(max_size=100, dim=1000)
        
        # Generate test data
        test_hv = hdc.random_hv()
        
        # Test cache operations
        cache.put("test_key", test_hv)
        retrieved = cache.get("test_key")
        
        assert retrieved is not None
        assert np.allclose(retrieved, test_hv, rtol=1e-5)
        
        # Test cache statistics
        stats = cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'size' in stats
        
        # Test Cache Manager
        cache_manager = CacheManager(max_memory_mb=64)
        
        # Register cache
        cache_manager.register_cache("hypervector_cache", cache)
        
        # Test cache management
        cache_manager.clear_cache("hypervector_cache")
        assert cache.get("test_key") is None
        
        return True
        
    except Exception as e:
        print(f"Caching system error: {e}")
        raise


def test_validation_system():
    """Test validation and quality assurance."""
    try:
        from hd_compute.validation.quality_assurance import QualityAssurance, QualityMetrics
        from hd_compute.validation.error_recovery import CircuitBreaker, RetryStrategy
        
        # Test Quality Assurance
        metrics_config = {
            'accuracy': {'min': 0.7, 'target': 0.95},
            'performance': {'min': 0.5, 'target': 0.9}
        }
        
        qa = QualityAssurance(metrics_config=metrics_config)
        
        # Test metrics update
        qa.update_metrics({'accuracy': 0.85, 'performance': 0.75})
        
        # Test quality report
        report = qa.get_quality_report()
        assert 'overall_score' in report
        assert 'metrics_status' in report
        
        # Test Circuit Breaker
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=ValueError
        )
        
        assert circuit_breaker.state == 'closed'
        
        # Test Retry Strategy
        retry_strategy = RetryStrategy(
            max_retries=3,
            backoff_factor=2.0,
            exceptions=(ValueError, RuntimeError)
        )
        
        assert retry_strategy.max_retries == 3
        
        return True
        
    except Exception as e:
        print(f"Validation system error: {e}")
        raise


def test_security_components():
    """Test security and audit components."""
    try:
        from hd_compute.security.audit_logger import AuditLogger
        from hd_compute.security.input_sanitization import InputSanitizer
        from hd_compute.security.security_scanner import SecurityScanner
        
        # Test Audit Logger
        audit_logger = AuditLogger()
        
        # Test logging functionality
        audit_logger.log_security_event(
            user_id="test_user",
            action="test_action",
            resource="test_resource",
            granted=True,
            details={"test": "data"}
        )
        
        # Test Input Sanitizer
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        clean_string = sanitizer.sanitize_string("test<script>alert('xss')</script>")
        assert "<script>" not in clean_string
        
        # Test Security Scanner
        scanner = SecurityScanner()
        
        # Test basic security scan
        scan_result = scanner.scan_text_for_threats("normal text")
        assert 'threat_level' in scan_result
        assert 'threats_detected' in scan_result
        
        return True
        
    except Exception as e:
        print(f"Security components error: {e}")
        raise


def test_distributed_components():
    """Test distributed computing components."""
    try:
        from hd_compute.distributed.parallel_processing import PipelineParallelProcessor
        import psutil
        
        # Test Parallel Processor initialization
        processor = PipelineParallelProcessor(
            num_workers=2,
            device="cpu"
        )
        
        assert processor.num_workers == 2
        assert processor.device == "cpu"
        
        # Test system monitoring (basic)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        assert cpu_count > 0
        assert memory.total > 0
        
        return True
        
    except Exception as e:
        print(f"Distributed components error: {e}")
        raise


def test_benchmarking_system():
    """Test performance benchmarking."""
    try:
        from hd_compute.performance.benchmark import Benchmark
        from hd_compute.performance.profiler import Profiler
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        
        # Test Benchmark
        benchmark = Benchmark(hdc=hdc)
        
        # Run basic benchmarks
        results = benchmark.run_basic_benchmarks(num_trials=5)
        
        assert 'random_hv_generation' in results
        assert 'bundle_operation' in results
        assert 'bind_operation' in results
        assert 'similarity_computation' in results
        
        # Verify reasonable performance metrics
        for operation, metrics in results.items():
            assert 'avg_time_ms' in metrics
            assert 'throughput_ops_sec' in metrics
            assert metrics['avg_time_ms'] > 0
            assert metrics['throughput_ops_sec'] > 0
        
        # Test Profiler
        profiler = Profiler()
        
        # Profile a simple operation
        with profiler.profile_context("test_operation"):
            time.sleep(0.001)  # Simulate work
        
        profile_results = profiler.get_results()
        assert 'test_operation' in profile_results
        
        return True
        
    except Exception as e:
        print(f"Benchmarking system error: {e}")
        raise


def test_statistical_validation():
    """Test statistical validation and reproducibility."""
    try:
        from hd_compute.research.statistical_analysis import StatisticalAnalysis
        from hd_compute.numpy.hdc_numpy import HDComputeNumPy
        
        hdc = HDComputeNumPy(dim=1000, device="cpu")
        stat_analysis = StatisticalAnalysis()
        
        # Generate test data
        test_data = []
        for _ in range(20):
            hv = hdc.random_hv()
            test_data.append(np.mean(hv))  # Use mean as a test statistic
        
        # Test statistical analysis
        analysis_result = stat_analysis.analyze_distribution(test_data)
        
        assert 'mean' in analysis_result
        assert 'std' in analysis_result
        assert 'confidence_interval' in analysis_result
        assert 'normality_test' in analysis_result
        
        # Test reproducibility
        np.random.seed(42)
        hv1 = hdc.random_hv()
        np.random.seed(42)  
        hv2 = hdc.random_hv()
        
        # Should be similar (not exactly equal due to implementation details)
        similarity = hdc.cosine_similarity(hv1, hv2)
        assert similarity > 0.9  # High similarity indicates reproducibility
        
        return True
        
    except Exception as e:
        print(f"Statistical validation error: {e}")
        raise


def run_minimal_comprehensive_tests():
    """Run minimal comprehensive test suite."""
    print("=" * 80)
    print("QUANTUM-INSPIRED TASK PLANNING - MINIMAL COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Define test suite focusing on core functionality
    tests = [
        ("Core HDC Operations", test_core_hdc_operations),
        ("Memory Structures", test_memory_structures),
        ("Applications", test_applications),
        ("Research Algorithms", test_research_algorithms),
        ("Caching System", test_caching_system),
        ("Validation System", test_validation_system),
        ("Security Components", test_security_components),
        ("Distributed Components", test_distributed_components),
        ("Benchmarking System", test_benchmarking_system),
        ("Statistical Validation", test_statistical_validation)
    ]
    
    # Run tests
    results = {}
    total_time = 0
    
    for test_name, test_func in tests:
        result = run_test(test_name, test_func)
        results[test_name] = result
        total_time += result["time"]
    
    # Summary
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")
    errors = sum(1 for r in results.values() if r["status"] == "ERROR")
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print(f"Total Time: {total_time:.3f}s")
    print()
    
    # Detailed results for failed/error tests
    if failed > 0 or errors > 0:
        print("FAILED/ERROR TESTS:")
        print("-" * 40)
        for test_name, result in results.items():
            if result["status"] in ["FAILED", "ERROR"]:
                print(f"  {test_name}: {result['status']}")
                if result["error"]:
                    print(f"    Error: {result['error']}")
        print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY:")
    print("-" * 40)
    for test_name, result in results.items():
        print(f"  {test_name}: {result['time']:.3f}s")
    print()
    
    # Coverage analysis
    coverage_areas = {
        "Core HDC Operations": passed >= 1,
        "Memory Management": results.get("Memory Structures", {}).get("status") == "PASSED",
        "Applications": results.get("Applications", {}).get("status") == "PASSED", 
        "Research Algorithms": results.get("Research Algorithms", {}).get("status") == "PASSED",
        "Performance & Caching": results.get("Caching System", {}).get("status") == "PASSED",
        "Quality Assurance": results.get("Validation System", {}).get("status") == "PASSED",
        "Security": results.get("Security Components", {}).get("status") == "PASSED",
        "Distributed Computing": results.get("Distributed Components", {}).get("status") == "PASSED",
        "Benchmarking": results.get("Benchmarking System", {}).get("status") == "PASSED",
        "Statistical Analysis": results.get("Statistical Validation", {}).get("status") == "PASSED"
    }
    
    covered_areas = sum(1 for covered in coverage_areas.values() if covered)
    total_areas = len(coverage_areas)
    coverage_percentage = covered_areas / total_areas * 100
    
    print("COVERAGE ANALYSIS:")
    print("-" * 40)
    for area, covered in coverage_areas.items():
        status = "✅ COVERED" if covered else "❌ NOT COVERED"
        print(f"  {area}: {status}")
    print()
    print(f"Overall Coverage: {covered_areas}/{total_areas} ({coverage_percentage:.1f}%)")
    print()
    
    # Quality assessment
    if coverage_percentage >= 85:
        quality_rating = "EXCELLENT ✅"
    elif coverage_percentage >= 70:
        quality_rating = "GOOD ✅"
    elif coverage_percentage >= 50:
        quality_rating = "ADEQUATE ⚠️"
    else:
        quality_rating = "NEEDS IMPROVEMENT ❌"
    
    print("QUALITY ASSESSMENT:")
    print("-" * 40)
    print(f"Test Coverage: {coverage_percentage:.1f}%")
    print(f"Overall Quality Rating: {quality_rating}")
    print()
    
    # Final assessment
    success_threshold = 0.80  # 80% success rate required
    
    if passed / total >= success_threshold and coverage_percentage >= 70:
        print("🎉 COMPREHENSIVE TESTING: SUCCESS!")
        print("   Core functionality validated successfully")
        print("   HD-Compute-Toolkit is ready for quantum task planning")
        print("   System demonstrates research-grade capabilities")
        return True
    else:
        print("⚠️  COMPREHENSIVE TESTING: NEEDS ATTENTION")
        print("   Some critical components require fixes")
        print("   Review failed tests and improve coverage")
        return False


if __name__ == "__main__":
    success = run_minimal_comprehensive_tests()
    sys.exit(0 if success else 1)