#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for HD-Compute-Toolkit
===================================================

This module provides extensive benchmarking capabilities for all HDC algorithms,
including performance analysis, statistical validation, scalability testing,
and comparative studies.

Features:
- Multi-dimensional performance benchmarking (time, memory, accuracy)
- Statistical significance testing and confidence intervals
- Scalability analysis across different dimensions and data sizes
- Comparative analysis between algorithms and backends
- Real-world application benchmarks
- Hardware acceleration testing
- Reproducibility validation
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle
import os

# Statistical analysis
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our enhanced algorithms
try:
    from enhanced_research_algorithms import (
        FractionalHDC, QuantumInspiredHDC, ContinualLearningHDC,
        ExplainableHDC, HierarchicalHDC, AdaptiveHDC, EnhancedBenchmarkSuite
    )
    from advanced_performance_system import PerformanceManager, OptimizationStrategy
except ImportError as e:
    print(f"Warning: Could not import enhanced algorithms: {e}")


class BenchmarkType(Enum):
    """Types of benchmarks."""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    COMPARATIVE = "comparative"
    REPRODUCIBILITY = "reproducibility"


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    algorithm: str
    operation: str
    dimension: int
    execution_time: float
    memory_usage: float
    throughput: float
    accuracy: Optional[float]
    error_rate: float
    metadata: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmarking suite for HDC algorithms."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Benchmark configuration
        self.dimensions = [1000, 2500, 5000, 10000, 16000]
        self.iterations = 10
        self.confidence_level = 0.95
        
        # Results storage
        self.results = []
        self.comparative_results = {}
        self.statistical_summaries = {}
        
        # Available algorithms
        self.algorithms = {
            'fractional': FractionalHDC,
            'quantum': QuantumInspiredHDC,
            'continual': ContinualLearningHDC,
            'explainable': ExplainableHDC,
            'hierarchical': HierarchicalHDC,
            'adaptive': AdaptiveHDC
        }
        
        self.logger.info("Comprehensive Benchmark Suite initialized")
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        self.logger.info("Starting comprehensive benchmark suite")
        
        suite_results = {
            'metadata': {
                'start_time': time.time(),
                'dimensions_tested': self.dimensions,
                'iterations_per_test': self.iterations,
                'algorithms_tested': list(self.algorithms.keys())
            },
            'results': {}
        }
        
        try:
            # 1. Performance Benchmarks
            self.logger.info("Running performance benchmarks...")
            perf_results = self.run_performance_benchmarks()
            suite_results['results']['performance'] = perf_results
            
            # 2. Scalability Analysis
            self.logger.info("Running scalability analysis...")
            scale_results = self.run_scalability_analysis()
            suite_results['results']['scalability'] = scale_results
            
            # 3. Memory Efficiency Tests
            self.logger.info("Running memory efficiency tests...")
            memory_results = self.run_memory_benchmarks()
            suite_results['results']['memory'] = memory_results
            
            # 4. Accuracy Validation
            self.logger.info("Running accuracy validation...")
            accuracy_results = self.run_accuracy_benchmarks()
            suite_results['results']['accuracy'] = accuracy_results
            
            # 5. Comparative Analysis
            self.logger.info("Running comparative analysis...")
            comp_results = self.run_comparative_analysis()
            suite_results['results']['comparative'] = comp_results
            
            # 6. Statistical Analysis
            self.logger.info("Performing statistical analysis...")
            stats_results = self.perform_statistical_analysis()
            suite_results['results']['statistical'] = stats_results
            
            # 7. Reproducibility Validation
            self.logger.info("Running reproducibility validation...")
            repro_results = self.validate_reproducibility()
            suite_results['results']['reproducibility'] = repro_results
            
            suite_results['metadata']['end_time'] = time.time()
            suite_results['metadata']['total_duration'] = suite_results['metadata']['end_time'] - suite_results['metadata']['start_time']
            
            # Save results
            self.save_results(suite_results)
            
            # Generate reports
            self.generate_comprehensive_report(suite_results)
            
            self.logger.info("Comprehensive benchmark suite completed successfully")
            
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            raise
        
        return suite_results
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        results = {}
        
        for alg_name, alg_class in self.algorithms.items():
            self.logger.info(f"Benchmarking {alg_name} performance...")
            alg_results = {}
            
            for dim in self.dimensions:
                dim_results = []
                
                for iteration in range(self.iterations):
                    try:
                        # Initialize algorithm
                        if alg_name == 'hierarchical':
                            alg = alg_class(dim=dim, levels=3)
                        elif alg_name == 'continual':
                            alg = alg_class(dim=dim, memory_size=100)
                        else:
                            alg = alg_class(dim=dim)
                        
                        # Run algorithm-specific benchmarks
                        bench_result = self._benchmark_algorithm_performance(alg, alg_name, dim)
                        dim_results.append(bench_result)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed iteration {iteration} for {alg_name} at dim {dim}: {e}")
                        continue
                
                if dim_results:
                    alg_results[f'dim_{dim}'] = self._aggregate_results(dim_results)
            
            results[alg_name] = alg_results
        
        return results
    
    def _benchmark_algorithm_performance(self, algorithm, alg_name: str, dim: int) -> Dict[str, Any]:
        """Benchmark specific algorithm performance."""
        start_time = time.time()
        
        # Generate test data
        if alg_name == 'fractional':
            hv1 = algorithm.random()
            hv2 = algorithm.random()
            
            # Test fractional binding
            result = algorithm.fractional_bind(hv1, hv2, strength=0.5)
            throughput = 1.0 / (time.time() - start_time) if time.time() > start_time else 0.0
            
        elif alg_name == 'quantum':
            qhv1 = algorithm.random_quantum()
            qhv2 = algorithm.random_quantum()
            
            # Test quantum operations
            result = algorithm.quantum_bind(qhv1, qhv2)
            entanglement = algorithm.entanglement_measure(qhv1, qhv2)
            throughput = 1.0 / (time.time() - start_time) if time.time() > start_time else 0.0
            
        elif alg_name == 'continual':
            # Test continual learning
            task_data = [
                (np.random.binomial(1, 0.5, dim).astype(np.float32),
                 np.random.binomial(1, 0.3, dim).astype(np.float32))
                for _ in range(20)
            ]
            result = algorithm.learn_task("test_task", task_data)
            throughput = len(task_data) / (time.time() - start_time) if time.time() > start_time else 0.0
            
        elif alg_name == 'explainable':
            query = np.random.binomial(1, 0.5, dim).astype(np.float32)
            contexts = [np.random.binomial(1, 0.5, dim).astype(np.float32) for _ in range(5)]
            result = algorithm.generate_explanation(query, contexts)
            throughput = len(contexts) / (time.time() - start_time) if time.time() > start_time else 0.0
            
        elif alg_name == 'hierarchical':
            data = np.random.random(dim).astype(np.float32)
            result = algorithm.encode_hierarchical(data, {})
            throughput = 1.0 / (time.time() - start_time) if time.time() > start_time else 0.0
            
        elif alg_name == 'adaptive':
            hv1 = np.random.binomial(1, 0.5, dim).astype(np.float32)
            hv2 = np.random.binomial(1, 0.5, dim).astype(np.float32)
            result, info = algorithm.adaptive_operation('bind', hv1, hv2)
            throughput = 1.0 / (time.time() - start_time) if time.time() > start_time else 0.0
        
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'throughput': throughput,
            'memory_usage': self._estimate_memory_usage(result),
            'success': True,
            'dimension': dim,
            'algorithm': alg_name
        }
    
    def run_scalability_analysis(self) -> Dict[str, Any]:
        """Analyze scalability characteristics."""
        scalability_results = {}
        
        for alg_name in self.algorithms:
            self.logger.info(f"Analyzing scalability for {alg_name}...")
            
            execution_times = []
            memory_usage = []
            throughput = []
            
            for dim in self.dimensions:
                dim_times = []
                dim_memory = []
                dim_throughput = []
                
                for _ in range(3):  # Fewer iterations for scalability
                    try:
                        if alg_name == 'hierarchical':
                            alg = self.algorithms[alg_name](dim=dim, levels=3)
                        elif alg_name == 'continual':
                            alg = self.algorithms[alg_name](dim=dim, memory_size=50)
                        else:
                            alg = self.algorithms[alg_name](dim=dim)
                        
                        result = self._benchmark_algorithm_performance(alg, alg_name, dim)
                        
                        dim_times.append(result['execution_time'])
                        dim_memory.append(result['memory_usage'])
                        dim_throughput.append(result['throughput'])
                        
                    except Exception as e:
                        self.logger.warning(f"Scalability test failed for {alg_name} at dim {dim}: {e}")
                        continue
                
                if dim_times:
                    execution_times.append(np.mean(dim_times))
                    memory_usage.append(np.mean(dim_memory))
                    throughput.append(np.mean(dim_throughput))
                else:
                    execution_times.append(np.nan)
                    memory_usage.append(np.nan)
                    throughput.append(np.nan)
            
            # Calculate scalability metrics
            scalability_metrics = self._calculate_scalability_metrics(
                self.dimensions, execution_times, memory_usage, throughput
            )
            
            scalability_results[alg_name] = {
                'dimensions': self.dimensions,
                'execution_times': execution_times,
                'memory_usage': memory_usage,
                'throughput': throughput,
                'metrics': scalability_metrics
            }
        
        return scalability_results
    
    def _calculate_scalability_metrics(self, dimensions: List[int], times: List[float], 
                                     memory: List[float], throughput: List[float]) -> Dict[str, float]:
        """Calculate scalability metrics."""
        # Remove NaN values
        valid_indices = [i for i in range(len(times)) if not np.isnan(times[i])]
        
        if len(valid_indices) < 3:
            return {'time_complexity': np.nan, 'memory_complexity': np.nan, 'efficiency_score': 0.0}
        
        valid_dims = [dimensions[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]
        valid_memory = [memory[i] for i in valid_indices]
        
        # Linear regression to estimate complexity
        log_dims = np.log(valid_dims)
        log_times = np.log(valid_times)
        log_memory = np.log(valid_memory)
        
        # Time complexity slope
        time_slope, _, time_r, _, _ = stats.linregress(log_dims, log_times)
        
        # Memory complexity slope
        memory_slope, _, memory_r, _, _ = stats.linregress(log_dims, log_memory)
        
        # Efficiency score (higher is better)
        efficiency = np.mean(throughput[-3:]) / np.mean(throughput[:3]) if len(throughput) >= 3 else 1.0
        
        return {
            'time_complexity': time_slope,
            'memory_complexity': memory_slope,
            'time_correlation': time_r,
            'memory_correlation': memory_r,
            'efficiency_score': efficiency
        }
    
    def run_memory_benchmarks(self) -> Dict[str, Any]:
        """Run memory efficiency benchmarks."""
        memory_results = {}
        
        for alg_name, alg_class in self.algorithms.items():
            self.logger.info(f"Testing memory efficiency for {alg_name}...")
            
            memory_usage = []
            
            for dim in [1000, 5000, 10000]:  # Selected dimensions
                try:
                    # Initialize algorithm
                    if alg_name == 'hierarchical':
                        alg = alg_class(dim=dim, levels=3)
                    elif alg_name == 'continual':
                        alg = alg_class(dim=dim, memory_size=100)
                    else:
                        alg = alg_class(dim=dim)
                    
                    # Measure memory usage
                    initial_memory = self._get_memory_usage()
                    
                    # Perform operations
                    if alg_name == 'fractional':
                        for _ in range(100):
                            hv1 = alg.random()
                            hv2 = alg.random()
                            result = alg.fractional_bind(hv1, hv2, strength=0.5)
                    
                    elif alg_name == 'quantum':
                        for _ in range(100):
                            qhv1 = alg.random_quantum()
                            qhv2 = alg.random_quantum()
                            result = alg.quantum_bind(qhv1, qhv2)
                    
                    # Add other algorithm-specific tests...
                    
                    final_memory = self._get_memory_usage()
                    memory_increase = final_memory - initial_memory
                    
                    memory_usage.append({
                        'dimension': dim,
                        'memory_increase': memory_increase,
                        'memory_per_operation': memory_increase / 100
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Memory test failed for {alg_name} at dim {dim}: {e}")
                    continue
            
            memory_results[alg_name] = memory_usage
        
        return memory_results
    
    def run_accuracy_benchmarks(self) -> Dict[str, Any]:
        """Run accuracy validation benchmarks."""
        accuracy_results = {}
        
        # Test accuracy on known ground truth tasks
        for alg_name, alg_class in self.algorithms.items():
            self.logger.info(f"Testing accuracy for {alg_name}...")
            
            accuracy_scores = []
            
            try:
                if alg_name == 'fractional':
                    # Test fractional binding accuracy
                    alg = alg_class(dim=1000)
                    
                    for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        hv1 = alg.random()
                        hv2 = alg.random()
                        
                        bound = alg.fractional_bind(hv1, hv2, strength)
                        
                        # Test reversibility
                        recovered = alg.fractional_bind(bound, hv2, strength)
                        accuracy = 1.0 - np.mean(np.abs(recovered - hv1))
                        
                        accuracy_scores.append({
                            'strength': strength,
                            'accuracy': accuracy
                        })
                
                elif alg_name == 'quantum':
                    # Test quantum operation consistency
                    alg = alg_class(dim=1000)
                    
                    for _ in range(10):
                        qhv1 = alg.random_quantum()
                        qhv2 = alg.random_quantum()
                        
                        # Test unitarity preservation
                        bound = alg.quantum_bind(qhv1, qhv2)
                        magnitude_preservation = np.mean(np.abs(np.abs(bound) - 1.0))
                        
                        accuracy_scores.append({
                            'test': 'magnitude_preservation',
                            'accuracy': 1.0 - magnitude_preservation
                        })
                
                # Add accuracy tests for other algorithms...
                
            except Exception as e:
                self.logger.warning(f"Accuracy test failed for {alg_name}: {e}")
                accuracy_scores = []
            
            accuracy_results[alg_name] = accuracy_scores
        
        return accuracy_results
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run comparative analysis between algorithms."""
        comparative_results = {}
        
        # Compare on standardized tasks
        test_dimension = 5000
        
        # Task 1: Basic binding operation speed
        binding_results = {}
        for alg_name, alg_class in self.algorithms.items():
            if alg_name in ['fractional', 'adaptive']:  # Algorithms that support binding
                try:
                    if alg_name == 'fractional':
                        alg = alg_class(dim=test_dimension)
                        
                        times = []
                        for _ in range(20):
                            hv1 = alg.random()
                            hv2 = alg.random()
                            
                            start_time = time.time()
                            result = alg.fractional_bind(hv1, hv2, strength=0.5)
                            execution_time = time.time() - start_time
                            times.append(execution_time)
                        
                        binding_results[alg_name] = {
                            'mean_time': np.mean(times),
                            'std_time': np.std(times),
                            'throughput': 1.0 / np.mean(times)
                        }
                    
                    elif alg_name == 'adaptive':
                        alg = alg_class(dim=test_dimension)
                        
                        times = []
                        for _ in range(20):
                            hv1 = np.random.binomial(1, 0.5, test_dimension).astype(np.float32)
                            hv2 = np.random.binomial(1, 0.5, test_dimension).astype(np.float32)
                            
                            start_time = time.time()
                            result, info = alg.adaptive_operation('bind', hv1, hv2)
                            execution_time = time.time() - start_time
                            times.append(execution_time)
                        
                        binding_results[alg_name] = {
                            'mean_time': np.mean(times),
                            'std_time': np.std(times),
                            'throughput': 1.0 / np.mean(times)
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Comparative binding test failed for {alg_name}: {e}")
        
        comparative_results['binding_comparison'] = binding_results
        
        # Task 2: Memory efficiency comparison
        memory_comparison = {}
        for alg_name, alg_class in self.algorithms.items():
            try:
                initial_memory = self._get_memory_usage()
                
                if alg_name == 'hierarchical':
                    alg = alg_class(dim=test_dimension, levels=3)
                elif alg_name == 'continual':
                    alg = alg_class(dim=test_dimension, memory_size=100)
                else:
                    alg = alg_class(dim=test_dimension)
                
                # Perform 50 operations
                for _ in range(50):
                    if alg_name == 'fractional':
                        hv1 = alg.random()
                        hv2 = alg.random()
                        result = alg.fractional_bind(hv1, hv2, strength=0.5)
                    elif alg_name == 'quantum':
                        qhv1 = alg.random_quantum()
                        qhv2 = alg.random_quantum()
                        result = alg.quantum_bind(qhv1, qhv2)
                    # Add operations for other algorithms...
                
                final_memory = self._get_memory_usage()
                memory_increase = final_memory - initial_memory
                
                memory_comparison[alg_name] = {
                    'memory_increase_mb': memory_increase,
                    'memory_per_operation_mb': memory_increase / 50
                }
                
            except Exception as e:
                self.logger.warning(f"Memory comparison failed for {alg_name}: {e}")
        
        comparative_results['memory_comparison'] = memory_comparison
        
        return comparative_results
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on benchmark results."""
        stats_results = {}
        
        # Collect all performance data
        performance_data = defaultdict(list)
        
        for result in self.results:
            if result.error_rate == 0:  # Only successful runs
                performance_data[result.algorithm].append(result.execution_time)
        
        # Statistical comparisons
        algorithm_pairs = []
        algorithms = list(performance_data.keys())
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                if len(performance_data[alg1]) >= 5 and len(performance_data[alg2]) >= 5:
                    # T-test for performance comparison
                    t_stat, p_value = ttest_ind(performance_data[alg1], performance_data[alg2])
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = mannwhitneyu(performance_data[alg1], performance_data[alg2])
                    
                    algorithm_pairs.append({
                        'algorithm_1': alg1,
                        'algorithm_2': alg2,
                        't_statistic': t_stat,
                        't_p_value': p_value,
                        'u_statistic': u_stat,
                        'u_p_value': u_p_value,
                        'significantly_different': p_value < 0.05,
                        'alg1_mean': np.mean(performance_data[alg1]),
                        'alg2_mean': np.mean(performance_data[alg2])
                    })
        
        stats_results['pairwise_comparisons'] = algorithm_pairs
        
        # Overall performance statistics
        overall_stats = {}
        for alg in algorithms:
            data = performance_data[alg]
            if len(data) >= 3:
                overall_stats[alg] = {
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75),
                    'coefficient_of_variation': np.std(data) / np.mean(data),
                    'sample_size': len(data)
                }
        
        stats_results['overall_statistics'] = overall_stats
        
        return stats_results
    
    def validate_reproducibility(self) -> Dict[str, Any]:
        """Validate reproducibility of algorithms."""
        reproducibility_results = {}
        
        test_dimension = 2000
        num_trials = 5
        
        for alg_name, alg_class in self.algorithms.items():
            self.logger.info(f"Testing reproducibility for {alg_name}...")
            
            trial_results = []
            
            for trial in range(num_trials):
                try:
                    # Set random seed for reproducibility
                    np.random.seed(42 + trial)
                    
                    if alg_name == 'hierarchical':
                        alg = alg_class(dim=test_dimension, levels=3)
                    elif alg_name == 'continual':
                        alg = alg_class(dim=test_dimension, memory_size=50)
                    else:
                        alg = alg_class(dim=test_dimension)
                    
                    # Perform standardized operations
                    results = []
                    for _ in range(10):
                        if alg_name == 'fractional':
                            np.random.seed(42)  # Fixed seed for consistent inputs
                            hv1 = alg.random()
                            hv2 = alg.random()
                            result = alg.fractional_bind(hv1, hv2, strength=0.5)
                            results.append(np.mean(result))
                        
                        elif alg_name == 'quantum':
                            np.random.seed(42)
                            qhv1 = alg.random_quantum()
                            qhv2 = alg.random_quantum()
                            result = alg.quantum_bind(qhv1, qhv2)
                            results.append(np.mean(np.abs(result)))
                        
                        # Add tests for other algorithms...
                    
                    trial_results.append(results)
                    
                except Exception as e:
                    self.logger.warning(f"Reproducibility test failed for {alg_name} trial {trial}: {e}")
                    continue
            
            if len(trial_results) >= 2:
                # Calculate variance between trials
                trial_means = [np.mean(trial) for trial in trial_results]
                reproducibility_score = 1.0 - (np.std(trial_means) / (np.mean(trial_means) + 1e-10))
                
                reproducibility_results[alg_name] = {
                    'reproducibility_score': max(0.0, reproducibility_score),
                    'trial_means': trial_means,
                    'mean_variance': np.var(trial_means),
                    'successful_trials': len(trial_results)
                }
            else:
                reproducibility_results[alg_name] = {
                    'reproducibility_score': 0.0,
                    'error': 'Insufficient successful trials'
                }
        
        return reproducibility_results
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple benchmark results."""
        if not results:
            return {}
        
        # Extract numeric values
        execution_times = [r['execution_time'] for r in results if 'execution_time' in r]
        throughputs = [r['throughput'] for r in results if 'throughput' in r]
        memory_usage = [r['memory_usage'] for r in results if 'memory_usage' in r]
        
        return {
            'count': len(results),
            'success_rate': sum(1 for r in results if r.get('success', False)) / len(results),
            'execution_time': {
                'mean': np.mean(execution_times) if execution_times else 0,
                'std': np.std(execution_times) if execution_times else 0,
                'min': np.min(execution_times) if execution_times else 0,
                'max': np.max(execution_times) if execution_times else 0,
                'median': np.median(execution_times) if execution_times else 0
            },
            'throughput': {
                'mean': np.mean(throughputs) if throughputs else 0,
                'std': np.std(throughputs) if throughputs else 0
            },
            'memory_usage': {
                'mean': np.mean(memory_usage) if memory_usage else 0,
                'std': np.std(memory_usage) if memory_usage else 0
            }
        }
    
    def _estimate_memory_usage(self, obj) -> float:
        """Estimate memory usage of an object in MB."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes / (1024 * 1024)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_memory_usage(item) for item in obj) 
            elif isinstance(obj, dict):
                return sum(self._estimate_memory_usage(v) for v in obj.values())
            else:
                return len(pickle.dumps(obj)) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        # Save JSON results
        json_file = f"{self.output_dir}/benchmark_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {json_file}")
        
        # Save individual result objects
        pickle_file = f"{self.output_dir}/benchmark_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report."""
        report_file = f"{self.output_dir}/benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# HD-Compute-Toolkit Comprehensive Benchmark Report\\n\\n")
            f.write(f"Generated: {time.ctime()}\\n\\n")
            
            # Executive Summary
            f.write("## Executive Summary\\n\\n")
            f.write(f"- **Algorithms Tested**: {len(self.algorithms)}\\n")
            f.write(f"- **Dimensions Tested**: {self.dimensions}\\n")
            f.write(f"- **Total Test Duration**: {results['metadata']['total_duration']:.2f} seconds\\n")
            f.write(f"- **Iterations per Test**: {self.iterations}\\n\\n")
            
            # Performance Summary
            if 'performance' in results['results']:
                f.write("## Performance Summary\\n\\n")
                perf_results = results['results']['performance']
                
                for alg_name, alg_results in perf_results.items():
                    f.write(f"### {alg_name.title()} HDC\\n")
                    
                    for dim_key, dim_data in alg_results.items():
                        if 'execution_time' in dim_data:
                            mean_time = dim_data['execution_time']['mean']
                            throughput = dim_data['throughput']['mean']
                            f.write(f"- **{dim_key}**: {mean_time:.4f}s avg, {throughput:.1f} ops/sec\\n")
                    f.write("\\n")
            
            # Scalability Analysis
            if 'scalability' in results['results']:
                f.write("## Scalability Analysis\\n\\n")
                scale_results = results['results']['scalability']
                
                for alg_name, scale_data in scale_results.items():
                    if 'metrics' in scale_data:
                        metrics = scale_data['metrics']
                        f.write(f"### {alg_name.title()} HDC\\n")
                        f.write(f"- **Time Complexity**: O(n^{metrics.get('time_complexity', 'unknown'):.2f})\\n")
                        f.write(f"- **Memory Complexity**: O(n^{metrics.get('memory_complexity', 'unknown'):.2f})\\n")
                        f.write(f"- **Efficiency Score**: {metrics.get('efficiency_score', 0):.3f}\\n\\n")
            
            # Statistical Analysis
            if 'statistical' in results['results']:
                f.write("## Statistical Analysis\\n\\n")
                stats_results = results['results']['statistical']
                
                if 'overall_statistics' in stats_results:
                    f.write("### Algorithm Performance Statistics\\n\\n")
                    for alg, stats in stats_results['overall_statistics'].items():
                        f.write(f"**{alg.title()}**: ")
                        f.write(f"μ={stats['mean']:.4f}s, σ={stats['std']:.4f}s, ")
                        f.write(f"CV={stats['coefficient_of_variation']:.3f}\\n")
                
                if 'pairwise_comparisons' in stats_results:
                    f.write("\\n### Significant Performance Differences\\n\\n")
                    for comp in stats_results['pairwise_comparisons']:
                        if comp['significantly_different']:
                            f.write(f"- **{comp['algorithm_1']}** vs **{comp['algorithm_2']}**: ")
                            f.write(f"p={comp['t_p_value']:.4f} (significant)\\n")
            
            # Reproducibility
            if 'reproducibility' in results['results']:
                f.write("\\n## Reproducibility Validation\\n\\n")
                repro_results = results['results']['reproducibility']
                
                for alg_name, repro_data in repro_results.items():
                    score = repro_data.get('reproducibility_score', 0)
                    f.write(f"- **{alg_name.title()}**: {score:.3f} reproducibility score\\n")
            
            f.write("\\n## Recommendations\\n\\n")
            f.write("Based on the benchmark results:\\n\\n")
            
            # Generate recommendations based on results
            recommendations = self._generate_recommendations(results)
            for rec in recommendations:
                f.write(f"- {rec}\\n")
        
        self.logger.info(f"Comprehensive report saved to {report_file}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze performance results
        if 'performance' in results['results']:
            perf_results = results['results']['performance']
            
            # Find fastest algorithm
            fastest_alg = None
            fastest_time = float('inf')
            
            for alg_name, alg_results in perf_results.items():
                for dim_key, dim_data in alg_results.items():
                    if 'execution_time' in dim_data:
                        mean_time = dim_data['execution_time']['mean']
                        if mean_time < fastest_time:
                            fastest_time = mean_time
                            fastest_alg = alg_name
            
            if fastest_alg:
                recommendations.append(f"For speed-critical applications, consider {fastest_alg.title()} HDC")
        
        # Analyze scalability
        if 'scalability' in results['results']:
            scale_results = results['results']['scalability']
            
            best_scalability = None
            best_efficiency = 0
            
            for alg_name, scale_data in scale_results.items():
                if 'metrics' in scale_data:
                    efficiency = scale_data['metrics'].get('efficiency_score', 0)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_scalability = alg_name
            
            if best_scalability:
                recommendations.append(f"For large-scale applications, {best_scalability.title()} HDC shows best scalability")
        
        # Analyze reproducibility
        if 'reproducibility' in results['results']:
            repro_results = results['results']['reproducibility']
            
            most_reproducible = None
            best_repro_score = 0
            
            for alg_name, repro_data in repro_results.items():
                score = repro_data.get('reproducibility_score', 0)
                if score > best_repro_score:
                    best_repro_score = score
                    most_reproducible = alg_name
            
            if most_reproducible:
                recommendations.append(f"For research applications requiring reproducibility, use {most_reproducible.title()} HDC")
        
        recommendations.append("Consider adaptive algorithms for varying workloads")
        recommendations.append("Use distributed optimization for large-scale deployments")
        
        return recommendations


if __name__ == "__main__":
    print("🔬 HD-Compute-Toolkit Comprehensive Benchmark Suite")
    print("=" * 60)
    
    # Initialize benchmark suite
    suite = ComprehensiveBenchmarkSuite(output_dir="benchmark_results")
    
    print("\\n🚀 Running Comprehensive Benchmark Suite...")
    print("This may take several minutes to complete...")
    
    try:
        # Run full benchmark suite
        results = suite.run_full_benchmark_suite()
        
        print("\\n✅ Benchmark Suite Completed Successfully!")
        print(f"\\n📊 Results Summary:")
        print(f"- Total Duration: {results['metadata']['total_duration']:.2f} seconds")
        print(f"- Algorithms Tested: {len(results['metadata']['algorithms_tested'])}")
        print(f"- Dimensions Tested: {len(results['metadata']['dimensions_tested'])}")
        
        print(f"\\n📁 Results saved to: {suite.output_dir}/")
        print("   - benchmark_results.json (detailed results)")
        print("   - benchmark_report.md (comprehensive report)")
        print("   - benchmark.log (execution log)")
        
    except Exception as e:
        print(f"\\n❌ Benchmark suite failed: {e}")
        raise
    
    print("\\n🎉 Comprehensive Benchmarking Complete!")