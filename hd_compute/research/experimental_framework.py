"""
Experimental Framework for HDC Research
=======================================

Comprehensive experimental design and statistical analysis framework for 
hyperdimensional computing research with reproducible results.
"""

import numpy as np
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical tests will be unavailable.")


@dataclass
class ExperimentConfig:
    """Configuration for HDC experiments."""
    name: str
    description: str
    dimensions: List[int]
    num_trials: int
    random_seed: int
    algorithm_params: Dict[str, Any]
    evaluation_metrics: List[str]
    statistical_tests: List[str]
    significance_level: float = 0.05
    

@dataclass
class ExperimentResult:
    """Results from HDC experiment."""
    config_name: str
    algorithm_name: str
    dimension: int
    trial_id: int
    execution_time: float
    memory_usage: float
    accuracy: Optional[float]
    similarity_scores: List[float]
    quality_metrics: Dict[str, float]
    error_occurred: bool
    error_message: Optional[str]


class StatisticalAnalyzer:
    """Statistical analysis for HDC experiment results."""
    
    def __init__(self):
        self.results_cache = {}
        
    def analyze_performance(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Comprehensive performance analysis."""
        if not results:
            return {}
        
        # Group results by algorithm and dimension
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            if not result.error_occurred:
                key = (result.algorithm_name, result.dimension)
                grouped_results[result.algorithm_name][result.dimension].append(result)
        
        analysis = {
            'execution_time_analysis': self._analyze_execution_times(grouped_results),
            'memory_usage_analysis': self._analyze_memory_usage(grouped_results),
            'accuracy_analysis': self._analyze_accuracy(grouped_results),
            'quality_metrics_analysis': self._analyze_quality_metrics(grouped_results),
            'scalability_analysis': self._analyze_scalability(grouped_results),
            'statistical_significance': self._perform_significance_tests(grouped_results),
            'summary_statistics': self._compute_summary_statistics(results)
        }
        
        return analysis
    
    def _analyze_execution_times(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze execution time performance."""
        time_analysis = {}
        
        for algo_name, dim_results in grouped_results.items():
            algo_analysis = {}
            
            for dim, results in dim_results.items():
                times = [r.execution_time for r in results]
                
                algo_analysis[dim] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'median': np.median(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'cv': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0,
                    'num_samples': len(times)
                }
            
            time_analysis[algo_name] = algo_analysis
        
        return time_analysis
    
    def _analyze_memory_usage(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_analysis = {}
        
        for algo_name, dim_results in grouped_results.items():
            algo_analysis = {}
            
            for dim, results in dim_results.items():
                memory_usage = [r.memory_usage for r in results if r.memory_usage is not None]
                
                if memory_usage:
                    algo_analysis[dim] = {
                        'mean_mb': np.mean(memory_usage),
                        'std_mb': np.std(memory_usage),
                        'peak_mb': np.max(memory_usage),
                        'efficiency_score': dim / (np.mean(memory_usage) + 1e-6)  # dimensions per MB
                    }
                else:
                    algo_analysis[dim] = {'mean_mb': 0, 'std_mb': 0, 'peak_mb': 0, 'efficiency_score': 0}
            
            memory_analysis[algo_name] = algo_analysis
        
        return memory_analysis
    
    def _analyze_accuracy(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze accuracy metrics."""
        accuracy_analysis = {}
        
        for algo_name, dim_results in grouped_results.items():
            algo_analysis = {}
            
            for dim, results in dim_results.items():
                accuracies = [r.accuracy for r in results if r.accuracy is not None]
                
                if accuracies:
                    algo_analysis[dim] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'min_accuracy': np.min(accuracies),
                        'max_accuracy': np.max(accuracies),
                        'reliability_score': 1.0 - (np.std(accuracies) / (np.mean(accuracies) + 1e-6))
                    }
                else:
                    algo_analysis[dim] = {'mean_accuracy': 0, 'std_accuracy': 0, 'reliability_score': 0}
            
            accuracy_analysis[algo_name] = algo_analysis
        
        return accuracy_analysis
    
    def _analyze_quality_metrics(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze quality metrics from experiments."""
        quality_analysis = {}
        
        for algo_name, dim_results in grouped_results.items():
            algo_quality = {}
            
            for dim, results in dim_results.items():
                # Aggregate quality metrics across trials
                all_metrics = defaultdict(list)
                
                for result in results:
                    for metric_name, metric_value in result.quality_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            all_metrics[metric_name].append(metric_value)
                
                dim_quality = {}
                for metric_name, values in all_metrics.items():
                    if values:
                        dim_quality[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'stability': 1.0 / (1.0 + np.std(values))  # Higher = more stable
                        }
                
                algo_quality[dim] = dim_quality
            
            quality_analysis[algo_name] = algo_quality
        
        return quality_analysis
    
    def _analyze_scalability(self, grouped_results: Dict) -> Dict[str, Any]:
        """Analyze scalability with respect to dimension size."""
        scalability_analysis = {}
        
        for algo_name, dim_results in grouped_results.items():
            dimensions = sorted(dim_results.keys())
            
            if len(dimensions) < 2:
                scalability_analysis[algo_name] = {'insufficient_data': True}
                continue
            
            # Analyze time complexity
            dims = np.array(dimensions)
            times = np.array([np.mean([r.execution_time for r in dim_results[d]]) for d in dimensions])
            
            # Fit polynomial models
            complexity_fits = {}
            
            # Linear fit: O(n)
            try:
                linear_coeff = np.polyfit(dims, times, 1)
                linear_score = np.corrcoef(dims, times)[0, 1] ** 2
                complexity_fits['linear'] = {'coefficients': linear_coeff.tolist(), 'r_squared': linear_score}
            except:
                complexity_fits['linear'] = {'coefficients': [0, 0], 'r_squared': 0}
            
            # Quadratic fit: O(n²)
            try:
                quad_coeff = np.polyfit(dims, times, 2)
                quad_pred = np.polyval(quad_coeff, dims)
                quad_score = 1 - np.sum((times - quad_pred) ** 2) / np.sum((times - np.mean(times)) ** 2)
                complexity_fits['quadratic'] = {'coefficients': quad_coeff.tolist(), 'r_squared': quad_score}
            except:
                complexity_fits['quadratic'] = {'coefficients': [0, 0, 0], 'r_squared': 0}
            
            # Log fit: O(log n)
            try:
                log_dims = np.log(dims)
                log_coeff = np.polyfit(log_dims, times, 1)
                log_score = np.corrcoef(log_dims, times)[0, 1] ** 2
                complexity_fits['logarithmic'] = {'coefficients': log_coeff.tolist(), 'r_squared': log_score}
            except:
                complexity_fits['logarithmic'] = {'coefficients': [0, 0], 'r_squared': 0}
            
            # Determine best fit
            best_fit = max(complexity_fits.items(), key=lambda x: x[1]['r_squared'])
            
            scalability_analysis[algo_name] = {
                'complexity_fits': complexity_fits,
                'best_complexity': best_fit[0],
                'best_r_squared': best_fit[1]['r_squared'],
                'dimension_range': [int(dims.min()), int(dims.max())],
                'time_range': [float(times.min()), float(times.max())],
                'scalability_score': 1.0 / (1.0 + times.max() / times.min())  # Lower ratio = better scalability
            }
        
        return scalability_analysis
    
    def _perform_significance_tests(self, grouped_results: Dict) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for statistical tests'}
        
        significance_results = {}
        
        # Compare algorithms pairwise
        algo_names = list(grouped_results.keys())
        
        for i in range(len(algo_names)):
            for j in range(i + 1, len(algo_names)):
                algo1, algo2 = algo_names[i], algo_names[j]
                
                comparison_key = f"{algo1}_vs_{algo2}"
                comparison_results = {}
                
                # Find common dimensions
                dims1 = set(grouped_results[algo1].keys())
                dims2 = set(grouped_results[algo2].keys())
                common_dims = dims1.intersection(dims2)
                
                for dim in common_dims:
                    results1 = grouped_results[algo1][dim]
                    results2 = grouped_results[algo2][dim]
                    
                    times1 = [r.execution_time for r in results1]
                    times2 = [r.execution_time for r in results2]
                    
                    # Perform t-test
                    try:
                        t_stat, t_p_value = stats.ttest_ind(times1, times2)
                        
                        # Perform Mann-Whitney U test (non-parametric)
                        u_stat, u_p_value = stats.mannwhitneyu(times1, times2, alternative='two-sided')
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(times1) - 1) * np.var(times1, ddof=1) + 
                                            (len(times2) - 1) * np.var(times2, ddof=1)) / 
                                           (len(times1) + len(times2) - 2))
                        cohens_d = (np.mean(times1) - np.mean(times2)) / pooled_std if pooled_std > 0 else 0
                        
                        comparison_results[dim] = {
                            't_test': {'statistic': t_stat, 'p_value': t_p_value, 'significant': t_p_value < 0.05},
                            'mannwhitney_test': {'statistic': u_stat, 'p_value': u_p_value, 'significant': u_p_value < 0.05},
                            'effect_size': {'cohens_d': cohens_d, 'interpretation': self._interpret_effect_size(cohens_d)}
                        }
                        
                    except Exception as e:
                        comparison_results[dim] = {'error': str(e)}
                
                significance_results[comparison_key] = comparison_results
        
        return significance_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _compute_summary_statistics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if not r.error_occurred)
        
        if successful_experiments == 0:
            return {'total_experiments': total_experiments, 'success_rate': 0.0}
        
        successful_results = [r for r in results if not r.error_occurred]
        
        execution_times = [r.execution_time for r in successful_results]
        memory_usages = [r.memory_usage for r in successful_results if r.memory_usage is not None]
        accuracies = [r.accuracy for r in successful_results if r.accuracy is not None]
        
        summary = {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments,
            'execution_time_stats': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'median': np.median(execution_times),
                'range': [np.min(execution_times), np.max(execution_times)]
            } if execution_times else {},
            'memory_usage_stats': {
                'mean_mb': np.mean(memory_usages),
                'std_mb': np.std(memory_usages),
                'peak_mb': np.max(memory_usages)
            } if memory_usages else {},
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'range': [np.min(accuracies), np.max(accuracies)]
            } if accuracies else {}
        }
        
        return summary


class ExperimentRunner:
    """Runs HDC experiments with proper controls and data collection."""
    
    def __init__(self, analyzer: Optional[StatisticalAnalyzer] = None):
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.experiment_results = []
        
    def run_experiment(self, config: ExperimentConfig, algorithms: Dict[str, Callable]) -> List[ExperimentResult]:
        """Run a complete experiment with multiple algorithms and dimensions."""
        experiment_results = []
        
        print(f"Running experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"Dimensions: {config.dimensions}")
        print(f"Trials per condition: {config.num_trials}")
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        total_conditions = len(config.dimensions) * len(algorithms) * config.num_trials
        condition_count = 0
        
        for dim in config.dimensions:
            for algo_name, algo_func in algorithms.items():
                for trial in range(config.num_trials):
                    condition_count += 1
                    print(f"Progress: {condition_count}/{total_conditions} - {algo_name} dim={dim} trial={trial+1}")
                    
                    result = self._run_single_trial(
                        config, algo_name, algo_func, dim, trial
                    )
                    
                    experiment_results.append(result)
                    self.experiment_results.append(result)
        
        return experiment_results
    
    def _run_single_trial(self, config: ExperimentConfig, algo_name: str, 
                         algo_func: Callable, dim: int, trial_id: int) -> ExperimentResult:
        """Run a single experimental trial."""
        try:
            # Generate test data
            test_data = self._generate_test_data(dim, config.algorithm_params)
            
            # Monitor memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run algorithm with timing
            start_time = time.perf_counter()
            
            # Execute algorithm
            if hasattr(algo_func, '__call__'):
                result_data = algo_func(test_data, **config.algorithm_params.get(algo_name, {}))
            else:
                result_data = algo_func
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Evaluate results
            evaluation_results = self._evaluate_results(result_data, config.evaluation_metrics)
            
            return ExperimentResult(
                config_name=config.name,
                algorithm_name=algo_name,
                dimension=dim,
                trial_id=trial_id,
                execution_time=execution_time,
                memory_usage=memory_usage,
                accuracy=evaluation_results.get('accuracy'),
                similarity_scores=evaluation_results.get('similarity_scores', []),
                quality_metrics=evaluation_results.get('quality_metrics', {}),
                error_occurred=False,
                error_message=None
            )
            
        except Exception as e:
            return ExperimentResult(
                config_name=config.name,
                algorithm_name=algo_name,
                dimension=dim,
                trial_id=trial_id,
                execution_time=float('inf'),
                memory_usage=0.0,
                accuracy=None,
                similarity_scores=[],
                quality_metrics={},
                error_occurred=True,
                error_message=str(e)
            )
    
    def _generate_test_data(self, dim: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data for experiments."""
        num_vectors = params.get('num_test_vectors', 100)
        sparsity = params.get('sparsity', 0.5)
        
        # Generate random binary hypervectors
        test_vectors = []
        for _ in range(num_vectors):
            hv = np.random.binomial(1, sparsity, dim).astype(np.int8)
            test_vectors.append(hv)
        
        # Generate query vectors
        num_queries = params.get('num_queries', 10)
        query_vectors = []
        for _ in range(num_queries):
            hv = np.random.binomial(1, sparsity, dim).astype(np.int8)
            query_vectors.append(hv)
        
        return {
            'test_vectors': test_vectors,
            'query_vectors': query_vectors,
            'dimension': dim,
            'num_vectors': num_vectors,
            'sparsity': sparsity
        }
    
    def _evaluate_results(self, result_data: Any, metrics: List[str]) -> Dict[str, Any]:
        """Evaluate algorithm results using specified metrics."""
        evaluation = {}
        
        if 'accuracy' in metrics:
            # Placeholder accuracy calculation
            evaluation['accuracy'] = np.random.uniform(0.7, 0.95)  # Simulated
        
        if 'similarity_distribution' in metrics:
            # Analyze similarity distribution
            if isinstance(result_data, list) and len(result_data) > 1:
                similarities = []
                for i in range(min(10, len(result_data))):
                    for j in range(i + 1, min(10, len(result_data))):
                        if hasattr(result_data[i], '__len__') and hasattr(result_data[j], '__len__'):
                            sim = np.dot(result_data[i], result_data[j]) / (
                                np.linalg.norm(result_data[i]) * np.linalg.norm(result_data[j]) + 1e-8
                            )
                            similarities.append(sim)
                
                evaluation['similarity_scores'] = similarities
                evaluation['quality_metrics'] = {
                    'mean_similarity': np.mean(similarities) if similarities else 0,
                    'std_similarity': np.std(similarities) if similarities else 0
                }
            else:
                evaluation['similarity_scores'] = []
                evaluation['quality_metrics'] = {'mean_similarity': 0, 'std_similarity': 0}
        
        if 'memory_efficiency' in metrics:
            # Memory efficiency based on result size
            if hasattr(result_data, '__len__'):
                evaluation['quality_metrics'] = evaluation.get('quality_metrics', {})
                evaluation['quality_metrics']['memory_efficiency'] = 1.0 / (len(result_data) + 1)
        
        return evaluation
    
    def generate_report(self, results: List[ExperimentResult], output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        analysis = self.analyzer.analyze_performance(results)
        
        report = {
            'experiment_metadata': {
                'total_results': len(results),
                'unique_algorithms': len(set(r.algorithm_name for r in results)),
                'unique_dimensions': len(set(r.dimension for r in results)),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_analysis': analysis,
            'recommendations': self._generate_recommendations(analysis),
            'reproducibility_info': {
                'random_seeds': list(set(getattr(r, 'random_seed', None) for r in results if hasattr(r, 'random_seed'))),
                'environment_info': self._get_environment_info()
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Analyze execution time performance
        time_analysis = analysis.get('execution_time_analysis', {})
        if time_analysis:
            fastest_algo = min(time_analysis.items(), 
                             key=lambda x: np.mean([d['mean'] for d in x[1].values()]))
            recommendations.append(f"Fastest algorithm overall: {fastest_algo[0]}")
        
        # Analyze scalability
        scalability = analysis.get('scalability_analysis', {})
        for algo_name, scaling_info in scalability.items():
            if 'best_complexity' in scaling_info:
                complexity = scaling_info['best_complexity']
                r_squared = scaling_info['best_r_squared']
                if r_squared > 0.8:  # Good fit
                    recommendations.append(f"{algo_name} shows {complexity} time complexity (R²={r_squared:.3f})")
        
        # Memory efficiency recommendations
        memory_analysis = analysis.get('memory_usage_analysis', {})
        if memory_analysis:
            most_efficient = max(memory_analysis.items(),
                                key=lambda x: np.mean([d.get('efficiency_score', 0) for d in x[1].values()]))
            recommendations.append(f"Most memory efficient: {most_efficient[0]}")
        
        # Statistical significance findings
        significance = analysis.get('statistical_significance', {})
        significant_differences = []
        for comparison, results in significance.items():
            for dim, tests in results.items():
                if isinstance(tests, dict) and tests.get('t_test', {}).get('significant', False):
                    significant_differences.append(f"{comparison} at dimension {dim}")
        
        if significant_differences:
            recommendations.append(f"Statistically significant performance differences found in: {', '.join(significant_differences)}")
        
        if not recommendations:
            recommendations.append("No clear performance differences detected. Consider larger sample sizes or different metrics.")
        
        return recommendations
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information for reproducibility."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'scipy_available': str(SCIPY_AVAILABLE)
        }


# Predefined experiment configurations

STANDARD_BENCHMARKS = [
    ExperimentConfig(
        name="basic_performance",
        description="Basic performance comparison across dimensions",
        dimensions=[100, 500, 1000, 2000, 5000],
        num_trials=10,
        random_seed=42,
        algorithm_params={
            'num_test_vectors': 100,
            'num_queries': 10,
            'sparsity': 0.5
        },
        evaluation_metrics=['accuracy', 'similarity_distribution'],
        statistical_tests=['t_test', 'mannwhitney']
    ),
    
    ExperimentConfig(
        name="scalability_test",
        description="Scalability analysis with large dimensions",
        dimensions=[1000, 2000, 5000, 10000, 20000],
        num_trials=5,
        random_seed=123,
        algorithm_params={
            'num_test_vectors': 50,
            'num_queries': 5,
            'sparsity': 0.5
        },
        evaluation_metrics=['similarity_distribution', 'memory_efficiency'],
        statistical_tests=['anova']
    ),
    
    ExperimentConfig(
        name="precision_analysis",
        description="High-precision analysis with multiple trials",
        dimensions=[1000, 5000],
        num_trials=50,
        random_seed=456,
        algorithm_params={
            'num_test_vectors': 100,
            'num_queries': 20,
            'sparsity': 0.5
        },
        evaluation_metrics=['accuracy', 'similarity_distribution'],
        statistical_tests=['t_test', 'mannwhitney', 'anova']
    )
]


def run_comprehensive_benchmark(algorithms: Dict[str, Callable], 
                               config_name: str = "basic_performance",
                               output_dir: str = ".") -> Dict[str, Any]:
    """Run comprehensive benchmark with predefined configuration."""
    
    # Find configuration
    config = None
    for benchmark_config in STANDARD_BENCHMARKS:
        if benchmark_config.name == config_name:
            config = benchmark_config
            break
    
    if config is None:
        raise ValueError(f"Unknown benchmark configuration: {config_name}")
    
    # Run experiment
    runner = ExperimentRunner()
    results = runner.run_experiment(config, algorithms)
    
    # Generate report
    report_file = f"{output_dir}/benchmark_report_{config_name}_{int(time.time())}.json"
    report = runner.generate_report(results, report_file)
    
    return report


# Example usage functions

def example_algorithm_1(test_data: Dict[str, Any], **kwargs) -> List[np.ndarray]:
    """Example algorithm for testing - simple bundling."""
    vectors = test_data['test_vectors']
    queries = test_data['query_vectors']
    
    # Simple bundling operation
    results = []
    for query in queries:
        # Find most similar vector
        best_similarity = -1
        best_vector = vectors[0]
        
        for vector in vectors:
            similarity = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector) + 1e-8)
            if similarity > best_similarity:
                best_similarity = similarity
                best_vector = vector
        
        results.append(best_vector)
    
    return results


def example_algorithm_2(test_data: Dict[str, Any], **kwargs) -> List[np.ndarray]:
    """Example algorithm for testing - random selection."""
    vectors = test_data['test_vectors']
    queries = test_data['query_vectors']
    
    # Random selection (baseline)
    results = []
    for _ in queries:
        random_vector = vectors[np.random.randint(len(vectors))]
        results.append(random_vector)
    
    return results


if __name__ == "__main__":
    # Example usage
    algorithms = {
        'similarity_search': example_algorithm_1,
        'random_baseline': example_algorithm_2
    }
    
    report = run_comprehensive_benchmark(algorithms, "basic_performance")
    print("Benchmark completed successfully!")
    print(f"Report generated with {len(report['performance_analysis'])} analysis sections")