"""Quality assurance, reproducibility, and performance monitoring for HDC research."""

import numpy as np
import pandas as pd
import hashlib
import json
import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import pickle
import logging


@dataclass
class QualityMetrics:
    """Quality metrics for HDC operations and experiments."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    execution_time: float
    memory_usage: float
    numerical_stability: float
    reproducibility_score: float
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    operation_name: str
    execution_time_ms: float
    memory_delta_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    latency_percentiles: Dict[str, float]  # P50, P95, P99
    error_rate: float
    success_rate: float
    resource_efficiency: float
    timestamp: float


@dataclass
class ReproducibilityReport:
    """Reproducibility analysis report."""
    experiment_id: str
    original_results: List[float]
    reproduction_results: List[float]
    correlation_coefficient: float
    mean_absolute_error: float
    relative_error_percentage: float
    statistical_test_p_value: float
    reproducibility_score: float  # 0-1 scale
    environment_hash: str
    parameter_hash: str
    is_reproducible: bool
    deviations: List[str]
    recommendations: List[str]


class QualityAssuranceFramework:
    """Comprehensive quality assurance framework for HDC research."""
    
    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        self.quality_thresholds = quality_thresholds or {
            'accuracy': 0.7,
            'precision': 0.7,
            'recall': 0.7,
            'f1_score': 0.7,
            'reproducibility_score': 0.95,
            'statistical_significance': 0.05,
            'numerical_stability': 0.99
        }
        
        self.quality_history = []
        self.failed_quality_checks = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def assess_quality(self, experiment_results: Dict[str, Any], 
                      expected_results: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """Assess overall quality of experimental results."""
        
        # Extract basic metrics
        accuracy = experiment_results.get('accuracy', 0.0)
        precision = experiment_results.get('precision', 0.0)
        recall = experiment_results.get('recall', 0.0)
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Performance metrics
        execution_time = experiment_results.get('execution_time', 0.0)
        memory_usage = experiment_results.get('memory_usage', 0.0)
        
        # Statistical metrics
        statistical_significance = experiment_results.get('p_value', 1.0)
        effect_size = experiment_results.get('effect_size', 0.0)
        
        # Calculate numerical stability
        numerical_stability = self._assess_numerical_stability(experiment_results)
        
        # Calculate reproducibility score
        reproducibility_score = self._assess_reproducibility(experiment_results, expected_results)
        
        # Confidence interval
        confidence_interval = experiment_results.get('confidence_interval', (0.0, 1.0))
        sample_size = experiment_results.get('sample_size', 0)
        
        quality_metrics = QualityMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            execution_time=execution_time,
            memory_usage=memory_usage,
            numerical_stability=numerical_stability,
            reproducibility_score=reproducibility_score,
            statistical_significance=statistical_significance,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            sample_size=sample_size,
            timestamp=time.time()
        )
        
        # Store quality assessment
        self.quality_history.append(quality_metrics)
        
        # Check quality thresholds
        quality_check_result = self._check_quality_thresholds(quality_metrics)
        if not quality_check_result['passed']:
            self.failed_quality_checks.append({
                'metrics': quality_metrics,
                'failed_checks': quality_check_result['failed_checks'],
                'timestamp': time.time()
            })
        
        return quality_metrics
    
    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Overall quality score (weighted average)
        weights = {
            'accuracy': 0.2,
            'precision': 0.15,
            'recall': 0.15,
            'f1_score': 0.2,
            'numerical_stability': 0.1,
            'reproducibility_score': 0.2
        }
        
        overall_score = sum(
            getattr(metrics, metric) * weight 
            for metric, weight in weights.items()
        )
        
        # Quality grade
        if overall_score >= 0.9:
            quality_grade = 'A'
        elif overall_score >= 0.8:
            quality_grade = 'B'
        elif overall_score >= 0.7:
            quality_grade = 'C'
        elif overall_score >= 0.6:
            quality_grade = 'D'
        else:
            quality_grade = 'F'
        
        # Performance analysis
        performance_analysis = self._analyze_performance(metrics)
        
        # Recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'metrics': asdict(metrics),
            'performance_analysis': performance_analysis,
            'threshold_compliance': self._check_quality_thresholds(metrics),
            'recommendations': recommendations,
            'quality_trends': self._analyze_quality_trends(),
            'timestamp': time.time()
        }
    
    def _assess_numerical_stability(self, results: Dict[str, Any]) -> float:
        """Assess numerical stability of results."""
        stability_score = 1.0
        
        # Check for NaN/Inf values
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    stability_score *= 0.5
            elif isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    stability_score *= 0.5
        
        # Check for extreme values
        numeric_values = []
        for value in results.values():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                numeric_values.append(abs(value))
            elif isinstance(value, (list, np.ndarray)):
                valid_values = [v for v in np.array(value).flatten() 
                              if not (np.isnan(v) or np.isinf(v))]
                numeric_values.extend([abs(v) for v in valid_values])
        
        if numeric_values:
            max_value = max(numeric_values)
            if max_value > 1e10:  # Very large values
                stability_score *= 0.8
            elif max_value > 1e6:
                stability_score *= 0.9
        
        return stability_score
    
    def _assess_reproducibility(self, current_results: Dict[str, Any], 
                               expected_results: Optional[Dict[str, Any]]) -> float:
        """Assess reproducibility score."""
        if expected_results is None:
            return 1.0  # Assume perfect reproducibility if no baseline
        
        reproducibility_scores = []
        
        for key in current_results.keys():
            if key in expected_results:
                current_val = current_results[key]
                expected_val = expected_results[key]
                
                if isinstance(current_val, (int, float)) and isinstance(expected_val, (int, float)):
                    if expected_val != 0:
                        relative_error = abs(current_val - expected_val) / abs(expected_val)
                        score = max(0, 1 - relative_error)
                    else:
                        score = 1.0 if abs(current_val - expected_val) < 1e-10 else 0.0
                    
                    reproducibility_scores.append(score)
        
        return np.mean(reproducibility_scores) if reproducibility_scores else 1.0
    
    def _check_quality_thresholds(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Check if quality metrics meet thresholds."""
        failed_checks = []
        
        for metric_name, threshold in self.quality_thresholds.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                
                # Handle statistical significance (lower is better)
                if metric_name == 'statistical_significance':
                    if value > threshold:
                        failed_checks.append(f"{metric_name}: {value:.4f} > {threshold}")
                else:
                    if value < threshold:
                        failed_checks.append(f"{metric_name}: {value:.4f} < {threshold}")
        
        return {
            'passed': len(failed_checks) == 0,
            'failed_checks': failed_checks,
            'total_checks': len(self.quality_thresholds)
        }
    
    def _analyze_performance(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        return {
            'execution_efficiency': 'good' if metrics.execution_time < 1.0 else 'needs_improvement',
            'memory_efficiency': 'good' if metrics.memory_usage < 100 else 'high',
            'statistical_power': 'sufficient' if metrics.sample_size > 30 else 'low',
            'effect_magnitude': 'large' if abs(metrics.effect_size) > 0.8 else 'medium' if abs(metrics.effect_size) > 0.5 else 'small'
        }
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if metrics.accuracy < 0.8:
            recommendations.append("Consider improving model architecture or feature engineering")
        
        if metrics.reproducibility_score < 0.95:
            recommendations.append("Improve experimental reproducibility by fixing random seeds and documenting environment")
        
        if metrics.statistical_significance > 0.05:
            recommendations.append("Results not statistically significant - consider increasing sample size or effect size")
        
        if metrics.numerical_stability < 0.95:
            recommendations.append("Address numerical stability issues - check for overflow/underflow")
        
        if metrics.execution_time > 10.0:
            recommendations.append("Optimize performance - consider algorithmic improvements or parallelization")
        
        if metrics.sample_size < 30:
            recommendations.append("Increase sample size for more reliable statistical inference")
        
        return recommendations
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze trends in quality metrics over time."""
        if len(self.quality_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        recent_metrics = self.quality_history[-10:]  # Last 10 assessments
        
        # Calculate trends
        accuracy_trend = np.polyfit(range(len(recent_metrics)), 
                                   [m.accuracy for m in recent_metrics], 1)[0]
        
        return {
            'accuracy_trend': 'improving' if accuracy_trend > 0 else 'declining',
            'trend_slope': accuracy_trend,
            'total_assessments': len(self.quality_history),
            'recent_average_quality': np.mean([m.accuracy for m in recent_metrics])
        }


class ReproducibilityChecker:
    """Advanced reproducibility checker for HDC experiments."""
    
    def __init__(self, storage_dir: str = "./reproducibility_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_registry = {}
        
    def register_experiment(self, experiment_id: str, 
                          config: Dict[str, Any],
                          environment_info: Optional[Dict[str, Any]] = None) -> str:
        """Register experiment for reproducibility tracking."""
        
        # Generate environment hash
        if environment_info is None:
            environment_info = self._capture_environment()
        
        env_hash = self._generate_hash(environment_info)
        config_hash = self._generate_hash(config)
        
        experiment_record = {
            'experiment_id': experiment_id,
            'config': config,
            'config_hash': config_hash,
            'environment_info': environment_info,
            'environment_hash': env_hash,
            'registration_time': time.time(),
            'results_stored': False
        }
        
        # Store experiment record
        self.experiment_registry[experiment_id] = experiment_record
        
        # Save to disk
        record_path = self.storage_dir / f"{experiment_id}_record.json"
        with open(record_path, 'w') as f:
            json.dump(experiment_record, f, indent=2, default=str)
        
        return env_hash
    
    def store_results(self, experiment_id: str, results: Any) -> None:
        """Store experiment results for future reproduction."""
        if experiment_id not in self.experiment_registry:
            raise ValueError(f"Experiment {experiment_id} not registered")
        
        # Serialize and store results
        results_path = self.storage_dir / f"{experiment_id}_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Update registry
        self.experiment_registry[experiment_id]['results_stored'] = True
        self.experiment_registry[experiment_id]['results_path'] = str(results_path)
        
        # Update record file
        record_path = self.storage_dir / f"{experiment_id}_record.json"
        with open(record_path, 'w') as f:
            json.dump(self.experiment_registry[experiment_id], f, indent=2, default=str)
    
    def reproduce_experiment(self, experiment_id: str, 
                           new_results: Any,
                           tolerance: float = 0.05) -> ReproducibilityReport:
        """Reproduce experiment and compare results."""
        
        if experiment_id not in self.experiment_registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_record = self.experiment_registry[experiment_id]
        
        if not experiment_record.get('results_stored', False):
            raise ValueError(f"Original results for {experiment_id} not available")
        
        # Load original results
        results_path = Path(experiment_record['results_path'])
        with open(results_path, 'rb') as f:
            original_results = pickle.load(f)
        
        # Compare results
        comparison = self._compare_results(original_results, new_results, tolerance)
        
        # Generate current environment hash
        current_env = self._capture_environment()
        current_env_hash = self._generate_hash(current_env)
        
        # Create reproducibility report
        report = ReproducibilityReport(
            experiment_id=experiment_id,
            original_results=self._extract_numeric_results(original_results),
            reproduction_results=self._extract_numeric_results(new_results),
            correlation_coefficient=comparison['correlation'],
            mean_absolute_error=comparison['mae'],
            relative_error_percentage=comparison['relative_error'],
            statistical_test_p_value=comparison['statistical_test_p'],
            reproducibility_score=comparison['reproducibility_score'],
            environment_hash=current_env_hash,
            parameter_hash=experiment_record['config_hash'],
            is_reproducible=comparison['is_reproducible'],
            deviations=comparison['deviations'],
            recommendations=comparison['recommendations']
        )
        
        # Store reproduction attempt
        reproduction_path = self.storage_dir / f"{experiment_id}_reproduction_{int(time.time())}.json"
        with open(reproduction_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        return report
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture current environment information."""
        try:
            import platform
            import sys
            
            env_info = {
                'python_version': sys.version,
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'timestamp': time.time()
            }
            
            # Try to capture package versions
            try:
                import pkg_resources
                installed_packages = {pkg.key: pkg.version 
                                    for pkg in pkg_resources.working_set}
                env_info['packages'] = installed_packages
            except ImportError:
                pass
            
            # Hardware info
            try:
                env_info['cpu_count'] = psutil.cpu_count()
                env_info['memory_total'] = psutil.virtual_memory().total
            except ImportError:
                pass
            
            return env_info
            
        except Exception as e:
            return {'error': f"Failed to capture environment: {str(e)}"}
    
    def _generate_hash(self, data: Any) -> str:
        """Generate deterministic hash for data."""
        # Convert to JSON string with sorted keys for deterministic hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _compare_results(self, original: Any, reproduction: Any, 
                        tolerance: float) -> Dict[str, Any]:
        """Compare original and reproduction results."""
        
        # Extract numeric results
        orig_numeric = self._extract_numeric_results(original)
        repro_numeric = self._extract_numeric_results(reproduction)
        
        if len(orig_numeric) != len(repro_numeric):
            return {
                'is_reproducible': False,
                'reproducibility_score': 0.0,
                'correlation': 0.0,
                'mae': float('inf'),
                'relative_error': float('inf'),
                'statistical_test_p': 1.0,
                'deviations': ['Result array lengths differ'],
                'recommendations': ['Check experimental setup for consistency']
            }
        
        if len(orig_numeric) == 0:
            return {
                'is_reproducible': True,
                'reproducibility_score': 1.0,
                'correlation': 1.0,
                'mae': 0.0,
                'relative_error': 0.0,
                'statistical_test_p': 1.0,
                'deviations': [],
                'recommendations': []
            }
        
        # Calculate comparison metrics
        orig_array = np.array(orig_numeric)
        repro_array = np.array(repro_numeric)
        
        # Correlation
        correlation = np.corrcoef(orig_array, repro_array)[0, 1] if len(orig_array) > 1 else 1.0
        
        # Mean Absolute Error
        mae = np.mean(np.abs(orig_array - repro_array))
        
        # Relative Error
        non_zero_orig = orig_array[orig_array != 0]
        if len(non_zero_orig) > 0:
            relative_errors = np.abs((orig_array - repro_array) / orig_array)[orig_array != 0]
            relative_error = np.mean(relative_errors) * 100  # Percentage
        else:
            relative_error = 0.0 if mae < 1e-10 else 100.0
        
        # Statistical test (paired t-test)
        try:
            from scipy import stats
            if len(orig_array) > 1:
                _, p_value = stats.ttest_rel(orig_array, repro_array)
            else:
                p_value = 0.0 if mae < tolerance else 1.0
        except ImportError:
            p_value = 0.0 if mae < tolerance else 1.0
        
        # Reproducibility score
        reproducibility_score = max(0, 1 - relative_error / 100)
        
        # Is reproducible check
        is_reproducible = (mae < tolerance and 
                          correlation > 0.95 and 
                          relative_error < tolerance * 100)
        
        # Identify deviations
        deviations = []
        if mae >= tolerance:
            deviations.append(f"High mean absolute error: {mae:.6f}")
        if correlation < 0.95:
            deviations.append(f"Low correlation: {correlation:.4f}")
        if relative_error >= tolerance * 100:
            deviations.append(f"High relative error: {relative_error:.2f}%")
        
        # Generate recommendations
        recommendations = []
        if not is_reproducible:
            recommendations.append("Check random seed settings for consistency")
            recommendations.append("Verify identical experimental parameters")
            recommendations.append("Consider environment differences (hardware, software versions)")
            
            if correlation < 0.8:
                recommendations.append("Large systematic differences detected - review methodology")
        
        return {
            'is_reproducible': is_reproducible,
            'reproducibility_score': reproducibility_score,
            'correlation': correlation,
            'mae': mae,
            'relative_error': relative_error,
            'statistical_test_p': p_value,
            'deviations': deviations,
            'recommendations': recommendations
        }
    
    def _extract_numeric_results(self, results: Any) -> List[float]:
        """Extract numeric values from results for comparison."""
        numeric_values = []
        
        def extract_recursive(obj):
            if isinstance(obj, (int, float)):
                if not (np.isnan(obj) or np.isinf(obj)):
                    numeric_values.append(float(obj))
            elif isinstance(obj, (list, tuple, np.ndarray)):
                for item in np.array(obj).flatten():
                    extract_recursive(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
        
        extract_recursive(results)
        return numeric_values


class PerformanceMonitor:
    """Real-time performance monitoring for HDC operations."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.operation_stats = defaultdict(list)
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def record_operation(self, operation_name: str, execution_time: float,
                        memory_usage: float, success: bool = True) -> None:
        """Record individual operation performance."""
        
        # Get current system metrics
        cpu_usage = psutil.cpu_percent()
        
        # Calculate throughput (operations per second)
        recent_ops = [m for m in self.metrics_buffer 
                     if m.operation_name == operation_name 
                     and time.time() - m.timestamp < 60]  # Last minute
        
        throughput = len(recent_ops) / 60.0 if recent_ops else 0.0
        
        # Calculate latency percentiles for this operation
        recent_times = [m.execution_time_ms for m in recent_ops[-100:]]  # Last 100 ops
        if recent_times:
            latency_percentiles = {
                'p50': np.percentile(recent_times, 50),
                'p95': np.percentile(recent_times, 95),
                'p99': np.percentile(recent_times, 99)
            }
        else:
            latency_percentiles = {'p50': execution_time, 'p95': execution_time, 'p99': execution_time}
        
        # Calculate error rate
        recent_ops_all = [m for m in self.metrics_buffer 
                         if m.operation_name == operation_name 
                         and time.time() - m.timestamp < 300]  # Last 5 minutes
        
        if recent_ops_all:
            error_rate = 1 - np.mean([m.success_rate for m in recent_ops_all])
            success_rate = np.mean([m.success_rate for m in recent_ops_all])
        else:
            error_rate = 0.0 if success else 1.0
            success_rate = 1.0 if success else 0.0
        
        # Resource efficiency (ops per MB per second)
        resource_efficiency = throughput / max(memory_usage, 1.0)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time_ms=execution_time * 1000,  # Convert to ms
            memory_delta_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            error_rate=error_rate,
            success_rate=success_rate,
            resource_efficiency=resource_efficiency,
            timestamp=time.time()
        )
        
        # Store metrics
        self.metrics_buffer.append(metrics)
        self.operation_stats[operation_name].append(metrics)
    
    def get_performance_summary(self, operation_name: Optional[str] = None,
                              time_window: float = 3600) -> Dict[str, Any]:
        """Get performance summary for operations."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter metrics by time window and operation
        if operation_name:
            relevant_metrics = [m for m in self.metrics_buffer 
                              if m.operation_name == operation_name and m.timestamp > cutoff_time]
        else:
            relevant_metrics = [m for m in self.metrics_buffer if m.timestamp > cutoff_time]
        
        if not relevant_metrics:
            return {'message': 'No metrics available for specified criteria'}
        
        # Aggregate statistics
        execution_times = [m.execution_time_ms for m in relevant_metrics]
        memory_usage = [m.memory_delta_mb for m in relevant_metrics]
        cpu_usage = [m.cpu_usage_percent for m in relevant_metrics]
        throughputs = [m.throughput_ops_per_sec for m in relevant_metrics]
        error_rates = [m.error_rate for m in relevant_metrics]
        
        summary = {
            'operation_name': operation_name or 'all_operations',
            'time_window_hours': time_window / 3600,
            'total_operations': len(relevant_metrics),
            'execution_time_stats': {
                'mean_ms': np.mean(execution_times),
                'median_ms': np.median(execution_times),
                'std_ms': np.std(execution_times),
                'min_ms': np.min(execution_times),
                'max_ms': np.max(execution_times),
                'p95_ms': np.percentile(execution_times, 95),
                'p99_ms': np.percentile(execution_times, 99)
            },
            'memory_stats': {
                'mean_mb': np.mean(memory_usage),
                'max_mb': np.max(memory_usage),
                'total_mb': np.sum(memory_usage)
            },
            'cpu_stats': {
                'mean_percent': np.mean(cpu_usage),
                'max_percent': np.max(cpu_usage)
            },
            'throughput_stats': {
                'mean_ops_per_sec': np.mean(throughputs),
                'peak_ops_per_sec': np.max(throughputs)
            },
            'reliability_stats': {
                'mean_error_rate': np.mean(error_rates),
                'success_rate': 1 - np.mean(error_rates)
            },
            'performance_trends': self._calculate_performance_trends(relevant_metrics)
        }
        
        return summary
    
    def get_operation_comparison(self, operation_names: List[str],
                               metric: str = 'execution_time_ms') -> Dict[str, Any]:
        """Compare performance across different operations."""
        comparison = {}
        
        for op_name in operation_names:
            op_metrics = [m for m in self.metrics_buffer if m.operation_name == op_name]
            
            if op_metrics:
                if hasattr(op_metrics[0], metric):
                    values = [getattr(m, metric) for m in op_metrics[-100:]]  # Last 100
                    comparison[op_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                else:
                    comparison[op_name] = {'error': f'Metric {metric} not found'}
            else:
                comparison[op_name] = {'error': 'No metrics available'}
        
        # Add comparative analysis
        if len(comparison) > 1 and all('error' not in v for v in comparison.values()):
            means = [v['mean'] for v in comparison.values()]
            best_op = operation_names[np.argmin(means)]  # Assuming lower is better
            worst_op = operation_names[np.argmax(means)]
            
            comparison['analysis'] = {
                'best_performing': best_op,
                'worst_performing': worst_op,
                'performance_ratio': max(means) / min(means) if min(means) > 0 else float('inf')
            }
        
        return comparison
    
    def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # System-level monitoring
                system_metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'timestamp': time.time()
                }
                
                # Could store system metrics separately
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {str(e)}")
                time.sleep(interval)
    
    def _calculate_performance_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, str]:
        """Calculate performance trends over time."""
        if len(metrics) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Calculate trends for key metrics
        timestamps = [m.timestamp for m in sorted_metrics]
        execution_times = [m.execution_time_ms for m in sorted_metrics]
        memory_usage = [m.memory_delta_mb for m in sorted_metrics]
        
        # Linear regression for trends
        time_trend_exec = np.polyfit(timestamps, execution_times, 1)[0]
        time_trend_memory = np.polyfit(timestamps, memory_usage, 1)[0]
        
        trends = {}
        
        # Execution time trend
        if abs(time_trend_exec) < 0.01:
            trends['execution_time'] = 'stable'
        elif time_trend_exec > 0:
            trends['execution_time'] = 'degrading'
        else:
            trends['execution_time'] = 'improving'
        
        # Memory usage trend
        if abs(time_trend_memory) < 0.1:
            trends['memory_usage'] = 'stable'
        elif time_trend_memory > 0:
            trends['memory_usage'] = 'increasing'
        else:
            trends['memory_usage'] = 'decreasing'
        
        return trends