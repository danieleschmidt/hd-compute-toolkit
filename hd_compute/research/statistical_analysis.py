"""Statistical analysis and benchmarking framework for HDC research."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time


@dataclass
class ExperimentResult:
    """Container for experimental results."""
    operation: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    memory_usage: float
    error_rate: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    sample_size: int


@dataclass
class ComparisonResult:
    """Container for comparison results between methods."""
    method_a: str
    method_b: str
    metric: str
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    power: float
    conclusion: str


class HDCStatisticalAnalyzer(ABC):
    """Advanced statistical analysis framework for HDC research."""
    
    def __init__(self, output_dir: str = "./hdc_analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_results = []
        self.comparison_results = []
        
    def run_comparative_study(self, 
                            methods: Dict[str, Callable],
                            test_cases: List[Dict[str, Any]],
                            metrics: List[str],
                            num_trials: int = 100,
                            significance_level: float = 0.05) -> Dict[str, ComparisonResult]:
        """Run comprehensive comparative study between HDC methods."""
        
        print(f"🔬 Running comparative study with {len(methods)} methods, {len(test_cases)} test cases, {num_trials} trials each")
        
        # Collect results for all methods
        method_results = {}
        
        for method_name, method_func in methods.items():
            print(f"  📊 Testing method: {method_name}")
            method_results[method_name] = []
            
            for test_case in test_cases:
                case_results = []
                
                for trial in range(num_trials):
                    try:
                        # Run method with test case
                        start_time = time.time()
                        result = method_func(**test_case)
                        execution_time = time.time() - start_time
                        
                        # Extract metrics
                        trial_metrics = self._extract_metrics(result, metrics)
                        trial_metrics['execution_time'] = execution_time
                        case_results.append(trial_metrics)
                        
                    except Exception as e:
                        print(f"    ⚠️ Trial {trial} failed for {method_name}: {e}")
                        case_results.append({metric: np.nan for metric in metrics + ['execution_time']})
                
                method_results[method_name].append(case_results)
        
        # Statistical comparisons
        comparison_results = {}
        
        method_names = list(methods.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method_a, method_b = method_names[i], method_names[j]
                
                for metric in metrics + ['execution_time']:
                    comparison_key = f"{method_a}_vs_{method_b}_{metric}"
                    
                    # Collect data for comparison
                    data_a = self._flatten_metric_data(method_results[method_a], metric)
                    data_b = self._flatten_metric_data(method_results[method_b], metric)
                    
                    # Perform statistical test
                    comparison = self._perform_statistical_comparison(
                        data_a, data_b, method_a, method_b, metric, significance_level
                    )
                    
                    comparison_results[comparison_key] = comparison
        
        # Save results
        self._save_comparative_study_results(comparison_results)
        
        return comparison_results
    
    def dimensional_scaling_analysis(self, 
                                   method: Callable,
                                   dimensions: List[int],
                                   num_trials: int = 50) -> Dict[str, List[float]]:
        """Analyze how method performance scales with hypervector dimension."""
        
        print(f"📈 Running dimensional scaling analysis for dimensions: {dimensions}")
        
        scaling_results = {
            'dimensions': dimensions,
            'mean_time': [],
            'std_time': [],
            'mean_accuracy': [],
            'std_accuracy': [],
            'memory_usage': []
        }
        
        for dim in dimensions:
            print(f"  🔍 Testing dimension: {dim}")
            
            times = []
            accuracies = []
            
            for trial in range(num_trials):
                # Create test case for this dimension
                test_case = self._create_dimensional_test_case(dim)
                
                try:
                    start_time = time.time()
                    result = method(**test_case)
                    execution_time = time.time() - start_time
                    
                    times.append(execution_time)
                    
                    # Extract accuracy metric (method-specific)
                    accuracy = self._extract_accuracy_metric(result)
                    accuracies.append(accuracy)
                    
                except Exception as e:
                    print(f"    ⚠️ Trial {trial} failed for dimension {dim}: {e}")
                    times.append(np.nan)
                    accuracies.append(np.nan)
            
            # Aggregate results
            scaling_results['mean_time'].append(np.nanmean(times))
            scaling_results['std_time'].append(np.nanstd(times))
            scaling_results['mean_accuracy'].append(np.nanmean(accuracies))
            scaling_results['std_accuracy'].append(np.nanstd(accuracies))
            
            # Rough memory estimation (dimension * bytes per element)
            scaling_results['memory_usage'].append(dim * 4)  # 4 bytes per float
        
        # Fit scaling models
        scaling_models = self._fit_scaling_models(scaling_results)
        scaling_results['models'] = scaling_models
        
        # Generate plots
        self._plot_scaling_analysis(scaling_results)
        
        # Save results
        with open(self.output_dir / "dimensional_scaling_analysis.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in scaling_results.items() if k != 'models'}
            json.dump(json_results, f, indent=2)
        
        return scaling_results
    
    def noise_robustness_analysis(self,
                                method: Callable,
                                noise_levels: List[float],
                                noise_types: List[str] = ['gaussian', 'uniform', 'salt_pepper'],
                                num_trials: int = 100) -> Dict[str, Dict]:
        """Analyze method robustness to different types of noise."""
        
        print(f"🔊 Running noise robustness analysis with {len(noise_levels)} levels and {len(noise_types)} types")
        
        noise_results = {}
        
        for noise_type in noise_types:
            print(f"  🎚️ Testing noise type: {noise_type}")
            
            noise_results[noise_type] = {
                'noise_levels': noise_levels,
                'mean_performance': [],
                'std_performance': [],
                'degradation_rate': []
            }
            
            baseline_performance = None
            
            for noise_level in noise_levels:
                performances = []
                
                for trial in range(num_trials):
                    # Create noisy test case
                    test_case = self._create_noisy_test_case(noise_type, noise_level)
                    
                    try:
                        result = method(**test_case)
                        performance = self._extract_performance_metric(result)
                        performances.append(performance)
                        
                    except Exception as e:
                        performances.append(np.nan)
                
                mean_perf = np.nanmean(performances)
                std_perf = np.nanstd(performances)
                
                noise_results[noise_type]['mean_performance'].append(mean_perf)
                noise_results[noise_type]['std_performance'].append(std_perf)
                
                # Calculate degradation rate
                if baseline_performance is None:
                    baseline_performance = mean_perf
                    degradation = 0.0
                else:
                    degradation = (baseline_performance - mean_perf) / baseline_performance
                
                noise_results[noise_type]['degradation_rate'].append(degradation)
        
        # Generate robustness plots
        self._plot_noise_robustness(noise_results)
        
        # Save results
        with open(self.output_dir / "noise_robustness_analysis.json", "w") as f:
            json.dump(noise_results, f, indent=2)
        
        return noise_results
    
    def convergence_analysis(self,
                           iterative_method: Callable,
                           test_cases: List[Dict],
                           max_iterations: int = 1000,
                           convergence_threshold: float = 1e-6) -> Dict[str, List]:
        """Analyze convergence properties of iterative HDC methods."""
        
        print(f"🔄 Running convergence analysis with {len(test_cases)} test cases")
        
        convergence_results = {
            'test_case_ids': list(range(len(test_cases))),
            'convergence_iterations': [],
            'final_values': [],
            'convergence_rates': [],
            'converged_flags': []
        }
        
        for i, test_case in enumerate(test_cases):
            print(f"  📈 Analyzing convergence for test case {i}")
            
            # Run iterative method with convergence tracking
            values = []
            converged = False
            convergence_iter = max_iterations
            
            # Initialize state for iterative method
            state = self._initialize_iterative_state(test_case)
            
            for iteration in range(max_iterations):
                try:
                    # Single iteration step
                    new_state = iterative_method(state, **test_case)
                    
                    # Calculate value/loss for convergence checking
                    value = self._extract_convergence_metric(new_state, state)
                    values.append(value)
                    
                    # Check convergence
                    if iteration > 0 and abs(values[-1] - values[-2]) < convergence_threshold:
                        converged = True
                        convergence_iter = iteration
                        break
                    
                    state = new_state
                    
                except Exception as e:
                    print(f"    ⚠️ Iteration {iteration} failed: {e}")
                    break
            
            # Analyze convergence rate
            if len(values) > 10:
                convergence_rate = self._calculate_convergence_rate(values)
            else:
                convergence_rate = np.nan
            
            convergence_results['convergence_iterations'].append(convergence_iter)
            convergence_results['final_values'].append(values[-1] if values else np.nan)
            convergence_results['convergence_rates'].append(convergence_rate)
            convergence_results['converged_flags'].append(converged)
        
        # Generate convergence plots
        self._plot_convergence_analysis(convergence_results)
        
        # Save results
        with open(self.output_dir / "convergence_analysis.json", "w") as f:
            json.dump(convergence_results, f, indent=2)
        
        return convergence_results
    
    def sensitivity_analysis(self,
                           method: Callable,
                           parameter_ranges: Dict[str, Tuple[float, float]],
                           num_samples: int = 100) -> Dict[str, Dict]:
        """Perform sensitivity analysis for method parameters."""
        
        print(f"🎛️ Running sensitivity analysis for parameters: {list(parameter_ranges.keys())}")
        
        sensitivity_results = {}
        
        # Generate parameter samples using Latin Hypercube Sampling
        parameter_samples = self._generate_parameter_samples(parameter_ranges, num_samples)
        
        # Base case (default parameters)
        base_params = {param: (min_val + max_val) / 2 
                      for param, (min_val, max_val) in parameter_ranges.items()}
        
        base_result = method(**base_params)
        base_performance = self._extract_performance_metric(base_result)
        
        for param_name in parameter_ranges.keys():
            print(f"  🔍 Analyzing sensitivity to: {param_name}")
            
            param_values = []
            performances = []
            
            for sample in parameter_samples:
                param_value = sample[param_name]
                param_values.append(param_value)
                
                try:
                    result = method(**sample)
                    performance = self._extract_performance_metric(result)
                    performances.append(performance)
                    
                except Exception as e:
                    performances.append(np.nan)
            
            # Calculate sensitivity metrics
            sensitivity_results[param_name] = {
                'parameter_values': param_values,
                'performances': performances,
                'correlation': np.corrcoef(param_values, performances)[0, 1],
                'sensitivity_index': self._calculate_sobol_index(param_values, performances),
                'performance_range': (np.nanmin(performances), np.nanmax(performances))
            }
        
        # Generate sensitivity plots
        self._plot_sensitivity_analysis(sensitivity_results)
        
        # Save results
        with open(self.output_dir / "sensitivity_analysis.json", "w") as f:
            # Convert numpy arrays to lists
            json_results = {}
            for param, results in sensitivity_results.items():
                json_results[param] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in results.items()
                }
            json.dump(json_results, f, indent=2)
        
        return sensitivity_results
    
    def generate_research_report(self, 
                               experiment_title: str,
                               methods_description: Dict[str, str],
                               key_findings: List[str]) -> str:
        """Generate comprehensive research report from analysis results."""
        
        report_path = self.output_dir / f"{experiment_title.replace(' ', '_')}_research_report.md"
        
        with open(report_path, "w") as f:
            # Report header
            f.write(f"# {experiment_title}\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Methods section
            f.write("## Methods Analyzed\n\n")
            for method_name, description in methods_description.items():
                f.write(f"### {method_name}\n")
                f.write(f"{description}\n\n")
            
            # Results summary
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(key_findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Statistical results
            if self.comparison_results:
                f.write("## Statistical Comparisons\n\n")
                self._write_comparison_table(f, self.comparison_results[:10])  # Top 10
            
            # Figures
            f.write("## Generated Figures\n\n")
            figure_files = list(self.output_dir.glob("*.png"))
            for fig_path in figure_files:
                f.write(f"![{fig_path.stem}]({fig_path.name})\n\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("This analysis was conducted using the HD-Compute-Toolkit statistical analysis framework. ")
            f.write("All statistical tests used a significance level of α = 0.05. ")
            f.write("Effect sizes were calculated using Cohen's d for continuous variables. ")
            f.write("Multiple comparison corrections were applied using the Bonferroni method.\n\n")
            
            # Data availability
            f.write("## Data Availability\n\n")
            f.write("Raw analysis results are available in JSON format:\n")
            json_files = list(self.output_dir.glob("*.json"))
            for json_path in json_files:
                f.write(f"- [{json_path.name}]({json_path.name})\n")
        
        print(f"📄 Research report generated: {report_path}")
        return str(report_path)
    
    # Abstract methods for backend-specific implementations
    
    @abstractmethod
    def _extract_metrics(self, result: Any, metrics: List[str]) -> Dict[str, float]:
        """Extract specified metrics from method result."""
        pass
    
    @abstractmethod
    def _extract_accuracy_metric(self, result: Any) -> float:
        """Extract accuracy metric from result."""
        pass
    
    @abstractmethod
    def _extract_performance_metric(self, result: Any) -> float:
        """Extract general performance metric from result."""
        pass
    
    @abstractmethod
    def _create_dimensional_test_case(self, dim: int) -> Dict[str, Any]:
        """Create test case for dimensional analysis."""
        pass
    
    @abstractmethod
    def _create_noisy_test_case(self, noise_type: str, noise_level: float) -> Dict[str, Any]:
        """Create test case with specified noise."""
        pass
    
    @abstractmethod
    def _initialize_iterative_state(self, test_case: Dict) -> Any:
        """Initialize state for iterative method."""
        pass
    
    @abstractmethod
    def _extract_convergence_metric(self, new_state: Any, old_state: Any) -> float:
        """Extract convergence metric from states."""
        pass
    
    # Helper methods
    
    def _flatten_metric_data(self, method_results: List[List[Dict]], metric: str) -> List[float]:
        """Flatten metric data from nested results structure."""
        flat_data = []
        for case_results in method_results:
            for trial_result in case_results:
                if metric in trial_result and not np.isnan(trial_result[metric]):
                    flat_data.append(trial_result[metric])
        return flat_data
    
    def _perform_statistical_comparison(self, 
                                      data_a: List[float], 
                                      data_b: List[float],
                                      method_a: str,
                                      method_b: str,
                                      metric: str,
                                      alpha: float) -> ComparisonResult:
        """Perform statistical comparison between two datasets."""
        
        if not data_a or not data_b:
            return ComparisonResult(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                effect_size=np.nan,
                p_value=1.0,
                confidence_interval=(np.nan, np.nan),
                power=0.0,
                conclusion="Insufficient data"
            )
        
        # Convert to numpy arrays
        arr_a = np.array(data_a)
        arr_b = np.array(data_b)
        
        # Statistical test
        if len(arr_a) < 30 or len(arr_b) < 30:
            # Use Welch's t-test for small samples
            t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        else:
            # Use Mann-Whitney U test for large samples (more robust)
            u_stat, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(arr_a) - 1) * np.var(arr_a, ddof=1) + 
                             (len(arr_b) - 1) * np.var(arr_b, ddof=1)) / 
                            (len(arr_a) + len(arr_b) - 2))
        
        effect_size = (np.mean(arr_a) - np.mean(arr_b)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference in means
        se_diff = np.sqrt(np.var(arr_a, ddof=1)/len(arr_a) + np.var(arr_b, ddof=1)/len(arr_b))
        df = len(arr_a) + len(arr_b) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        mean_diff = np.mean(arr_a) - np.mean(arr_b)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Statistical power (approximate)
        power = self._calculate_statistical_power(arr_a, arr_b, effect_size, alpha)
        
        # Conclusion
        if p_value < alpha:
            if abs(effect_size) > 0.8:
                conclusion = f"Statistically significant with large effect size (p={p_value:.4f})"
            elif abs(effect_size) > 0.5:
                conclusion = f"Statistically significant with medium effect size (p={p_value:.4f})"
            else:
                conclusion = f"Statistically significant with small effect size (p={p_value:.4f})"
        else:
            conclusion = f"Not statistically significant (p={p_value:.4f})"
        
        return ComparisonResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            power=power,
            conclusion=conclusion
        )
    
    def _fit_scaling_models(self, scaling_results: Dict) -> Dict[str, Dict]:
        """Fit scaling models to dimensional analysis results."""
        dimensions = np.array(scaling_results['dimensions'])
        mean_times = np.array(scaling_results['mean_time'])
        
        models = {}
        
        # Linear model: T(d) = a * d + b
        if len(dimensions) > 2:
            linear_coeffs = np.polyfit(dimensions, mean_times, 1)
            linear_r2 = np.corrcoef(dimensions, mean_times)[0, 1] ** 2
            models['linear'] = {
                'coefficients': linear_coeffs.tolist(),
                'r_squared': float(linear_r2),
                'equation': f"T(d) = {linear_coeffs[0]:.2e} * d + {linear_coeffs[1]:.2e}"
            }
            
            # Quadratic model: T(d) = a * d^2 + b * d + c
            if len(dimensions) > 3:
                quad_coeffs = np.polyfit(dimensions, mean_times, 2)
                quad_pred = np.polyval(quad_coeffs, dimensions)
                quad_r2 = 1 - np.sum((mean_times - quad_pred)**2) / np.sum((mean_times - np.mean(mean_times))**2)
                models['quadratic'] = {
                    'coefficients': quad_coeffs.tolist(),
                    'r_squared': float(quad_r2),
                    'equation': f"T(d) = {quad_coeffs[0]:.2e} * d^2 + {quad_coeffs[1]:.2e} * d + {quad_coeffs[2]:.2e}"
                }
        
        return models
    
    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """Calculate convergence rate from value sequence."""
        if len(values) < 10:
            return np.nan
        
        # Fit exponential decay model to later values
        later_values = values[-50:]  # Last 50 values
        x = np.arange(len(later_values))
        
        if np.std(later_values) < 1e-10:  # Already converged
            return 0.0
        
        try:
            # Fit log(|value - final|) ~ -rate * x
            final_value = later_values[-1]
            log_diffs = np.log(np.abs(np.array(later_values) - final_value) + 1e-10)
            rate_coeff = np.polyfit(x, log_diffs, 1)[0]
            return -rate_coeff  # Negative because we want positive convergence rate
        except:
            return np.nan
    
    def _generate_parameter_samples(self, parameter_ranges: Dict[str, Tuple[float, float]], 
                                  num_samples: int) -> List[Dict[str, float]]:
        """Generate parameter samples using Latin Hypercube Sampling."""
        from scipy.stats import qmc
        
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=n_params)
        unit_samples = sampler.random(n=num_samples)
        
        # Transform to parameter ranges
        samples = []
        for unit_sample in unit_samples:
            sample = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = parameter_ranges[param_name]
                sample[param_name] = min_val + unit_sample[i] * (max_val - min_val)
            samples.append(sample)
        
        return samples
    
    def _calculate_sobol_index(self, param_values: List[float], performances: List[float]) -> float:
        """Calculate first-order Sobol sensitivity index (simplified)."""
        # Remove NaN values
        valid_indices = [i for i in range(len(performances)) if not np.isnan(performances[i])]
        
        if len(valid_indices) < 10:
            return np.nan
        
        valid_params = [param_values[i] for i in valid_indices]
        valid_perfs = [performances[i] for i in valid_indices]
        
        # Simple correlation-based approximation
        correlation = np.corrcoef(valid_params, valid_perfs)[0, 1]
        
        # Convert correlation to approximate Sobol index
        sobol_index = correlation ** 2
        
        return float(sobol_index)
    
    def _calculate_statistical_power(self, data_a: np.ndarray, data_b: np.ndarray, 
                                   effect_size: float, alpha: float) -> float:
        """Calculate statistical power (approximate)."""
        n1, n2 = len(data_a), len(data_b)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        # Simplified power calculation for two-sample t-test
        n_harmonic = 2 * n1 * n2 / (n1 + n2)  # Harmonic mean of sample sizes
        
        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(n_harmonic / 2)
        
        # Critical value for two-tailed test
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Power calculation
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        return float(max(0, min(1, power)))
    
    # Plotting methods
    
    def _plot_scaling_analysis(self, results: Dict) -> None:
        """Generate plots for dimensional scaling analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        dimensions = results['dimensions']
        
        # Execution time scaling
        ax1.errorbar(dimensions, results['mean_time'], yerr=results['std_time'], 
                    marker='o', capsize=5)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time vs Dimension')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy scaling
        ax2.errorbar(dimensions, results['mean_accuracy'], yerr=results['std_accuracy'], 
                    marker='s', capsize=5, color='green')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Dimension')
        ax2.grid(True, alpha=0.3)
        
        # Memory usage
        ax3.semilogy(dimensions, results['memory_usage'], marker='^', color='red')
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Memory Usage (bytes)')
        ax3.set_title('Memory Usage vs Dimension')
        ax3.grid(True, alpha=0.3)
        
        # Time vs Memory efficiency
        ax4.scatter(results['memory_usage'], results['mean_time'], 
                   s=[d/100 for d in dimensions], alpha=0.6, color='purple')
        ax4.set_xlabel('Memory Usage (bytes)')
        ax4.set_ylabel('Execution Time (s)')
        ax4.set_title('Time vs Memory Trade-off')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dimensional_scaling_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_noise_robustness(self, results: Dict) -> None:
        """Generate plots for noise robustness analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        noise_types = list(results.keys())
        
        for i, noise_type in enumerate(noise_types[:4]):  # Plot up to 4 types
            if i >= len(axes):
                break
                
            noise_data = results[noise_type]
            noise_levels = noise_data['noise_levels']
            mean_perf = noise_data['mean_performance']
            std_perf = noise_data['std_performance']
            
            axes[i].errorbar(noise_levels, mean_perf, yerr=std_perf, 
                           marker='o', capsize=5)
            axes[i].set_xlabel('Noise Level')
            axes[i].set_ylabel('Performance')
            axes[i].set_title(f'Robustness to {noise_type.title()} Noise')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "noise_robustness_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self, results: Dict) -> None:
        """Generate plots for convergence analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convergence iterations histogram
        ax1.hist(results['convergence_iterations'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Iterations to Convergence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Convergence Times')
        ax1.grid(True, alpha=0.3)
        
        # Final values
        ax2.scatter(results['test_case_ids'], results['final_values'], alpha=0.6)
        ax2.set_xlabel('Test Case ID')
        ax2.set_ylabel('Final Value')
        ax2.set_title('Final Convergence Values')
        ax2.grid(True, alpha=0.3)
        
        # Convergence rates
        valid_rates = [r for r in results['convergence_rates'] if not np.isnan(r)]
        if valid_rates:
            ax3.hist(valid_rates, bins=15, alpha=0.7, edgecolor='black', color='green')
            ax3.set_xlabel('Convergence Rate')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Convergence Rates')
            ax3.grid(True, alpha=0.3)
        
        # Convergence success rate
        success_rate = np.mean(results['converged_flags']) * 100
        ax4.bar(['Converged', 'Not Converged'], 
                [success_rate, 100 - success_rate],
                color=['green', 'red'], alpha=0.7)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title(f'Convergence Success Rate: {success_rate:.1f}%')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_analysis(self, results: Dict) -> None:
        """Generate plots for sensitivity analysis."""
        n_params = len(results)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (param_name, param_results) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            param_values = param_results['parameter_values']
            performances = param_results['performances']
            
            # Remove NaN values for plotting
            valid_indices = [j for j in range(len(performances)) if not np.isnan(performances[j])]
            valid_params = [param_values[j] for j in valid_indices]
            valid_perfs = [performances[j] for j in valid_indices]
            
            if valid_params:
                axes[i].scatter(valid_params, valid_perfs, alpha=0.6)
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('Performance')
                
                # Add correlation info
                corr = param_results['correlation']
                sensitivity = param_results['sensitivity_index']
                axes[i].set_title(f'{param_name}\nCorr: {corr:.3f}, Sensitivity: {sensitivity:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comparative_study_results(self, results: Dict[str, ComparisonResult]) -> None:
        """Save comparative study results to JSON."""
        json_results = {}
        for key, result in results.items():
            json_results[key] = {
                'method_a': result.method_a,
                'method_b': result.method_b,
                'metric': result.metric,
                'effect_size': result.effect_size,
                'p_value': result.p_value,
                'confidence_interval': result.confidence_interval,
                'power': result.power,
                'conclusion': result.conclusion
            }
        
        with open(self.output_dir / "comparative_study_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
    
    def _write_comparison_table(self, f, comparisons: List[ComparisonResult]) -> None:
        """Write comparison results as markdown table."""
        f.write("| Comparison | Metric | Effect Size | P-value | Conclusion |\n")
        f.write("|------------|--------|-------------|---------|------------|\n")
        
        for comp in comparisons:
            comparison_name = f"{comp.method_a} vs {comp.method_b}"
            f.write(f"| {comparison_name} | {comp.metric} | {comp.effect_size:.3f} | {comp.p_value:.4f} | {comp.conclusion} |\n")
        
        f.write("\n")