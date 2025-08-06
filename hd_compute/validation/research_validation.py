"""Research-grade validation for hyperdimensional computing operations."""

import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy import stats
import hashlib


@dataclass
class ValidationResult:
    """Result of validation check."""
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Dict[str, Any]
    timestamp: float
    

@dataclass
class IntegrityReport:
    """Hypervector integrity analysis report."""
    is_valid: bool
    dimension_check: bool
    sparsity_check: bool
    numerical_stability: bool
    statistical_properties: Dict[str, float]
    anomalies: List[str]
    recommendations: List[str]


class ResearchValidator(ABC):
    """Advanced validation framework for research-grade HDC operations."""
    
    def __init__(self, strict_mode: bool = True, log_level: int = logging.WARNING):
        self.strict_mode = strict_mode
        self.validation_history = []
        self.error_counts = defaultdict(int)
        
        # Setup logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def validate_operation(self, operation_name: str, *args, **kwargs) -> ValidationResult:
        """Validate HDC operation with comprehensive checks."""
        try:
            # Pre-operation validation
            pre_result = self._validate_inputs(operation_name, args, kwargs)
            if not pre_result.passed and self.strict_mode:
                return pre_result
            
            # Operation-specific validation
            operation_result = self._validate_operation_specific(operation_name, args, kwargs)
            if not operation_result.passed and self.strict_mode:
                return operation_result
            
            # Statistical validation
            stats_result = self._validate_statistical_properties(operation_name, args, kwargs)
            if not stats_result.passed and self.strict_mode:
                return stats_result
            
            # Combine results
            overall_result = ValidationResult(
                passed=pre_result.passed and operation_result.passed and stats_result.passed,
                message=f"Operation '{operation_name}' validation complete",
                severity='info' if pre_result.passed and operation_result.passed and stats_result.passed else 'warning',
                details={
                    'pre_validation': pre_result.details,
                    'operation_validation': operation_result.details,
                    'statistical_validation': stats_result.details
                },
                timestamp=self._current_timestamp()
            )
            
            # Record validation
            self.validation_history.append(overall_result)
            
            if not overall_result.passed:
                self.error_counts[operation_name] += 1
                
            return overall_result
            
        except Exception as e:
            error_result = ValidationResult(
                passed=False,
                message=f"Validation failed for '{operation_name}': {str(e)}",
                severity='error',
                details={'exception': str(e), 'type': type(e).__name__},
                timestamp=self._current_timestamp()
            )
            
            self.validation_history.append(error_result)
            self.error_counts[operation_name] += 1
            
            if self.strict_mode:
                raise
            
            return error_result
    
    def validate_research_hypothesis(self, hypothesis: str, 
                                   experimental_data: Dict[str, Any],
                                   expected_properties: Dict[str, Any]) -> ValidationResult:
        """Validate research hypothesis against experimental data."""
        validation_checks = []
        
        # Check data completeness
        required_fields = ['method_results', 'baseline_results', 'sample_size', 'metrics']
        for field in required_fields:
            if field not in experimental_data:
                validation_checks.append(f"Missing required field: {field}")
        
        # Statistical significance check
        if 'method_results' in experimental_data and 'baseline_results' in experimental_data:
            method_data = experimental_data['method_results']
            baseline_data = experimental_data['baseline_results']
            
            # Perform statistical test
            if len(method_data) > 1 and len(baseline_data) > 1:
                try:
                    statistic, p_value = stats.ttest_ind(method_data, baseline_data)
                    
                    if p_value > 0.05:
                        validation_checks.append(f"Results not statistically significant (p={p_value:.4f})")
                    
                    # Effect size check
                    effect_size = self._calculate_cohens_d(method_data, baseline_data)
                    if abs(effect_size) < 0.2:
                        validation_checks.append(f"Small effect size detected ({effect_size:.3f})")
                        
                except Exception as e:
                    validation_checks.append(f"Statistical test failed: {str(e)}")
        
        # Sample size adequacy
        if 'sample_size' in experimental_data:
            sample_size = experimental_data['sample_size']
            min_sample_size = expected_properties.get('min_sample_size', 30)
            
            if sample_size < min_sample_size:
                validation_checks.append(f"Insufficient sample size: {sample_size} < {min_sample_size}")
        
        # Expected property validation
        for prop_name, expected_value in expected_properties.items():
            if prop_name in experimental_data:
                actual_value = experimental_data[prop_name]
                
                if isinstance(expected_value, (int, float)):
                    tolerance = expected_properties.get(f"{prop_name}_tolerance", 0.1)
                    if abs(actual_value - expected_value) > tolerance:
                        validation_checks.append(
                            f"Property '{prop_name}' deviation: {actual_value} vs {expected_value} (tolerance: {tolerance})"
                        )
        
        # Create validation result
        passed = len(validation_checks) == 0
        
        return ValidationResult(
            passed=passed,
            message=f"Hypothesis '{hypothesis}' {'validated' if passed else 'failed validation'}",
            severity='info' if passed else 'warning',
            details={
                'hypothesis': hypothesis,
                'validation_checks': validation_checks,
                'experimental_data': experimental_data,
                'expected_properties': expected_properties
            },
            timestamp=self._current_timestamp()
        )
    
    def validate_reproducibility(self, experiment_config: Dict[str, Any],
                               results_set_1: List[Any],
                               results_set_2: List[Any],
                               tolerance: float = 0.05) -> ValidationResult:
        """Validate reproducibility of experimental results."""
        if len(results_set_1) != len(results_set_2):
            return ValidationResult(
                passed=False,
                message="Result sets have different lengths",
                severity='error',
                details={'length_1': len(results_set_1), 'length_2': len(results_set_2)},
                timestamp=self._current_timestamp()
            )
        
        # Calculate reproducibility metrics
        differences = []
        relative_errors = []
        
        for r1, r2 in zip(results_set_1, results_set_2):
            if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
                diff = abs(r1 - r2)
                differences.append(diff)
                
                if abs(r1) > 1e-10:  # Avoid division by very small numbers
                    rel_error = diff / abs(r1)
                    relative_errors.append(rel_error)
        
        # Statistical measures of reproducibility
        mean_abs_diff = np.mean(differences) if differences else np.inf
        max_abs_diff = np.max(differences) if differences else np.inf
        mean_rel_error = np.mean(relative_errors) if relative_errors else np.inf
        
        # Reproducibility tests
        reproducible = (mean_abs_diff < tolerance and mean_rel_error < tolerance)
        
        # Additional checks for hypervector results
        if hasattr(results_set_1[0], '__len__') and hasattr(results_set_2[0], '__len__'):
            # Hypervector similarity check
            similarities = []
            for r1, r2 in zip(results_set_1, results_set_2):
                similarity = self._calculate_similarity(r1, r2)
                similarities.append(similarity)
            
            mean_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            similarity_threshold = 1.0 - tolerance
            reproducible = reproducible and (mean_similarity > similarity_threshold)
        else:
            mean_similarity = None
            min_similarity = None
        
        return ValidationResult(
            passed=reproducible,
            message=f"Reproducibility {'validated' if reproducible else 'failed'}",
            severity='info' if reproducible else 'error',
            details={
                'mean_absolute_difference': mean_abs_diff,
                'max_absolute_difference': max_abs_diff,
                'mean_relative_error': mean_rel_error,
                'mean_similarity': mean_similarity,
                'min_similarity': min_similarity,
                'tolerance': tolerance,
                'experiment_config': experiment_config
            },
            timestamp=self._current_timestamp()
        )
    
    @abstractmethod
    def _validate_inputs(self, operation_name: str, args: tuple, kwargs: dict) -> ValidationResult:
        """Validate operation inputs."""
        pass
    
    @abstractmethod
    def _validate_operation_specific(self, operation_name: str, args: tuple, kwargs: dict) -> ValidationResult:
        """Operation-specific validation."""
        pass
    
    @abstractmethod
    def _validate_statistical_properties(self, operation_name: str, args: tuple, kwargs: dict) -> ValidationResult:
        """Validate statistical properties of operation."""
        pass
    
    @abstractmethod
    def _calculate_similarity(self, hv1: Any, hv2: Any) -> float:
        """Calculate similarity between hypervectors."""
        pass
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _current_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_history:
            return {'message': 'No validations performed'}
        
        passed_count = sum(1 for result in self.validation_history if result.passed)
        failed_count = len(self.validation_history) - passed_count
        
        severity_counts = defaultdict(int)
        for result in self.validation_history:
            severity_counts[result.severity] += 1
        
        return {
            'total_validations': len(self.validation_history),
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': passed_count / len(self.validation_history),
            'severity_breakdown': dict(severity_counts),
            'error_counts_by_operation': dict(self.error_counts),
            'most_recent': self.validation_history[-1].message if self.validation_history else None
        }


class ExperimentalValidation:
    """Validation for experimental research methodologies."""
    
    def __init__(self):
        self.experimental_standards = {
            'min_sample_size': 30,
            'min_effect_size': 0.2,
            'significance_level': 0.05,
            'statistical_power': 0.8
        }
    
    def validate_experimental_design(self, design: Dict[str, Any]) -> ValidationResult:
        """Validate experimental design for research."""
        issues = []
        
        # Check for control group
        if 'control_group' not in design or not design['control_group']:
            issues.append("Missing or empty control group")
        
        # Check randomization
        if not design.get('randomized', False):
            issues.append("Experiment not randomized")
        
        # Check blinding
        if not design.get('blinded', False):
            issues.append("Experiment not blinded - may introduce bias")
        
        # Check sample size calculation
        if 'power_analysis' not in design:
            issues.append("Missing power analysis for sample size determination")
        
        # Check multiple comparisons
        if design.get('multiple_comparisons', 0) > 5:
            if 'correction_method' not in design:
                issues.append("Multiple comparisons without correction method")
        
        # Check outcome measures
        if 'primary_outcome' not in design:
            issues.append("Primary outcome measure not specified")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            passed=passed,
            message=f"Experimental design {'validated' if passed else 'has issues'}",
            severity='info' if passed else 'warning',
            details={
                'issues': issues,
                'design': design,
                'standards': self.experimental_standards
            },
            timestamp=time.time()
        )
    
    def validate_statistical_analysis_plan(self, analysis_plan: Dict[str, Any]) -> ValidationResult:
        """Validate statistical analysis plan."""
        issues = []
        
        # Check analysis methods
        required_methods = ['primary_analysis', 'assumptions_check', 'effect_size_calculation']
        for method in required_methods:
            if method not in analysis_plan:
                issues.append(f"Missing {method} in analysis plan")
        
        # Check for pre-specification
        if not analysis_plan.get('pre_specified', False):
            issues.append("Analysis plan not pre-specified - risk of p-hacking")
        
        # Check handling of missing data
        if 'missing_data_strategy' not in analysis_plan:
            issues.append("No strategy specified for handling missing data")
        
        # Check sensitivity analyses
        if 'sensitivity_analyses' not in analysis_plan:
            issues.append("No sensitivity analyses planned")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            passed=passed,
            message=f"Statistical analysis plan {'validated' if passed else 'needs improvement'}",
            severity='info' if passed else 'warning',
            details={
                'issues': issues,
                'analysis_plan': analysis_plan
            },
            timestamp=time.time()
        )


class StatisticalValidation:
    """Statistical validation for HDC research."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def validate_distribution_assumptions(self, data: np.ndarray, 
                                        test_normality: bool = True,
                                        test_homogeneity: bool = True) -> ValidationResult:
        """Validate statistical distribution assumptions."""
        issues = []
        test_results = {}
        
        if len(data) < 3:
            return ValidationResult(
                passed=False,
                message="Insufficient data for distribution testing",
                severity='error',
                details={'data_length': len(data)},
                timestamp=time.time()
            )
        
        # Test normality
        if test_normality and len(data) >= 8:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                test_results['shapiro_test'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                
                if shapiro_p < self.significance_level:
                    issues.append(f"Data may not be normally distributed (Shapiro p={shapiro_p:.4f})")
            except Exception as e:
                issues.append(f"Normality test failed: {str(e)}")
        
        # Test for outliers
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        
        outliers = data[(data < q1 - outlier_threshold) | (data > q3 + outlier_threshold)]
        outlier_proportion = len(outliers) / len(data)
        
        test_results['outlier_analysis'] = {
            'outlier_count': len(outliers),
            'outlier_proportion': outlier_proportion,
            'threshold': outlier_threshold
        }
        
        if outlier_proportion > 0.1:  # More than 10% outliers
            issues.append(f"High proportion of outliers detected ({outlier_proportion:.2%})")
        
        # Test homogeneity (if data can be split)
        if test_homogeneity and len(data) >= 20:
            try:
                # Split data in half and test variance equality
                mid = len(data) // 2
                group1, group2 = data[:mid], data[mid:]
                
                levene_stat, levene_p = stats.levene(group1, group2)
                test_results['levene_test'] = {'statistic': levene_stat, 'p_value': levene_p}
                
                if levene_p < self.significance_level:
                    issues.append(f"Unequal variances detected (Levene p={levene_p:.4f})")
            except Exception as e:
                issues.append(f"Homogeneity test failed: {str(e)}")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            passed=passed,
            message=f"Distribution assumptions {'validated' if passed else 'violated'}",
            severity='info' if passed else 'warning',
            details={
                'issues': issues,
                'test_results': test_results,
                'data_summary': {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
            },
            timestamp=time.time()
        )
    
    def validate_effect_size(self, effect_size: float, 
                           effect_type: str = 'cohens_d',
                           context: str = 'behavioral') -> ValidationResult:
        """Validate effect size magnitude and interpretation."""
        # Cohen's conventions (adjusted for context)
        if context == 'behavioral':
            thresholds = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
        elif context == 'medical':
            thresholds = {'small': 0.1, 'medium': 0.3, 'large': 0.5}
        else:  # computational/technical
            thresholds = {'small': 0.1, 'medium': 0.25, 'large': 0.4}
        
        abs_effect = abs(effect_size)
        
        if abs_effect < thresholds['small']:
            magnitude = 'negligible'
            interpretation = 'Effect size is very small and may not be practically significant'
            severity = 'warning'
        elif abs_effect < thresholds['medium']:
            magnitude = 'small'
            interpretation = 'Small effect size - may have limited practical significance'
            severity = 'info'
        elif abs_effect < thresholds['large']:
            magnitude = 'medium'
            interpretation = 'Medium effect size - moderate practical significance'
            severity = 'info'
        else:
            magnitude = 'large'
            interpretation = 'Large effect size - high practical significance'
            severity = 'info'
        
        return ValidationResult(
            passed=abs_effect >= thresholds['small'],
            message=f"Effect size ({effect_size:.3f}) classified as {magnitude}",
            severity=severity,
            details={
                'effect_size': effect_size,
                'magnitude': magnitude,
                'interpretation': interpretation,
                'context': context,
                'thresholds': thresholds
            },
            timestamp=time.time()
        )


class HypervectorIntegrityChecker:
    """Specialized checker for hypervector data integrity."""
    
    def __init__(self, expected_dim: int, expected_sparsity: Optional[float] = None):
        self.expected_dim = expected_dim
        self.expected_sparsity = expected_sparsity
    
    def check_integrity(self, hv: Any, hv_name: str = "hypervector") -> IntegrityReport:
        """Comprehensive integrity check for hypervector."""
        issues = []
        recommendations = []
        
        # Convert to numpy for analysis
        if hasattr(hv, 'numpy'):
            hv_array = hv.numpy()
        elif hasattr(hv, 'cpu'):
            hv_array = hv.cpu().numpy()
        else:
            hv_array = np.array(hv)
        
        # Dimension check
        dimension_ok = len(hv_array.shape) == 1 and hv_array.shape[0] == self.expected_dim
        if not dimension_ok:
            issues.append(f"Dimension mismatch: expected {self.expected_dim}, got {hv_array.shape}")
        
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(hv_array))
        has_inf = np.any(np.isinf(hv_array))
        
        numerical_stability = not (has_nan or has_inf)
        if has_nan:
            issues.append("Contains NaN values")
            recommendations.append("Check for numerical overflow or invalid operations")
        if has_inf:
            issues.append("Contains infinite values")
            recommendations.append("Implement numerical bounds checking")
        
        # Sparsity check
        if np.all(hv_array == 0):
            issues.append("Hypervector is all zeros")
            recommendations.append("Verify random generation or binding operations")
        
        if self.expected_sparsity is not None:
            actual_sparsity = np.mean(hv_array == 0)
            sparsity_tolerance = 0.1
            sparsity_ok = abs(actual_sparsity - self.expected_sparsity) <= sparsity_tolerance
            
            if not sparsity_ok:
                issues.append(f"Sparsity mismatch: expected {self.expected_sparsity:.2f}, got {actual_sparsity:.2f}")
        else:
            sparsity_ok = True
        
        # Statistical properties
        stats_props = {
            'mean': float(np.mean(hv_array)),
            'std': float(np.std(hv_array)),
            'min': float(np.min(hv_array)),
            'max': float(np.max(hv_array)),
            'norm': float(np.linalg.norm(hv_array)),
            'sparsity': float(np.mean(hv_array == 0)),
            'unique_values': int(len(np.unique(hv_array)))
        }
        
        # Check for suspicious patterns
        if stats_props['std'] < 1e-10:
            issues.append("Extremely low variance - may be constant vector")
        
        if stats_props['unique_values'] < 2:
            issues.append(f"Only {stats_props['unique_values']} unique values")
        
        # Dynamic range check
        if stats_props['max'] - stats_props['min'] < 1e-6:
            issues.append("Very small dynamic range")
        
        # Norm check (for normalized vectors)
        if abs(stats_props['norm'] - 1.0) > 0.1:
            recommendations.append("Consider normalizing hypervector")
        
        # Check for binary hypervector properties
        unique_vals = np.unique(hv_array)
        if len(unique_vals) == 2 and set(unique_vals) == {-1, 1}:
            recommendations.append("Detected binary hypervector (+1/-1)")
        elif len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            recommendations.append("Detected binary hypervector (0/1)")
        
        is_valid = len(issues) == 0
        
        return IntegrityReport(
            is_valid=is_valid,
            dimension_check=dimension_ok,
            sparsity_check=sparsity_ok,
            numerical_stability=numerical_stability,
            statistical_properties=stats_props,
            anomalies=issues,
            recommendations=recommendations
        )
    
    def batch_integrity_check(self, hvs: List[Any], 
                            names: Optional[List[str]] = None) -> Dict[str, IntegrityReport]:
        """Check integrity of multiple hypervectors."""
        if names is None:
            names = [f"hv_{i}" for i in range(len(hvs))]
        
        reports = {}
        for hv, name in zip(hvs, names):
            reports[name] = self.check_integrity(hv, name)
        
        return reports
    
    def generate_integrity_summary(self, reports: Dict[str, IntegrityReport]) -> Dict[str, Any]:
        """Generate summary of integrity checks."""
        total_hvs = len(reports)
        valid_hvs = sum(1 for report in reports.values() if report.is_valid)
        
        common_issues = defaultdict(int)
        all_recommendations = set()
        
        for report in reports.values():
            for anomaly in report.anomalies:
                common_issues[anomaly] += 1
            all_recommendations.update(report.recommendations)
        
        return {
            'total_hypervectors': total_hvs,
            'valid_hypervectors': valid_hvs,
            'validation_rate': valid_hvs / total_hvs if total_hvs > 0 else 0,
            'common_issues': dict(common_issues),
            'recommendations': list(all_recommendations),
            'summary': f"{valid_hvs}/{total_hvs} hypervectors passed integrity checks"
        }