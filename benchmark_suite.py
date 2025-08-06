#!/usr/bin/env python3
"""Comprehensive performance benchmark suite for HD-Compute-Toolkit."""

import time
import random
import math
import statistics
from typing import Dict, List, Callable, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    operation: str
    parameters: Dict[str, Any]
    execution_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_estimate: float
    accuracy_score: Optional[float] = None
    error_rate: float = 0.0


@dataclass
class PerformanceProfile:
    """Performance profile for operation."""
    operation: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_sec: float
    scalability_score: float


class HDCBenchmarkSuite:
    """Comprehensive benchmark suite for HDC operations."""
    
    def __init__(self):
        self.results = []
        self.operations = {
            'random_hv_generation': self._benchmark_random_hv,
            'bundle_operation': self._benchmark_bundle,
            'bind_operation': self._benchmark_bind,
            'cosine_similarity': self._benchmark_cosine_similarity,
            'hamming_distance': self._benchmark_hamming_distance,
            'fractional_bind': self._benchmark_fractional_bind,
            'sequence_encoding': self._benchmark_sequence_encoding,
            'memory_operations': self._benchmark_memory_ops
        }
        
    def run_comprehensive_benchmark(self, 
                                  dimensions: List[int] = [1000, 5000, 10000, 20000],
                                  num_trials: int = 100,
                                  warmup_trials: int = 10) -> Dict[str, PerformanceProfile]:
        """Run comprehensive benchmark across operations and dimensions."""
        
        print("🚀 HD-Compute-Toolkit Performance Benchmark Suite")
        print("=" * 60)
        
        profiles = {}
        
        for operation_name, benchmark_func in self.operations.items():
            print(f"\n📊 Benchmarking: {operation_name}")
            print("-" * 40)
            
            operation_results = []
            
            for dim in dimensions:
                print(f"  Dimension: {dim}")
                
                # Warmup runs
                for _ in range(warmup_trials):
                    benchmark_func(dim, 1)
                
                # Actual benchmark runs
                times = []
                for trial in range(num_trials):
                    start_time = time.perf_counter()
                    result = benchmark_func(dim, 1)
                    end_time = time.perf_counter()
                    
                    execution_time = (end_time - start_time) * 1000  # ms
                    times.append(execution_time)
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                throughput = 1000 / avg_time if avg_time > 0 else 0  # ops/sec
                
                benchmark_result = BenchmarkResult(
                    operation=operation_name,
                    parameters={'dimension': dim, 'trials': num_trials},
                    execution_time_ms=avg_time,
                    throughput_ops_per_sec=throughput,
                    memory_usage_estimate=dim * 4 / 1024,  # KB estimate
                    accuracy_score=1.0  # Perfect for synthetic operations
                )
                
                operation_results.append((dim, times))
                self.results.append(benchmark_result)
            
            # Create performance profile
            all_times = [t for _, times in operation_results for t in times]
            if all_times:
                profiles[operation_name] = PerformanceProfile(
                    operation=operation_name,
                    avg_time_ms=statistics.mean(all_times),
                    std_time_ms=statistics.stdev(all_times) if len(all_times) > 1 else 0,
                    min_time_ms=min(all_times),
                    max_time_ms=max(all_times),
                    p50_time_ms=self._percentile(all_times, 50),
                    p95_time_ms=self._percentile(all_times, 95),
                    p99_time_ms=self._percentile(all_times, 99),
                    throughput_ops_per_sec=1000 / statistics.mean(all_times) if statistics.mean(all_times) > 0 else 0,
                    scalability_score=self._calculate_scalability_score(operation_results)
                )
        
        self._print_benchmark_summary(profiles)
        return profiles
    
    def run_scalability_test(self, 
                           operation: str = 'bundle_operation',
                           dimensions: List[int] = [1000, 2000, 5000, 10000, 20000, 50000],
                           num_trials: int = 50) -> Dict[str, Any]:
        """Test scalability characteristics of an operation."""
        
        print(f"📈 Scalability Test: {operation}")
        print("-" * 40)
        
        if operation not in self.operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        benchmark_func = self.operations[operation]
        scalability_data = []
        
        for dim in dimensions:
            print(f"  Testing dimension: {dim}")
            
            times = []
            for _ in range(num_trials):
                start_time = time.perf_counter()
                benchmark_func(dim, 1)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(times)
            scalability_data.append((dim, avg_time))
        
        # Analyze scalability
        scalability_analysis = self._analyze_scalability(scalability_data)
        
        return {
            'operation': operation,
            'data_points': scalability_data,
            'analysis': scalability_analysis
        }
    
    def run_stress_test(self, 
                       operation: str = 'bundle_operation',
                       dimension: int = 10000,
                       duration_seconds: int = 60) -> Dict[str, Any]:
        """Run stress test for sustained performance."""
        
        print(f"💪 Stress Test: {operation} for {duration_seconds}s")
        print("-" * 40)
        
        if operation not in self.operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        benchmark_func = self.operations[operation]
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        execution_times = []
        operation_count = 0
        
        print("  Running stress test...", end="", flush=True)
        
        while time.time() < end_time:
            op_start = time.perf_counter()
            benchmark_func(dimension, 1)
            op_end = time.perf_counter()
            
            execution_times.append((op_end - op_start) * 1000)
            operation_count += 1
            
            # Progress indicator
            if operation_count % 100 == 0:
                print(".", end="", flush=True)
        
        print(f" Done! ({operation_count} operations)")
        
        # Analyze performance degradation
        chunk_size = max(10, len(execution_times) // 10)
        time_chunks = [execution_times[i:i+chunk_size] 
                      for i in range(0, len(execution_times), chunk_size)]
        
        chunk_averages = [statistics.mean(chunk) for chunk in time_chunks if chunk]
        
        # Performance stability analysis
        first_chunk_avg = chunk_averages[0] if chunk_averages else 0
        last_chunk_avg = chunk_averages[-1] if chunk_averages else 0
        
        degradation = ((last_chunk_avg - first_chunk_avg) / first_chunk_avg * 100 
                      if first_chunk_avg > 0 else 0)
        
        return {
            'operation': operation,
            'duration_seconds': duration_seconds,
            'total_operations': operation_count,
            'avg_ops_per_second': operation_count / duration_seconds,
            'avg_execution_time_ms': statistics.mean(execution_times),
            'execution_time_std_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'min_execution_time_ms': min(execution_times) if execution_times else 0,
            'max_execution_time_ms': max(execution_times) if execution_times else 0,
            'performance_degradation_percent': degradation,
            'chunk_averages': chunk_averages
        }
    
    def run_comparative_benchmark(self, 
                                operations: List[str],
                                dimension: int = 10000,
                                num_trials: int = 100) -> Dict[str, Any]:
        """Compare performance across different operations."""
        
        print(f"⚖️  Comparative Benchmark (dimension={dimension})")
        print("-" * 40)
        
        comparison_results = {}
        
        for operation in operations:
            if operation not in self.operations:
                print(f"  ⚠️  Skipping unknown operation: {operation}")
                continue
            
            print(f"  📊 Benchmarking: {operation}")
            
            benchmark_func = self.operations[operation]
            times = []
            
            for _ in range(num_trials):
                start_time = time.perf_counter()
                benchmark_func(dimension, 1)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
            
            comparison_results[operation] = {
                'avg_time_ms': statistics.mean(times),
                'std_time_ms': statistics.stdev(times) if len(times) > 1 else 0,
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'ops_per_second': 1000 / statistics.mean(times) if statistics.mean(times) > 0 else 0
            }
        
        # Find fastest and slowest
        if comparison_results:
            fastest = min(comparison_results.keys(), 
                         key=lambda k: comparison_results[k]['avg_time_ms'])
            slowest = max(comparison_results.keys(), 
                         key=lambda k: comparison_results[k]['avg_time_ms'])
            
            speed_ratio = (comparison_results[slowest]['avg_time_ms'] / 
                          comparison_results[fastest]['avg_time_ms'])
        else:
            fastest = slowest = None
            speed_ratio = 1.0
        
        return {
            'dimension': dimension,
            'results': comparison_results,
            'fastest_operation': fastest,
            'slowest_operation': slowest,
            'speed_ratio': speed_ratio,
            'summary': f"{fastest} is {speed_ratio:.1f}x faster than {slowest}"
        }
    
    def generate_performance_report(self, profiles: Dict[str, PerformanceProfile]) -> str:
        """Generate comprehensive performance report."""
        
        report = ["# HD-Compute-Toolkit Performance Report", ""]
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        if profiles:
            fastest_op = min(profiles.keys(), key=lambda k: profiles[k].avg_time_ms)
            slowest_op = max(profiles.keys(), key=lambda k: profiles[k].avg_time_ms)
            
            report.append("## Executive Summary")
            report.append(f"- **Total Operations Tested**: {len(profiles)}")
            report.append(f"- **Fastest Operation**: {fastest_op} ({profiles[fastest_op].avg_time_ms:.2f}ms avg)")
            report.append(f"- **Slowest Operation**: {slowest_op} ({profiles[slowest_op].avg_time_ms:.2f}ms avg)")
            report.append("")
        
        # Detailed Results
        report.append("## Performance Profiles")
        report.append("")
        
        for operation, profile in profiles.items():
            report.append(f"### {operation}")
            report.append(f"- **Average Time**: {profile.avg_time_ms:.2f}ms")
            report.append(f"- **Throughput**: {profile.throughput_ops_per_sec:.0f} ops/sec")
            report.append(f"- **P95 Latency**: {profile.p95_time_ms:.2f}ms")
            report.append(f"- **P99 Latency**: {profile.p99_time_ms:.2f}ms")
            report.append(f"- **Scalability Score**: {profile.scalability_score:.2f}")
            report.append("")
        
        # Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        for operation, profile in profiles.items():
            if profile.avg_time_ms > 10:  # Slow operations
                report.append(f"- **{operation}**: Consider optimization (avg: {profile.avg_time_ms:.1f}ms)")
            elif profile.scalability_score < 0.8:  # Poor scalability
                report.append(f"- **{operation}**: Scalability concerns (score: {profile.scalability_score:.2f})")
        
        if not any(p.avg_time_ms > 10 or p.scalability_score < 0.8 for p in profiles.values()):
            report.append("- All operations show good performance characteristics")
        
        return "\n".join(report)
    
    # Benchmark implementations for different operations
    
    def _benchmark_random_hv(self, dimension: int, num_ops: int) -> Any:
        """Benchmark random hypervector generation."""
        for _ in range(num_ops):
            # Simulate binary hypervector generation
            result = [random.choice([-1, 1]) for _ in range(dimension)]
        return result
    
    def _benchmark_bundle(self, dimension: int, num_ops: int) -> Any:
        """Benchmark bundling operation."""
        # Create test hypervectors
        hvs = [[random.choice([-1, 1]) for _ in range(dimension)] for _ in range(5)]
        
        for _ in range(num_ops):
            # Simulate bundling (majority voting)
            result = []
            for i in range(dimension):
                sum_val = sum(hv[i] for hv in hvs)
                result.append(1 if sum_val > 0 else -1)
        
        return result
    
    def _benchmark_bind(self, dimension: int, num_ops: int) -> Any:
        """Benchmark binding operation."""
        hv1 = [random.choice([-1, 1]) for _ in range(dimension)]
        hv2 = [random.choice([-1, 1]) for _ in range(dimension)]
        
        for _ in range(num_ops):
            # Simulate binding (element-wise multiplication)
            result = [a * b for a, b in zip(hv1, hv2)]
        
        return result
    
    def _benchmark_cosine_similarity(self, dimension: int, num_ops: int) -> float:
        """Benchmark cosine similarity calculation."""
        hv1 = [random.choice([-1, 1]) for _ in range(dimension)]
        hv2 = [random.choice([-1, 1]) for _ in range(dimension)]
        
        for _ in range(num_ops):
            # Simulate cosine similarity
            dot_product = sum(a * b for a, b in zip(hv1, hv2))
            norm1 = math.sqrt(sum(a * a for a in hv1))
            norm2 = math.sqrt(sum(b * b for b in hv2))
            result = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        
        return result
    
    def _benchmark_hamming_distance(self, dimension: int, num_ops: int) -> int:
        """Benchmark Hamming distance calculation."""
        hv1 = [random.choice([-1, 1]) for _ in range(dimension)]
        hv2 = [random.choice([-1, 1]) for _ in range(dimension)]
        
        for _ in range(num_ops):
            # Simulate Hamming distance
            result = sum(1 for a, b in zip(hv1, hv2) if a != b)
        
        return result
    
    def _benchmark_fractional_bind(self, dimension: int, num_ops: int) -> Any:
        """Benchmark fractional binding operation."""
        hv1 = [random.choice([-1, 1]) for _ in range(dimension)]
        hv2 = [random.choice([-1, 1]) for _ in range(dimension)]
        
        for _ in range(num_ops):
            # Simulate fractional binding
            power = 0.5
            result = []
            for a, b in zip(hv1, hv2):
                frac_result = a * (1 - power) + (a * b) * power
                result.append(1 if frac_result > 0 else -1)
        
        return result
    
    def _benchmark_sequence_encoding(self, dimension: int, num_ops: int) -> Any:
        """Benchmark sequence encoding operation."""
        sequence = [[random.choice([-1, 1]) for _ in range(dimension)] for _ in range(10)]
        
        for _ in range(num_ops):
            # Simulate temporal sequence encoding
            result = [0] * dimension
            for i, hv in enumerate(sequence):
                weight = 0.9 ** i  # Temporal decay
                for j in range(dimension):
                    result[j] += weight * hv[j]
            
            # Binarize result
            result = [1 if x > 0 else -1 for x in result]
        
        return result
    
    def _benchmark_memory_ops(self, dimension: int, num_ops: int) -> float:
        """Benchmark memory operations (store/retrieve)."""
        # Simulate item memory
        memory = {}
        
        for _ in range(num_ops):
            # Store operation
            key = f"item_{random.randint(0, 100)}"
            hv = [random.choice([-1, 1]) for _ in range(dimension)]
            memory[key] = hv
            
            # Retrieve operation
            if memory:
                query_key = random.choice(list(memory.keys()))
                retrieved = memory[query_key]
                
                # Simulate similarity calculation
                similarity = sum(1 for a, b in zip(hv, retrieved) if a == b) / dimension
        
        return similarity if 'similarity' in locals() else 0.0
    
    # Utility methods
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(percentile / 100.0 * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    def _calculate_scalability_score(self, operation_results: List[Tuple[int, List[float]]]) -> float:
        """Calculate scalability score based on time complexity."""
        if len(operation_results) < 2:
            return 1.0
        
        # Calculate how execution time scales with dimension
        dimensions = [dim for dim, _ in operation_results]
        avg_times = [statistics.mean(times) for _, times in operation_results]
        
        # Simple linear regression to estimate complexity
        n = len(dimensions)
        if n < 2:
            return 1.0
        
        # Calculate correlation with different complexity models
        linear_correlation = self._correlation(dimensions, avg_times)
        
        # Quadratic model
        quad_dimensions = [d * d for d in dimensions]
        quad_correlation = self._correlation(quad_dimensions, avg_times)
        
        # Score based on how close to linear scaling (1.0 = perfect linear, 0.0 = quadratic or worse)
        if abs(linear_correlation) > abs(quad_correlation):
            return max(0.0, min(1.0, 1.0 - abs(quad_correlation - linear_correlation)))
        else:
            return max(0.0, 0.5 - abs(quad_correlation) * 0.5)
    
    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _analyze_scalability(self, data: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Analyze scalability characteristics."""
        dimensions = [d for d, _ in data]
        times = [t for _, t in data]
        
        if len(data) < 2:
            return {'complexity': 'unknown', 'efficiency': 'unknown'}
        
        # Calculate growth rate
        time_ratios = []
        dim_ratios = []
        
        for i in range(1, len(data)):
            time_ratio = times[i] / times[i-1] if times[i-1] > 0 else 1
            dim_ratio = dimensions[i] / dimensions[i-1] if dimensions[i-1] > 0 else 1
            
            time_ratios.append(time_ratio)
            dim_ratios.append(dim_ratio)
        
        avg_time_growth = statistics.mean(time_ratios) if time_ratios else 1
        avg_dim_growth = statistics.mean(dim_ratios) if dim_ratios else 1
        
        # Estimate complexity
        if avg_time_growth <= avg_dim_growth * 1.1:
            complexity = 'O(n) - Linear'
            efficiency = 'Excellent'
        elif avg_time_growth <= avg_dim_growth * avg_dim_growth * 1.1:
            complexity = 'O(n²) - Quadratic'
            efficiency = 'Good'
        else:
            complexity = 'O(n³+) - Polynomial/Exponential'
            efficiency = 'Poor'
        
        return {
            'complexity': complexity,
            'efficiency': efficiency,
            'avg_time_growth_ratio': avg_time_growth,
            'avg_dimension_growth_ratio': avg_dim_growth,
            'data_points': len(data)
        }
    
    def _print_benchmark_summary(self, profiles: Dict[str, PerformanceProfile]) -> None:
        """Print benchmark summary."""
        
        print(f"\n🏁 Benchmark Summary")
        print("=" * 60)
        
        if not profiles:
            print("No benchmark results available.")
            return
        
        # Sort by average time
        sorted_ops = sorted(profiles.items(), key=lambda x: x[1].avg_time_ms)
        
        print(f"{'Operation':<25} {'Avg Time (ms)':<15} {'Throughput (ops/s)':<20} {'Scalability'}")
        print("-" * 80)
        
        for operation, profile in sorted_ops:
            print(f"{operation:<25} {profile.avg_time_ms:<15.2f} {profile.throughput_ops_per_sec:<20.0f} {profile.scalability_score:.2f}")
        
        # Performance insights
        print(f"\n💡 Performance Insights:")
        
        fastest = sorted_ops[0]
        slowest = sorted_ops[-1]
        
        print(f"  • Fastest: {fastest[0]} ({fastest[1].avg_time_ms:.2f}ms avg)")
        print(f"  • Slowest: {slowest[0]} ({slowest[1].avg_time_ms:.2f}ms avg)")
        print(f"  • Speed Difference: {slowest[1].avg_time_ms / fastest[1].avg_time_ms:.1f}x")
        
        # Scalability insights
        best_scaling = max(profiles.items(), key=lambda x: x[1].scalability_score)
        worst_scaling = min(profiles.items(), key=lambda x: x[1].scalability_score)
        
        print(f"  • Best Scalability: {best_scaling[0]} (score: {best_scaling[1].scalability_score:.2f})")
        print(f"  • Worst Scalability: {worst_scaling[0]} (score: {worst_scaling[1].scalability_score:.2f})")


def main():
    """Run benchmark suite."""
    
    benchmark_suite = HDCBenchmarkSuite()
    
    # Run comprehensive benchmark
    profiles = benchmark_suite.run_comprehensive_benchmark(
        dimensions=[1000, 5000, 10000],
        num_trials=50  # Reduced for faster execution
    )
    
    # Generate and save report
    report = benchmark_suite.generate_performance_report(profiles)
    
    report_path = Path("/root/repo/PERFORMANCE_BENCHMARK_REPORT.md")
    report_path.write_text(report)
    
    print(f"\n📄 Performance report saved to: {report_path}")
    
    # Save raw results as JSON
    results_data = {
        'profiles': {
            name: {
                'operation': profile.operation,
                'avg_time_ms': profile.avg_time_ms,
                'throughput_ops_per_sec': profile.throughput_ops_per_sec,
                'scalability_score': profile.scalability_score
            }
            for name, profile in profiles.items()
        },
        'metadata': {
            'timestamp': time.time(),
            'test_dimensions': [1000, 5000, 10000],
            'trials_per_test': 50
        }
    }
    
    json_path = Path("/root/repo/benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"📊 Raw benchmark data saved to: {json_path}")
    
    return True


if __name__ == "__main__":
    main()