#!/usr/bin/env python3
"""
Automated metrics collection script for HD-Compute-Toolkit.
Collects performance, quality, and operational metrics for dashboard reporting.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects various metrics for the HD-Compute-Toolkit project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics = {}
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        logger.info("Starting comprehensive metrics collection...")
        
        try:
            self.metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "code_quality": self.collect_code_quality_metrics(),
                "performance": self.collect_performance_metrics(),
                "test_coverage": self.collect_test_coverage_metrics(),
                "security": self.collect_security_metrics(),
                "dependencies": self.collect_dependency_metrics(),
                "git": self.collect_git_metrics(),
                "build": self.collect_build_metrics(),
                "operational": self.collect_operational_metrics(),
            }
            
            logger.info("Metrics collection completed successfully")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics using various tools."""
        logger.info("Collecting code quality metrics...")
        
        metrics = {}
        
        # Flake8 linting
        try:
            result = subprocess.run(
                ["flake8", "hd_compute", "--format=json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            flake8_issues = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["linting_issues"] = flake8_issues
        except Exception as e:
            logger.warning(f"Could not collect flake8 metrics: {e}")
            metrics["linting_issues"] = None
        
        # MyPy type checking
        try:
            result = subprocess.run(
                ["mypy", "hd_compute", "--json-report", "/tmp/mypy-report"],
                capture_output=True, text=True, cwd=self.project_root
            )
            # Parse mypy JSON report
            try:
                with open("/tmp/mypy-report/index.txt", "r") as f:
                    mypy_summary = f.read()
                    metrics["type_check_errors"] = mypy_summary.count("error:")
            except FileNotFoundError:
                metrics["type_check_errors"] = 0 if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"Could not collect mypy metrics: {e}")
            metrics["type_check_errors"] = None
        
        # Code complexity (using radon)
        try:
            result = subprocess.run(
                ["radon", "cc", "hd_compute", "-j"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = sum(
                    func["complexity"] for file_data in complexity_data.values()
                    for func in file_data
                )
                function_count = sum(len(file_data) for file_data in complexity_data.values())
                metrics["average_complexity"] = total_complexity / function_count if function_count > 0 else 0
            else:
                metrics["average_complexity"] = None
        except Exception as e:
            logger.warning(f"Could not collect complexity metrics: {e}")
            metrics["average_complexity"] = None
        
        # Lines of code
        try:
            result = subprocess.run(
                ["find", "hd_compute", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = int(lines[-1].split()[0]) if lines else 0
                metrics["lines_of_code"] = total_lines
            else:
                metrics["lines_of_code"] = None
        except Exception as e:
            logger.warning(f"Could not collect LOC metrics: {e}")
            metrics["lines_of_code"] = None
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance benchmarks."""
        logger.info("Collecting performance metrics...")
        
        metrics = {}
        
        try:
            # Run benchmark suite
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/performance/", "-m", "benchmark", 
                 "--benchmark-json=/tmp/benchmark-results.json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0:
                with open("/tmp/benchmark-results.json", "r") as f:
                    benchmark_data = json.load(f)
                
                benchmarks = {}
                for bench in benchmark_data.get("benchmarks", []):
                    name = bench["name"]
                    stats = bench["stats"]
                    benchmarks[name] = {
                        "mean_ms": stats["mean"] * 1000,
                        "min_ms": stats["min"] * 1000,
                        "max_ms": stats["max"] * 1000,
                        "stddev_ms": stats["stddev"] * 1000,
                        "iterations": stats["iterations"]
                    }
                
                metrics["benchmarks"] = benchmarks
                
                # Calculate overall performance score
                cpu_times = [b["mean_ms"] for b in benchmarks.values() if "cpu" in b]
                gpu_times = [b["mean_ms"] for b in benchmarks.values() if "gpu" in b]
                
                if cpu_times and gpu_times:
                    avg_cpu = sum(cpu_times) / len(cpu_times)
                    avg_gpu = sum(gpu_times) / len(gpu_times)
                    metrics["gpu_acceleration_factor"] = avg_cpu / avg_gpu
                
            else:
                logger.warning("Benchmark tests failed")
                metrics["benchmarks"] = {}
                
        except Exception as e:
            logger.warning(f"Could not collect performance metrics: {e}")
            metrics["benchmarks"] = {}
        
        return metrics
    
    def collect_test_coverage_metrics(self) -> Dict[str, Any]:
        """Collect test coverage information."""
        logger.info("Collecting test coverage metrics...")
        
        metrics = {}
        
        try:
            # Run tests with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=hd_compute", "--cov-report=json:/tmp/coverage.json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0:
                with open("/tmp/coverage.json", "r") as f:
                    coverage_data = json.load(f)
                
                metrics.update({
                    "line_coverage": coverage_data["totals"]["percent_covered"],
                    "lines_covered": coverage_data["totals"]["covered_lines"],
                    "lines_total": coverage_data["totals"]["num_statements"],
                    "missing_lines": coverage_data["totals"]["missing_lines"],
                    "excluded_lines": coverage_data["totals"]["excluded_lines"]
                })
                
                # File-level coverage
                file_coverage = {}
                for filename, file_data in coverage_data["files"].items():
                    if filename.startswith("hd_compute/"):
                        file_coverage[filename] = file_data["summary"]["percent_covered"]
                
                metrics["file_coverage"] = file_coverage
                
        except Exception as e:
            logger.warning(f"Could not collect coverage metrics: {e}")
            metrics = {"line_coverage": None}
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        
        metrics = {}
        
        # Bandit security analysis
        try:
            result = subprocess.run(
                ["bandit", "-r", "hd_compute/", "-f", "json", "-o", "/tmp/bandit-report.json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            with open("/tmp/bandit-report.json", "r") as f:
                bandit_data = json.load(f)
            
            metrics["security_issues"] = {
                "high": len([r for r in bandit_data.get("results", []) if r["issue_severity"] == "HIGH"]),
                "medium": len([r for r in bandit_data.get("results", []) if r["issue_severity"] == "MEDIUM"]),
                "low": len([r for r in bandit_data.get("results", []) if r["issue_severity"] == "LOW"]),
                "total": len(bandit_data.get("results", []))
            }
            
        except Exception as e:
            logger.warning(f"Could not collect bandit security metrics: {e}")
            metrics["security_issues"] = None
        
        # Safety vulnerability scan
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                metrics["vulnerabilities"] = len(safety_data)
            else:
                metrics["vulnerabilities"] = 0
                
        except Exception as e:
            logger.warning(f"Could not collect safety vulnerability metrics: {e}")
            metrics["vulnerabilities"] = None
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        logger.info("Collecting dependency metrics...")
        
        metrics = {}
        
        try:
            # Parse pyproject.toml for dependencies
            import tomli
            
            with open(self.project_root / "pyproject.toml", "rb") as f:
                pyproject_data = tomli.load(f)
            
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
            
            metrics.update({
                "total_dependencies": len(dependencies),
                "optional_dependency_groups": len(optional_deps),
                "total_optional_dependencies": sum(len(deps) for deps in optional_deps.values()),
                "dependency_groups": list(optional_deps.keys())
            })
            
        except Exception as e:
            logger.warning(f"Could not collect dependency metrics: {e}")
            metrics = {"total_dependencies": None}
        
        return metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect git repository metrics."""
        logger.info("Collecting git metrics...")
        
        metrics = {}
        
        try:
            # Commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            )
            metrics["total_commits"] = int(result.stdout.strip()) if result.returncode == 0 else None
            
            # Contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                contributors = len(result.stdout.strip().split('\n'))
                metrics["contributors"] = contributors
            
            # Recent activity (last 30 days)
            result = subprocess.run(
                ["git", "log", "--since='30 days ago'", "--oneline"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                metrics["commits_last_30_days"] = recent_commits
            
            # Current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=self.project_root
            )
            metrics["current_branch"] = result.stdout.strip() if result.returncode == 0 else None
            
        except Exception as e:
            logger.warning(f"Could not collect git metrics: {e}")
        
        return metrics
    
    def collect_build_metrics(self) -> Dict[str, Any]:
        """Collect build-related metrics."""
        logger.info("Collecting build metrics...")
        
        metrics = {}
        
        try:
            # Build time estimation
            start_time = time.time()
            
            result = subprocess.run(
                ["python", "-m", "build", "--wheel", "--outdir", "/tmp/build-test"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            build_time = time.time() - start_time
            
            metrics.update({
                "build_success": result.returncode == 0,
                "build_time_seconds": build_time,
                "build_output_size_mb": self._get_wheel_size("/tmp/build-test")
            })
            
        except Exception as e:
            logger.warning(f"Could not collect build metrics: {e}")
            metrics = {"build_success": None}
        
        return metrics
    
    def collect_operational_metrics(self) -> Dict[str, Any]:
        """Collect operational/runtime metrics."""
        logger.info("Collecting operational metrics...")
        
        metrics = {}
        
        try:
            # Import test to check basic functionality
            start_time = time.time()
            result = subprocess.run(
                ["python", "-c", "import hd_compute; print('Import successful')"],
                capture_output=True, text=True, cwd=self.project_root
            )
            import_time = time.time() - start_time
            
            metrics.update({
                "import_success": result.returncode == 0,
                "import_time_seconds": import_time
            })
            
            # Check GPU availability
            gpu_result = subprocess.run(
                ["python", "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if gpu_result.returncode == 0:
                metrics["gpu_available"] = gpu_result.stdout.strip() == "True"
            
        except Exception as e:
            logger.warning(f"Could not collect operational metrics: {e}")
        
        return metrics
    
    def _get_wheel_size(self, build_dir: str) -> Optional[float]:
        """Get the size of the built wheel in MB."""
        try:
            wheel_files = list(Path(build_dir).glob("*.whl"))
            if wheel_files:
                size_bytes = wheel_files[0].stat().st_size
                return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            pass
        return None
    
    def save_metrics(self, output_file: Path) -> None:
        """Save collected metrics to JSON file."""
        logger.info(f"Saving metrics to {output_file}")
        
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def print_summary(self) -> None:
        """Print a summary of collected metrics."""
        print("\n" + "="*60)
        print("HD-COMPUTE-TOOLKIT METRICS SUMMARY")
        print("="*60)
        
        # Code Quality
        if "code_quality" in self.metrics:
            cq = self.metrics["code_quality"]
            print(f"\n📊 Code Quality:")
            print(f"  • Lines of Code: {cq.get('lines_of_code', 'N/A')}")
            print(f"  • Linting Issues: {cq.get('linting_issues', 'N/A')}")
            print(f"  • Type Check Errors: {cq.get('type_check_errors', 'N/A')}")
            print(f"  • Average Complexity: {cq.get('average_complexity', 'N/A'):.2f}" if cq.get('average_complexity') else "  • Average Complexity: N/A")
        
        # Test Coverage
        if "test_coverage" in self.metrics:
            tc = self.metrics["test_coverage"]
            coverage = tc.get('line_coverage')
            print(f"\n🧪 Test Coverage:")
            print(f"  • Line Coverage: {coverage:.1f}%" if coverage else "  • Line Coverage: N/A")
            print(f"  • Lines Covered: {tc.get('lines_covered', 'N/A')}")
            print(f"  • Total Lines: {tc.get('lines_total', 'N/A')}")
        
        # Performance
        if "performance" in self.metrics:
            perf = self.metrics["performance"]
            benchmarks = perf.get("benchmarks", {})
            print(f"\n⚡ Performance:")
            print(f"  • Benchmarks Run: {len(benchmarks)}")
            if "gpu_acceleration_factor" in perf:
                print(f"  • GPU Acceleration: {perf['gpu_acceleration_factor']:.1f}x faster")
        
        # Security
        if "security" in self.metrics:
            sec = self.metrics["security"]
            sec_issues = sec.get("security_issues")
            if sec_issues:
                total_issues = sec_issues.get("total", 0)
                print(f"\n🔒 Security:")
                print(f"  • Security Issues: {total_issues}")
                print(f"  • Vulnerabilities: {sec.get('vulnerabilities', 'N/A')}")
        
        # Git
        if "git" in self.metrics:
            git = self.metrics["git"]
            print(f"\n📈 Repository:")
            print(f"  • Total Commits: {git.get('total_commits', 'N/A')}")
            print(f"  • Contributors: {git.get('contributors', 'N/A')}")
            print(f"  • Recent Commits (30d): {git.get('commits_last_30_days', 'N/A')}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Collect metrics for HD-Compute-Toolkit")
    parser.add_argument("--output", "-o", default="metrics-report.json", 
                       help="Output file for metrics (default: metrics-report.json)")
    parser.add_argument("--project-root", default=".", 
                       help="Path to project root (default: current directory)")
    parser.add_argument("--summary", action="store_true", 
                       help="Print summary to console")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not (project_root / "pyproject.toml").exists():
        logger.error("pyproject.toml not found. Are you in the project root?")
        sys.exit(1)
    
    collector = MetricsCollector(project_root)
    
    try:
        collector.collect_all_metrics()
        collector.save_metrics(Path(args.output))
        
        if args.summary:
            collector.print_summary()
        
        logger.info(f"Metrics collection completed successfully. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()