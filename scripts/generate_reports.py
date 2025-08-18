#!/usr/bin/env python3
"""
Automated report generation for HD-Compute-Toolkit metrics.
Generates weekly/monthly reports in multiple formats.
"""

import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive reports from collected metrics."""
    
    def __init__(self, metrics_data: Dict[str, Any]):
        self.metrics = metrics_data
        self.timestamp = datetime.utcnow()
        
    def generate_weekly_report(self) -> str:
        """Generate a weekly summary report."""
        logger.info("Generating weekly report...")
        
        report = []
        report.append("# HD-Compute-Toolkit Weekly Report")
        report.append(f"**Report Date**: {self.timestamp.strftime('%Y-%m-%d')}")
        report.append(f"**Reporting Period**: {(self.timestamp - timedelta(days=7)).strftime('%Y-%m-%d')} to {self.timestamp.strftime('%Y-%m-%d')}")
        report.append("")
        
        # Executive Summary
        report.append("## 📊 Executive Summary")
        report.append("")
        
        # Code quality summary
        if "code_quality" in self.metrics:
            cq = self.metrics["code_quality"]
            report.append("### Code Quality")
            report.append(f"- **Lines of Code**: {cq.get('lines_of_code', 'N/A'):,}")
            report.append(f"- **Linting Issues**: {cq.get('linting_issues', 'N/A')}")
            report.append(f"- **Type Check Errors**: {cq.get('type_check_errors', 'N/A')}")
            if cq.get('average_complexity'):
                report.append(f"- **Average Complexity**: {cq['average_complexity']:.2f}")
            report.append("")
        
        # Test coverage
        if "test_coverage" in self.metrics:
            tc = self.metrics["test_coverage"]
            coverage = tc.get('line_coverage')
            if coverage:
                status = "🟢 Excellent" if coverage >= 90 else "🟡 Good" if coverage >= 80 else "🔴 Needs Improvement"
                report.append("### Test Coverage")
                report.append(f"- **Line Coverage**: {coverage:.1f}% ({status})")
                report.append(f"- **Lines Covered**: {tc.get('lines_covered', 'N/A'):,}")
                report.append(f"- **Total Lines**: {tc.get('lines_total', 'N/A'):,}")
                report.append("")
        
        # Performance benchmarks
        if "performance" in self.metrics:
            perf = self.metrics["performance"]
            report.append("### Performance Metrics")
            benchmarks = perf.get("benchmarks", {})
            report.append(f"- **Benchmarks Executed**: {len(benchmarks)}")
            
            if benchmarks:
                # Find fastest and slowest benchmarks
                times = [(name, data["mean_ms"]) for name, data in benchmarks.items()]
                fastest = min(times, key=lambda x: x[1])
                slowest = max(times, key=lambda x: x[1])
                
                report.append(f"- **Fastest Operation**: {fastest[0]} ({fastest[1]:.2f}ms)")
                report.append(f"- **Slowest Operation**: {slowest[0]} ({slowest[1]:.2f}ms)")
            
            if "gpu_acceleration_factor" in perf:
                report.append(f"- **GPU Acceleration**: {perf['gpu_acceleration_factor']:.1f}x faster than CPU")
            
            report.append("")
        
        # Security status
        if "security" in self.metrics:
            sec = self.metrics["security"]
            report.append("### Security Status")
            
            sec_issues = sec.get("security_issues")
            if sec_issues:
                total = sec_issues.get("total", 0)
                status = "🟢 Secure" if total == 0 else "🟡 Monitor" if total <= 5 else "🔴 Action Required"
                report.append(f"- **Security Issues**: {total} ({status})")
                
                if total > 0:
                    report.append(f"  - High: {sec_issues.get('high', 0)}")
                    report.append(f"  - Medium: {sec_issues.get('medium', 0)}")
                    report.append(f"  - Low: {sec_issues.get('low', 0)}")
            
            vuln_count = sec.get("vulnerabilities", 0)
            vuln_status = "🟢 Clean" if vuln_count == 0 else "🔴 Vulnerabilities Found"
            report.append(f"- **Dependency Vulnerabilities**: {vuln_count} ({vuln_status})")
            report.append("")
        
        # Repository activity
        if "git" in self.metrics:
            git = self.metrics["git"]
            report.append("### Repository Activity")
            report.append(f"- **Total Commits**: {git.get('total_commits', 'N/A'):,}")
            report.append(f"- **Contributors**: {git.get('contributors', 'N/A')}")
            report.append(f"- **Recent Activity (30 days)**: {git.get('commits_last_30_days', 'N/A')} commits")
            report.append(f"- **Current Branch**: `{git.get('current_branch', 'unknown')}`")
            report.append("")
        
        # Detailed sections
        report.append("---")
        report.append("")
        
        # Performance details
        if "performance" in self.metrics and self.metrics["performance"].get("benchmarks"):
            report.append("## ⚡ Detailed Performance Analysis")
            report.append("")
            report.append("| Benchmark | Mean (ms) | Min (ms) | Max (ms) | StdDev (ms) |")
            report.append("|-----------|-----------|----------|----------|-------------|")
            
            for name, data in self.metrics["performance"]["benchmarks"].items():
                report.append(f"| {name} | {data['mean_ms']:.2f} | {data['min_ms']:.2f} | {data['max_ms']:.2f} | {data['stddev_ms']:.2f} |")
            
            report.append("")
        
        # File-level coverage details
        if "test_coverage" in self.metrics and "file_coverage" in self.metrics["test_coverage"]:
            file_cov = self.metrics["test_coverage"]["file_coverage"]
            if file_cov:
                report.append("## 🧪 Test Coverage by File")
                report.append("")
                report.append("| File | Coverage % |")
                report.append("|------|------------|")
                
                # Sort by coverage percentage
                sorted_files = sorted(file_cov.items(), key=lambda x: x[1], reverse=True)
                for filename, coverage in sorted_files[:10]:  # Top 10 files
                    short_name = filename.replace("hd_compute/", "")
                    report.append(f"| {short_name} | {coverage:.1f}% |")
                
                report.append("")
                
                # Files needing attention (low coverage)
                low_coverage = [(f, c) for f, c in file_cov.items() if c < 80]
                if low_coverage:
                    report.append("### Files Needing Coverage Attention")
                    report.append("")
                    for filename, coverage in sorted(low_coverage, key=lambda x: x[1]):
                        short_name = filename.replace("hd_compute/", "")
                        report.append(f"- `{short_name}`: {coverage:.1f}%")
                    report.append("")
        
        # Action items and recommendations
        report.append("## 📝 Action Items & Recommendations")
        report.append("")
        
        action_items = []
        
        # Code quality recommendations
        if "code_quality" in self.metrics:
            cq = self.metrics["code_quality"]
            if cq.get("linting_issues", 0) > 0:
                action_items.append(f"🔧 **Code Quality**: Fix {cq['linting_issues']} linting issues")
            
            if cq.get("type_check_errors", 0) > 0:
                action_items.append(f"🔧 **Type Safety**: Resolve {cq['type_check_errors']} type check errors")
            
            if cq.get("average_complexity", 0) > 5:
                action_items.append("🔧 **Code Complexity**: Consider refactoring complex functions")
        
        # Coverage recommendations
        if "test_coverage" in self.metrics:
            coverage = self.metrics["test_coverage"].get("line_coverage", 0)
            if coverage < 90:
                action_items.append(f"🧪 **Test Coverage**: Increase coverage from {coverage:.1f}% to 90%+")
        
        # Security recommendations
        if "security" in self.metrics:
            sec = self.metrics["security"]
            if sec.get("vulnerabilities", 0) > 0:
                action_items.append("🔒 **Security**: Update vulnerable dependencies")
            
            sec_issues = sec.get("security_issues", {})
            if sec_issues.get("high", 0) > 0:
                action_items.append("🔒 **Security**: Address high-severity security issues immediately")
        
        if action_items:
            for item in action_items:
                report.append(f"- {item}")
        else:
            report.append("- ✅ **All systems green** - No immediate action items identified")
        
        report.append("")
        
        # Next week's focus
        report.append("## 🎯 Next Week's Focus")
        report.append("")
        report.append("Based on current metrics, prioritize:")
        report.append("1. Address any security vulnerabilities")
        report.append("2. Improve test coverage in low-coverage files") 
        report.append("3. Continue performance optimization efforts")
        report.append("4. Review and refactor high-complexity code")
        report.append("")
        
        # Footer
        report.append("---")
        report.append("")
        report.append(f"*Report generated automatically by HD-Compute-Toolkit metrics system on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC*")
        report.append("")
        report.append("💡 **Need help interpreting these metrics?** See the [metrics documentation](../docs/metrics.md)")
        
        return "\n".join(report)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for dashboard visualization."""
        logger.info("Generating dashboard data...")
        
        dashboard_data = {
            "timestamp": self.timestamp.isoformat(),
            "summary": {},
            "charts": {},
            "alerts": []
        }
        
        # Summary cards
        if "test_coverage" in self.metrics:
            coverage = self.metrics["test_coverage"].get("line_coverage")
            if coverage:
                dashboard_data["summary"]["test_coverage"] = {
                    "value": coverage,
                    "unit": "%",
                    "status": "good" if coverage >= 80 else "warning" if coverage >= 70 else "critical"
                }
        
        if "code_quality" in self.metrics:
            cq = self.metrics["code_quality"]
            issues = cq.get("linting_issues", 0) + cq.get("type_check_errors", 0)
            dashboard_data["summary"]["code_issues"] = {
                "value": issues,
                "unit": "issues",
                "status": "good" if issues == 0 else "warning" if issues <= 5 else "critical"
            }
        
        if "security" in self.metrics:
            sec_issues = self.metrics["security"].get("security_issues", {})
            total_security = sec_issues.get("total", 0)
            dashboard_data["summary"]["security_score"] = {
                "value": max(0, 100 - total_security * 10),
                "unit": "/100",
                "status": "good" if total_security == 0 else "warning" if total_security <= 2 else "critical"
            }
        
        # Performance chart data
        if "performance" in self.metrics and "benchmarks" in self.metrics["performance"]:
            benchmarks = self.metrics["performance"]["benchmarks"]
            dashboard_data["charts"]["performance"] = {
                "type": "bar",
                "title": "Benchmark Performance (ms)",
                "data": [
                    {"name": name, "value": data["mean_ms"]}
                    for name, data in benchmarks.items()
                ]
            }
        
        # Coverage trend (would need historical data)
        if "test_coverage" in self.metrics:
            dashboard_data["charts"]["coverage_trend"] = {
                "type": "line",
                "title": "Test Coverage Trend",
                "data": [
                    {"date": self.timestamp.strftime('%Y-%m-%d'), 
                     "value": self.metrics["test_coverage"].get("line_coverage", 0)}
                ]
            }
        
        # Generate alerts
        alerts = []
        
        # Coverage alerts
        if "test_coverage" in self.metrics:
            coverage = self.metrics["test_coverage"].get("line_coverage", 0)
            if coverage < 80:
                alerts.append({
                    "level": "warning",
                    "message": f"Test coverage is {coverage:.1f}%, below target of 80%",
                    "category": "quality"
                })
        
        # Security alerts
        if "security" in self.metrics:
            vuln_count = self.metrics["security"].get("vulnerabilities", 0)
            if vuln_count > 0:
                alerts.append({
                    "level": "critical",
                    "message": f"{vuln_count} security vulnerabilities found in dependencies",
                    "category": "security"
                })
        
        dashboard_data["alerts"] = alerts
        
        return dashboard_data
    
    def save_report(self, content: str, output_path: Path, format_type: str = "markdown") -> None:
        """Save report to file."""
        logger.info(f"Saving {format_type} report to {output_path}")
        
        with open(output_path, "w") as f:
            f.write(content)
    
    def save_dashboard_data(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save dashboard data as JSON."""
        logger.info(f"Saving dashboard data to {output_path}")
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def load_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate reports from HD-Compute-Toolkit metrics")
    parser.add_argument("--metrics", "-m", required=True,
                       help="Path to metrics JSON file")
    parser.add_argument("--output-dir", "-o", default="reports",
                       help="Output directory for reports")
    parser.add_argument("--format", choices=["markdown", "html", "json", "all"], 
                       default="markdown", help="Report format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load metrics
    metrics_file = Path(args.metrics)
    metrics_data = load_metrics(metrics_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate reports
    generator = ReportGenerator(metrics_data)
    
    try:
        if args.format in ["markdown", "all"]:
            # Weekly report
            weekly_report = generator.generate_weekly_report()
            weekly_path = output_dir / f"weekly-report-{datetime.now().strftime('%Y-%m-%d')}.md"
            generator.save_report(weekly_report, weekly_path, "markdown")
            
            logger.info(f"Markdown report saved to {weekly_path}")
        
        if args.format in ["json", "all"]:
            # Dashboard data
            dashboard_data = generator.generate_dashboard_data()
            dashboard_path = output_dir / "dashboard-data.json"
            generator.save_dashboard_data(dashboard_data, dashboard_path)
            
            logger.info(f"Dashboard data saved to {dashboard_path}")
        
        if args.format in ["html", "all"]:
            # HTML report (would require template engine like Jinja2)
            logger.warning("HTML format not yet implemented")
        
        logger.info("Report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()