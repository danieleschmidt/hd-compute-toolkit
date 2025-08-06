#!/usr/bin/env python3
"""Security scan for HD-Compute-Toolkit."""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
import hashlib
import subprocess


class SecurityScanner:
    """Comprehensive security scanner for Python codebase."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.findings = []
        
        # Security patterns to detect
        self.security_patterns = {
            'sql_injection': [
                r'\.execute\s*\(\s*["\'].*%.*["\']',
                r'\.execute\s*\(\s*.*\.format\s*\(',
                r'cursor\.execute.*\+',
            ],
            'command_injection': [
                r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
                r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True',
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\./.*["\']',
                r'os\.path\.join\s*\([^)]*\.\.',
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_?key\s*=\s*["\'][^"\']{20,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']',
            ],
            'unsafe_eval': [
                r'\beval\s*\(',
                r'\bexec\s*\(',
                r'__import__\s*\(',
            ],
            'weak_crypto': [
                r'hashlib\.md5\s*\(',
                r'hashlib\.sha1\s*\(',
                r'random\.random\s*\(',  # For crypto purposes
            ],
            'unsafe_pickle': [
                r'pickle\.loads?\s*\(',
                r'cPickle\.loads?\s*\(',
            ],
            'debug_code': [
                r'print\s*\([^)]*password',
                r'print\s*\([^)]*secret',
                r'print\s*\([^)]*token',
                r'logging\.debug\s*\([^)]*password',
            ]
        }
        
    def scan_all(self) -> Dict[str, List[Dict]]:
        """Run comprehensive security scan."""
        
        print("🔍 Starting Security Scan for HD-Compute-Toolkit")
        print("=" * 60)
        
        results = {
            'pattern_matches': self.scan_security_patterns(),
            'file_permissions': self.check_file_permissions(),
            'dependency_check': self.check_dependencies(),
            'sensitive_files': self.check_sensitive_files(),
            'code_quality': self.check_code_quality(),
        }
        
        # Generate summary
        total_issues = sum(len(issues) for issues in results.values())
        
        print(f"\n📊 Security Scan Summary:")
        print(f"Total Issues Found: {total_issues}")
        
        for category, issues in results.items():
            if issues:
                print(f"  {category}: {len(issues)} issues")
        
        if total_issues == 0:
            print("✅ No security issues detected!")
        else:
            print("⚠️  Security issues found - review required")
        
        return results
    
    def scan_security_patterns(self) -> List[Dict]:
        """Scan for security anti-patterns in code."""
        
        findings = []
        python_files = list(self.root_dir.glob("**/*.py"))
        
        print(f"🔎 Scanning {len(python_files)} Python files for security patterns...")
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check each security pattern
                for category, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            findings.append({
                                'type': 'security_pattern',
                                'category': category,
                                'file': str(file_path.relative_to(self.root_dir)),
                                'line': line_num,
                                'pattern': pattern,
                                'match': match.group(),
                                'severity': self._get_severity(category)
                            })
            
            except Exception as e:
                findings.append({
                    'type': 'scan_error',
                    'file': str(file_path),
                    'error': str(e),
                    'severity': 'low'
                })
        
        return findings
    
    def check_file_permissions(self) -> List[Dict]:
        """Check for overly permissive file permissions."""
        
        findings = []
        
        print("🔐 Checking file permissions...")
        
        # Check for world-writable files
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode & 0o777
                    
                    # Check for world-writable (002)
                    if mode & 0o002:
                        findings.append({
                            'type': 'file_permission',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'permission': oct(mode),
                            'issue': 'world_writable',
                            'severity': 'high'
                        })
                    
                    # Check for group-writable sensitive files
                    if (mode & 0o020 and 
                        any(pattern in str(file_path) for pattern in ['.env', 'config', 'secret', 'key'])):
                        findings.append({
                            'type': 'file_permission',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'permission': oct(mode),
                            'issue': 'group_writable_sensitive',
                            'severity': 'medium'
                        })
                
                except OSError:
                    continue  # Skip files we can't access
        
        return findings
    
    def check_dependencies(self) -> List[Dict]:
        """Check for vulnerable dependencies."""
        
        findings = []
        
        print("📦 Checking dependencies...")
        
        # Check requirements files
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']
        
        for req_file in req_files:
            req_path = self.root_dir / req_file
            if req_path.exists():
                findings.extend(self._check_requirement_file(req_path))
        
        return findings
    
    def _check_requirement_file(self, file_path: Path) -> List[Dict]:
        """Check a specific requirements file."""
        
        findings = []
        
        # Known vulnerable packages (simplified list)
        vulnerable_packages = {
            'django': ['<2.2.13', '<3.0.7'],
            'flask': ['<1.1.4'],
            'requests': ['<2.20.0'],
            'pyyaml': ['<5.4'],
            'pillow': ['<8.1.1'],
            'cryptography': ['<3.3.2'],
        }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract package names and versions
            if file_path.name == 'pyproject.toml':
                # Basic TOML parsing for dependencies
                import re
                deps = re.findall(r'"([^"]+)>=?([^"]*)"', content)
            else:
                # Simple parsing for other formats
                deps = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            pkg, version = line.split('==', 1)
                            deps.append((pkg.strip(), version.strip()))
                        elif '>=' in line:
                            pkg, version = line.split('>=', 1)
                            deps.append((pkg.strip(), version.strip()))
            
            # Check for vulnerable versions
            for pkg_name, version in deps:
                if pkg_name.lower() in vulnerable_packages:
                    vuln_versions = vulnerable_packages[pkg_name.lower()]
                    for vuln_pattern in vuln_versions:
                        findings.append({
                            'type': 'dependency_vulnerability',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'package': pkg_name,
                            'version': version,
                            'vulnerability': f'Potentially vulnerable version {vuln_pattern}',
                            'severity': 'medium'
                        })
        
        except Exception as e:
            findings.append({
                'type': 'dependency_check_error',
                'file': str(file_path.relative_to(self.root_dir)),
                'error': str(e),
                'severity': 'low'
            })
        
        return findings
    
    def check_sensitive_files(self) -> List[Dict]:
        """Check for sensitive files that shouldn't be in repo."""
        
        findings = []
        
        print("🕵️ Checking for sensitive files...")
        
        sensitive_patterns = [
            r'\.env$',
            r'\.env\..*$',
            r'id_rsa$',
            r'id_dsa$',
            r'\.pem$',
            r'\.key$',
            r'\.p12$',
            r'\.pfx$',
            r'password.*\.txt$',
            r'secret.*\.txt$',
            r'credentials.*\.json$',
        ]
        
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file():
                filename = file_path.name.lower()
                
                for pattern in sensitive_patterns:
                    if re.match(pattern, filename):
                        findings.append({
                            'type': 'sensitive_file',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'pattern': pattern,
                            'severity': 'high' if any(x in pattern for x in ['key', 'pem', 'p12']) else 'medium'
                        })
        
        return findings
    
    def check_code_quality(self) -> List[Dict]:
        """Check for code quality issues that may lead to security problems."""
        
        findings = []
        
        print("🧹 Checking code quality...")
        
        python_files = list(self.root_dir.glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for potential issues
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # TODO comments in production code
                    if re.search(r'#\s*(TODO|FIXME|HACK)', line, re.IGNORECASE):
                        findings.append({
                            'type': 'code_quality',
                            'category': 'todo_in_code',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'line': i,
                            'content': line.strip(),
                            'severity': 'low'
                        })
                    
                    # Commented out debug code
                    if re.search(r'#\s*(print|console\.log)', line):
                        findings.append({
                            'type': 'code_quality',
                            'category': 'commented_debug',
                            'file': str(file_path.relative_to(self.root_dir)),
                            'line': i,
                            'content': line.strip(),
                            'severity': 'low'
                        })
            
            except Exception:
                continue
        
        return findings
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for security category."""
        
        high_severity = ['sql_injection', 'command_injection', 'unsafe_eval', 'unsafe_pickle']
        medium_severity = ['hardcoded_secrets', 'weak_crypto', 'path_traversal']
        
        if category in high_severity:
            return 'high'
        elif category in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def generate_security_report(self, results: Dict) -> str:
        """Generate detailed security report."""
        
        report = ["# HD-Compute-Toolkit Security Scan Report", ""]
        report.append(f"**Scan Date**: {__import__('datetime').datetime.now().isoformat()}")
        report.append("")
        
        # Executive Summary
        total_issues = sum(len(issues) for issues in results.values())
        high_issues = sum(1 for issues in results.values() for issue in issues 
                         if issue.get('severity') == 'high')
        
        report.append("## Executive Summary")
        report.append(f"- **Total Issues**: {total_issues}")
        report.append(f"- **High Severity**: {high_issues}")
        report.append("")
        
        if total_issues == 0:
            report.append("✅ **No security issues detected.** The codebase appears secure.")
        else:
            report.append("⚠️ **Security issues detected.** Review and remediation required.")
        
        report.append("")
        
        # Detailed Findings
        for category, issues in results.items():
            if issues:
                report.append(f"## {category.replace('_', ' ').title()}")
                report.append("")
                
                for issue in issues:
                    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.get('severity', 'low'), '⚪')
                    report.append(f"### {severity_emoji} {issue.get('type', 'Unknown')}")
                    
                    if 'file' in issue:
                        report.append(f"**File**: `{issue['file']}`")
                    if 'line' in issue:
                        report.append(f"**Line**: {issue['line']}")
                    if 'category' in issue:
                        report.append(f"**Category**: {issue['category']}")
                    if 'match' in issue:
                        report.append(f"**Match**: `{issue['match']}`")
                    if 'error' in issue:
                        report.append(f"**Error**: {issue['error']}")
                    
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if high_issues > 0:
            report.append("1. **Priority**: Address all high-severity issues immediately")
            report.append("2. **Code Review**: Implement mandatory security code reviews")
            report.append("3. **Static Analysis**: Integrate automated security scanning in CI/CD")
        else:
            report.append("1. **Maintenance**: Continue regular security scans")
            report.append("2. **Monitoring**: Monitor dependencies for new vulnerabilities")
            report.append("3. **Training**: Ensure team understands secure coding practices")
        
        return "\n".join(report)


def main():
    """Run security scan."""
    
    scanner = SecurityScanner("/root/repo")
    results = scanner.scan_all()
    
    # Generate report
    report = scanner.generate_security_report(results)
    
    # Save report
    report_path = Path("/root/repo/SECURITY_SCAN_REPORT.md")
    report_path.write_text(report)
    
    print(f"\n📄 Security report saved to: {report_path}")
    
    # Return exit code based on findings
    high_severity_count = sum(1 for issues in results.values() for issue in issues 
                             if issue.get('severity') == 'high')
    
    return 0 if high_severity_count == 0 else 1


if __name__ == "__main__":
    exit(main())