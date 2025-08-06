"""Security scanning and validation utilities."""

import hashlib
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Security scanner for detecting potential vulnerabilities."""
    
    def __init__(self):
        self.scan_results = []
        self.severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        # Patterns for detecting potential security issues
        self.security_patterns = {
            "hardcoded_secrets": [
                (r"password\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded password"),
                (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded API key"),
                (r"secret\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded secret"),
                (r"token\s*=\s*['\"][^'\"]+['\"]", "Potential hardcoded token"),
            ],
            "unsafe_operations": [
                (r"eval\s*\(", "Use of eval() function"),
                (r"exec\s*\(", "Use of exec() function"),
                (r"pickle\.loads?\s*\(", "Use of pickle.load() - potential code execution"),
                (r"subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True", "Shell command injection risk"),
            ],
            "file_operations": [
                (r"open\s*\([^)]*['\"]\/[^'\"]*['\"]", "Absolute path in file operations"),
                (r"os\.system\s*\(", "Use of os.system() - command injection risk"),
                (r"os\.popen\s*\(", "Use of os.popen() - command injection risk"),
            ],
            "network_operations": [
                (r"urllib\.request\.urlopen\s*\([^)]*verify\s*=\s*False", "SSL verification disabled"),
                (r"requests\.[get|post|put|delete]+\s*\([^)]*verify\s*=\s*False", "SSL verification disabled"),
            ]
        }
    
    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Scan a single file for security issues.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of security findings
        """
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for category, patterns in self.security_patterns.items():
                    for pattern, description in patterns:
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                severity = self._determine_severity(category, pattern)
                                
                                finding = {
                                    'file': file_path,
                                    'line': line_num,
                                    'category': category,
                                    'description': description,
                                    'severity': severity,
                                    'pattern': pattern,
                                    'code': line.strip()
                                }
                                findings.append(finding)
                
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
        
        return findings
    
    def scan_directory(self, directory_path: str, file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Scan directory for security issues.
        
        Args:
            directory_path: Path to directory to scan
            file_extensions: List of file extensions to scan (default: ['.py'])
            
        Returns:
            List of all security findings
        """
        if file_extensions is None:
            file_extensions = ['.py']
        
        all_findings = []
        directory = Path(directory_path)
        
        for ext in file_extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if file_path.is_file():
                    findings = self.scan_file(str(file_path))
                    all_findings.extend(findings)
        
        self.scan_results = all_findings
        return all_findings
    
    def _determine_severity(self, category: str, pattern: str) -> str:
        """Determine severity level based on category and pattern."""
        severity_map = {
            "hardcoded_secrets": "HIGH",
            "unsafe_operations": "CRITICAL",
            "file_operations": "MEDIUM",
            "network_operations": "HIGH"
        }
        
        # Specific pattern overrides
        if "eval(" in pattern or "exec(" in pattern:
            return "CRITICAL"
        elif "pickle" in pattern:
            return "HIGH"
        elif "shell=True" in pattern:
            return "HIGH"
        
        return severity_map.get(category, "MEDIUM")
    
    def validate_dependencies(self, requirements_file: str = None) -> Dict[str, Any]:
        """Validate dependencies for known vulnerabilities.
        
        Args:
            requirements_file: Path to requirements file
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'vulnerable_packages': [],
            'outdated_packages': [],
            'recommendations': []
        }
        
        # This is a simplified implementation
        # In practice, you'd integrate with vulnerability databases
        known_vulnerable_patterns = [
            "pillow==8.0.0",  # Example vulnerable version
            "requests<2.20.0",  # Example vulnerable version
        ]
        
        try:
            if requirements_file and os.path.exists(requirements_file):
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                    
                    for vulnerable_pattern in known_vulnerable_patterns:
                        if vulnerable_pattern in requirements:
                            validation_results['vulnerable_packages'].append(vulnerable_pattern)
            
            # Add general recommendations
            validation_results['recommendations'] = [
                "Regularly update dependencies",
                "Use dependency scanning tools like safety or snyk",
                "Pin dependency versions in production",
                "Review dependencies for licensing compatibility"
            ]
                            
        except Exception as e:
            logger.error(f"Error validating dependencies: {e}")
        
        return validation_results
    
    def check_file_permissions(self, directory_path: str) -> Dict[str, Any]:
        """Check for potentially dangerous file permissions.
        
        Args:
            directory_path: Path to check
            
        Returns:
            Dictionary with permission analysis
        """
        permission_issues = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        stat_info = os.stat(file_path)
                        mode = stat_info.st_mode
                        
                        # Check for world-writable files
                        if mode & 0o002:  # World writable
                            permission_issues.append({
                                'file': file_path,
                                'issue': 'World writable',
                                'severity': 'HIGH',
                                'permissions': oct(mode)[-3:]
                            })
                        
                        # Check for executable Python files
                        if file.endswith('.py') and mode & 0o111:
                            permission_issues.append({
                                'file': file_path,
                                'issue': 'Executable Python file',
                                'severity': 'LOW',
                                'permissions': oct(mode)[-3:]
                            })
                            
                    except OSError as e:
                        logger.warning(f"Cannot check permissions for {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
        
        return {
            'issues': permission_issues,
            'total_issues': len(permission_issues),
            'high_severity_count': len([i for i in permission_issues if i['severity'] == 'HIGH'])
        }
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for security issues.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            Validation results
        """
        config_issues = []
        
        # Check for insecure configurations
        insecure_configs = [
            ('debug', True, "Debug mode enabled in configuration"),
            ('disable_telemetry', False, "Telemetry not disabled - potential privacy issue"),
            ('ssl_verify', False, "SSL verification disabled"),
            ('log_level', 'DEBUG', "Debug logging enabled - may expose sensitive info"),
        ]
        
        for key, dangerous_value, message in insecure_configs:
            if config_data.get(key) == dangerous_value:
                config_issues.append({
                    'key': key,
                    'value': dangerous_value,
                    'message': message,
                    'severity': 'MEDIUM'
                })
        
        # Check for potentially sensitive keys in config
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential']
        for config_key, config_value in config_data.items():
            if any(pattern in config_key.lower() for pattern in sensitive_patterns):
                if isinstance(config_value, str) and len(config_value) > 0:
                    config_issues.append({
                        'key': config_key,
                        'message': f"Potential sensitive data in configuration: {config_key}",
                        'severity': 'HIGH'
                    })
        
        return {
            'issues': config_issues,
            'total_issues': len(config_issues),
            'recommendations': [
                "Avoid storing secrets in configuration files",
                "Use environment variables for sensitive data",
                "Disable debug mode in production",
                "Enable SSL verification",
                "Use appropriate logging levels for production"
            ]
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Returns:
            Security report dictionary
        """
        report = {
            'timestamp': os.times(),
            'total_findings': len(self.scan_results),
            'severity_breakdown': {level: 0 for level in self.severity_levels},
            'category_breakdown': {},
            'critical_issues': [],
            'recommendations': [],
            'files_scanned': set()
        }
        
        # Analyze findings
        for finding in self.scan_results:
            # Count by severity
            severity = finding['severity']
            report['severity_breakdown'][severity] += 1
            
            # Count by category
            category = finding['category']
            if category not in report['category_breakdown']:
                report['category_breakdown'][category] = 0
            report['category_breakdown'][category] += 1
            
            # Track files
            report['files_scanned'].add(finding['file'])
            
            # Collect critical issues
            if severity == 'CRITICAL':
                report['critical_issues'].append(finding)
        
        report['files_scanned'] = len(report['files_scanned'])
        
        # Generate recommendations
        if report['severity_breakdown']['CRITICAL'] > 0:
            report['recommendations'].append("Address critical security issues immediately")
        
        if report['severity_breakdown']['HIGH'] > 0:
            report['recommendations'].append("Review and fix high-severity security issues")
        
        if report['category_breakdown'].get('hardcoded_secrets', 0) > 0:
            report['recommendations'].append("Remove hardcoded secrets and use secure secret management")
        
        if report['category_breakdown'].get('unsafe_operations', 0) > 0:
            report['recommendations'].append("Replace unsafe operations with secure alternatives")
        
        # General recommendations
        report['recommendations'].extend([
            "Implement regular security scanning in CI/CD pipeline",
            "Conduct code reviews with security focus",
            "Keep dependencies updated",
            "Use static analysis security testing (SAST) tools"
        ])
        
        return report
    
    def export_findings(self, filename: str, format: str = 'json'):
        """Export security findings to file.
        
        Args:
            filename: Output filename
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            import json
            
            export_data = {
                'security_scan_results': self.scan_results,
                'summary': self.generate_security_report()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            
            if self.scan_results:
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.scan_results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.scan_results)
        
        logger.info(f"Security findings exported to {filename}")
    
    def clear_results(self):
        """Clear scan results."""
        self.scan_results = []