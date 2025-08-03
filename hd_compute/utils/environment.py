"""Environment management and validation utilities."""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment setup and validation for HD-Compute-Toolkit."""
    
    def __init__(self):
        self.requirements_check = {}
        self.warnings = []
        self.errors = []
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment setup.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'python_version': self._check_python_version(),
            'dependencies': self._check_dependencies(),
            'hardware': self._check_hardware(),
            'environment_variables': self._check_environment_variables(),
            'file_permissions': self._check_file_permissions(),
            'disk_space': self._check_disk_space(),
            'warnings': self.warnings.copy(),
            'errors': self.errors.copy()
        }
        
        validation_results['overall_status'] = 'pass' if not self.errors else 'fail'
        
        return validation_results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        min_version = (3, 8)
        max_version = (3, 12)
        
        current_version = sys.version_info[:2]
        
        result = {
            'current': f"{current_version[0]}.{current_version[1]}",
            'supported': f"{min_version[0]}.{min_version[1]}+ to {max_version[0]}.{max_version[1]}",
            'compatible': min_version <= current_version <= max_version
        }
        
        if not result['compatible']:
            error_msg = f"Python {result['current']} not supported. Required: {result['supported']}"
            self.errors.append(error_msg)
            logger.error(error_msg)
        else:
            logger.info(f"Python version {result['current']} is compatible")
        
        return result
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required and optional dependencies."""
        dependencies = {
            'required': {
                'numpy': '>=1.21.0',
                'torch': '>=1.12.0',
                'jax': '>=0.3.0',
                'jaxlib': '>=0.3.0'
            },
            'optional': {
                'librosa': '>=0.9.0',  # For speech processing
                'psutil': '>=5.8.0',   # For system monitoring
                'wandb': '>=0.13.0',   # For experiment tracking
                'tensorboard': '>=2.10.0',  # For logging
                'matplotlib': '>=3.5.0',    # For plotting
                'seaborn': '>=0.11.0',      # For visualization
            },
            'development': {
                'pytest': '>=7.0.0',
                'black': '>=22.0.0',
                'flake8': '>=4.0.0',
                'mypy': '>=0.991',
                'pre-commit': '>=2.20.0'
            }
        }
        
        results = {}
        
        for category, deps in dependencies.items():
            results[category] = {}
            
            for dep_name, version_spec in deps.items():
                try:
                    module = __import__(dep_name)
                    version = getattr(module, '__version__', 'unknown')
                    
                    results[category][dep_name] = {
                        'available': True,
                        'version': version,
                        'required': version_spec
                    }
                    
                    # Basic version checking (simplified)
                    if version_spec.startswith('>='):
                        min_version = version_spec[2:]
                        # This is a simplified check - in practice you'd use packaging.version
                        results[category][dep_name]['version_ok'] = version >= min_version
                    else:
                        results[category][dep_name]['version_ok'] = True
                
                except ImportError:
                    results[category][dep_name] = {
                        'available': False,
                        'version': None,
                        'required': version_spec,
                        'version_ok': False
                    }
                    
                    if category == 'required':
                        error_msg = f"Required dependency '{dep_name}' not found"
                        self.errors.append(error_msg)
                        logger.error(error_msg)
                    else:
                        warning_msg = f"Optional dependency '{dep_name}' not found"
                        self.warnings.append(warning_msg)
                        logger.warning(warning_msg)
        
        return results
    
    def _check_hardware(self) -> Dict[str, Any]:
        """Check hardware capabilities."""
        from .device_utils import get_device_info
        
        device_info = get_device_info()
        
        # Check minimum requirements
        min_memory_gb = 4.0
        min_cores = 2
        
        hardware_status = {
            'device_info': device_info,
            'meets_minimum': True,
            'recommendations': []
        }
        
        # Check memory
        if 'memory_total_gb' in device_info:
            if device_info['memory_total_gb'] < min_memory_gb:
                hardware_status['meets_minimum'] = False
                error_msg = f"Insufficient memory: {device_info['memory_total_gb']:.1f}GB < {min_memory_gb}GB required"
                self.errors.append(error_msg)
        
        # Check CPU cores
        if 'cpu_logical_cores' in device_info:
            if device_info['cpu_logical_cores'] < min_cores:
                hardware_status['meets_minimum'] = False
                error_msg = f"Insufficient CPU cores: {device_info['cpu_logical_cores']} < {min_cores} required"
                self.errors.append(error_msg)
        
        # Recommendations
        if not device_info.get('cuda_available', False):
            hardware_status['recommendations'].append(
                "CUDA GPU not available. Consider using a GPU for better performance."
            )
        
        if device_info.get('memory_total_gb', 0) < 16:
            hardware_status['recommendations'].append(
                "Consider using a system with 16GB+ RAM for large-scale experiments."
            )
        
        return hardware_status
    
    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check environment variable configuration."""
        env_vars = {
            'optional': [
                'HDC_DEBUG',
                'HDC_DEFAULT_DEVICE',
                'HDC_DEFAULT_DIMENSION',
                'HDC_CACHE_DIR',
                'HDC_DATA_DIR',
                'HDC_LOG_LEVEL',
                'CUDA_VISIBLE_DEVICES',
                'JAX_PLATFORM_NAME'
            ],
            'recommended': [
                'HDC_CACHE_DIR',
                'HDC_DATA_DIR'
            ]
        }
        
        env_status = {
            'set_variables': {},
            'missing_recommended': []
        }
        
        for var in env_vars['optional']:
            value = os.getenv(var)
            env_status['set_variables'][var] = value
        
        for var in env_vars['recommended']:
            if not os.getenv(var):
                env_status['missing_recommended'].append(var)
                warning_msg = f"Recommended environment variable '{var}' not set"
                self.warnings.append(warning_msg)
        
        return env_status
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions for required directories."""
        from .config import get_config
        
        config = get_config()
        paths_to_check = config.get_paths_config()
        
        permission_status = {
            'writable_paths': {},
            'permission_errors': []
        }
        
        for path_name, path_value in paths_to_check.items():
            try:
                path = Path(path_value)
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write permission
                test_file = path / '.permission_test'
                test_file.write_text('test')
                test_file.unlink()
                
                permission_status['writable_paths'][path_name] = {
                    'path': str(path),
                    'writable': True
                }
                
            except (PermissionError, OSError) as e:
                permission_status['writable_paths'][path_name] = {
                    'path': path_value,
                    'writable': False,
                    'error': str(e)
                }
                
                error_msg = f"Cannot write to {path_name} directory: {path_value}"
                permission_status['permission_errors'].append(error_msg)
                self.errors.append(error_msg)
        
        return permission_status
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        disk_status = {
            'space_info': {},
            'space_warnings': []
        }
        
        min_space_gb = 5.0  # Minimum 5GB free space
        
        try:
            import psutil
            
            # Check space for current directory
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            disk_status['space_info']['current_directory'] = {
                'free_gb': free_gb,
                'total_gb': total_gb,
                'percent_free': (free_gb / total_gb) * 100
            }
            
            if free_gb < min_space_gb:
                warning_msg = f"Low disk space: {free_gb:.1f}GB free (minimum {min_space_gb}GB recommended)"
                disk_status['space_warnings'].append(warning_msg)
                self.warnings.append(warning_msg)
        
        except ImportError:
            warning_msg = "Cannot check disk space (psutil not available)"
            disk_status['space_warnings'].append(warning_msg)
            self.warnings.append(warning_msg)
        
        return disk_status
    
    def setup_environment(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Setup environment with recommended settings.
        
        Args:
            config: Optional configuration override
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Setup directories
            from .config import get_config
            hdc_config = get_config()
            hdc_config.ensure_directories()
            
            # Setup logging
            from .logging_config import setup_logging, configure_external_loggers
            
            log_level = hdc_config.get('log_level', 'INFO')
            log_file = hdc_config.get('log_file')
            
            setup_logging(log_level=log_level, log_file=log_file)
            configure_external_loggers()
            
            # Set environment variables if needed
            if not os.getenv('OMP_NUM_THREADS'):
                os.environ['OMP_NUM_THREADS'] = str(hdc_config.get('num_threads', 4))
            
            if not os.getenv('MKL_NUM_THREADS'):
                os.environ['MKL_NUM_THREADS'] = str(hdc_config.get('num_threads', 4))
            
            # Disable warnings for cleaner output
            if not hdc_config.get('dev_mode', True):
                warnings.filterwarnings('ignore', category=UserWarning)
            
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def generate_setup_report(self) -> str:
        """Generate a comprehensive setup report.
        
        Returns:
            Formatted report string
        """
        validation_results = self.validate_environment()
        
        report_lines = [
            "HD-Compute-Toolkit Environment Report",
            "=" * 40,
            "",
            f"Overall Status: {validation_results['overall_status'].upper()}",
            ""
        ]
        
        # Python version
        python_info = validation_results['python_version']
        report_lines.extend([
            f"Python Version: {python_info['current']}",
            f"Compatible: {'✓' if python_info['compatible'] else '✗'}",
            ""
        ])
        
        # Dependencies
        deps = validation_results['dependencies']
        for category, category_deps in deps.items():
            report_lines.append(f"{category.title()} Dependencies:")
            for dep_name, dep_info in category_deps.items():
                status = "✓" if dep_info['available'] and dep_info.get('version_ok', True) else "✗"
                version = dep_info.get('version', 'N/A')
                report_lines.append(f"  {status} {dep_name}: {version}")
            report_lines.append("")
        
        # Hardware
        hardware = validation_results['hardware']
        device_info = hardware['device_info']
        report_lines.extend([
            "Hardware Information:",
            f"  CPU Cores: {device_info.get('cpu_logical_cores', 'Unknown')}",
            f"  Memory: {device_info.get('memory_total_gb', 0):.1f} GB",
            f"  CUDA Available: {'✓' if device_info.get('cuda_available') else '✗'}",
            f"  JAX Available: {'✓' if device_info.get('jax_available') else '✗'}",
            ""
        ])
        
        # Warnings and errors
        if validation_results['warnings']:
            report_lines.append("Warnings:")
            for warning in validation_results['warnings']:
                report_lines.append(f"  ⚠ {warning}")
            report_lines.append("")
        
        if validation_results['errors']:
            report_lines.append("Errors:")
            for error in validation_results['errors']:
                report_lines.append(f"  ✗ {error}")
            report_lines.append("")
        
        # Recommendations
        recommendations = hardware.get('recommendations', [])
        if recommendations:
            report_lines.append("Recommendations:")
            for rec in recommendations:
                report_lines.append(f"  • {rec}")
        
        return "\n".join(report_lines)