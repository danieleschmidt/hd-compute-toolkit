"""Configuration management for HD-Compute-Toolkit."""

import os
from typing import Any, Optional, Dict, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for HD-Compute-Toolkit."""
    
    def __init__(self, config_file: Optional[str] = None, load_env: bool = True):
        """Initialize configuration.
        
        Args:
            config_file: Path to JSON configuration file
            load_env: Whether to load environment variables
        """
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        if load_env:
            self._load_from_env()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            # General settings
            'debug': False,
            'default_device': 'cpu',
            'default_dimension': 10000,
            'random_seed': 42,
            
            # Performance settings
            'memory_pool_size': 1000,
            'batch_size': 100,
            'num_threads': 4,
            'memory_profiling': False,
            
            # Hardware acceleration
            'cuda_enabled': True,
            'fpga_enabled': False,
            'vulkan_enabled': False,
            'tpu_enabled': False,
            
            # Logging and monitoring
            'log_level': 'INFO',
            'log_file': 'logs/hdc.log',
            'metrics_enabled': False,
            'metrics_endpoint': 'http://localhost:8080/metrics',
            
            # Data paths
            'data_dir': 'data',
            'model_dir': 'models',
            'cache_dir': '.cache/hdc',
            'download_dir': 'downloads',
            
            # Testing and benchmarking
            'test_data_dir': 'tests/data',
            'benchmark_dir': 'benchmarks/results',
            'run_slow_tests': False,
            'benchmark_timeout': 300,
            'benchmark_iterations': 1000,
            
            # Development settings
            'dev_mode': True,
            'auto_reload': True,
            'strict_mode': False,
            'profile_dir': 'profiling',
            
            # External services
            'wandb_project': 'hd-compute-toolkit',
            'wandb_entity': 'your-username',
            'wandb_mode': 'offline',
            'mlflow_tracking_uri': 'file:./mlruns',
            'mlflow_experiment_name': 'hdc-experiments',
            
            # Security
            'disable_telemetry': False,
            
            # Database
            'database_url': 'sqlite:///./hdc_experiments.db',
            
            # Cache settings
            'cache_max_size_mb': 500,
            'cache_cleanup_interval_hours': 24,
            
            # Speech processing
            'speech_sample_rate': 16000,
            'speech_n_mfcc': 13,
            'speech_n_fft': 2048,
            'speech_hop_length': 512,
        }
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            self._config.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'HDC_DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'HDC_DEFAULT_DEVICE': ('default_device', str),
            'HDC_DEFAULT_DIMENSION': ('default_dimension', int),
            'HDC_RANDOM_SEED': ('random_seed', int),
            'HDC_MEMORY_POOL_SIZE': ('memory_pool_size', int),
            'HDC_BATCH_SIZE': ('batch_size', int),
            'HDC_NUM_THREADS': ('num_threads', int),
            'HDC_MEMORY_PROFILING': ('memory_profiling', lambda x: x.lower() == 'true'),
            'HDC_FPGA_ENABLED': ('fpga_enabled', lambda x: x.lower() == 'true'),
            'HDC_VULKAN_ENABLED': ('vulkan_enabled', lambda x: x.lower() == 'true'),
            'HDC_LOG_LEVEL': ('log_level', str),
            'HDC_LOG_FILE': ('log_file', str),
            'HDC_METRICS_ENABLED': ('metrics_enabled', lambda x: x.lower() == 'true'),
            'HDC_METRICS_ENDPOINT': ('metrics_endpoint', str),
            'HDC_DATA_DIR': ('data_dir', str),
            'HDC_MODEL_DIR': ('model_dir', str),
            'HDC_CACHE_DIR': ('cache_dir', str),
            'HDC_DOWNLOAD_DIR': ('download_dir', str),
            'HDC_TEST_DATA_DIR': ('test_data_dir', str),
            'HDC_BENCHMARK_DIR': ('benchmark_dir', str),
            'HDC_RUN_SLOW_TESTS': ('run_slow_tests', lambda x: x.lower() == 'true'),
            'HDC_BENCHMARK_TIMEOUT': ('benchmark_timeout', int),
            'HDC_BENCHMARK_ITERATIONS': ('benchmark_iterations', int),
            'HDC_DEV_MODE': ('dev_mode', lambda x: x.lower() == 'true'),
            'HDC_AUTO_RELOAD': ('auto_reload', lambda x: x.lower() == 'true'),
            'HDC_STRICT_MODE': ('strict_mode', lambda x: x.lower() == 'true'),
            'HDC_PROFILE_DIR': ('profile_dir', str),
            'WANDB_PROJECT': ('wandb_project', str),
            'WANDB_ENTITY': ('wandb_entity', str),
            'WANDB_MODE': ('wandb_mode', str),
            'MLFLOW_TRACKING_URI': ('mlflow_tracking_uri', str),
            'MLFLOW_EXPERIMENT_NAME': ('mlflow_experiment_name', str),
            'HDC_DISABLE_TELEMETRY': ('disable_telemetry', lambda x: x.lower() == 'true'),
            'DATABASE_URL': ('database_url', str),
            'SPEECH_SAMPLE_RATE': ('speech_sample_rate', int),
            'SPEECH_N_MFCC': ('speech_n_mfcc', int),
            'SPEECH_N_FFT': ('speech_n_fft', int),
            'SPEECH_HOP_LENGTH': ('speech_hop_length', int),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    self._config[config_key] = converter(env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration with dictionary."""
        self._config.update(config_dict)
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device-related configuration."""
        return {
            'default_device': self.get('default_device'),
            'cuda_enabled': self.get('cuda_enabled'),
            'fpga_enabled': self.get('fpga_enabled'),
            'vulkan_enabled': self.get('vulkan_enabled'),
            'tpu_enabled': self.get('tpu_enabled'),
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            'memory_pool_size': self.get('memory_pool_size'),
            'batch_size': self.get('batch_size'),
            'num_threads': self.get('num_threads'),
            'memory_profiling': self.get('memory_profiling'),
        }
    
    def get_paths_config(self) -> Dict[str, str]:
        """Get path-related configuration."""
        return {
            'data_dir': self.get('data_dir'),
            'model_dir': self.get('model_dir'),
            'cache_dir': self.get('cache_dir'),
            'download_dir': self.get('download_dir'),
            'test_data_dir': self.get('test_data_dir'),
            'benchmark_dir': self.get('benchmark_dir'),
            'profile_dir': self.get('profile_dir'),
        }
    
    def ensure_directories(self):
        """Ensure all configured directories exist."""
        paths = self.get_paths_config()
        
        for path_name, path_value in paths.items():
            try:
                Path(path_value).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path_value}")
            except Exception as e:
                logger.warning(f"Failed to create directory {path_value}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self._config, f, indent=2, sort_keys=True)
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors."""
        errors = {}
        
        # Validate dimension
        if not isinstance(self.get('default_dimension'), int) or self.get('default_dimension') <= 0:
            errors['default_dimension'] = "Must be a positive integer"
        
        # Validate batch size
        if not isinstance(self.get('batch_size'), int) or self.get('batch_size') <= 0:
            errors['batch_size'] = "Must be a positive integer"
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.get('log_level') not in valid_log_levels:
            errors['log_level'] = f"Must be one of: {valid_log_levels}"
        
        # Validate device
        valid_devices = ['cpu', 'cuda', 'mps', 'auto']
        if self.get('default_device') not in valid_devices:
            errors['default_device'] = f"Must be one of: {valid_devices}"
        
        return errors


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_file: Optional[str] = None, reload: bool = False) -> Config:
    """Get global configuration instance.
    
    Args:
        config_file: Path to configuration file (only used on first call)
        reload: Whether to reload configuration
        
    Returns:
        Configuration instance
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = Config(config_file)
        
        # Ensure directories exist
        _global_config.ensure_directories()
        
        # Validate configuration
        errors = _global_config.validate()
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(
                f"  {key}: {msg}" for key, msg in errors.items()
            )
            logger.warning(error_msg)
    
    return _global_config