"""Database models for HD-Compute-Toolkit."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class ExperimentModel:
    """Model for experiment tracking."""
    
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    status: str = "pending"
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'config': json.dumps(self.config) if self.config else None
        }
    
    @classmethod
    def from_row(cls, row) -> 'ExperimentModel':
        """Create model from database row."""
        return cls(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            status=row['status'],
            config=json.loads(row['config']) if row['config'] else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            completed_at=row['completed_at']
        )


@dataclass
class ModelMetricsModel:
    """Model for storing training/evaluation metrics."""
    
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    metric_name: str = ""
    metric_value: float = 0.0
    step: int = 0
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'experiment_id': self.experiment_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'step': self.step
        }
    
    @classmethod
    def from_row(cls, row) -> 'ModelMetricsModel':
        """Create model from database row."""
        return cls(
            id=row['id'],
            experiment_id=row['experiment_id'],
            metric_name=row['metric_name'],
            metric_value=row['metric_value'],
            step=row['step'],
            timestamp=row['timestamp']
        )


@dataclass
class BenchmarkResultModel:
    """Model for storing benchmark results."""
    
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    operation_name: str = ""
    dimension: int = 0
    device: str = ""
    backend: str = ""
    execution_time_ms: float = 0.0
    memory_usage_mb: Optional[float] = None
    iterations: int = 1
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'experiment_id': self.experiment_id,
            'operation_name': self.operation_name,
            'dimension': self.dimension,
            'device': self.device,
            'backend': self.backend,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'iterations': self.iterations
        }
    
    @classmethod
    def from_row(cls, row) -> 'BenchmarkResultModel':
        """Create model from database row."""
        return cls(
            id=row['id'],
            experiment_id=row['experiment_id'],
            operation_name=row['operation_name'],
            dimension=row['dimension'],
            device=row['device'],
            backend=row['backend'],
            execution_time_ms=row['execution_time_ms'],
            memory_usage_mb=row['memory_usage_mb'],
            iterations=row['iterations'],
            timestamp=row['timestamp']
        )


@dataclass
class HypervectorCacheModel:
    """Model for caching hypervectors."""
    
    id: Optional[int] = None
    cache_key: str = ""
    dimension: int = 0
    data: bytes = b""
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'cache_key': self.cache_key,
            'dimension': self.dimension,
            'data': self.data,
            'metadata': json.dumps(self.metadata) if self.metadata else None
        }
    
    @classmethod
    def from_row(cls, row) -> 'HypervectorCacheModel':
        """Create model from database row."""
        return cls(
            id=row['id'],
            cache_key=row['cache_key'],
            dimension=row['dimension'],
            data=row['data'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None,
            created_at=row['created_at'],
            accessed_at=row['accessed_at'],
            access_count=row['access_count']
        )