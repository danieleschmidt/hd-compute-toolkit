"""Repository classes for data access operations."""

import hashlib
import pickle
from typing import List, Optional, Dict, Any
import numpy as np

from .connection import DatabaseConnection
from .models import ExperimentModel, ModelMetricsModel, BenchmarkResultModel, HypervectorCacheModel


class BaseRepository:
    """Base repository class with common functionality."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection


class ExperimentRepository(BaseRepository):
    """Repository for experiment management."""
    
    def create_experiment(self, experiment: ExperimentModel) -> int:
        """Create a new experiment."""
        query = """
            INSERT INTO experiments (name, description, status, config)
            VALUES (?, ?, ?, ?)
        """
        data = experiment.to_dict()
        self.db.execute_update(
            query, 
            (data['name'], data['description'], data['status'], data['config'])
        )
        return self.db.get_last_insert_id()
    
    def get_experiment(self, experiment_id: int) -> Optional[ExperimentModel]:
        """Get experiment by ID."""
        query = "SELECT * FROM experiments WHERE id = ?"
        results = self.db.execute_query(query, (experiment_id,))
        
        if results:
            return ExperimentModel.from_row(results[0])
        return None
    
    def get_experiments(self, status: Optional[str] = None) -> List[ExperimentModel]:
        """Get all experiments, optionally filtered by status."""
        if status:
            query = "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC"
            results = self.db.execute_query(query, (status,))
        else:
            query = "SELECT * FROM experiments ORDER BY created_at DESC"
            results = self.db.execute_query(query)
        
        return [ExperimentModel.from_row(row) for row in results]
    
    def update_experiment_status(self, experiment_id: int, status: str) -> bool:
        """Update experiment status."""
        query = """
            UPDATE experiments 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        affected = self.db.execute_update(query, (status, experiment_id))
        return affected > 0
    
    def complete_experiment(self, experiment_id: int) -> bool:
        """Mark experiment as completed."""
        query = """
            UPDATE experiments 
            SET status = 'completed', 
                completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        affected = self.db.execute_update(query, (experiment_id,))
        return affected > 0
    
    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete experiment and related data."""
        # Delete related metrics and benchmarks first
        self.db.execute_update("DELETE FROM model_metrics WHERE experiment_id = ?", (experiment_id,))
        self.db.execute_update("DELETE FROM benchmark_results WHERE experiment_id = ?", (experiment_id,))
        
        # Delete experiment
        affected = self.db.execute_update("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        return affected > 0


class MetricsRepository(BaseRepository):
    """Repository for model metrics."""
    
    def log_metric(self, metric: ModelMetricsModel) -> int:
        """Log a single metric."""
        query = """
            INSERT INTO model_metrics (experiment_id, metric_name, metric_value, step)
            VALUES (?, ?, ?, ?)
        """
        data = metric.to_dict()
        self.db.execute_update(
            query,
            (data['experiment_id'], data['metric_name'], data['metric_value'], data['step'])
        )
        return self.db.get_last_insert_id()
    
    def log_metrics(self, metrics: List[ModelMetricsModel]):
        """Log multiple metrics efficiently."""
        query = """
            INSERT INTO model_metrics (experiment_id, metric_name, metric_value, step)
            VALUES (?, ?, ?, ?)
        """
        
        with self.db.get_cursor() as cursor:
            for metric in metrics:
                data = metric.to_dict()
                cursor.execute(query, (
                    data['experiment_id'], 
                    data['metric_name'], 
                    data['metric_value'], 
                    data['step']
                ))
    
    def get_experiment_metrics(self, experiment_id: int) -> List[ModelMetricsModel]:
        """Get all metrics for an experiment."""
        query = """
            SELECT * FROM model_metrics 
            WHERE experiment_id = ? 
            ORDER BY step ASC, timestamp ASC
        """
        results = self.db.execute_query(query, (experiment_id,))
        return [ModelMetricsModel.from_row(row) for row in results]
    
    def get_metric_history(self, experiment_id: int, metric_name: str) -> List[ModelMetricsModel]:
        """Get history of a specific metric."""
        query = """
            SELECT * FROM model_metrics 
            WHERE experiment_id = ? AND metric_name = ?
            ORDER BY step ASC, timestamp ASC
        """
        results = self.db.execute_query(query, (experiment_id, metric_name))
        return [ModelMetricsModel.from_row(row) for row in results]
    
    def get_latest_metrics(self, experiment_id: int) -> Dict[str, float]:
        """Get latest value for each metric in an experiment."""
        query = """
            SELECT metric_name, metric_value
            FROM model_metrics m1
            WHERE m1.experiment_id = ? 
            AND m1.step = (
                SELECT MAX(m2.step) 
                FROM model_metrics m2 
                WHERE m2.experiment_id = m1.experiment_id 
                AND m2.metric_name = m1.metric_name
            )
        """
        results = self.db.execute_query(query, (experiment_id,))
        return {row['metric_name']: row['metric_value'] for row in results}


class BenchmarkRepository(BaseRepository):
    """Repository for benchmark results."""
    
    def save_benchmark_result(self, result: BenchmarkResultModel) -> int:
        """Save a benchmark result."""
        query = """
            INSERT INTO benchmark_results 
            (experiment_id, operation_name, dimension, device, backend, 
             execution_time_ms, memory_usage_mb, iterations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = result.to_dict()
        self.db.execute_update(query, (
            data['experiment_id'], data['operation_name'], data['dimension'],
            data['device'], data['backend'], data['execution_time_ms'],
            data['memory_usage_mb'], data['iterations']
        ))
        return self.db.get_last_insert_id()
    
    def get_benchmark_results(
        self, 
        experiment_id: Optional[int] = None,
        operation_name: Optional[str] = None,
        backend: Optional[str] = None
    ) -> List[BenchmarkResultModel]:
        """Get benchmark results with optional filtering."""
        conditions = []
        params = []
        
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        
        if operation_name:
            conditions.append("operation_name = ?")
            params.append(operation_name)
        
        if backend:
            conditions.append("backend = ?")
            params.append(backend)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM benchmark_results 
            WHERE {where_clause}
            ORDER BY timestamp DESC
        """
        
        results = self.db.execute_query(query, params)
        return [BenchmarkResultModel.from_row(row) for row in results]
    
    def get_performance_comparison(self, operation_name: str) -> List[Dict[str, Any]]:
        """Get performance comparison across backends and devices."""
        query = """
            SELECT backend, device, dimension,
                   AVG(execution_time_ms) as avg_time,
                   MIN(execution_time_ms) as min_time,
                   MAX(execution_time_ms) as max_time,
                   COUNT(*) as run_count
            FROM benchmark_results 
            WHERE operation_name = ?
            GROUP BY backend, device, dimension
            ORDER BY dimension, avg_time
        """
        results = self.db.execute_query(query, (operation_name,))
        return [dict(row) for row in results]


class HypervectorCacheRepository(BaseRepository):
    """Repository for hypervector caching."""
    
    def _generate_cache_key(self, dimension: int, seed: int, metadata: Dict[str, Any]) -> str:
        """Generate a unique cache key for hypervector."""
        key_data = f"{dimension}_{seed}_{str(sorted(metadata.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def store_hypervector(
        self, 
        hypervector: np.ndarray, 
        dimension: int, 
        seed: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store hypervector in cache."""
        metadata = metadata or {}
        cache_key = self._generate_cache_key(dimension, seed, metadata)
        
        # Serialize hypervector
        hv_data = pickle.dumps(hypervector)
        
        # Try to insert or update if exists
        query = """
            INSERT OR REPLACE INTO hypervector_cache 
            (cache_key, dimension, data, metadata, accessed_at, access_count)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 
                    COALESCE((SELECT access_count + 1 FROM hypervector_cache WHERE cache_key = ?), 1))
        """
        
        cache_model = HypervectorCacheModel(
            cache_key=cache_key,
            dimension=dimension,
            data=hv_data,
            metadata=metadata
        )
        
        data = cache_model.to_dict()
        self.db.execute_update(query, (
            cache_key, dimension, hv_data, data['metadata'], cache_key
        ))
        
        return cache_key
    
    def get_hypervector(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve hypervector from cache."""
        # Update access time and count
        self.db.execute_update("""
            UPDATE hypervector_cache 
            SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
            WHERE cache_key = ?
        """, (cache_key,))
        
        # Get hypervector data
        query = "SELECT data FROM hypervector_cache WHERE cache_key = ?"
        results = self.db.execute_query(query, (cache_key,))
        
        if results:
            return pickle.loads(results[0]['data'])
        return None
    
    def cleanup_old_cache(self, max_age_days: int = 30, max_entries: int = 1000):
        """Clean up old cache entries."""
        # Remove old entries
        self.db.execute_update("""
            DELETE FROM hypervector_cache 
            WHERE created_at < datetime('now', '-{} days')
        """.format(max_age_days))
        
        # Keep only most recently accessed entries if over limit
        count_query = "SELECT COUNT(*) as count FROM hypervector_cache"
        count_result = self.db.execute_query(count_query)
        
        if count_result and count_result[0]['count'] > max_entries:
            self.db.execute_update("""
                DELETE FROM hypervector_cache 
                WHERE id NOT IN (
                    SELECT id FROM hypervector_cache 
                    ORDER BY accessed_at DESC 
                    LIMIT ?
                )
            """, (max_entries,))
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        query = """
            SELECT 
                COUNT(*) as total_entries,
                SUM(LENGTH(data)) as total_size_bytes,
                AVG(access_count) as avg_access_count,
                MAX(accessed_at) as last_access
            FROM hypervector_cache
        """
        result = self.db.execute_query(query)
        
        if result:
            stats = dict(result[0])
            stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024) if stats['total_size_bytes'] else 0
            return stats
        
        return {
            'total_entries': 0,
            'total_size_bytes': 0,
            'total_size_mb': 0,
            'avg_access_count': 0,
            'last_access': None
        }