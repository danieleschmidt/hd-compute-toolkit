"""Database connection management."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Thread-safe database connection manager."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses environment variable.
        """
        self.db_path = db_path or os.getenv('DATABASE_URL', 'sqlite:///./hdc_experiments.db')
        
        # Extract actual file path from SQLite URL
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path.replace('sqlite:///', '')
        
        self._local = threading.local()
        self._ensure_directory_exists()
        self._init_database()
    
    def _ensure_directory_exists(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign key support
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
        return self._local.connection
    
    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Get database cursor with automatic transaction management."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_cursor() as cursor:
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    config JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Model metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Benchmark results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    operation_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    device TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    memory_usage_mb REAL,
                    iterations INTEGER DEFAULT 1,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Hypervector storage table (for caching)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hypervector_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    dimension INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name 
                ON experiments (name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metrics_experiment 
                ON model_metrics (experiment_id, metric_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_benchmark_results_experiment 
                ON benchmark_results (experiment_id, operation_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypervector_cache_key 
                ON hypervector_cache (cache_key)
            """)
            
            logger.info("Database schema initialized successfully")
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
    
    def execute_query(self, query: str, params=None):
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_update(self, query: str, params=None) -> int:
        """Execute an update/insert query and return affected rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.rowcount
    
    def get_last_insert_id(self) -> int:
        """Get the ID of the last inserted row."""
        conn = self._get_connection()
        return conn.lastrowid