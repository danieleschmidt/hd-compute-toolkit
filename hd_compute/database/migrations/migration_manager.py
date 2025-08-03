"""Database migration management system."""

import os
import sqlite3
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Migration:
    """Database migration class."""
    
    def __init__(self, version: int, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
    
    def apply(self, cursor: sqlite3.Cursor):
        """Apply migration."""
        logger.info(f"Applying migration {self.version}: {self.name}")
        cursor.executescript(self.up_sql)
    
    def rollback(self, cursor: sqlite3.Cursor):
        """Rollback migration."""
        if self.down_sql:
            logger.info(f"Rolling back migration {self.version}: {self.name}")
            cursor.executescript(self.down_sql)
        else:
            logger.warning(f"No rollback SQL for migration {self.version}: {self.name}")


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations: List[Migration] = []
        self._init_migrations_table()
        self._load_migrations()
    
    def _init_migrations_table(self):
        """Initialize migrations tracking table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_migrations(self):
        """Load migration definitions."""
        # Migration 001: Create initial hypervector storage table
        self.migrations.append(Migration(
            version=1,
            name="add_hypervector_metadata_index",
            up_sql="""
                CREATE INDEX IF NOT EXISTS idx_hypervector_cache_metadata 
                ON hypervector_cache (metadata);
                
                CREATE INDEX IF NOT EXISTS idx_hypervector_cache_dimension 
                ON hypervector_cache (dimension);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_hypervector_cache_metadata;
                DROP INDEX IF EXISTS idx_hypervector_cache_dimension;
            """
        ))
        
        # Migration 002: Add experiment tags
        self.migrations.append(Migration(
            version=2,
            name="add_experiment_tags",
            up_sql="""
                ALTER TABLE experiments ADD COLUMN tags TEXT;
                CREATE INDEX IF NOT EXISTS idx_experiments_tags ON experiments (tags);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_experiments_tags;
                -- SQLite doesn't support dropping columns easily
                -- This would require recreating the table
            """
        ))
        
        # Migration 003: Add benchmark environment info
        self.migrations.append(Migration(
            version=3,
            name="add_benchmark_environment",
            up_sql="""
                ALTER TABLE benchmark_results ADD COLUMN cpu_info TEXT;
                ALTER TABLE benchmark_results ADD COLUMN gpu_info TEXT;
                ALTER TABLE benchmark_results ADD COLUMN memory_total_mb REAL;
                ALTER TABLE benchmark_results ADD COLUMN python_version TEXT;
                ALTER TABLE benchmark_results ADD COLUMN framework_version TEXT;
                
                CREATE INDEX IF NOT EXISTS idx_benchmark_device_backend 
                ON benchmark_results (device, backend);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_benchmark_device_backend;
                -- Column removal would require table recreation in SQLite
            """
        ))
        
        # Migration 004: Add model checkpoints table
        self.migrations.append(Migration(
            version=4,
            name="add_model_checkpoints",
            up_sql="""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    checkpoint_name TEXT NOT NULL,
                    epoch INTEGER,
                    step INTEGER,
                    model_state BLOB,
                    optimizer_state BLOB,
                    metrics JSON,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment 
                ON model_checkpoints (experiment_id);
                
                CREATE INDEX IF NOT EXISTS idx_checkpoints_step 
                ON model_checkpoints (experiment_id, step);
            """,
            down_sql="""
                DROP TABLE IF EXISTS model_checkpoints;
            """
        ))
        
        # Migration 005: Add hypervector operation logs
        self.migrations.append(Migration(
            version=5,
            name="add_operation_logs",
            up_sql="""
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    operation_type TEXT NOT NULL,
                    operation_params JSON,
                    execution_time_ms REAL,
                    memory_usage_mb REAL,
                    input_dimensions TEXT,
                    output_dimensions TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_operation_logs_type 
                ON operation_logs (operation_type);
                
                CREATE INDEX IF NOT EXISTS idx_operation_logs_experiment 
                ON operation_logs (experiment_id, timestamp);
            """,
            down_sql="""
                DROP TABLE IF EXISTS operation_logs;
            """
        ))
    
    def get_current_version(self) -> int:
        """Get current database schema version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(version) FROM schema_migrations")
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] if result[0] is not None else 0
    
    def get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        results = cursor.fetchall()
        
        conn.close()
        
        return [row[0] for row in results]
    
    def migrate_to_latest(self) -> List[int]:
        """Apply all pending migrations."""
        current_version = self.get_current_version()
        applied_migrations = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for migration in self.migrations:
                if migration.version > current_version:
                    migration.apply(cursor)
                    
                    # Record migration as applied
                    cursor.execute("""
                        INSERT INTO schema_migrations (version, name)
                        VALUES (?, ?)
                    """, (migration.version, migration.name))
                    
                    applied_migrations.append(migration.version)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()
        
        return applied_migrations
    
    def migrate_to_version(self, target_version: int) -> List[int]:
        """Migrate to specific version."""
        current_version = self.get_current_version()
        applied_migrations = []
        
        if target_version == current_version:
            logger.info(f"Already at version {target_version}")
            return applied_migrations
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if target_version > current_version:
                # Migrate up
                for migration in self.migrations:
                    if current_version < migration.version <= target_version:
                        migration.apply(cursor)
                        
                        cursor.execute("""
                            INSERT INTO schema_migrations (version, name)
                            VALUES (?, ?)
                        """, (migration.version, migration.name))
                        
                        applied_migrations.append(migration.version)
            
            else:
                # Migrate down (rollback)
                applied = self.get_applied_migrations()
                migrations_to_rollback = [m for m in self.migrations if target_version < m.version <= current_version]
                migrations_to_rollback.sort(key=lambda m: m.version, reverse=True)
                
                for migration in migrations_to_rollback:
                    if migration.version in applied:
                        migration.rollback(cursor)
                        
                        cursor.execute("""
                            DELETE FROM schema_migrations WHERE version = ?
                        """, (migration.version,))
                        
                        applied_migrations.append(-migration.version)  # Negative to indicate rollback
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()
        
        return applied_migrations
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status information."""
        current_version = self.get_current_version()
        applied_migrations = self.get_applied_migrations()
        
        pending_migrations = [
            m for m in self.migrations 
            if m.version not in applied_migrations
        ]
        
        return {
            'current_version': current_version,
            'latest_available_version': max(m.version for m in self.migrations) if self.migrations else 0,
            'applied_migrations': applied_migrations,
            'pending_migrations': [m.version for m in pending_migrations],
            'migration_count': len(self.migrations),
            'pending_count': len(pending_migrations)
        }