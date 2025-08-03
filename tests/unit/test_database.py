"""Tests for database functionality."""

import pytest
import tempfile
from pathlib import Path

from hd_compute.database import DatabaseConnection, ExperimentRepository, MetricsRepository, BenchmarkRepository
from hd_compute.database.models import ExperimentModel, ModelMetricsModel, BenchmarkResultModel


class TestDatabaseConnection:
    """Test DatabaseConnection functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_initialization(self, temp_db_path):
        """Test database initialization."""
        db = DatabaseConnection(temp_db_path)
        assert db.db_path == temp_db_path
        
        # Check that tables were created
        tables = db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [table['name'] for table in tables]
        
        expected_tables = ['experiments', 'model_metrics', 'benchmark_results', 'hypervector_cache']
        for table in expected_tables:
            assert table in table_names
    
    def test_execute_query(self, test_database):
        """Test executing queries."""
        # Insert test data
        test_database.execute_update(
            "INSERT INTO experiments (name, description) VALUES (?, ?)",
            ("test_exp", "Test experiment")
        )
        
        # Query data
        results = test_database.execute_query(
            "SELECT name FROM experiments WHERE description = ?",
            ("Test experiment",)
        )
        
        assert len(results) == 1
        assert results[0]['name'] == 'test_exp'
    
    def test_execute_update(self, test_database):
        """Test executing update queries."""
        affected = test_database.execute_update(
            "INSERT INTO experiments (name) VALUES (?)",
            ("test_exp",)
        )
        
        assert affected == 1
        
        # Check that data was inserted
        results = test_database.execute_query("SELECT COUNT(*) as count FROM experiments")
        assert results[0]['count'] == 1
    
    def test_transaction_rollback(self, test_database):
        """Test transaction rollback on error."""
        with pytest.raises(Exception):
            with test_database.get_cursor() as cursor:
                cursor.execute("INSERT INTO experiments (name) VALUES (?)", ("test1",))
                cursor.execute("INSERT INTO invalid_table (name) VALUES (?)", ("test2",))
        
        # First insert should have been rolled back
        results = test_database.execute_query("SELECT COUNT(*) as count FROM experiments")
        assert results[0]['count'] == 0


class TestExperimentRepository:
    """Test ExperimentRepository functionality."""
    
    @pytest.fixture
    def experiment_repo(self, test_database):
        """Create ExperimentRepository instance."""
        return ExperimentRepository(test_database)
    
    def test_create_experiment(self, experiment_repo):
        """Test creating an experiment."""
        experiment = ExperimentModel(
            name="test_experiment",
            description="Test description",
            config={"param1": "value1"}
        )
        
        experiment_id = experiment_repo.create_experiment(experiment)
        
        assert experiment_id > 0
        
        # Verify experiment was created
        retrieved = experiment_repo.get_experiment(experiment_id)
        assert retrieved is not None
        assert retrieved.name == "test_experiment"
        assert retrieved.description == "Test description"
        assert retrieved.config == {"param1": "value1"}
    
    def test_get_experiment(self, experiment_repo):
        """Test getting an experiment by ID."""
        # Create experiment
        experiment = ExperimentModel(name="test_exp")
        exp_id = experiment_repo.create_experiment(experiment)
        
        # Retrieve experiment
        retrieved = experiment_repo.get_experiment(exp_id)
        assert retrieved is not None
        assert retrieved.id == exp_id
        assert retrieved.name == "test_exp"
        
        # Test non-existent experiment
        assert experiment_repo.get_experiment(99999) is None
    
    def test_get_experiments(self, experiment_repo):
        """Test getting all experiments."""
        # Create multiple experiments
        exp1 = ExperimentModel(name="exp1", status="pending")
        exp2 = ExperimentModel(name="exp2", status="running")
        exp3 = ExperimentModel(name="exp3", status="completed")
        
        experiment_repo.create_experiment(exp1)
        experiment_repo.create_experiment(exp2)
        experiment_repo.create_experiment(exp3)
        
        # Get all experiments
        all_experiments = experiment_repo.get_experiments()
        assert len(all_experiments) == 3
        
        # Get experiments by status
        pending_experiments = experiment_repo.get_experiments(status="pending")
        assert len(pending_experiments) == 1
        assert pending_experiments[0].name == "exp1"
    
    def test_update_experiment_status(self, experiment_repo):
        """Test updating experiment status."""
        experiment = ExperimentModel(name="test_exp", status="pending")
        exp_id = experiment_repo.create_experiment(experiment)
        
        # Update status
        success = experiment_repo.update_experiment_status(exp_id, "running")
        assert success
        
        # Verify update
        retrieved = experiment_repo.get_experiment(exp_id)
        assert retrieved.status == "running"
    
    def test_complete_experiment(self, experiment_repo):
        """Test completing an experiment."""
        experiment = ExperimentModel(name="test_exp", status="running")
        exp_id = experiment_repo.create_experiment(experiment)
        
        # Complete experiment
        success = experiment_repo.complete_experiment(exp_id)
        assert success
        
        # Verify completion
        retrieved = experiment_repo.get_experiment(exp_id)
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None
    
    def test_delete_experiment(self, experiment_repo):
        """Test deleting an experiment."""
        experiment = ExperimentModel(name="test_exp")
        exp_id = experiment_repo.create_experiment(experiment)
        
        # Delete experiment
        success = experiment_repo.delete_experiment(exp_id)
        assert success
        
        # Verify deletion
        assert experiment_repo.get_experiment(exp_id) is None


class TestMetricsRepository:
    """Test MetricsRepository functionality."""
    
    @pytest.fixture
    def metrics_repo(self, test_database):
        """Create MetricsRepository instance."""
        return MetricsRepository(test_database)
    
    @pytest.fixture
    def test_experiment_id(self, test_database):
        """Create test experiment and return its ID."""
        db = test_database
        db.execute_update(
            "INSERT INTO experiments (name) VALUES (?)",
            ("test_exp",)
        )
        return db.get_last_insert_id()
    
    def test_log_metric(self, metrics_repo, test_experiment_id):
        """Test logging a single metric."""
        metric = ModelMetricsModel(
            experiment_id=test_experiment_id,
            metric_name="accuracy",
            metric_value=0.85,
            step=10
        )
        
        metric_id = metrics_repo.log_metric(metric)
        assert metric_id > 0
        
        # Verify metric was logged
        metrics = metrics_repo.get_experiment_metrics(test_experiment_id)
        assert len(metrics) == 1
        assert metrics[0].metric_name == "accuracy"
        assert metrics[0].metric_value == 0.85
    
    def test_log_multiple_metrics(self, metrics_repo, test_experiment_id):
        """Test logging multiple metrics."""
        metrics = [
            ModelMetricsModel(
                experiment_id=test_experiment_id,
                metric_name="accuracy",
                metric_value=0.8,
                step=1
            ),
            ModelMetricsModel(
                experiment_id=test_experiment_id,
                metric_name="loss",
                metric_value=0.5,
                step=1
            )
        ]
        
        metrics_repo.log_metrics(metrics)
        
        # Verify metrics were logged
        logged_metrics = metrics_repo.get_experiment_metrics(test_experiment_id)
        assert len(logged_metrics) == 2
    
    def test_get_metric_history(self, metrics_repo, test_experiment_id):
        """Test getting history of a specific metric."""
        # Log multiple values for same metric
        for step, value in enumerate([0.7, 0.8, 0.85], 1):
            metric = ModelMetricsModel(
                experiment_id=test_experiment_id,
                metric_name="accuracy",
                metric_value=value,
                step=step
            )
            metrics_repo.log_metric(metric)
        
        # Get history
        history = metrics_repo.get_metric_history(test_experiment_id, "accuracy")
        assert len(history) == 3
        
        # Should be ordered by step
        assert history[0].step == 1
        assert history[1].step == 2
        assert history[2].step == 3
        
        # Values should match
        assert history[0].metric_value == 0.7
        assert history[2].metric_value == 0.85
    
    def test_get_latest_metrics(self, metrics_repo, test_experiment_id):
        """Test getting latest value for each metric."""
        # Log metrics at different steps
        metrics = [
            ModelMetricsModel(test_experiment_id, "accuracy", 0.7, 1),
            ModelMetricsModel(test_experiment_id, "accuracy", 0.8, 2),
            ModelMetricsModel(test_experiment_id, "loss", 0.5, 1),
            ModelMetricsModel(test_experiment_id, "loss", 0.3, 2),
        ]
        
        for metric in metrics:
            metrics_repo.log_metric(metric)
        
        # Get latest values
        latest = metrics_repo.get_latest_metrics(test_experiment_id)
        
        assert len(latest) == 2
        assert latest["accuracy"] == 0.8  # Latest accuracy from step 2
        assert latest["loss"] == 0.3      # Latest loss from step 2


class TestBenchmarkRepository:
    """Test BenchmarkRepository functionality."""
    
    @pytest.fixture
    def benchmark_repo(self, test_database):
        """Create BenchmarkRepository instance."""
        return BenchmarkRepository(test_database)
    
    def test_save_benchmark_result(self, benchmark_repo):
        """Test saving a benchmark result."""
        result = BenchmarkResultModel(
            operation_name="random_hv",
            dimension=10000,
            device="cpu",
            backend="pytorch",
            execution_time_ms=5.2,
            memory_usage_mb=100.0,
            iterations=1000
        )
        
        result_id = benchmark_repo.save_benchmark_result(result)
        assert result_id > 0
        
        # Verify result was saved
        results = benchmark_repo.get_benchmark_results()
        assert len(results) == 1
        assert results[0].operation_name == "random_hv"
        assert results[0].execution_time_ms == 5.2
    
    def test_get_benchmark_results_filtered(self, benchmark_repo):
        """Test getting benchmark results with filters."""
        # Create multiple results
        results = [
            BenchmarkResultModel(
                operation_name="random_hv", dimension=10000, device="cpu", 
                backend="pytorch", execution_time_ms=5.0
            ),
            BenchmarkResultModel(
                operation_name="random_hv", dimension=10000, device="cuda", 
                backend="pytorch", execution_time_ms=2.0
            ),
            BenchmarkResultModel(
                operation_name="bundle", dimension=10000, device="cpu", 
                backend="pytorch", execution_time_ms=10.0
            ),
        ]
        
        for result in results:
            benchmark_repo.save_benchmark_result(result)
        
        # Filter by operation
        random_hv_results = benchmark_repo.get_benchmark_results(operation_name="random_hv")
        assert len(random_hv_results) == 2
        
        # Filter by backend
        pytorch_results = benchmark_repo.get_benchmark_results(backend="pytorch")
        assert len(pytorch_results) == 3
        
        # Multiple filters
        cpu_random_results = benchmark_repo.get_benchmark_results(
            operation_name="random_hv", device="cpu"
        )
        assert len(cpu_random_results) == 1
        assert cpu_random_results[0].execution_time_ms == 5.0
    
    def test_get_performance_comparison(self, benchmark_repo):
        """Test getting performance comparison data."""
        # Create results for comparison
        results = [
            BenchmarkResultModel(
                operation_name="random_hv", dimension=10000, device="cpu",
                backend="pytorch", execution_time_ms=5.0
            ),
            BenchmarkResultModel(
                operation_name="random_hv", dimension=10000, device="cpu",
                backend="pytorch", execution_time_ms=6.0
            ),
            BenchmarkResultModel(
                operation_name="random_hv", dimension=10000, device="cuda",
                backend="pytorch", execution_time_ms=2.0
            ),
        ]
        
        for result in results:
            benchmark_repo.save_benchmark_result(result)
        
        # Get comparison
        comparison = benchmark_repo.get_performance_comparison("random_hv")
        
        assert len(comparison) == 2  # CPU and CUDA groups
        
        # Find CPU and CUDA results
        cpu_result = next(r for r in comparison if r['device'] == 'cpu')
        cuda_result = next(r for r in comparison if r['device'] == 'cuda')
        
        assert cpu_result['avg_time'] == 5.5  # Average of 5.0 and 6.0
        assert cpu_result['run_count'] == 2
        assert cuda_result['avg_time'] == 2.0
        assert cuda_result['run_count'] == 1