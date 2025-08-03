"""Integration tests for the REST API."""

import pytest
import json
from typing import Generator

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from hd_compute.api.server import create_app


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    config = {
        'database_url': 'sqlite:///:memory:',
        'cache_dir': '/tmp/test_cache',
        'default_device': 'cpu',
        'default_dimension': 1000
    }
    return create_app(config)


@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(test_app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns success."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"


class TestHDCOperations:
    """Test HDC operation endpoints."""
    
    def test_generate_random_hypervector(self, client):
        """Test random hypervector generation."""
        request_data = {
            "dimension": 1000,
            "sparsity": 0.5,
            "seed": 42
        }
        
        response = client.post("/hdc/random", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "hypervector" in data
        assert data["dimension"] == 1000
        assert data["sparsity"] == 0.5
        assert len(data["hypervector"]) == 1000
        assert isinstance(data["cache_hit"], bool)
    
    def test_generate_random_hypervector_cached(self, client):
        """Test that same seed produces cached result."""
        request_data = {
            "dimension": 1000,
            "sparsity": 0.5,
            "seed": 42
        }
        
        # First request
        response1 = client.post("/hdc/random", json=request_data)
        assert response1.status_code == 200
        data1 = response1.json()
        assert not data1["cache_hit"]  # First time should not be cached
        
        # Second request with same parameters
        response2 = client.post("/hdc/random", json=request_data)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["cache_hit"]  # Should be from cache
        
        # Hypervectors should be identical
        assert data1["hypervector"] == data2["hypervector"]
    
    def test_bundle_hypervectors(self, client):
        """Test bundling hypervectors."""
        # First generate some hypervectors
        hv1_response = client.post("/hdc/random", json={"dimension": 100, "seed": 1})
        hv2_response = client.post("/hdc/random", json={"dimension": 100, "seed": 2})
        
        hv1 = hv1_response.json()["hypervector"]
        hv2 = hv2_response.json()["hypervector"]
        
        # Bundle them
        bundle_request = {
            "hypervectors": [hv1, hv2]
        }
        
        response = client.post("/hdc/bundle", json=bundle_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "hypervector" in data
        assert data["dimension"] == 100
        assert data["input_count"] == 2
        assert len(data["hypervector"]) == 100
    
    def test_bundle_empty_list(self, client):
        """Test bundling empty list returns error."""
        bundle_request = {"hypervectors": []}
        
        response = client.post("/hdc/bundle", json=bundle_request)
        assert response.status_code == 400
        assert "No hypervectors provided" in response.json()["detail"]
    
    def test_bind_hypervectors(self, client):
        """Test binding hypervectors."""
        # Generate two hypervectors
        hv1_response = client.post("/hdc/random", json={"dimension": 100, "seed": 1})
        hv2_response = client.post("/hdc/random", json={"dimension": 100, "seed": 2})
        
        hv1 = hv1_response.json()["hypervector"]
        hv2 = hv2_response.json()["hypervector"]
        
        # Bind them
        bind_request = {
            "hypervector1": hv1,
            "hypervector2": hv2
        }
        
        response = client.post("/hdc/bind", json=bind_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "hypervector" in data
        assert data["dimension"] == 100
        assert len(data["hypervector"]) == 100
    
    def test_bind_mismatched_dimensions(self, client):
        """Test binding hypervectors with different dimensions fails."""
        hv1 = [1.0] * 100
        hv2 = [1.0] * 200  # Different dimension
        
        bind_request = {
            "hypervector1": hv1,
            "hypervector2": hv2
        }
        
        response = client.post("/hdc/bind", json=bind_request)
        assert response.status_code == 400
        assert "same dimension" in response.json()["detail"]
    
    def test_compute_similarity(self, client):
        """Test similarity computation."""
        # Generate two hypervectors
        hv1_response = client.post("/hdc/random", json={"dimension": 100, "seed": 1})
        hv2_response = client.post("/hdc/random", json={"dimension": 100, "seed": 2})
        
        hv1 = hv1_response.json()["hypervector"]
        hv2 = hv2_response.json()["hypervector"]
        
        # Compute similarity
        similarity_request = {
            "hypervector1": hv1,
            "hypervector2": hv2
        }
        
        response = client.post("/hdc/similarity", json=similarity_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "similarity" in data
        assert data["metric"] == "cosine"
        assert -1.0 <= data["similarity"] <= 1.0
    
    def test_self_similarity(self, client):
        """Test that self-similarity is 1.0."""
        # Generate hypervector
        hv_response = client.post("/hdc/random", json={"dimension": 100, "seed": 1})
        hv = hv_response.json()["hypervector"]
        
        # Compute self-similarity
        similarity_request = {
            "hypervector1": hv,
            "hypervector2": hv
        }
        
        response = client.post("/hdc/similarity", json=similarity_request)
        assert response.status_code == 200
        
        data = response.json()
        assert abs(data["similarity"] - 1.0) < 1e-6


class TestExperimentAPI:
    """Test experiment management endpoints."""
    
    def test_create_experiment(self, client):
        """Test creating an experiment."""
        experiment_data = {
            "name": "test_experiment",
            "description": "Test experiment description",
            "config": {"param1": "value1", "param2": 42}
        }
        
        response = client.post("/experiments/", json=experiment_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test_experiment"
        assert data["description"] == "Test experiment description"
        assert data["config"] == {"param1": "value1", "param2": 42}
        assert data["status"] == "pending"
        assert "id" in data
        assert data["id"] > 0
    
    def test_get_experiment(self, client):
        """Test getting an experiment by ID."""
        # Create experiment first
        experiment_data = {"name": "test_experiment"}
        create_response = client.post("/experiments/", json=experiment_data)
        experiment_id = create_response.json()["id"]
        
        # Get experiment
        response = client.get(f"/experiments/{experiment_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == experiment_id
        assert data["name"] == "test_experiment"
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting non-existent experiment returns 404."""
        response = client.get("/experiments/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_list_experiments(self, client):
        """Test listing experiments."""
        # Create a few experiments
        for i in range(3):
            experiment_data = {"name": f"experiment_{i}"}
            client.post("/experiments/", json=experiment_data)
        
        # List all experiments
        response = client.get("/experiments/")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) >= 3
        assert all("id" in exp for exp in data)
        assert all("name" in exp for exp in data)
    
    def test_list_experiments_by_status(self, client):
        """Test listing experiments filtered by status."""
        response = client.get("/experiments/?status=pending")
        assert response.status_code == 200
        
        data = response.json()
        assert all(exp["status"] == "pending" for exp in data)
    
    def test_log_metric(self, client):
        """Test logging metrics for an experiment."""
        # Create experiment first
        experiment_data = {"name": "metric_test_experiment"}
        create_response = client.post("/experiments/", json=experiment_data)
        experiment_id = create_response.json()["id"]
        
        # Log metric
        metric_data = {
            "metric_name": "accuracy",
            "metric_value": 0.85,
            "step": 10
        }
        
        response = client.post(f"/experiments/{experiment_id}/metrics", json=metric_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "metric_id" in data
        assert data["status"] == "logged"
    
    def test_log_metric_nonexistent_experiment(self, client):
        """Test logging metric for non-existent experiment fails."""
        metric_data = {
            "metric_name": "accuracy",
            "metric_value": 0.85,
            "step": 10
        }
        
        response = client.post("/experiments/99999/metrics", json=metric_data)
        assert response.status_code == 404


class TestCacheAPI:
    """Test cache management endpoints."""
    
    def test_get_cache_stats(self, client):
        """Test getting cache statistics."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "memory_cache_size" in data
        assert "file_cache_size" in data
        assert "total_size_mb" in data
        assert isinstance(data["memory_cache_size"], int)
        assert isinstance(data["total_size_mb"], float)
    
    def test_clear_cache(self, client):
        """Test clearing cache."""
        # First generate some data to cache
        client.post("/hdc/random", json={"dimension": 100, "seed": 42})
        
        # Clear cache
        response = client.delete("/cache/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert "cleared all cache" in data["status"]
    
    def test_clear_cache_namespace(self, client):
        """Test clearing specific cache namespace."""
        response = client.delete("/cache/clear?namespace=test_namespace")
        assert response.status_code == 200
        
        data = response.json()
        assert "cleared namespace 'test_namespace'" in data["status"]


class TestBenchmarkAPI:
    """Test benchmark management endpoints."""
    
    def test_save_benchmark_result(self, client):
        """Test saving benchmark result."""
        result_data = {
            "operation_name": "random_hv",
            "dimension": 10000,
            "device": "cpu",
            "backend": "pytorch",
            "execution_time_ms": 5.2,
            "memory_usage_mb": 100.0,
            "iterations": 1000
        }
        
        response = client.post("/benchmarks/results", json=result_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "result_id" in data
        assert data["status"] == "saved"
    
    def test_get_benchmark_results(self, client):
        """Test getting benchmark results."""
        # Save a benchmark result first
        result_data = {
            "operation_name": "bundle",
            "dimension": 5000,
            "device": "cpu",
            "backend": "pytorch",
            "execution_time_ms": 10.5,
            "iterations": 500
        }
        client.post("/benchmarks/results", json=result_data)
        
        # Get results
        response = client.get("/benchmarks/results")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        # Check structure of first result
        if data:
            result = data[0]
            assert "id" in result
            assert "operation_name" in result
            assert "execution_time_ms" in result
    
    def test_get_benchmark_results_filtered(self, client):
        """Test getting filtered benchmark results."""
        response = client.get("/benchmarks/results?operation_name=random_hv&backend=pytorch")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # All results should match filter criteria
        for result in data:
            if "operation_name" in result:
                assert result["operation_name"] == "random_hv"
            if "backend" in result:
                assert result["backend"] == "pytorch"


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self, client):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/hdc/random", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_validation_errors(self, client):
        """Test request validation errors."""
        # Invalid dimension (too large)
        invalid_request = {
            "dimension": 1000000,  # Exceeds limit
            "sparsity": 0.5
        }
        
        response = client.post("/hdc/random", json=invalid_request)
        assert response.status_code == 422
        
        # Invalid sparsity (out of range)
        invalid_request = {
            "dimension": 1000,
            "sparsity": 1.5  # > 1.0
        }
        
        response = client.post("/hdc/random", json=invalid_request)
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.put("/health")  # GET only endpoint
        assert response.status_code == 405
    
    def test_not_found(self, client):
        """Test 404 for non-existent endpoints."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404