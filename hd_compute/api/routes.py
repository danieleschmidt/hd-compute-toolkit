"""API routes for HD-Compute-Toolkit."""

import time
from typing import List, Optional, Dict, Any
import logging

try:
    from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    HTTPException = None
    BaseModel = None

from ..torch import HDComputeTorch
from ..database import DatabaseConnection, ExperimentRepository, MetricsRepository, BenchmarkRepository
from ..database.models import ExperimentModel, ModelMetricsModel, BenchmarkResultModel
from ..cache import CacheManager, HypervectorCache
from .server import get_database_connection, get_cache_manager, get_hdc_backend

logger = logging.getLogger(__name__)


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class HealthResponse(BaseModel):
        status: str = "healthy"
        timestamp: float = Field(default_factory=time.time)
        version: str = "0.1.0"
    
    class HypervectorRequest(BaseModel):
        dimension: int = Field(gt=0, le=100000)
        sparsity: float = Field(ge=0.0, le=1.0, default=0.5)
        seed: Optional[int] = None
        batch_size: Optional[int] = Field(None, gt=0, le=1000)
    
    class HypervectorResponse(BaseModel):
        hypervector: List[float]
        dimension: int
        sparsity: float
        cache_hit: bool = False
    
    class BundleRequest(BaseModel):
        hypervectors: List[List[float]]
        threshold: Optional[float] = None
    
    class BindRequest(BaseModel):
        hypervector1: List[float]
        hypervector2: List[float]
    
    class SimilarityRequest(BaseModel):
        hypervector1: List[float]
        hypervector2: List[float]
    
    class SimilarityResponse(BaseModel):
        similarity: float
        metric: str = "cosine"
    
    class ExperimentCreate(BaseModel):
        name: str
        description: Optional[str] = None
        config: Optional[Dict[str, Any]] = None
    
    class ExperimentResponse(BaseModel):
        id: int
        name: str
        description: Optional[str]
        status: str
        config: Optional[Dict[str, Any]]
        created_at: Optional[str]
        updated_at: Optional[str]
        completed_at: Optional[str]
    
    class MetricLog(BaseModel):
        metric_name: str
        metric_value: float
        step: int = 0
    
    class BenchmarkResult(BaseModel):
        operation_name: str
        dimension: int
        device: str
        backend: str
        execution_time_ms: float
        memory_usage_mb: Optional[float] = None
        iterations: int = 1
    
    class CacheStats(BaseModel):
        memory_cache_size: int
        file_cache_size: int
        total_size_mb: float
        hit_rate: Optional[float] = None


def setup_routes(app):
    """Setup API routes."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")
    
    # Health endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()
    
    # HDC Operations Router
    hdc_router = APIRouter(prefix="/hdc", tags=["HDC Operations"])
    
    @hdc_router.post("/random", response_model=HypervectorResponse)
    async def generate_random_hypervector(
        request: HypervectorRequest,
        backend: HDComputeTorch = Depends(get_hdc_backend),
        cache: CacheManager = Depends(get_cache_manager)
    ):
        """Generate random hypervector(s)."""
        try:
            hv_cache = HypervectorCache(cache)
            
            if request.batch_size:
                # Batch generation
                hvs = backend.random_hv(
                    sparsity=request.sparsity,
                    batch_size=request.batch_size
                )
                # Return first hypervector for simplicity
                hv = hvs[0]
            else:
                # Single hypervector - check cache first
                if request.seed is not None:
                    cached_hv = hv_cache.get_hypervector(
                        request.dimension, request.sparsity, request.seed
                    )
                    if cached_hv is not None:
                        return HypervectorResponse(
                            hypervector=cached_hv.tolist(),
                            dimension=request.dimension,
                            sparsity=request.sparsity,
                            cache_hit=True
                        )
                
                hv = backend.random_hv(sparsity=request.sparsity)
                
                # Cache if seed provided
                if request.seed is not None:
                    hv_cache.store_hypervector(
                        hv.cpu().numpy(), request.dimension, request.sparsity, request.seed
                    )
            
            return HypervectorResponse(
                hypervector=hv.cpu().numpy().tolist(),
                dimension=request.dimension,
                sparsity=request.sparsity,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Error generating hypervector: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @hdc_router.post("/bundle")
    async def bundle_hypervectors(
        request: BundleRequest,
        backend: HDComputeTorch = Depends(get_hdc_backend)
    ):
        """Bundle multiple hypervectors."""
        try:
            import torch
            
            # Convert to tensors
            hvs = [torch.tensor(hv, dtype=backend.dtype, device=backend.device) 
                   for hv in request.hypervectors]
            
            if not hvs:
                raise HTTPException(status_code=400, detail="No hypervectors provided")
            
            bundled = backend.bundle(hvs, threshold=request.threshold)
            
            return {
                "hypervector": bundled.cpu().numpy().tolist(),
                "dimension": len(bundled),
                "input_count": len(hvs)
            }
            
        except Exception as e:
            logger.error(f"Error bundling hypervectors: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @hdc_router.post("/bind")
    async def bind_hypervectors(
        request: BindRequest,
        backend: HDComputeTorch = Depends(get_hdc_backend)
    ):
        """Bind two hypervectors."""
        try:
            import torch
            
            hv1 = torch.tensor(request.hypervector1, dtype=backend.dtype, device=backend.device)
            hv2 = torch.tensor(request.hypervector2, dtype=backend.dtype, device=backend.device)
            
            if len(hv1) != len(hv2):
                raise HTTPException(
                    status_code=400, 
                    detail="Hypervectors must have same dimension"
                )
            
            bound = backend.bind(hv1, hv2)
            
            return {
                "hypervector": bound.cpu().numpy().tolist(),
                "dimension": len(bound)
            }
            
        except Exception as e:
            logger.error(f"Error binding hypervectors: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @hdc_router.post("/similarity", response_model=SimilarityResponse)
    async def compute_similarity(
        request: SimilarityRequest,
        backend: HDComputeTorch = Depends(get_hdc_backend)
    ):
        """Compute similarity between two hypervectors."""
        try:
            import torch
            
            hv1 = torch.tensor(request.hypervector1, dtype=backend.dtype, device=backend.device)
            hv2 = torch.tensor(request.hypervector2, dtype=backend.dtype, device=backend.device)
            
            if len(hv1) != len(hv2):
                raise HTTPException(
                    status_code=400,
                    detail="Hypervectors must have same dimension"
                )
            
            similarity = backend.cosine_similarity(hv1, hv2)
            
            return SimilarityResponse(similarity=float(similarity))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    app.include_router(hdc_router)
    
    # Experiments Router
    experiments_router = APIRouter(prefix="/experiments", tags=["Experiments"])
    
    @experiments_router.post("/", response_model=ExperimentResponse)
    async def create_experiment(
        experiment: ExperimentCreate,
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """Create a new experiment."""
        try:
            repo = ExperimentRepository(db)
            exp_model = ExperimentModel(
                name=experiment.name,
                description=experiment.description,
                config=experiment.config
            )
            exp_id = repo.create_experiment(exp_model)
            
            created_exp = repo.get_experiment(exp_id)
            return ExperimentResponse(
                id=created_exp.id,
                name=created_exp.name,
                description=created_exp.description,
                status=created_exp.status,
                config=created_exp.config,
                created_at=str(created_exp.created_at) if created_exp.created_at else None,
                updated_at=str(created_exp.updated_at) if created_exp.updated_at else None,
                completed_at=str(created_exp.completed_at) if created_exp.completed_at else None
            )
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @experiments_router.get("/{experiment_id}", response_model=ExperimentResponse)
    async def get_experiment(
        experiment_id: int = Path(..., gt=0),
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """Get experiment by ID."""
        try:
            repo = ExperimentRepository(db)
            experiment = repo.get_experiment(experiment_id)
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            return ExperimentResponse(
                id=experiment.id,
                name=experiment.name,
                description=experiment.description,
                status=experiment.status,
                config=experiment.config,
                created_at=str(experiment.created_at) if experiment.created_at else None,
                updated_at=str(experiment.updated_at) if experiment.updated_at else None,
                completed_at=str(experiment.completed_at) if experiment.completed_at else None
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting experiment: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @experiments_router.get("/", response_model=List[ExperimentResponse])
    async def list_experiments(
        status: Optional[str] = Query(None),
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """List experiments."""
        try:
            repo = ExperimentRepository(db)
            experiments = repo.get_experiments(status=status)
            
            return [
                ExperimentResponse(
                    id=exp.id,
                    name=exp.name,
                    description=exp.description,
                    status=exp.status,
                    config=exp.config,
                    created_at=str(exp.created_at) if exp.created_at else None,
                    updated_at=str(exp.updated_at) if exp.updated_at else None,
                    completed_at=str(exp.completed_at) if exp.completed_at else None
                )
                for exp in experiments
            ]
            
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @experiments_router.post("/{experiment_id}/metrics")
    async def log_metric(
        experiment_id: int = Path(..., gt=0),
        metric: MetricLog = Body(...),
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """Log a metric for an experiment."""
        try:
            # Verify experiment exists
            exp_repo = ExperimentRepository(db)
            experiment = exp_repo.get_experiment(experiment_id)
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # Log metric
            metrics_repo = MetricsRepository(db)
            metric_model = ModelMetricsModel(
                experiment_id=experiment_id,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                step=metric.step
            )
            metric_id = metrics_repo.log_metric(metric_model)
            
            return {"metric_id": metric_id, "status": "logged"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error logging metric: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    app.include_router(experiments_router)
    
    # Cache Router
    cache_router = APIRouter(prefix="/cache", tags=["Cache"])
    
    @cache_router.get("/stats", response_model=CacheStats)
    async def get_cache_stats(
        cache: CacheManager = Depends(get_cache_manager)
    ):
        """Get cache statistics."""
        try:
            hv_cache = HypervectorCache(cache)
            stats = hv_cache.get_cache_stats()
            
            return CacheStats(
                memory_cache_size=stats.get('memory_cache_size', 0),
                file_cache_size=stats.get('file_cache_size', 0),
                total_size_mb=stats.get('total_size_mb', 0.0),
                hit_rate=stats.get('hit_rate')
            )
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @cache_router.delete("/clear")
    async def clear_cache(
        namespace: Optional[str] = Query(None),
        cache: CacheManager = Depends(get_cache_manager)
    ):
        """Clear cache entries."""
        try:
            if namespace:
                cache.clear(namespace=namespace)
                return {"status": f"cleared namespace '{namespace}'"}
            else:
                cache.clear()
                return {"status": "cleared all cache"}
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    app.include_router(cache_router)
    
    # Benchmarks Router
    benchmarks_router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])
    
    @benchmarks_router.post("/results")
    async def save_benchmark_result(
        result: BenchmarkResult,
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """Save a benchmark result."""
        try:
            repo = BenchmarkRepository(db)
            result_model = BenchmarkResultModel(
                operation_name=result.operation_name,
                dimension=result.dimension,
                device=result.device,
                backend=result.backend,
                execution_time_ms=result.execution_time_ms,
                memory_usage_mb=result.memory_usage_mb,
                iterations=result.iterations
            )
            result_id = repo.save_benchmark_result(result_model)
            
            return {"result_id": result_id, "status": "saved"}
            
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @benchmarks_router.get("/results")
    async def get_benchmark_results(
        operation_name: Optional[str] = Query(None),
        backend: Optional[str] = Query(None),
        limit: int = Query(100, le=1000),
        db: DatabaseConnection = Depends(get_database_connection)
    ):
        """Get benchmark results."""
        try:
            repo = BenchmarkRepository(db)
            results = repo.get_benchmark_results(
                operation_name=operation_name,
                backend=backend
            )
            
            # Limit results
            results = results[:limit]
            
            return [
                {
                    "id": r.id,
                    "operation_name": r.operation_name,
                    "dimension": r.dimension,
                    "device": r.device,
                    "backend": r.backend,
                    "execution_time_ms": r.execution_time_ms,
                    "memory_usage_mb": r.memory_usage_mb,
                    "iterations": r.iterations,
                    "timestamp": str(r.timestamp) if r.timestamp else None
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting benchmark results: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    app.include_router(benchmarks_router)