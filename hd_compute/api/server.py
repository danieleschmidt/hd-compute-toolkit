"""FastAPI server for HD-Compute-Toolkit API."""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    Depends = None
    uvicorn = None

from ..utils.config import get_config
from ..utils.logging_config import setup_logging, get_logger
from ..database import DatabaseConnection, ExperimentRepository, MetricsRepository, BenchmarkRepository
from ..cache import CacheManager, HypervectorCache
from ..torch import HDComputeTorch
from .routes import setup_routes

logger = get_logger("api")


def get_database_connection() -> DatabaseConnection:
    """Dependency to get database connection."""
    config = get_config()
    db_path = config.get('database_url', 'sqlite:///./hdc_experiments.db')
    return DatabaseConnection(db_path)


def get_cache_manager() -> CacheManager:
    """Dependency to get cache manager."""
    config = get_config()
    cache_dir = config.get('cache_dir', '.cache/hdc')
    return CacheManager(cache_dir=cache_dir)


def get_hdc_backend() -> HDComputeTorch:
    """Dependency to get HDC backend."""
    config = get_config()
    device = config.get('default_device', 'cpu')
    dimension = config.get('default_dimension', 10000)
    return HDComputeTorch(dim=dimension, device=device)


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    # Setup configuration and logging
    hdc_config = get_config()
    if config:
        hdc_config.update(config)
    
    setup_logging(
        log_level=hdc_config.get('log_level', 'INFO'),
        log_file=hdc_config.get('log_file')
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="HD-Compute-Toolkit API",
        description="REST API for Hyperdimensional Computing operations",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Setup routes
    setup_routes(app)
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("HD-Compute-Toolkit API starting up")
        
        # Initialize database
        try:
            db = get_database_connection()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        
        # Initialize cache
        try:
            cache = get_cache_manager()
            logger.info("Cache manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("HD-Compute-Toolkit API shutting down")
        
        # Cleanup resources
        try:
            db = get_database_connection()
            db.close()
        except Exception as e:
            logger.warning(f"Error closing database: {e}")
    
    return app


class HDCAPIServer:
    """HD-Compute-Toolkit API server."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize API server.
        
        Args:
            config: Optional configuration dictionary
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        self.config = config or {}
        self.app = create_app(self.config)
        self.server = None
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
        **kwargs
    ):
        """Run the API server.
        
        Args:
            host: Host address to bind to
            port: Port to bind to
            reload: Enable auto-reload for development
            workers: Number of worker processes
            **kwargs: Additional uvicorn configuration
        """
        uvicorn_config = {
            "app": self.app,
            "host": host,
            "port": port,
            "reload": reload,
            "workers": workers if not reload else 1,  # Can't use workers with reload
            "log_level": "info",
            **kwargs
        }
        
        logger.info(f"Starting HD-Compute-Toolkit API server on {host}:{port}")
        uvicorn.run(**uvicorn_config)
    
    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start server asynchronously.
        
        Args:
            host: Host address to bind to
            port: Port to bind to
        """
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    async def stop(self):
        """Stop server asynchronously."""
        if self.server:
            await self.server.shutdown()
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# CLI interface for running the server
def main():
    """Main entry point for API server CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HD-Compute-Toolkit API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create and run server
    server = HDCAPIServer(config)
    server.run(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()