"""
Main application entry point for the AI Crypto Trading System
"""

import asyncio
import signal
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import api_router
from app.core.scheduler import TradingScheduler
from app.database.models import init_database
from app.monitoring.metrics import setup_metrics


logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting AI Crypto Trading Application")
    
    # Initialize database
    await init_database()
    
    # Setup monitoring
    setup_metrics()
    
    # Start trading scheduler
    scheduler = TradingScheduler()
    scheduler_task = asyncio.create_task(scheduler.start())
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        scheduler_task.cancel()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Crypto Trading Application")
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass


# Create FastAPI application
app = FastAPI(
    title="AI Crypto Trading System",
    description="Automated cryptocurrency trading with ML predictions and risk management",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Crypto Trading System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "trading_enabled": settings.trading.enabled,
        "mode": settings.trading.mode
    }


def main():
    """Main entry point"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()