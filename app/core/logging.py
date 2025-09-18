"""
Structured logging configuration for the crypto trading application
"""

import logging
import logging.handlers
import structlog
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings


def setup_logging(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Setup structured logging for the application
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    
    # Ensure logs directory exists
    log_path = Path(settings.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper()),
        format=settings.logging.format,
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                settings.logging.file,
                maxBytes=settings.logging.max_bytes,
                backupCount=settings.logging.backup_count
            )
        ]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name or __name__)


def log_trade_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    symbol: str,
    **kwargs: Any
) -> None:
    """
    Log trading events with consistent structure
    
    Args:
        logger: Structured logger instance
        event_type: Type of trading event (buy, sell, stop_loss, etc.)
        symbol: Trading pair symbol
        **kwargs: Additional event data
    """
    logger.info(
        "Trading event",
        event_type=event_type,
        symbol=symbol,
        **kwargs
    )


def log_ml_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    model_name: str,
    **kwargs: Any
) -> None:
    """
    Log ML model events with consistent structure
    
    Args:
        logger: Structured logger instance
        event_type: Type of ML event (train, predict, evaluate, etc.)
        model_name: Name of the ML model
        **kwargs: Additional event data
    """
    logger.info(
        "ML event",
        event_type=event_type,
        model_name=model_name,
        **kwargs
    )


def log_error_with_context(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    context: Dict[str, Any] = None
) -> None:
    """
    Log errors with additional context
    
    Args:
        logger: Structured logger instance
        error: Exception that occurred
        context: Additional context data
    """
    logger.error(
        "Application error",
        error_type=type(error).__name__,
        error_message=str(error),
        **(context or {})
    )