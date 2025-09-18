"""
Prometheus metrics for monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import setup_logging


logger = setup_logging(__name__)


# Trading metrics
trades_total = Counter('crypto_trades_total', 'Total number of trades executed', ['symbol', 'side', 'status'])
trade_pnl = Histogram('crypto_trade_pnl', 'Trade P&L distribution', ['symbol'])
portfolio_value = Gauge('crypto_portfolio_value_usd', 'Total portfolio value in USD')
position_count = Gauge('crypto_open_positions', 'Number of open positions')
daily_pnl = Gauge('crypto_daily_pnl_usd', 'Daily P&L in USD')

# ML model metrics
model_predictions = Counter('crypto_ml_predictions_total', 'ML model predictions made', ['model', 'symbol'])
model_accuracy = Gauge('crypto_ml_model_accuracy', 'Model prediction accuracy', ['model'])
model_training_duration = Histogram('crypto_ml_training_duration_seconds', 'Model training time', ['model'])

# Risk metrics
risk_events = Counter('crypto_risk_events_total', 'Risk management events', ['event_type', 'severity'])
position_size_ratio = Gauge('crypto_position_size_ratio', 'Current position size ratio', ['symbol'])
daily_loss_ratio = Gauge('crypto_daily_loss_ratio', 'Daily loss as ratio of portfolio')

# Exchange metrics
api_requests = Counter('crypto_exchange_requests_total', 'Exchange API requests', ['endpoint', 'status'])
api_response_time = Histogram('crypto_exchange_response_time_seconds', 'Exchange API response time', ['endpoint'])
websocket_messages = Counter('crypto_websocket_messages_total', 'WebSocket messages received', ['channel'])

# System metrics
application_info = Gauge('crypto_application_info', 'Application information')
uptime_seconds = Gauge('crypto_uptime_seconds', 'Application uptime in seconds')


class MetricsCollector:
    """Metrics collection utility"""
    
    @staticmethod
    def record_trade(symbol: str, side: str, status: str, pnl: float = None):
        """Record trade execution metrics"""
        trades_total.labels(symbol=symbol, side=side, status=status).inc()
        
        if pnl is not None:
            trade_pnl.labels(symbol=symbol).observe(pnl)
    
    @staticmethod
    def update_portfolio_metrics(total_value: float, position_count_val: int, daily_pnl_val: float):
        """Update portfolio-related metrics"""
        portfolio_value.set(total_value)
        position_count.set(position_count_val)
        daily_pnl.set(daily_pnl_val)
    
    @staticmethod
    def record_prediction(model: str, symbol: str):
        """Record ML prediction"""
        model_predictions.labels(model=model, symbol=symbol).inc()
    
    @staticmethod
    def update_model_accuracy(model: str, accuracy: float):
        """Update model accuracy metric"""
        model_accuracy.labels(model=model).set(accuracy)
    
    @staticmethod
    def record_training_time(model: str, duration: float):
        """Record model training duration"""
        model_training_duration.labels(model=model).observe(duration)
    
    @staticmethod
    def record_risk_event(event_type: str, severity: str):
        """Record risk management event"""
        risk_events.labels(event_type=event_type, severity=severity).inc()
    
    @staticmethod
    def update_risk_metrics(position_ratios: Dict[str, float], daily_loss_ratio_val: float):
        """Update risk-related metrics"""
        for symbol, ratio in position_ratios.items():
            position_size_ratio.labels(symbol=symbol).set(ratio)
        
        daily_loss_ratio.set(daily_loss_ratio_val)
    
    @staticmethod
    def record_api_request(endpoint: str, status: str, response_time: float):
        """Record API request metrics"""
        api_requests.labels(endpoint=endpoint, status=status).inc()
        api_response_time.labels(endpoint=endpoint).observe(response_time)
    
    @staticmethod
    def record_websocket_message(channel: str):
        """Record WebSocket message"""
        websocket_messages.labels(channel=channel).inc()


def setup_metrics():
    """Setup Prometheus metrics server"""
    try:
        if settings.monitoring.enabled:
            port = settings.monitoring.prometheus_port
            start_http_server(port)
            logger.info("Prometheus metrics server started", port=port)
            
            # Set application info
            application_info.set(1)
    
    except Exception as e:
        logger.error("Failed to start metrics server", error=str(e))