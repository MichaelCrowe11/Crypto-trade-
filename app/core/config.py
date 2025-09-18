"""
Configuration management for the crypto trading application
"""

import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    """Exchange configuration"""
    name: str = "kraken"
    api_url: str = "https://api.kraken.com"
    websocket_url: str = "wss://ws.kraken.com"
    sandbox: bool = True
    rate_limit: float = 1.0


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    stop_loss: float = 0.02
    take_profit: float = 0.04
    max_open_positions: int = 3


class TradingConfig(BaseModel):
    """Trading configuration"""
    enabled: bool = False
    mode: str = "paper"
    base_currency: str = "USD"
    trading_pairs: List[str] = ["BTC/USD", "ETH/USD", "ADA/USD"]
    risk: RiskConfig = RiskConfig()


class MLConfig(BaseModel):
    """Machine learning configuration"""
    lstm_enabled: bool = True
    ensemble_enabled: bool = True
    retraining_frequency_hours: int = 24
    min_data_points: int = 1000


class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str = "sqlite:///crypto_trading.db"
    echo: bool = False
    pool_size: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/crypto_trading.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class Settings(BaseSettings):
    """Main application settings"""
    
    # API Keys from environment
    kraken_api_key: Optional[str] = Field(None, env="KRAKEN_API_KEY")
    kraken_secret_key: Optional[str] = Field(None, env="KRAKEN_SECRET_KEY")
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    
    # Application settings
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    paper_trading: bool = Field(True, env="PAPER_TRADING")
    
    # Configuration objects
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    ml: MLConfig = MLConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config() -> Settings:
    """Load configuration from YAML and environment variables"""
    config_path = Path("config.yaml")
    
    # Load base configuration from YAML
    config_data = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    
    # Create settings object (will also load from environment)
    settings = Settings()
    
    # Override with YAML config if available
    if config_data:
        # Update exchange config
        if "exchange" in config_data:
            settings.exchange = ExchangeConfig(**config_data["exchange"])
        
        # Update trading config
        if "trading" in config_data:
            trading_data = config_data["trading"]
            if "risk" in trading_data:
                risk_config = RiskConfig(**trading_data["risk"])
                trading_data["risk"] = risk_config
            settings.trading = TradingConfig(**trading_data)
        
        # Update ML config
        if "ml_models" in config_data:
            ml_data = config_data["ml_models"]
            settings.ml = MLConfig(
                lstm_enabled=ml_data.get("lstm", {}).get("enabled", True),
                ensemble_enabled=ml_data.get("ensemble", {}).get("enabled", True),
                retraining_frequency_hours=ml_data.get("retraining", {}).get("frequency_hours", 24),
                min_data_points=ml_data.get("retraining", {}).get("min_data_points", 1000)
            )
        
        # Update database config
        if "database" in config_data:
            settings.database = DatabaseConfig(**config_data["database"])
        
        # Update logging config
        if "logging" in config_data:
            settings.logging = LoggingConfig(**config_data["logging"])
    
    return settings


# Global settings instance
settings = load_config()