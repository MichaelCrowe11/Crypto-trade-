"""
Basic tests for the crypto trading application
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.core.config import settings
from app.ml.data_processor import DataProcessor, TechnicalIndicators
from app.core.trading_engine import RiskManager, SignalType, TradingSignal


class TestConfiguration:
    """Test configuration loading"""
    
    def test_settings_loaded(self):
        """Test that settings are loaded correctly"""
        assert settings.trading.mode in ["paper", "live"]
        assert isinstance(settings.trading.trading_pairs, list)
        assert len(settings.trading.trading_pairs) > 0


class TestTechnicalIndicators:
    """Test technical indicators"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        
        # Generate realistic price data
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.rand(100) * 200,
            'low': prices - np.random.rand(100) * 200,
            'close': prices + np.random.randn(100) * 50,
            'volume': np.random.rand(100) * 1000 + 100
        })
        
        # Ensure high >= close >= low and high >= open >= low
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
        
        return df
    
    def test_sma_calculation(self, sample_data):
        """Test Simple Moving Average calculation"""
        result = TechnicalIndicators.add_sma(sample_data.copy())
        
        assert 'sma_7' in result.columns
        assert 'sma_21' in result.columns
        assert 'sma_50' in result.columns
        
        # Check that SMA values are reasonable
        assert not result['sma_7'].isna().all()
        assert result['sma_7'].iloc[-1] > 0
    
    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation"""
        result = TechnicalIndicators.add_rsi(sample_data.copy())
        
        assert 'rsi' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)
    
    def test_macd_calculation(self, sample_data):
        """Test MACD calculation"""
        result = TechnicalIndicators.add_macd(sample_data.copy())
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns


class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create data processor instance"""
        return DataProcessor(sequence_length=20)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        
        prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.rand(200) * 200,
            'low': prices - np.random.rand(200) * 200,
            'close': prices + np.random.randn(200) * 50,
            'volume': np.random.rand(200) * 1000 + 100
        })
        
        # Ensure price constraints
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
        
        return df
    
    def test_feature_preparation(self, processor, sample_data):
        """Test feature preparation"""
        result = processor.prepare_features(sample_data)
        
        # Should have more columns than original (technical indicators added)
        assert len(result.columns) > len(sample_data.columns)
        
        # Should have technical indicators
        assert any('sma' in col for col in result.columns)
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
    
    def test_data_preparation(self, processor, sample_data):
        """Test data preparation for ML"""
        X_train, X_test, y_train, y_test, features = processor.prepare_data(sample_data)
        
        # Check shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check feature names
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_sequence_creation(self, processor):
        """Test sequence creation for LSTM"""
        data = np.random.randn(100, 10)
        target = np.random.randn(100)
        
        X_seq, y_seq = processor.create_sequences(data, target, sequence_length=20)
        
        assert X_seq.shape[0] == 80  # 100 - 20
        assert X_seq.shape[1] == 20  # sequence length
        assert X_seq.shape[2] == 10  # features
        assert len(y_seq) == 80


class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        return RiskManager()
    
    def test_position_size_check(self, risk_manager):
        """Test position size limits"""
        portfolio_value = 100000
        
        # Within limits
        assert risk_manager.check_position_size("BTC/USD", 5000, portfolio_value)
        
        # Exceeds limits
        assert not risk_manager.check_position_size("BTC/USD", 15000, portfolio_value)
    
    def test_daily_loss_check(self, risk_manager):
        """Test daily loss limits"""
        portfolio_value = 100000
        
        # Small loss - within limits
        risk_manager.daily_pnl = -1000
        assert risk_manager.check_daily_loss_limit(portfolio_value)
        
        # Large loss - exceeds limits
        risk_manager.daily_pnl = -6000
        assert not risk_manager.check_daily_loss_limit(portfolio_value)
    
    def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        signal = TradingSignal(
            symbol="BTC/USD",
            signal=SignalType.BUY,
            confidence=0.8,
            price_prediction=55000,
            current_price=50000
        )
        
        portfolio_value = 100000
        current_price = 50000
        
        position_size = risk_manager.calculate_position_size(signal, portfolio_value, current_price)
        
        assert position_size > 0
        assert position_size * current_price <= portfolio_value * settings.trading.risk.max_position_size


class TestTradingSignal:
    """Test trading signal functionality"""
    
    def test_signal_creation(self):
        """Test trading signal creation"""
        signal = TradingSignal(
            symbol="BTC/USD",
            signal=SignalType.BUY,
            confidence=0.75,
            price_prediction=55000,
            current_price=50000,
            reasoning={"test": "data"}
        )
        
        assert signal.symbol == "BTC/USD"
        assert signal.signal == SignalType.BUY
        assert signal.confidence == 0.75
        assert signal.price_prediction == 55000
        assert signal.current_price == 50000
        assert "test" in signal.reasoning
        assert signal.timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__])