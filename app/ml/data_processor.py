"""
Data processing utilities for ML models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta

from app.core.logging import setup_logging


logger = setup_logging(__name__)


class TechnicalIndicators:
    """Technical analysis indicators for feature engineering"""
    
    @staticmethod
    def add_sma(df: pd.DataFrame, windows: List[int] = [7, 21, 50]) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for window in windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators"""
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands"""
        bb = ta.volatility.BollingerBands(df['close'], window=window)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=window)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume Price Trend
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        return df
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_ma'] = df['price_change'].rolling(window=5).mean()
        
        # High-Low ratio
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close ratio
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volatility (standard deviation of returns)
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df


class DataProcessor:
    """Data processor for ML model preparation"""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize data processor
        
        Args:
            sequence_length: Length of sequences for LSTM models
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.feature_columns = []
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Preparing features for ML models", shape=df.shape)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Add technical indicators
        df = TechnicalIndicators.add_sma(df)
        df = TechnicalIndicators.add_ema(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_price_features(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        logger.info("Feature engineering completed", final_shape=df.shape)
        return df
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        target: np.ndarray,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Feature data
            target: Target values
            sequence_length: Length of sequences (uses self.sequence_length if None)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        seq_len = sequence_length or self.sequence_length
        
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML training
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Prepare features
        df_features = self.prepare_features(df.copy())
        
        # Define feature columns (exclude target and non-numeric columns)
        exclude_cols = [target_column, 'timestamp'] if 'timestamp' in df_features.columns else [target_column]
        self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        # Extract features and target
        X = df_features[self.feature_columns].values
        y = df_features[target_column].values
        
        # Handle infinite and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit scalers
        if not self.is_fitted:
            self.scaler.fit(X)
            self.price_scaler.fit(y.reshape(-1, 1))
            self.is_fitted = True
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        y_scaled = self.price_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_test = y_scaled[split_idx:]
        
        logger.info(
            "Data preparation completed",
            train_shape=X_train.shape,
            test_shape=X_test.shape,
            features=len(self.feature_columns)
        )
        
        return X_train, X_test, y_train, y_test, self.feature_columns
    
    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM training
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) with sequences
        """
        X_train, X_test, y_train, y_test, _ = self.prepare_data(df, target_column, test_size)
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        logger.info(
            "LSTM data preparation completed",
            train_sequences=X_train_seq.shape,
            test_sequences=X_test_seq.shape
        )
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call prepare_data first.")
        
        return self.price_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def get_latest_sequence(self, df: pd.DataFrame, target_column: str = 'close') -> np.ndarray:
        """
        Get the latest sequence for making predictions
        
        Args:
            df: DataFrame with recent data
            target_column: Name of target column
            
        Returns:
            Latest sequence for prediction
        """
        df_features = self.prepare_features(df.copy())
        
        if len(df_features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        # Get features
        X = df_features[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get latest sequence
        latest_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        return latest_sequence