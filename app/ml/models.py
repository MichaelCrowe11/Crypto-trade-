"""
Machine learning models for cryptocurrency price prediction
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ML imports
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

from app.core.config import settings
from app.core.logging import setup_logging, log_ml_event
from app.ml.data_processor import DataProcessor


logger = setup_logging(__name__)


class BaseModel(ABC):
    """Base class for ML models"""
    
    def __init__(self, name: str):
        """
        Initialize base model
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.model_path = f"models/{name}_model.pkl"
        self.metrics = {}
    
    @abstractmethod
    def build_model(self, input_shape: Tuple) -> Any:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': float(mean_squared_error(y_test, predictions)),
            'mae': float(mean_absolute_error(y_test, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'r2': float(r2_score(y_test, predictions))
        }
        
        self.metrics = metrics
        log_ml_event(logger, "evaluate", self.name, metrics=metrics)
        
        return metrics
    
    def save_model(self) -> None:
        """Save trained model to disk"""
        os.makedirs("models", exist_ok=True)
        
        if hasattr(self.model, 'save') and callable(self.model.save):
            # TensorFlow/Keras model
            self.model.save(self.model_path.replace('.pkl', '.h5'))
        else:
            # Scikit-learn or XGBoost model
            joblib.dump(self.model, self.model_path)
        
        log_ml_event(logger, "save", self.name, path=self.model_path)
    
    def load_model(self) -> bool:
        """
        Load trained model from disk
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                log_ml_event(logger, "load", self.name, path=self.model_path)
                return True
            elif os.path.exists(self.model_path.replace('.pkl', '.h5')):
                # TensorFlow/Keras model
                self.model = keras.models.load_model(self.model_path.replace('.pkl', '.h5'))
                self.is_trained = True
                log_ml_event(logger, "load", self.name, path=self.model_path.replace('.pkl', '.h5'))
                return True
            else:
                logger.warning("Model file not found", model=self.name, path=self.model_path)
                return False
        except Exception as e:
            logger.error("Failed to load model", model=self.name, error=str(e))
            return False


class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            hidden_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: Tuple) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(
                self.hidden_units,
                return_sequences=True,
                input_shape=input_shape
            ),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.hidden_units, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(25, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history metrics
        """
        log_ml_event(logger, "train_start", self.name, 
                    train_shape=X_train.shape, epochs=epochs, batch_size=batch_size)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        
        # Extract final metrics
        final_metrics = {
            'loss': float(history.history['loss'][-1]),
            'mae': float(history.history['mae'][-1])
        }
        
        if validation_data:
            final_metrics.update({
                'val_loss': float(history.history['val_loss'][-1]),
                'val_mae': float(history.history['val_mae'][-1])
            })
        
        log_ml_event(logger, "train_complete", self.name, metrics=final_metrics)
        return final_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with LSTM model
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple regressors"""
    
    def __init__(self, models: List[str] = None):
        """
        Initialize ensemble model
        
        Args:
            models: List of model names to include in ensemble
        """
        super().__init__("ensemble")
        self.model_names = models or ["random_forest", "xgboost", "linear_regression"]
        self.base_models = {}
    
    def build_model(self, input_shape: Tuple) -> VotingRegressor:
        """
        Build ensemble model
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Voting regressor ensemble
        """
        estimators = []
        
        if "random_forest" in self.model_names:
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(("rf", rf))
            self.base_models["random_forest"] = rf
        
        if "xgboost" in self.model_names:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(("xgb", xgb_model))
            self.base_models["xgboost"] = xgb_model
        
        if "linear_regression" in self.model_names:
            lr = LinearRegression()
            estimators.append(("lr", lr))
            self.base_models["linear_regression"] = lr
        
        return VotingRegressor(estimators=estimators)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train ensemble model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Training metrics
        """
        log_ml_event(logger, "train_start", self.name, 
                    train_shape=X_train.shape, models=self.model_names)
        
        # Build model
        input_shape = (X_train.shape[1],)
        self.model = self.build_model(input_shape)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        metrics = {
            'train_mse': float(mean_squared_error(y_train, train_predictions)),
            'train_mae': float(mean_absolute_error(y_train, train_predictions)),
            'train_r2': float(r2_score(y_train, train_predictions))
        }
        
        log_ml_event(logger, "train_complete", self.name, metrics=metrics)
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with ensemble model
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models
        
        Returns:
            Dictionary of feature importance arrays
        """
        importance_dict = {}
        
        if "random_forest" in self.base_models and hasattr(self.base_models["random_forest"], 'feature_importances_'):
            importance_dict["random_forest"] = self.base_models["random_forest"].feature_importances_
        
        if "xgboost" in self.base_models and hasattr(self.base_models["xgboost"], 'feature_importances_'):
            importance_dict["xgboost"] = self.base_models["xgboost"].feature_importances_
        
        return importance_dict


class ModelManager:
    """Manager for multiple ML models"""
    
    def __init__(self):
        """Initialize model manager"""
        self.models: Dict[str, BaseModel] = {}
        self.data_processor = DataProcessor()
        self.last_training_time = None
    
    def register_model(self, model: BaseModel) -> None:
        """
        Register a model with the manager
        
        Args:
            model: Model instance to register
        """
        self.models[model.name] = model
        logger.info("Model registered", name=model.name)
    
    def initialize_models(self) -> None:
        """Initialize default models based on configuration"""
        if settings.ml.lstm_enabled:
            lstm_model = LSTMModel(
                sequence_length=60,
                hidden_units=50,
                dropout_rate=0.2
            )
            self.register_model(lstm_model)
        
        if settings.ml.ensemble_enabled:
            ensemble_model = EnsembleModel()
            self.register_model(ensemble_model)
    
    def train_models(self, df: pd.DataFrame, target_column: str = 'close') -> Dict[str, Dict[str, float]]:
        """
        Train all registered models
        
        Args:
            df: Training data
            target_column: Name of target column
            
        Returns:
            Dictionary of training metrics for each model
        """
        if len(df) < settings.ml.min_data_points:
            logger.warning("Insufficient data for training", 
                         required=settings.ml.min_data_points, 
                         available=len(df))
            return {}
        
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info("Training model", name=name)
                
                if isinstance(model, LSTMModel):
                    # Prepare LSTM data
                    X_train, X_test, y_train, y_test = self.data_processor.prepare_lstm_data(
                        df, target_column
                    )
                    
                    # Train with validation
                    split_idx = int(len(X_train) * 0.8)
                    X_val = X_train[split_idx:]
                    y_val = y_train[split_idx:]
                    X_train = X_train[:split_idx]
                    y_train = y_train[:split_idx]
                    
                    train_metrics = model.train(X_train, y_train, X_val, y_val)
                    eval_metrics = model.evaluate(X_test, y_test)
                    
                else:
                    # Prepare standard ML data
                    X_train, X_test, y_train, y_test, _ = self.data_processor.prepare_data(
                        df, target_column
                    )
                    
                    train_metrics = model.train(X_train, y_train)
                    eval_metrics = model.evaluate(X_test, y_test)
                
                # Combine metrics
                all_metrics = {**train_metrics, **eval_metrics}
                results[name] = all_metrics
                
                # Save model
                model.save_model()
                
            except Exception as e:
                logger.error("Model training failed", name=name, error=str(e))
                results[name] = {"error": str(e)}
        
        self.last_training_time = datetime.now()
        logger.info("Model training completed", results=list(results.keys()))
        
        return results
    
    def get_predictions(
        self,
        df: pd.DataFrame,
        target_column: str = 'close'
    ) -> Dict[str, float]:
        """
        Get predictions from all trained models
        
        Args:
            df: Recent data for prediction
            target_column: Name of target column
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        
        for name, model in self.models.items():
            if not model.is_trained:
                if not model.load_model():
                    logger.warning("Model not trained or loaded", name=name)
                    continue
            
            try:
                if isinstance(model, LSTMModel):
                    # Get latest sequence for LSTM
                    latest_sequence = self.data_processor.get_latest_sequence(df, target_column)
                    scaled_prediction = model.predict(latest_sequence)
                    prediction = self.data_processor.inverse_transform_predictions(scaled_prediction)[0]
                else:
                    # Get latest features for ensemble
                    df_features = self.data_processor.prepare_features(df.copy())
                    latest_features = df_features[self.data_processor.feature_columns].iloc[-1:].values
                    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=1e6, neginf=-1e6)
                    scaled_features = self.data_processor.scaler.transform(latest_features)
                    scaled_prediction = model.predict(scaled_features)
                    prediction = self.data_processor.inverse_transform_predictions(scaled_prediction)[0]
                
                predictions[name] = float(prediction)
                
            except Exception as e:
                logger.error("Prediction failed", name=name, error=str(e))
        
        return predictions
    
    def should_retrain(self) -> bool:
        """
        Check if models should be retrained based on configuration
        
        Returns:
            True if retraining is needed
        """
        if self.last_training_time is None:
            return True
        
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training > timedelta(hours=settings.ml.retraining_frequency_hours)
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all models
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        
        for name, model in self.models.items():
            status[name] = {
                "trained": model.is_trained,
                "metrics": model.metrics,
                "last_training": self.last_training_time.isoformat() if self.last_training_time else None
            }
        
        return status