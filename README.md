# AI Automated Crypto Trading System

A production-grade cryptocurrency trading application that combines machine learning predictions, automated trading, and comprehensive risk management with compliance-friendly guardrails.

## 🚀 Features

### Core Functionality
- **Multi-Exchange Support**: Primary integration with Kraken API (easily extensible)
- **AI-Powered Predictions**: LSTM neural networks and ensemble ML models
- **Automated Trading**: Real-time signal generation and order execution
- **Risk Management**: Position sizing, stop-loss, take-profit, and daily loss limits
- **Real-time Monitoring**: WebSocket data feeds and portfolio tracking

### Machine Learning
- **LSTM Models**: Time series prediction with technical indicators
- **Ensemble Methods**: Random Forest, XGBoost, and Linear Regression
- **Feature Engineering**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Automated Retraining**: Scheduled model updates with new data

### Risk & Compliance
- **Position Limits**: Maximum position size and open positions controls
- **Daily Loss Limits**: Automatic trading halt on excessive losses
- **Audit Trail**: Comprehensive logging for compliance requirements
- **Paper Trading**: Safe testing mode before live trading

### Monitoring & Observability
- **Prometheus Metrics**: Comprehensive performance and system metrics
- **Grafana Dashboards**: Real-time visualization and alerting
- **Structured Logging**: JSON logs with trade and ML event tracking
- **Health Checks**: API endpoints for system status monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI Web   │    │  Trading Engine │    │   ML Models     │
│   Interface     │───▶│  & Scheduler    │───▶│   (LSTM +       │
│                 │    │                 │    │    Ensemble)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        │
         │              ┌─────────────────┐                │
         │              │  Risk Manager   │                │
         │              │  & Compliance   │                │
         │              └─────────────────┘                │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │  Kraken API     │    │  Monitoring     │
│   (SQLite)      │    │  Integration    │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚦 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (recommended)
- Kraken API credentials (for live trading)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MichaelCrowe11/Crypto-trade-.git
   cd Crypto-trade-
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Run with Docker (Recommended)**
   ```bash
   docker-compose up -d
   ```

4. **Or run locally**
   ```bash
   pip install -r requirements.txt
   python -m app.main
   ```

### Configuration

Edit `config.yaml` to customize:
- Trading pairs and parameters
- Risk management settings  
- ML model configuration
- Monitoring preferences

**⚠️ IMPORTANT**: Trading is disabled by default. Set `trading.enabled: true` in config.yaml only after thorough testing.

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Application status
- `GET /health` - Health check
- `GET /api/v1/status` - Trading system status
- `GET /api/v1/portfolio` - Portfolio information
- `GET /api/v1/positions` - Current positions
- `GET /api/v1/trades` - Trade history

### ML & Predictions
- `GET /api/v1/predictions/{symbol}` - Get ML predictions
- `GET /api/v1/models/status` - Model status and metrics
- `POST /api/v1/models/retrain` - Trigger model retraining

### Market Data
- `GET /api/v1/market-data/{symbol}` - Historical OHLCV data
- `GET /api/v1/performance` - Performance metrics

### Risk & Monitoring
- `GET /api/v1/risk-events` - Risk management events
- `GET /api/v1/config` - Current configuration

## 🛡️ Safety Features

### Built-in Safeguards
- **Paper Trading Mode**: Default safe testing environment
- **Position Size Limits**: Maximum 10% of portfolio per position
- **Daily Loss Limits**: Automatic halt at 5% daily loss
- **Stop Loss**: Automatic 2% stop loss on all positions
- **Trade Frequency Limits**: Maximum trades per day
- **API Rate Limiting**: Respect exchange limits

### Risk Management
- Real-time position monitoring
- Automated stop-loss and take-profit
- Portfolio-level risk metrics
- Compliance audit logging

## 📈 Monitoring

Access monitoring dashboards:
- **API**: http://localhost:8080
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000 (admin/admin123)

### Key Metrics
- Portfolio value and P&L
- Trade execution statistics
- ML model performance
- Risk event tracking
- System health metrics

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Integration tests
pytest tests/integration/
```

## 🔧 Development

### Project Structure
```
app/
├── api/          # FastAPI routes and endpoints
├── core/         # Core business logic and configuration
├── exchange/     # Exchange API integrations (Kraken)
├── ml/           # Machine learning models and data processing
├── database/     # Database models and operations
├── monitoring/   # Metrics and observability
└── utils/        # Utility functions
```

### Adding New Exchanges
1. Implement the exchange client in `app/exchange/`
2. Follow the `KrakenClient` interface pattern
3. Add configuration in `config.yaml`
4. Update the trading engine to support the new exchange

### Custom ML Models
1. Inherit from `BaseModel` in `app/ml/models.py`
2. Implement required methods: `build_model`, `train`, `predict`
3. Register with `ModelManager`
4. Configure in `config.yaml`

## 📋 Configuration Reference

### Trading Settings
```yaml
trading:
  enabled: false          # Enable/disable trading
  mode: "paper"          # paper/live
  trading_pairs:         # Supported pairs
    - "BTC/USD"
    - "ETH/USD"
  risk:
    max_position_size: 0.1    # 10% max position
    max_daily_loss: 0.05      # 5% daily loss limit
    stop_loss: 0.02           # 2% stop loss
    take_profit: 0.04         # 4% take profit
```

### ML Configuration
```yaml
ml_models:
  lstm:
    enabled: true
    sequence_length: 60
    hidden_units: 50
    epochs: 100
  ensemble:
    enabled: true
    models: ["random_forest", "xgboost", "linear_regression"]
```

## 🚨 Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always test thoroughly in paper trading mode before considering live trading.**

- Past performance does not guarantee future results
- ML predictions are not investment advice
- Use at your own risk and responsibility
- Comply with your local financial regulations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🆘 Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/MichaelCrowe11/Crypto-trade-/issues)
- 💬 [Discussions](https://github.com/MichaelCrowe11/Crypto-trade-/discussions)

---

**⭐ If you find this project useful, please consider giving it a star!**