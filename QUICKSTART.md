# 🚀 Quick Start Guide

## Instant Setup (5 minutes)

### 1. Prerequisites
- Docker & Docker Compose installed
- (Optional) Kraken API credentials for live data

### 2. Clone & Setup
```bash
git clone https://github.com/MichaelCrowe11/Crypto-trade-.git
cd Crypto-trade-
./setup.sh
```

### 3. Start the Application
```bash
docker-compose up -d
```

### 4. Access the System
- **API Interface**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)  
- **Prometheus Metrics**: http://localhost:9090

## 🎯 Key Endpoints

### System Status
```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/status
```

### Portfolio & Trading
```bash
curl http://localhost:8080/api/v1/portfolio
curl http://localhost:8080/api/v1/positions
curl http://localhost:8080/api/v1/trades
```

### ML Predictions
```bash
curl http://localhost:8080/api/v1/predictions/BTCUSD
curl http://localhost:8080/api/v1/models/status
```

### Market Data
```bash
curl http://localhost:8080/api/v1/market-data/BTCUSD?limit=100
```

## ⚙️ Configuration

### Basic Settings (config.yaml)
```yaml
trading:
  enabled: false      # SAFETY: Keep false until ready
  mode: "paper"       # paper/live  
  trading_pairs:
    - "BTC/USD"
    - "ETH/USD"
```

### API Keys (.env)
```bash
# Optional - only needed for live trading
KRAKEN_API_KEY=your_api_key
KRAKEN_SECRET_KEY=your_secret_key
```

## 🛡️ Safety Features

✅ **Trading DISABLED by default**  
✅ **Paper trading mode enabled**  
✅ **Position size limits (10% max)**  
✅ **Daily loss limits (5% max)**  
✅ **Stop-loss protection (2%)**  
✅ **Comprehensive audit logging**  

## 🧪 Testing Flow

1. **Start with paper trading** (default)
2. **Monitor ML predictions** at `/api/v1/predictions/{symbol}`
3. **Check portfolio tracking** at `/api/v1/portfolio`
4. **Review risk events** at `/api/v1/risk-events`
5. **Only enable live trading** after thorough validation

## 📊 Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:8080/health

# System metrics  
curl http://localhost:8080/api/v1/performance
```

### Logs
```bash
# View application logs
docker-compose logs -f crypto-trading-app

# View all logs
docker-compose logs -f
```

## 🔧 Development

### Local Python Development
```bash
pip install -r requirements.txt
python -m app.main
```

### Run Tests
```bash
pytest tests/
```

### Stop Services
```bash
docker-compose down
```

## 🆘 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Database locked"**
```bash
docker-compose down
docker-compose up -d
```

**"API connection failed"**
- Check .env file has correct credentials
- Verify Kraken API key permissions

### Getting Help
- 📚 [Full Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/MichaelCrowe11/Crypto-trade-/issues)
- 💬 [Discussions](https://github.com/MichaelCrowe11/Crypto-trade-/discussions)

---

**⭐ Star the repo if this helps you!**