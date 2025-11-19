# 📊 PSX AI Trading System

Fully automated AI-powered stock prediction and trading system for Pakistan Stock Exchange (PSX).

## 🎯 Features

- ✅ **Automated Data Collection** - Daily updates from Sarmaaya API
- ✅ **AI Predictions** - LSTM + XGBoost ensemble models
- ✅ **Live Dashboard** - Real-time Streamlit interface
- ✅ **Auto-Training** - Monthly model retraining
- ✅ **Historical Data** - Preserves all past data
- ✅ **GitHub Auto-Sync** - Automatic commits daily
- ✅ **Zero Maintenance** - Runs completely automated

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/psx-predictor.git
cd psx-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Daily Usage
```bash
# Update prices and predictions (30 seconds)
python automated_system.py

# Launch dashboard
streamlit run dashboard/app.py
```

### First-Time Setup
```bash
# Train models (takes 2-20 hours depending on stock count)
python train_all.py --max 50  # Train top 50 stocks
```

## 📁 Project Structure
```
psx-predictor/
├── data/
│   ├── raw/historical/      # Growing historical price data
│   └── raw/snapshots/       # Daily backups
├── models/
│   ├── saved/               # Current trained models
│   └── versions/            # Model version history
├── reports/
│   ├── trading_signals.csv  # Latest signals
│   └── history/             # Signal history
├── dashboard/
│   └── app.py              # Streamlit dashboard
└── automated_system.py      # Main automation script
```

## 🤖 Automation

The system runs automatically via GitHub Actions:
- **Daily at 8:30 AM** - Updates data and predictions
- **Monthly** - Retrains models
- **Always** - Preserves historical data

## 📊 Dashboard

View live predictions at: `http://localhost:8501`

Or deploy to Streamlit Cloud for 24/7 access.

## 🛠️ Technologies

- **Python 3.11**
- **TensorFlow** - LSTM neural networks
- **XGBoost** - Direction classification
- **Streamlit** - Interactive dashboard
- **Sarmaaya API** - Live PSX data
- **GitHub Actions** - Automation

## 📈 Performance

- **409 stocks** tracked
- **4+ years** historical data
- **~50% accuracy** on direction prediction
- **Updates daily** automatically

## 🔒 Data Safety

All data is preserved:
- Daily snapshots in `data/raw/snapshots/`
- Prediction history in `reports/history/`
- Model versions in `models/versions/`

**Nothing is ever deleted or overwritten!**

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Please open an issue first.

## 📧 Contact

Created by [Your Name] - [Your Email]

---

⭐ **Star this repo if you find it useful!**