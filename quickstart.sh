#!/bin/bash
# Quick Start Script for PSX Stock Predictor

echo "======================================"
echo "PSX Stock Predictor - Quick Start"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
python -c "from src.utils import create_directories; create_directories()"
echo "✓ Directories created"
echo ""

echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train a model:"
echo "   python train.py --symbol HBL --days 365"
echo ""
echo "2. Or train on KSE-100:"
echo "   python train.py --kse100 --days 365"
echo ""
echo "3. Start the API server:"
echo "   python src/api/api_server.py"
echo ""
echo "4. Launch the dashboard:"
echo "   streamlit run dashboard/app.py"
echo ""
echo "For more information, see README.md"
echo ""
