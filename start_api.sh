#!/bin/bash

# Startup script for deepfake text detection API

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Deepfake Text Detection API                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "api/app.py" ]; then
    echo -e "${RED}Error: api/app.py not found!${NC}"
    echo "Please run this script from the deepfake-text-detector directory"
    exit 1
fi

# Check if saved_models directory exists
if [ ! -d "saved_models" ]; then
    echo -e "${YELLOW}Warning: saved_models/ directory not found!${NC}"
    echo "Creating directory..."
    mkdir -p saved_models
fi

# Check for trained models
MODEL_COUNT=$(ls -1 saved_models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: No trained models found in saved_models/${NC}"
    echo ""
    echo "You should train at least one model first:"
    echo "  python scripts/train_and_save_detector.py"
    echo ""
    echo -n "Continue anyway? (y/N) "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 0
    fi
else
    echo -e "${GREEN}✓${NC} Found $MODEL_COUNT trained model(s)"
fi

# Check Python dependencies
echo -e "\nChecking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}FastAPI not installed. Installing dependencies...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓${NC} Dependencies OK"

# Check GPU availability
echo -e "\nChecking GPU availability..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME"
else
    echo -e "${YELLOW}⚠${NC}  No GPU detected - will use CPU (slower)"
fi

# Get configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-true}"

echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Auto-reload: $RELOAD"
echo ""

# Start the server
echo -e "${GREEN}🚀 Starting API server...${NC}"
echo ""
echo "API will be available at:"
echo "  - Local:   http://localhost:$PORT"
echo "  - Network: http://$(hostname -I | awk '{print $1}'):$PORT"
echo "  - Docs:    http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Build uvicorn command
UVICORN_CMD="uvicorn api.app:app --host $HOST --port $PORT --workers $WORKERS"

if [ "$RELOAD" = "true" ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

# Run the server
$UVICORN_CMD
