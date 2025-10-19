#!/bin/bash

# Script to deploy API to VPS
# Usage: ./deploy_to_vps.sh [user@host:/path]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Deploy Deepfake API to VPS                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "api/app.py" ]; then
    echo -e "${RED}Error: api/app.py not found!${NC}"
    echo "Please run this script from the deepfake-text-detector directory"
    exit 1
fi

# Get VPS destination
VPS_DESTINATION="${1}"

if [ -z "$VPS_DESTINATION" ]; then
    echo "Usage: $0 user@host:/path/to/backend"
    echo ""
    echo "Example:"
    echo "  $0 myuser@123.456.78.90:/home/myuser/deepfake-backend"
    echo ""
    echo "Or set VPS_DESTINATION environment variable:"
    echo "  export VPS_DESTINATION=user@host:/path"
    echo "  $0"
    exit 1
fi

echo -e "Destination: ${GREEN}$VPS_DESTINATION${NC}"
echo ""

# Confirm
echo -n "Deploy to this destination? (y/N) "
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check for trained models
MODEL_COUNT=$(ls -1 saved_models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: No trained models found!${NC}"
    echo ""
    echo "You should train at least one model first:"
    echo "  python scripts/train_and_save_detector.py"
    echo ""
    echo -n "Continue anyway? (y/N) "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo -e "${GREEN}✓${NC} Found $MODEL_COUNT trained model(s)"
    echo ""
    echo "Models to deploy:"
    ls -lh saved_models/*.pkl | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
fi

# What to deploy
echo "What would you like to deploy?"
echo "  1) Everything (first time setup)"
echo "  2) Only models (quick update)"
echo "  3) Only API code (code changes)"
echo "  4) Custom (select what to sync)"
echo ""
echo -n "Choice (1-4): "
read -r choice

case $choice in
    1)
        echo -e "\n${YELLOW}Deploying everything...${NC}"
        rsync -avz --progress \
            --exclude 'data/' \
            --exclude 'archive/' \
            --exclude 'scripts/' \
            --exclude '.git/' \
            --exclude '__pycache__/' \
            --exclude '*.ipynb' \
            --exclude 'results/' \
            --exclude 'tests/' \
            . "$VPS_DESTINATION"
        ;;
    2)
        echo -e "\n${YELLOW}Deploying models only...${NC}"
        rsync -avz --progress \
            saved_models/ \
            "$VPS_DESTINATION/saved_models/"
        ;;
    3)
        echo -e "\n${YELLOW}Deploying API code...${NC}"
        rsync -avz --progress \
            api/ models/ utils/ \
            "$VPS_DESTINATION/"
        ;;
    4)
        echo -e "\n${YELLOW}Custom deployment...${NC}"
        echo "Deploying: api/, models/, utils/, requirements.txt, Dockerfile"
        rsync -avz --progress \
            --include='api/***' \
            --include='models/***' \
            --include='utils/***' \
            --include='saved_models/***' \
            --include='Dockerfile' \
            --include='docker-compose.yml' \
            --include='requirements.txt' \
            --include='start_api.sh' \
            --include='.env.example' \
            --exclude='*' \
            . "$VPS_DESTINATION"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. SSH into your VPS:"
echo "     ssh ${VPS_DESTINATION%%:*}"
echo ""
echo "  2. Navigate to the directory:"
echo "     cd ${VPS_DESTINATION##*:}"
echo ""
echo "  3. Check files were copied:"
echo "     ls -la"
echo "     ls -la saved_models/"
echo ""
echo "  4. Test locally on VPS (optional):"
echo "     ./start_api.sh"
echo ""
echo "  5. Deploy in Coolify:"
echo "     - Point to the directory: ${VPS_DESTINATION##*:}"
echo "     - Add volume: ./saved_models:/app/saved_models"
echo "     - Click Deploy!"
echo ""
echo -e "${GREEN}🚀 Ready for Coolify deployment!${NC}"
