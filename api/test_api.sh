#!/bin/bash

# Quick test script for API deployment

set -e

echo "🚀 Starting API Backend Test..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8010}"

echo "Testing API at: $API_URL"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}[1/4]${NC} Testing health endpoint..."
if curl -f -s "$API_URL/health" > /dev/null; then
    echo -e "${GREEN}✓${NC} Health check passed"
else
    echo -e "${RED}✗${NC} Health check failed"
    exit 1
fi

# Test 2: Root endpoint
echo -e "${YELLOW}[2/4]${NC} Testing root endpoint..."
if curl -f -s "$API_URL/" > /dev/null; then
    echo -e "${GREEN}✓${NC} Root endpoint passed"
else
    echo -e "${RED}✗${NC} Root endpoint failed"
    exit 1
fi

# Test 3: Models endpoint
echo -e "${YELLOW}[3/4]${NC} Testing models endpoint..."
if curl -f -s "$API_URL/models" > /dev/null; then
    echo -e "${GREEN}✓${NC} Models endpoint passed"
else
    echo -e "${RED}✗${NC} Models endpoint failed"
    exit 1
fi

# Test 4: Detection endpoint
echo -e "${YELLOW}[4/4]${NC} Testing detection endpoint..."
RESPONSE=$(curl -s -X POST "$API_URL/detect" \
    -H "Content-Type: application/json" \
    -d '{"text":"This is a test sentence."}')

if echo "$RESPONSE" | grep -q "prediction"; then
    echo -e "${GREEN}✓${NC} Detection endpoint passed"
    echo ""
    echo "Response preview:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null | head -n 10
else
    echo -e "${RED}✗${NC} Detection endpoint failed"
    echo "Response: $RESPONSE"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 All tests passed!${NC}"
echo ""
echo "API is ready for deployment 🚀"
