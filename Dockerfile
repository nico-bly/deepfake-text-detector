# Multi-stage Dockerfile for deepfake text detection backend
# Optimized for Coolify deployment with GPU support

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ /app/models/
COPY utils/ /app/utils/
COPY api/ /app/api/

# Copy saved models (if they exist locally)
COPY saved_models/ /app/saved_models/ 2>/dev/null || mkdir -p /app/saved_models

# Create necessary directories
RUN mkdir -p /app/saved_models /app/logs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.app_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
