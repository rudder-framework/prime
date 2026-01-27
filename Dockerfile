FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for server
RUN pip install --no-cache-dir fastapi uvicorn anthropic

# Copy application
COPY . .

# Railway injects PORT env var
ENV PORT=8000

# Run ORTHON server (serves static + API)
CMD python -m orthon.server
