# Vyuha AI - Enterprise AI Model Optimization Platform
# Multi-stage Docker build for production deployment

# Stage 1: Base Python environment
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
RUN chown -R app:app /app

# Stage 2: Development environment
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Install package in development mode
RUN pip install -e .

# Switch to app user
USER app

# Expose port for potential web interface
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "vyuha.cli", "--help"]

# Stage 3: Production environment
FROM base as production

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Install package
RUN pip install -e .

# Switch to app user
USER app

# Create directories for model storage
RUN mkdir -p /app/models /app/output

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import vyuha; print('Vyuha AI is healthy')" || exit 1

# Default command
CMD ["python", "-m", "vyuha.cli", "--help"]
