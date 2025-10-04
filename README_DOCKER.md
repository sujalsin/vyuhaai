# Vyuha AI - Docker Setup Guide

This guide explains how to run Vyuha AI using Docker for easy deployment and development.

## 🐳 Quick Start

### Prerequisites
- Docker (20.10+)
- Docker Compose (2.0+)

### 1. Build and Run
```bash
# Build the Docker image
./docker-run.sh build

# Run optimization
./docker-run.sh run --model microsoft/DialoGPT-medium

# Start development environment
./docker-run.sh dev
```

### 2. Test Everything Works
```bash
# Run comprehensive tests
./docker-test.sh
```

## 🚀 Available Commands

### Docker Run Script (`./docker-run.sh`)
```bash
# Build Docker image
./docker-run.sh build

# Run optimization pipeline
./docker-run.sh run --model <model> --task <task> --output <dir>

# Start development environment
./docker-run.sh dev

# Run tests
./docker-run.sh test

# Start Jupyter notebook
./docker-run.sh jupyter

# Clean up resources
./docker-run.sh clean

# Show logs
./docker-run.sh logs

# Open shell in container
./docker-run.sh shell
```

### Docker Compose Commands
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d vyuha-ai

# View logs
docker-compose logs -f vyuha-ai

# Stop services
docker-compose down

# Clean up everything
docker-compose down -v
```

## 🏗️ Docker Architecture

### Multi-Stage Build
- **Base**: Python 3.9-slim with system dependencies
- **Development**: Includes dev tools (vim, htop, jupyter)
- **Production**: Optimized for deployment

### Services
- **vyuha-ai**: Main optimization platform
- **vyuha-dev**: Development environment
- **jupyter**: Jupyter notebook for experimentation
- **redis**: Caching (optional)
- **postgres**: Metadata storage (optional)

## 📁 Directory Structure

```
vyuha/
├── Dockerfile              # Multi-stage production build
├── Dockerfile.dev          # Development-optimized build
├── docker-compose.yml      # Service orchestration
├── docker-run.sh          # Easy runner script
├── docker-test.sh         # Test runner script
├── .dockerignore          # Docker ignore file
└── README_DOCKER.md       # This file
```

## 🔧 Development Workflow

### 1. Start Development Environment
```bash
# Start development container
./docker-run.sh dev

# Access the container
docker exec -it vyuha-dev bash

# Run tests inside container
python simple_test.py
```

### 2. Jupyter Notebook Development
```bash
# Start Jupyter
./docker-run.sh jupyter

# Access at http://localhost:8888
# Create notebooks in ./notebooks/ directory
```

### 3. Production Testing
```bash
# Build production image
./docker-run.sh build

# Test production image
./docker-run.sh test

# Run optimization
./docker-run.sh run --model microsoft/DialoGPT-small
```

## 🧪 Testing in Docker

### Automated Tests
```bash
# Run all tests
./docker-test.sh

# Run specific test
docker run --rm vyuha-ai:test python -m pytest tests/test_optimizer.py
```

### Manual Testing
```bash
# Interactive testing
docker run -it --rm vyuha-ai:test bash

# Inside container:
python -c "import vyuha; print('Vyuha AI loaded successfully')"
python simple_test.py
```

## 🚀 Production Deployment

### 1. Build Production Image
```bash
docker build -t vyuha-ai:latest -f Dockerfile .
```

### 2. Run Production Container
```bash
docker run -d \
  --name vyuha-ai \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  vyuha-ai:latest
```

### 3. Health Check
```bash
# Check container health
docker ps

# View logs
docker logs vyuha-ai

# Test optimization
docker exec vyuha-ai python -m vyuha.cli optimize --help
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Rebuild with proper permissions
docker-compose build --no-cache
```

#### 2. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000

# Use different ports
docker-compose up -d -p 8001:8000
```

#### 3. Memory Issues
```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory
```

#### 4. Model Download Issues
```bash
# Check internet connection
docker run --rm vyuha-ai:test ping -c 3 huggingface.co

# Use pre-downloaded models
# Mount model directory: -v ./models:/app/models
```

### Debug Commands
```bash
# Check container status
docker ps -a

# View detailed logs
docker-compose logs --tail=100 vyuha-ai

# Access container shell
docker exec -it vyuha-ai bash

# Check disk usage
docker system df

# Clean up unused resources
docker system prune -a
```

## 📊 Performance Optimization

### Docker Build Optimization
```bash
# Use build cache
docker build --cache-from vyuha-ai:latest .

# Multi-stage build optimization
docker build --target production .
```

### Runtime Optimization
```bash
# Limit memory usage
docker run --memory=4g vyuha-ai:latest

# Use specific CPU cores
docker run --cpus=2 vyuha-ai:latest

# Enable GPU support (if available)
docker run --gpus all vyuha-ai:latest
```

## 🔒 Security Considerations

### Production Security
```bash
# Run as non-root user (already configured)
docker run --user app vyuha-ai:latest

# Use read-only filesystem
docker run --read-only vyuha-ai:latest

# Limit capabilities
docker run --cap-drop=ALL vyuha-ai:latest
```

### Network Security
```bash
# Use custom network
docker network create vyuha-network
docker run --network vyuha-network vyuha-ai:latest
```

## 📈 Monitoring and Logging

### Log Management
```bash
# View real-time logs
docker-compose logs -f

# Save logs to file
docker-compose logs > vyuha-logs.txt

# Rotate logs
docker run --log-opt max-size=10m --log-opt max-file=3 vyuha-ai:latest
```

### Health Monitoring
```bash
# Check container health
docker inspect vyuha-ai | grep Health

# Monitor resource usage
docker stats vyuha-ai

# Set up monitoring
docker run -d --name monitoring \
  -v /var/run/docker.sock:/var/run/docker.sock \
  prom/prometheus
```

## 🎯 Best Practices

### 1. Development
- Use `docker-run.sh dev` for development
- Mount source code as volume for live updates
- Use Jupyter for experimentation
- Run tests before committing

### 2. Production
- Use multi-stage builds for smaller images
- Run as non-root user
- Use health checks
- Monitor resource usage
- Keep images updated

### 3. CI/CD
- Use Docker for consistent testing
- Build images in CI pipeline
- Tag images with version numbers
- Use Docker registry for distribution

## 🆘 Support

### Getting Help
1. Check logs: `docker-compose logs -f`
2. Run tests: `./docker-test.sh`
3. Check documentation: `README.md`
4. Open issue on GitHub

### Useful Commands
```bash
# Quick health check
docker run --rm vyuha-ai:test python -c "import vyuha; print('OK')"

# Full test suite
./docker-test.sh

# Clean restart
docker-compose down -v && docker-compose up -d
```

---

**Vyuha AI Docker Setup Complete! 🎉**

Your enterprise AI optimization platform is now containerized and ready for deployment!
