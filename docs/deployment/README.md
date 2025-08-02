# Deployment Guide

This guide covers deployment options and strategies for HD-Compute-Toolkit.

## Deployment Options

### 1. Local Development

```bash
# Clone repository
git clone https://github.com/danieleschmidt/hd-compute-toolkit
cd hd-compute-toolkit

# Install in development mode
make install-dev

# Run tests
make test
```

### 2. Docker Containers

#### Development Container
```bash
# Build and run development environment
make docker-build
make docker-run

# Or use docker-compose
docker-compose up hdc-dev
```

#### Production Container
```bash
# Build production image
make docker-build-prod

# Run production container
docker run -d \
  --name hdc-toolkit \
  -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/models:/app/models \
  hd-compute-toolkit:latest
```

#### GPU-Enabled Container
```bash
# Build GPU image
make docker-build-gpu

# Run with GPU support
docker run -d \
  --name hdc-toolkit-gpu \
  --gpus all \
  -p 8001:8000 \
  -v /path/to/data:/app/data \
  hd-compute-toolkit:gpu
```

### 3. Cloud Deployment

#### AWS EC2
```bash
# Launch GPU-enabled instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge \
  --key-name your-key-pair \
  --security-groups hdc-sg

# Install Docker and run container
sudo docker run -d --gpus all \
  -p 80:8000 \
  hd-compute-toolkit:gpu
```

#### Google Cloud Platform
```bash
# Create VM with GPU
gcloud compute instances create hdc-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# Deploy container
gcloud compute ssh hdc-instance --command \
  "sudo docker run -d --gpus all -p 80:8000 hd-compute-toolkit:gpu"
```

#### Azure Container Instances
```bash
# Deploy container group
az container create \
  --resource-group myResourceGroup \
  --name hdc-toolkit \
  --image hd-compute-toolkit:latest \
  --cpu 4 \
  --memory 16 \
  --ports 8000
```

### 4. Kubernetes Deployment

#### Basic Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdc-toolkit
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hdc-toolkit
  template:
    metadata:
      labels:
        app: hdc-toolkit
    spec:
      containers:
      - name: hdc-toolkit
        image: hd-compute-toolkit:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: hdc-toolkit-service
spec:
  selector:
    app: hdc-toolkit
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### GPU Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdc-toolkit-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hdc-toolkit-gpu
  template:
    metadata:
      labels:
        app: hdc-toolkit-gpu
    spec:
      containers:
      - name: hdc-toolkit
        image: hd-compute-toolkit:gpu
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
```

## Environment Configuration

### Environment Variables
```bash
# Core settings
export HDC_ENV=production
export HDC_DEFAULT_DEVICE=cuda
export HDC_DEFAULT_DIMENSION=10000

# Performance settings
export HDC_MEMORY_POOL_SIZE=1000
export HDC_NUM_THREADS=4

# Logging
export HDC_LOG_LEVEL=INFO
export HDC_LOG_FILE=/app/logs/hdc.log

# Hardware acceleration
export CUDA_VISIBLE_DEVICES=0
export HDC_VULKAN_ENABLED=false
export HDC_FPGA_ENABLED=false
```

### Configuration Files
Create `/app/config/production.yaml`:
```yaml
hdc:
  dimension: 10000
  device: cuda
  seed: null
  
performance:
  memory_pool_size: 1000
  batch_size: 100
  num_threads: 4
  
logging:
  level: INFO
  file: /app/logs/hdc.log
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

hardware:
  cuda:
    enabled: true
    device_id: 0
  vulkan:
    enabled: false
  fpga:
    enabled: false
```

## Performance Optimization

### Resource Requirements

#### Minimum Requirements
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 10 GB
- Python: 3.8+

#### Recommended for Production
- CPU: 8 cores, 3.0 GHz
- RAM: 32 GB
- GPU: NVIDIA V100 or better
- Storage: 100 GB SSD
- Network: 1 Gbps

#### Large-Scale Deployment
- CPU: 16+ cores, 3.5 GHz
- RAM: 64+ GB
- GPU: Multiple A100s
- Storage: 1+ TB NVMe SSD
- Network: 10+ Gbps

### Scaling Strategies

#### Horizontal Scaling
```bash
# Deploy multiple instances
kubectl scale deployment hdc-toolkit --replicas=10

# Use load balancer
kubectl expose deployment hdc-toolkit \
  --type=LoadBalancer \
  --port=80 \
  --target-port=8000
```

#### Vertical Scaling
```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "8"
    nvidia.com/gpu: 2
  limits:
    memory: "32Gi"
    cpu: "16"
    nvidia.com/gpu: 2
```

### Monitoring and Observability

#### Health Checks
```bash
# HTTP health check
curl http://localhost:8000/health

# Container health check
docker exec hdc-toolkit python -c "import hd_compute; print('healthy')"
```

#### Metrics Collection
```yaml
# Prometheus scraping
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"
```

#### Logging
```bash
# Centralized logging
docker run -d \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  --log-opt tag="hdc.{{.Name}}" \
  hd-compute-toolkit:latest
```

## Security Considerations

### Container Security
```dockerfile
# Run as non-root user
USER appuser

# Read-only root filesystem
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /app/logs \
  hd-compute-toolkit:latest
```

### Network Security
```bash
# Limit network access
docker run --network=host \
  --publish 127.0.0.1:8000:8000 \
  hd-compute-toolkit:latest
```

### Secrets Management
```bash
# Use Docker secrets
echo "your-api-key" | docker secret create hdc-api-key -

# Mount secrets
docker run -d \
  --secret hdc-api-key \
  hd-compute-toolkit:latest
```

## Backup and Recovery

### Data Backup
```bash
# Backup models and data
docker run --rm \
  -v hdc-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/hdc-data-$(date +%Y%m%d).tar.gz /data
```

### Disaster Recovery
```bash
# Restore from backup
docker run --rm \
  -v hdc-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/hdc-data-20240101.tar.gz -C /
```

## Troubleshooting

### Common Issues

#### CUDA Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats hdc-toolkit

# Increase container memory
docker run -m 16g hd-compute-toolkit:latest
```

#### Performance Issues
```bash
# Profile performance
docker exec hdc-toolkit python -m cProfile -o profile.stats script.py

# Monitor CPU usage
top -p $(docker inspect -f '{{.State.Pid}}' hdc-toolkit)
```

### Debug Mode
```bash
# Run with debug logging
docker run -e HDC_DEBUG=true -e HDC_LOG_LEVEL=DEBUG hd-compute-toolkit:latest
```

## Maintenance

### Updates
```bash
# Pull latest image
docker pull hd-compute-toolkit:latest

# Rolling update in Kubernetes
kubectl set image deployment/hdc-toolkit hdc-toolkit=hd-compute-toolkit:v1.1.0
```

### Health Monitoring
```bash
# Automated health checks
*/5 * * * * curl -f http://localhost:8000/health || docker restart hdc-toolkit
```