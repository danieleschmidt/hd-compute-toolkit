# HD-Compute-Toolkit Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites
```bash
# System requirements
- Python 3.8+
- 4GB+ RAM recommended
- Optional: CUDA-capable GPU
- Optional: TPU access for JAX backend
```

### Installation Options

#### Option 1: Pip Installation (Recommended)
```bash
# Basic installation
pip install hd-compute-toolkit

# With GPU support
pip install hd-compute-toolkit[gpu]

# Full installation with all backends
pip install hd-compute-toolkit[all]
```

#### Option 2: Development Installation
```bash
git clone https://github.com/danieleschmidt/hd-compute-toolkit
cd hd-compute-toolkit
pip install -e ".[dev]"
```

#### Option 3: Docker Deployment
```bash
# Pull and run container
docker run -it danieleschmidt/hd-compute-toolkit:latest

# With GPU support
docker run --gpus all -it danieleschmidt/hd-compute-toolkit:gpu
```

## 🏗️ Deployment Architectures

### 1. Single Node Research Setup
Perfect for: Academic research, algorithm development, small-scale experiments

```python
from hd_compute import HDComputeTorch

# Initialize with GPU acceleration
hdc = HDComputeTorch(dim=10000, device='cuda')

# Basic operations
hv1 = hdc.random_hv()
hv2 = hdc.random_hv()
bundled = hdc.bundle([hv1, hv2])
similarity = hdc.cosine_similarity(hv1, bundled)
```

**Hardware Requirements**:
- CPU: 4+ cores
- RAM: 8GB+  
- GPU: Optional (NVIDIA with CUDA)
- Storage: 10GB+

### 2. Distributed Research Cluster
Perfect for: Large-scale experiments, comparative studies, high-throughput research

```python
from hd_compute.distributed import DistributedHDC, NodeInfo

# Initialize cluster coordinator
cluster = DistributedHDC()

# Add compute nodes
nodes = [
    NodeInfo(node_id="gpu-1", host="192.168.1.10", port=8080, gpu_available=True),
    NodeInfo(node_id="cpu-1", host="192.168.1.11", port=8080, gpu_available=False),
    NodeInfo(node_id="cpu-2", host="192.168.1.12", port=8080, gpu_available=False),
]

for node in nodes:
    cluster.add_node(node)

# Start distributed processing
cluster.start_cluster()

# Distributed operations
result = cluster.distributed_operation('bundle', large_dataset)
```

**Hardware Requirements per Node**:
- CPU: 8+ cores
- RAM: 16GB+
- Network: Gigabit Ethernet
- Optional: GPU nodes with high-bandwidth interconnect

### 3. Production Web Service
Perfect for: Real-time inference, API services, production applications

```python
from fastapi import FastAPI
from hd_compute import HDComputeTorch

app = FastAPI()
hdc = HDComputeTorch(dim=10000, device='cuda')

@app.post("/similarity")
async def compute_similarity(data: dict):
    hv1 = data['vector1'] 
    hv2 = data['vector2']
    similarity = hdc.cosine_similarity(hv1, hv2)
    return {"similarity": float(similarity)}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

**Production Requirements**:
- Load balancer (Nginx/HAProxy)
- Container orchestration (Kubernetes)
- Monitoring (Prometheus/Grafana)
- Logging (ELK Stack)

### 4. Edge Deployment
Perfect for: IoT devices, mobile apps, resource-constrained environments

```python
from hd_compute import HDComputePython  # Minimal dependencies

# Lightweight deployment
hdc = HDComputePython(dim=1000)  # Smaller dimension for edge

# Efficient operations
hv = hdc.random_hv(sparsity=0.8)  # Higher sparsity for memory efficiency
result = hdc.bind(hv, reference_vector)
```

**Edge Requirements**:
- CPU: ARM/x86 compatible
- RAM: 1GB minimum
- Storage: 100MB+
- Python 3.8+ (no additional dependencies)

## 🐳 Container Deployment

### Docker Images

#### Base Image
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY hd_compute/ ./hd_compute/
COPY examples/ ./examples/

CMD ["python", "-m", "hd_compute.cli.main"]
```

#### GPU Image
```dockerfile  
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements-gpu.txt /tmp/
RUN pip3 install -r /tmp/requirements-gpu.txt

COPY hd_compute/ /app/hd_compute/
WORKDIR /app

CMD ["python3", "-m", "hd_compute.cli.main", "--device", "cuda"]
```

### Docker Compose Setup
```yaml
version: '3.8'
services:
  hdc-coordinator:
    image: danieleschmidt/hd-compute-toolkit:latest
    ports:
      - "8080:8080"
    environment:
      - HDC_ROLE=coordinator
      - HDC_CLUSTER_SIZE=3
    
  hdc-worker-gpu:
    image: danieleschmidt/hd-compute-toolkit:gpu
    runtime: nvidia
    environment:
      - HDC_ROLE=worker
      - HDC_COORDINATOR=hdc-coordinator:8080
      - CUDA_VISIBLE_DEVICES=0
    
  hdc-worker-cpu:
    image: danieleschmidt/hd-compute-toolkit:cpu
    deploy:
      replicas: 2
    environment:
      - HDC_ROLE=worker  
      - HDC_COORDINATOR=hdc-coordinator:8080
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hdc-compute

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: hdc-config
  namespace: hdc-compute
data:
  config.yaml: |
    cluster:
      heartbeat_interval: 30
      node_timeout: 120
      load_balance_strategy: least_loaded
    performance:
      default_dimension: 10000
      cache_size: 1000
```

### Coordinator Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdc-coordinator
  namespace: hdc-compute
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hdc-coordinator
  template:
    metadata:
      labels:
        app: hdc-coordinator
    spec:
      containers:
      - name: coordinator
        image: danieleschmidt/hdc-compute-toolkit:latest
        ports:
        - containerPort: 8080
        env:
        - name: HDC_ROLE
          value: "coordinator"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: hdc-coordinator-service
  namespace: hdc-compute
spec:
  selector:
    app: hdc-coordinator
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### Worker Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdc-workers
  namespace: hdc-compute
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hdc-worker
  template:
    metadata:
      labels:
        app: hdc-worker
    spec:
      containers:
      - name: worker
        image: danieleschmidt/hd-compute-toolkit:latest
        env:
        - name: HDC_ROLE
          value: "worker"
        - name: HDC_COORDINATOR
          value: "hdc-coordinator-service:8080"
        resources:
          requests:
            memory: "4Gi" 
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hdc-worker-hpa
  namespace: hdc-compute
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hdc-workers
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🌐 Cloud Provider Deployments

### AWS Deployment

#### EC2 Instance Setup
```bash
# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge \
  --key-name my-key \
  --security-groups hdc-security-group \
  --user-data file://setup-script.sh
```

#### EKS Cluster
```bash
# Create EKS cluster
eksctl create cluster \
  --name hdc-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --instance-types=p3.2xlarge \
  --nodes=2 \
  --nodes-min=1 \
  --nodes-max=5
```

#### Lambda Function (Edge)
```python
import json
from hd_compute import HDComputePython

hdc = HDComputePython(dim=1000)

def lambda_handler(event, context):
    # Extract vectors from request
    hv1 = event['vector1']
    hv2 = event['vector2']
    
    # Compute similarity
    similarity = hdc.cosine_similarity(hv1, hv2)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'similarity': float(similarity)
        })
    }
```

### Google Cloud Platform

#### GKE with TPUs
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hdc-tpu-workers
spec:
  template:
    spec:
      containers:
      - name: worker
        image: gcr.io/my-project/hd-compute-toolkit:tpu
        resources:
          limits:
            cloud-tpus.google.com/v3: 1
        env:
        - name: HDC_BACKEND
          value: "jax"
        - name: HDC_DEVICE
          value: "tpu"
```

#### Cloud Run Serverless
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: hdc-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
    spec:
      containers:
      - image: gcr.io/my-project/hdc-compute-toolkit:api
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "2Gi"
            cpu: "1"
```

### Microsoft Azure

#### AKS with GPU Nodes
```bash
# Create AKS cluster with GPU support
az aks create \
  --resource-group myResourceGroup \
  --name hdcCluster \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --generate-ssh-keys
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name hdc-instance \
  --image danieleschmidt/hd-compute-toolkit:gpu \
  --cpu 4 \
  --memory 16 \
  --gpu-count 1 \
  --gpu-sku V100
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core configuration
export HDC_DEFAULT_DIMENSION=10000
export HDC_DEFAULT_BACKEND=torch
export HDC_DEVICE=auto  # auto, cpu, cuda, tpu

# Cluster configuration  
export HDC_CLUSTER_ENABLED=true
export HDC_COORDINATOR_HOST=localhost
export HDC_COORDINATOR_PORT=8080

# Performance tuning
export HDC_CACHE_SIZE=1000
export HDC_BATCH_SIZE=32
export HDC_NUM_WORKERS=4

# Security
export HDC_ENABLE_AUTH=true
export HDC_API_KEY_FILE=/etc/hdc/api-keys.txt

# Monitoring
export HDC_METRICS_ENABLED=true
export HDC_METRICS_PORT=9090
export HDC_LOG_LEVEL=INFO
```

### Configuration File
```yaml
# hdc-config.yaml
cluster:
  enabled: true
  coordinator:
    host: "localhost"
    port: 8080
  heartbeat_interval: 30
  node_timeout: 120
  
backends:
  default: "torch"
  torch:
    device: "auto"
    dtype: "float32"
  jax:
    platform: "gpu"
    memory_fraction: 0.8
    
performance:
  default_dimension: 10000
  cache_size: 1000
  batch_size: 32
  num_workers: 4
  
security:
  enable_auth: false
  audit_logging: true
  input_validation: strict
  
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  performance_tracking: true
  log_level: "INFO"
```

## 📊 Monitoring and Observability

### Prometheus Metrics
```python
# Custom metrics endpoint
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
operation_counter = Counter('hdc_operations_total', 'Total HDC operations')
operation_duration = Histogram('hdc_operation_duration_seconds', 'HDC operation duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard
Key metrics to monitor:
- Operations per second
- Average operation latency  
- Memory usage per node
- Cache hit rate
- Error rate
- Queue depth
- Node health status

### Logging Configuration
```python
import logging
from hd_compute.utils import setup_logging

# Configure structured logging
setup_logging(
    level="INFO",
    format="json",
    output="/var/log/hdc/application.log"
)

logger = logging.getLogger(__name__)
logger.info("HDC service starting", extra={
    "dimension": 10000,
    "backend": "torch",
    "device": "cuda"
})
```

## 🔐 Security Deployment

### TLS/SSL Setup
```yaml
# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hdc-ingress
spec:
  tls:
  - hosts:
    - api.hdc-compute.com
    secretName: hdc-tls-secret
  rules:
  - host: api.hdc-compute.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hdc-coordinator-service
            port:
              number: 8080
```

### Network Security
```yaml
# Network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hdc-network-policy
spec:
  podSelector:
    matchLabels:
      app: hdc-worker
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: hdc-coordinator
    ports:
    - protocol: TCP
      port: 8080
```

## 🚨 Disaster Recovery

### Backup Strategy
```bash
# Backup cluster state
kubectl get all -n hdc-compute -o yaml > hdc-cluster-backup.yaml

# Backup persistent volumes
kubectl get pvc -n hdc-compute -o yaml > hdc-pvc-backup.yaml

# Export configuration
kubectl get configmap hdc-config -o yaml > hdc-config-backup.yaml
```

### Recovery Procedures
```bash
# Restore from backup
kubectl apply -f hdc-cluster-backup.yaml
kubectl apply -f hdc-pvc-backup.yaml
kubectl apply -f hdc-config-backup.yaml

# Verify cluster health
kubectl get pods -n hdc-compute
kubectl logs -l app=hdc-coordinator -n hdc-compute
```

## 📈 Performance Tuning

### CPU Optimization
```python
# Optimize for CPU deployment
export HDC_NUM_WORKERS=$(nproc)
export HDC_BATCH_SIZE=64
export OMP_NUM_THREADS=$(nproc)
```

### GPU Optimization
```python
# Optimize for GPU deployment  
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HDC_GPU_MEMORY_FRACTION=0.8
export HDC_MIXED_PRECISION=true
```

### Memory Optimization
```python
# Memory-efficient configuration
export HDC_CACHE_SIZE=500
export HDC_DIMENSION=5000  # Smaller for memory-constrained environments
export HDC_SPARSE_MODE=true
```

## 🧪 Testing Deployment

### Health Checks
```python
# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test basic operation
        hdc = HDComputeTorch(dim=100)
        hv = hdc.random_hv()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "backend": "torch",
            "dimension": 100
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }
```

### Load Testing
```bash
# Apache Benchmark
ab -n 1000 -c 10 http://api.hdc-compute.com/similarity

# Custom load test
python load_test.py --url http://api.hdc-compute.com --requests 10000 --concurrency 50
```

---

## 🎯 Deployment Checklist

### Pre-deployment
- [ ] Requirements verified
- [ ] Configuration tested
- [ ] Security review completed
- [ ] Performance benchmarked
- [ ] Documentation updated

### Deployment
- [ ] Infrastructure provisioned
- [ ] Applications deployed
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Logging enabled

### Post-deployment
- [ ] End-to-end testing
- [ ] Performance validation
- [ ] Security scan
- [ ] Backup verified
- [ ] Team trained

**🚀 Ready for Production Deployment!**

This deployment guide provides comprehensive coverage for all deployment scenarios from single-node research setups to enterprise-grade distributed clusters. Choose the deployment architecture that best fits your use case and scale as needed.