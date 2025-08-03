# Operational Runbooks

Standard operating procedures for HD-Compute-Toolkit production operations.

## Overview

This directory contains runbooks for common operational scenarios, incident response procedures, and maintenance tasks for HD-Compute-Toolkit deployments.

## Runbook Structure

Each runbook follows this structure:
- **Scenario**: Description of the situation
- **Symptoms**: How to identify the issue
- **Diagnosis**: Steps to confirm the root cause
- **Resolution**: Step-by-step fix procedures
- **Prevention**: How to prevent future occurrences
- **Escalation**: When and how to escalate

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: +1-555-0123
- **DevOps Team**: devops@company.com
- **Security Team**: security@company.com

### Critical System Links
- **Monitoring Dashboard**: https://grafana.company.com/hdc
- **Log Aggregation**: https://kibana.company.com/hdc
- **Incident Management**: https://pagerduty.com/incidents
- **Status Page**: https://status.company.com

## Common Scenarios

### 1. High Latency (P95 > 100ms)

#### Symptoms
- Alert: "HDC_HighLatency" firing
- User reports slow response times
- Grafana dashboard shows latency spikes

#### Diagnosis
```bash
# Check current performance metrics
curl http://hdc-service:8080/metrics | grep hdc_latency

# Check resource utilization
kubectl top pods -l app=hdc-toolkit

# Review recent logs
kubectl logs -l app=hdc-toolkit --since=10m | grep -E "(error|ERROR|slow)"
```

#### Resolution
1. **Immediate**: Scale up replicas
   ```bash
   kubectl scale deployment hdc-toolkit --replicas=6
   ```

2. **Investigate resource constraints**:
   ```bash
   # Check memory usage
   kubectl describe pods -l app=hdc-toolkit | grep -A 5 "Limits:"
   
   # Check GPU utilization
   kubectl exec -it hdc-toolkit-pod -- nvidia-smi
   ```

3. **Optimize configuration**:
   ```bash
   # Increase batch size if memory allows
   kubectl set env deployment/hdc-toolkit HDC_BATCH_SIZE=200
   ```

#### Prevention
- Set up auto-scaling based on latency metrics
- Regular performance testing with realistic workloads
- Monitor resource utilization trends

### 2. Memory Exhaustion

#### Symptoms
- Pods being OOMKilled
- Alert: "HDC_HighMemoryUsage" firing
- Application crashes with memory errors

#### Diagnosis
```bash
# Check memory usage patterns
kubectl top pods -l app=hdc-toolkit

# Review OOMKill events
kubectl get events --field-selector reason=OOMKilling

# Check memory leaks
kubectl exec hdc-toolkit-pod -- python -c "
import tracemalloc
tracemalloc.start()
# Run operation
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.1f}MB, Peak: {peak / 1024 / 1024:.1f}MB')
"
```

#### Resolution
1. **Immediate**: Increase memory limits
   ```bash
   kubectl patch deployment hdc-toolkit -p '{"spec":{"template":{"spec":{"containers":[{"name":"hdc-toolkit","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
   ```

2. **Restart affected pods**:
   ```bash
   kubectl delete pods -l app=hdc-toolkit
   ```

3. **Investigate memory leaks**:
   ```bash
   # Profile memory usage
   kubectl exec hdc-toolkit-pod -- python -m memory_profiler script.py
   ```

#### Prevention
- Implement memory usage monitoring
- Regular memory leak testing
- Optimize hypervector caching strategies

### 3. CUDA Out of Memory

#### Symptoms
- Error logs: "CUDA out of memory"
- GPU operations failing
- Degraded performance on GPU-enabled nodes

#### Diagnosis
```bash
# Check GPU memory usage
kubectl exec hdc-gpu-pod -- nvidia-smi

# Review CUDA-related logs
kubectl logs hdc-gpu-pod | grep -i cuda

# Check GPU resource allocation
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```

#### Resolution
1. **Clear GPU memory cache**:
   ```bash
   kubectl exec hdc-gpu-pod -- python -c "
   import torch
   torch.cuda.empty_cache()
   print('GPU cache cleared')
   "
   ```

2. **Reduce batch sizes**:
   ```bash
   kubectl set env deployment/hdc-gpu HDC_GPU_BATCH_SIZE=50
   ```

3. **Restart GPU pods**:
   ```bash
   kubectl delete pods -l app=hdc-gpu
   ```

#### Prevention
- Monitor GPU memory usage trends
- Implement dynamic batch sizing
- Set appropriate GPU memory limits

### 4. Application Startup Failures

#### Symptoms
- Pods stuck in CrashLoopBackOff
- Health checks failing
- Application not responding

#### Diagnosis
```bash
# Check pod status
kubectl get pods -l app=hdc-toolkit

# Review startup logs
kubectl logs hdc-toolkit-pod --previous

# Check configuration
kubectl describe configmap hdc-config
```

#### Resolution
1. **Check dependencies**:
   ```bash
   # Verify PyTorch/JAX installation
   kubectl exec hdc-toolkit-pod -- python -c "import torch, jax; print('Dependencies OK')"
   ```

2. **Validate configuration**:
   ```bash
   # Check config syntax
   kubectl exec hdc-toolkit-pod -- python -c "
   import yaml
   with open('/app/config/production.yaml') as f:
       yaml.safe_load(f)
   print('Config valid')
   "
   ```

3. **Reset to known good state**:
   ```bash
   kubectl rollout undo deployment/hdc-toolkit
   ```

#### Prevention
- Automated configuration validation
- Health check improvements
- Staged deployments with canary releases

### 5. High Error Rate

#### Symptoms
- Alert: "HDC_HighErrorRate" firing
- Increased 5xx responses
- User-reported failures

#### Diagnosis
```bash
# Check error metrics
curl http://hdc-service:8080/metrics | grep hdc_operation_errors

# Review error logs
kubectl logs -l app=hdc-toolkit | grep -E "(ERROR|Exception|Failed)"

# Check external dependencies
curl -I http://external-api/health
```

#### Resolution
1. **Identify error sources**:
   ```bash
   # Analyze error patterns
   kubectl logs hdc-toolkit-pod | python -c "
   import sys, json
   from collections import Counter
   errors = [json.loads(line)['error_type'] for line in sys.stdin if 'error_type' in line]
   print(Counter(errors).most_common(5))
   "
   ```

2. **Apply hotfixes**:
   ```bash
   # Deploy hotfix if available
   kubectl set image deployment/hdc-toolkit hdc-toolkit=hdc-toolkit:hotfix-v1.0.1
   ```

3. **Circuit breaker activation**:
   ```bash
   # Enable circuit breaker for failing operations
   kubectl set env deployment/hdc-toolkit HDC_CIRCUIT_BREAKER_ENABLED=true
   ```

#### Prevention
- Comprehensive error monitoring
- Circuit breaker patterns
- Graceful degradation strategies

## Maintenance Procedures

### 1. Routine Health Checks

#### Daily Checks
```bash
#!/bin/bash
# daily-health-check.sh

echo "=== HD-Compute-Toolkit Daily Health Check ==="
echo "Date: $(date)"

# Check pod health
echo "Pod Status:"
kubectl get pods -l app=hdc-toolkit

# Check resource usage
echo "Resource Usage:"
kubectl top pods -l app=hdc-toolkit

# Check recent errors
echo "Recent Errors (last 24h):"
kubectl logs -l app=hdc-toolkit --since=24h | grep ERROR | tail -10

# Check external dependencies
echo "External Dependencies:"
curl -s -o /dev/null -w "%{http_code}" http://external-api/health

echo "=== Health Check Complete ==="
```

#### Weekly Checks
```bash
#!/bin/bash
# weekly-maintenance.sh

# Performance trend analysis
echo "Generating weekly performance report..."
python scripts/performance_report.py --period=7d

# Security updates
echo "Checking for security updates..."
kubectl get pods -o jsonpath='{.items[*].spec.containers[*].image}' | \
  xargs -n1 trivy image --severity HIGH,CRITICAL

# Backup verification
echo "Verifying backup integrity..."
python scripts/verify_backups.py --days=7
```

### 2. Deployment Procedures

#### Blue-Green Deployment
```bash
#!/bin/bash
# blue-green-deploy.sh

VERSION=$1
if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# Deploy to green environment
kubectl apply -f k8s/green-deployment.yaml
kubectl set image deployment/hdc-toolkit-green hdc-toolkit=hdc-toolkit:$VERSION

# Wait for green to be ready
kubectl rollout status deployment/hdc-toolkit-green

# Run health checks
kubectl exec deployment/hdc-toolkit-green -- python scripts/health_check.py

# Switch traffic
kubectl patch service hdc-toolkit -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for issues
sleep 300

# If successful, scale down blue
kubectl scale deployment hdc-toolkit-blue --replicas=0
```

#### Canary Deployment
```bash
#!/bin/bash
# canary-deploy.sh

VERSION=$1
CANARY_PERCENT=${2:-10}

# Deploy canary version
kubectl apply -f k8s/canary-deployment.yaml
kubectl set image deployment/hdc-toolkit-canary hdc-toolkit=hdc-toolkit:$VERSION

# Configure traffic split
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: hdc-toolkit
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: hdc-toolkit-canary
  - route:
    - destination:
        host: hdc-toolkit-stable
      weight: $((100-CANARY_PERCENT))
    - destination:
        host: hdc-toolkit-canary
      weight: $CANARY_PERCENT
EOF

# Monitor canary metrics
python scripts/monitor_canary.py --duration=30m --version=$VERSION
```

### 3. Backup and Recovery

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="/backups/hdc_$BACKUP_DATE"

# Create backup
kubectl exec postgres-pod -- pg_dump -U hdc_user hdc_experiments > $BACKUP_PATH.sql

# Compress and encrypt
gzip $BACKUP_PATH.sql
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric \
    --output $BACKUP_PATH.sql.gz.gpg $BACKUP_PATH.sql.gz

# Upload to cloud storage
aws s3 cp $BACKUP_PATH.sql.gz.gpg s3://hdc-backups/database/

# Clean up local files
rm $BACKUP_PATH.sql.gz $BACKUP_PATH.sql.gz.gpg
```

#### Model Backup
```bash
#!/bin/bash
# backup-models.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

# Backup trained models
kubectl cp hdc-toolkit-pod:/app/models /tmp/models_$BACKUP_DATE

# Create archive
tar -czf /tmp/models_$BACKUP_DATE.tar.gz -C /tmp models_$BACKUP_DATE

# Upload to cloud storage
aws s3 cp /tmp/models_$BACKUP_DATE.tar.gz s3://hdc-backups/models/

# Clean up
rm -rf /tmp/models_$BACKUP_DATE /tmp/models_$BACKUP_DATE.tar.gz
```

### 4. Disaster Recovery

#### Service Recovery
```bash
#!/bin/bash
# disaster-recovery.sh

echo "Starting HD-Compute-Toolkit disaster recovery..."

# 1. Restore from backup
kubectl apply -f k8s/disaster-recovery.yaml

# 2. Restore database
LATEST_BACKUP=$(aws s3 ls s3://hdc-backups/database/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://hdc-backups/database/$LATEST_BACKUP /tmp/
gpg --decrypt /tmp/$LATEST_BACKUP | gunzip | kubectl exec -i postgres-pod -- psql -U hdc_user -d hdc_experiments

# 3. Restore models
LATEST_MODELS=$(aws s3 ls s3://hdc-backups/models/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://hdc-backups/models/$LATEST_MODELS /tmp/
kubectl cp /tmp/$LATEST_MODELS hdc-toolkit-pod:/app/models.tar.gz
kubectl exec hdc-toolkit-pod -- tar -xzf /app/models.tar.gz -C /app/

# 4. Verify recovery
kubectl exec hdc-toolkit-pod -- python scripts/verify_recovery.py

echo "Disaster recovery complete. Verify all services are operational."
```

## Incident Response

### Severity Levels

#### P0 - Critical
- Complete service outage
- Data loss
- Security breach

**Response Time**: 15 minutes
**Escalation**: Immediate

#### P1 - High
- Major feature unavailable
- Significant performance degradation
- Multiple user impact

**Response Time**: 1 hour
**Escalation**: 2 hours

#### P2 - Medium
- Minor feature issues
- Single user impact
- Non-critical bugs

**Response Time**: 4 hours
**Escalation**: 24 hours

#### P3 - Low
- Cosmetic issues
- Enhancement requests
- Documentation updates

**Response Time**: Next business day
**Escalation**: 1 week

### Incident Checklist

1. **Acknowledge**: Confirm receipt within SLA
2. **Assess**: Determine severity and impact
3. **Communicate**: Update stakeholders
4. **Investigate**: Identify root cause
5. **Resolve**: Implement fix
6. **Verify**: Confirm resolution
7. **Document**: Write post-mortem
8. **Follow-up**: Implement preventive measures

## Contact Information

### Team Contacts
- **Primary On-call**: +1-555-0123 (PagerDuty)
- **Secondary On-call**: +1-555-0124 (PagerDuty)
- **Team Lead**: lead@company.com
- **DevOps Engineer**: devops@company.com

### Escalation Path
1. **L1**: On-call Engineer
2. **L2**: Team Lead
3. **L3**: Engineering Manager
4. **L4**: VP Engineering

### External Contacts
- **Cloud Provider Support**: support@cloudprovider.com
- **Vendor Support**: support@vendor.com
- **Legal/Compliance**: legal@company.com