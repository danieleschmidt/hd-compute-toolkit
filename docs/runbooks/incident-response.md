# HD-Compute-Toolkit Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in HD-Compute-Toolkit production environments.

## Incident Classification

### Severity Levels

#### Critical (P0)
- Complete service outage
- Data loss or corruption
- Security breach
- Response time: Immediate (< 5 minutes)

#### High (P1)
- Significant performance degradation
- Partial service unavailability
- Failed deployments affecting production
- Response time: < 15 minutes

#### Medium (P2)
- Minor performance issues
- Non-critical feature failures
- Monitoring/alerting issues
- Response time: < 1 hour

#### Low (P3)
- Documentation issues
- Minor cosmetic problems
- Enhancement requests
- Response time: < 24 hours

## Common Incidents and Responses

### 1. Service Outage

#### Symptoms
- Health check endpoints returning 5xx errors
- No response from application
- Container/pod not running

#### Investigation Steps
```bash
# Check container status
docker ps | grep hdc-toolkit
kubectl get pods -l app=hdc-toolkit

# Check logs
docker logs hdc-toolkit --tail=100
kubectl logs -l app=hdc-toolkit --tail=100

# Check resource usage
docker stats hdc-toolkit
kubectl top pods -l app=hdc-toolkit
```

#### Resolution Steps
1. **Immediate**
   ```bash
   # Restart container/pod
   docker restart hdc-toolkit
   kubectl rollout restart deployment/hdc-toolkit
   ```

2. **If restart fails**
   ```bash
   # Check system resources
   df -h  # Disk space
   free -h  # Memory
   top  # CPU usage
   
   # Check Docker daemon
   systemctl status docker
   
   # Check Kubernetes cluster
   kubectl cluster-info
   ```

3. **Recovery actions**
   ```bash
   # Scale up replicas
   kubectl scale deployment hdc-toolkit --replicas=3
   
   # Rollback to previous version if needed
   kubectl rollout undo deployment/hdc-toolkit
   ```

### 2. High Memory Usage

#### Symptoms
- `HDC_HighMemoryUsage` alert firing
- OOMKilled container restarts
- Slow response times

#### Investigation Steps
```bash
# Check memory metrics
curl http://localhost:8080/metrics | grep hdc_memory

# Check system memory
free -h
cat /proc/meminfo

# Check for memory leaks
docker exec hdc-toolkit python -c "
import psutil
import gc
gc.collect()
print(f'Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### Resolution Steps
1. **Immediate relief**
   ```bash
   # Force garbage collection
   docker exec hdc-toolkit python -c "import gc; gc.collect()"
   
   # Increase memory limits
   kubectl patch deployment hdc-toolkit -p '{"spec":{"template":{"spec":{"containers":[{"name":"hdc-toolkit","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```

2. **Long-term fixes**
   - Review hypervector dimensions and batch sizes
   - Implement memory pooling
   - Add memory profiling to identify leaks

### 3. GPU Not Available

#### Symptoms
- CUDA initialization failures
- `torch.cuda.is_available()` returns False
- GPU metrics show 0% utilization

#### Investigation Steps
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check Kubernetes GPU resources
kubectl describe nodes | grep -A 10 "Capacity:\|Allocatable:"
```

#### Resolution Steps
1. **Driver issues**
   ```bash
   # Restart NVIDIA services
   sudo systemctl restart nvidia-persistenced
   sudo systemctl restart nvidia-docker
   
   # Reload NVIDIA modules
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
   ```

2. **Container issues**
   ```bash
   # Restart with GPU access
   docker stop hdc-toolkit-gpu
   docker run -d --name hdc-toolkit-gpu --gpus all hd-compute-toolkit:gpu
   ```

3. **Kubernetes issues**
   ```bash
   # Check GPU device plugin
   kubectl get daemonset -n kube-system | grep nvidia
   
   # Redeploy GPU workload
   kubectl delete pod -l app=hdc-toolkit-gpu
   ```

### 4. Performance Degradation

#### Symptoms
- `HDC_HighLatency` alert firing
- Slow response times
- Low throughput metrics

#### Investigation Steps
```bash
# Check latency metrics
curl http://localhost:8080/metrics | grep hdc_latency

# Check system load
uptime
iostat 1 5

# Check application logs for slow operations
docker logs hdc-toolkit | grep -i "slow\|timeout\|performance"

# Profile running application
docker exec hdc-toolkit python -m cProfile -o /tmp/profile.stats -c "
import hd_compute
# Run typical workload
"
```

#### Resolution Steps
1. **Immediate**
   ```bash
   # Scale horizontally
   kubectl scale deployment hdc-toolkit --replicas=5
   
   # Check and fix resource limits
   kubectl get deployment hdc-toolkit -o yaml | grep -A 10 resources
   ```

2. **Optimization**
   - Review batch sizes and dimensions
   - Enable GPU acceleration if available
   - Implement caching for frequent operations
   - Review algorithm efficiency

### 5. Database Connection Issues

#### Symptoms
- Database connection timeouts
- MLflow tracking failures
- Experiment data not being saved

#### Investigation Steps
```bash
# Check database status
docker exec hdc-postgres pg_isready

# Check connection parameters
docker logs hdc-toolkit | grep -i "database\|postgres"

# Test database connectivity
docker exec hdc-toolkit python -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://hdc_user:hdc_password@postgres:5432/hdc_experiments')
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

#### Resolution Steps
1. **Database restart**
   ```bash
   docker restart hdc-postgres
   kubectl rollout restart deployment/postgres
   ```

2. **Connection pool issues**
   ```bash
   # Check active connections
   docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
   SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
   "
   
   # Kill long-running queries if necessary
   docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND state_change < now() - interval '1 hour';
   "
   ```

## Escalation Procedures

### Level 1: On-call Engineer
- Initial response and investigation
- Apply immediate fixes from runbooks
- Monitor for resolution

### Level 2: Senior Engineer
- Complex debugging and analysis
- Performance optimization
- Architecture decisions

### Level 3: Engineering Manager
- Resource allocation decisions
- Cross-team coordination
- Communication with stakeholders

### External Escalation
- Vendor support (NVIDIA, cloud providers)
- Security team (for security incidents)
- Legal team (for data breaches)

## Communication Templates

### Initial Incident Notification
```
🚨 INCIDENT ALERT 🚨
Severity: [P0/P1/P2/P3]
Service: HD-Compute-Toolkit
Impact: [Description of user impact]
Started: [Timestamp]
Investigating: [Name]

Status page: [URL]
Updates will be posted every 15 minutes.
```

### Status Update
```
📊 INCIDENT UPDATE
Incident: [Brief description]
Status: [Investigating/Identified/Monitoring/Resolved]
Update: [What we've learned/done]
Next: [Next steps]
ETA: [Estimated resolution time]

Latest update: [Timestamp]
```

### Resolution Notification
```
✅ INCIDENT RESOLVED
Incident: [Brief description]
Duration: [Total time]
Resolution: [What fixed it]
Follow-up: [Prevention measures]

Post-mortem will be shared within 24 hours.
```

## Post-Incident Procedures

### 1. Immediate (within 1 hour)
- [ ] Confirm service is fully restored
- [ ] Document timeline of events
- [ ] Gather logs and metrics
- [ ] Notify stakeholders of resolution

### 2. Short-term (within 24 hours)
- [ ] Write post-mortem report
- [ ] Identify root cause
- [ ] Create action items for fixes
- [ ] Share lessons learned

### 3. Long-term (within 1 week)
- [ ] Implement preventive measures
- [ ] Update monitoring and alerting
- [ ] Review and update runbooks
- [ ] Conduct team retrospective

## Prevention Strategies

### Monitoring
- Comprehensive health checks
- Proactive alerting on leading indicators
- Performance baseline tracking
- Automated anomaly detection

### Testing
- Regular disaster recovery drills
- Chaos engineering practices
- Load testing in staging
- Security vulnerability scanning

### Documentation
- Keep runbooks updated
- Document all configuration changes
- Maintain incident knowledge base
- Regular training on procedures

## Emergency Contacts

```
Primary On-call: +1-555-0123
Secondary On-call: +1-555-0124
Engineering Manager: +1-555-0125
Security Team: security@company.com
Infrastructure Team: infra@company.com
```

## Recovery Time Objectives (RTO)

| Incident Type | Target RTO | Maximum Downtime |
|---------------|------------|------------------|
| Service Outage | 15 minutes | 1 hour |
| Performance Issues | 30 minutes | 2 hours |
| GPU Unavailability | 1 hour | 4 hours |
| Database Issues | 30 minutes | 2 hours |

## Testing and Validation

### Monthly Drills
- [ ] Practice incident response procedures
- [ ] Test backup and recovery systems
- [ ] Validate monitoring and alerting
- [ ] Review and update contact information

### Quarterly Reviews
- [ ] Analyze incident trends
- [ ] Update severity classifications
- [ ] Review RTO/RPO objectives
- [ ] Conduct tabletop exercises

Remember: When in doubt, escalate early. It's better to involve more people than needed than to struggle alone during an incident.