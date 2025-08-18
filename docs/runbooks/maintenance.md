# HD-Compute-Toolkit Maintenance Runbook

## Overview

This runbook covers routine maintenance procedures for HD-Compute-Toolkit production environments to ensure optimal performance, security, and reliability.

## Maintenance Schedule

### Daily Tasks (Automated)
- [ ] Health check monitoring
- [ ] Log rotation and cleanup
- [ ] Backup verification
- [ ] Security scan results review
- [ ] Performance metrics collection

### Weekly Tasks
- [ ] Dependency security updates
- [ ] Performance baseline review
- [ ] Log analysis and alerting tuning
- [ ] Capacity planning review
- [ ] Documentation updates

### Monthly Tasks
- [ ] Full system backup and recovery test
- [ ] Security vulnerability assessment
- [ ] Performance optimization review
- [ ] Disaster recovery drill
- [ ] Monitoring system maintenance

### Quarterly Tasks
- [ ] Major dependency updates
- [ ] Architecture review and optimization
- [ ] Capacity planning and scaling decisions
- [ ] Security audit and penetration testing
- [ ] Business continuity plan review

## System Updates

### Dependency Updates

#### Security Updates (Weekly)
```bash
# Check for security vulnerabilities
make security

# Update Python dependencies
pip-audit
safety check

# Update base images
docker pull python:3.11-slim
docker pull nvidia/cuda:12.2-devel-ubuntu22.04

# Rebuild containers with updates
./scripts/build-all.sh
```

#### Regular Updates (Monthly)
```bash
# Update development dependencies
pip install --upgrade pip setuptools wheel
pip install --upgrade -e ".[dev]"

# Update container base images
docker system prune -f
docker pull --all-tags hd-compute-toolkit

# Test updated environment
make test-all
make benchmark
```

### Application Updates

#### Rolling Updates
```bash
# Kubernetes rolling update
kubectl set image deployment/hdc-toolkit hdc-toolkit=hd-compute-toolkit:v1.2.0
kubectl rollout status deployment/hdc-toolkit

# Docker Compose update
docker-compose pull
docker-compose up -d --no-deps hdc-toolkit
```

#### Rollback Procedures
```bash
# Kubernetes rollback
kubectl rollout undo deployment/hdc-toolkit
kubectl rollout status deployment/hdc-toolkit

# Docker rollback
docker tag hd-compute-toolkit:v1.1.0 hd-compute-toolkit:latest
docker-compose up -d --no-deps hdc-toolkit
```

## Performance Maintenance

### Performance Monitoring

#### Daily Performance Checks
```bash
# Check key metrics
curl -s http://localhost:8080/metrics | grep -E "(hdc_latency|hdc_throughput|hdc_memory)"

# Review performance alerts
curl -s http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.labels.alertname | contains("HDC_"))'

# System resource utilization
free -h
df -h
iostat -x 1 3
nvidia-smi
```

#### Performance Optimization
```bash
# Analyze performance trends
python scripts/analyze_performance.py --days 7

# Identify bottlenecks
make profile

# Memory optimization
python -c "
import gc
import psutil
import hd_compute

# Force garbage collection
gc.collect()

# Check memory usage by component
memory_info = psutil.Process().memory_info()
print(f'Total memory: {memory_info.rss / 1024 / 1024:.2f} MB')
print(f'Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB')
"
```

### Capacity Planning

#### Resource Utilization Analysis
```bash
# CPU and memory trends (last 30 days)
curl -G 'http://prometheus:9090/api/v1/query_range' \
  --data-urlencode 'query=rate(hdc_cpu_usage_total[5m])' \
  --data-urlencode 'start=2024-01-01T00:00:00Z' \
  --data-urlencode 'end=2024-01-31T23:59:59Z' \
  --data-urlencode 'step=3600' | jq '.'

# GPU utilization trends
curl -G 'http://prometheus:9090/api/v1/query_range' \
  --data-urlencode 'query=hdc_gpu_utilization_percent' \
  --data-urlencode 'start=2024-01-01T00:00:00Z' \
  --data-urlencode 'end=2024-01-31T23:59:59Z' \
  --data-urlencode 'step=3600' | jq '.'
```

#### Scaling Decisions
```bash
# Horizontal Pod Autoscaler setup
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hdc-toolkit-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hdc-toolkit
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
EOF
```

## Data Maintenance

### Database Maintenance

#### Daily Database Health
```bash
# Check database connections
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
"

# Check database size
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
SELECT pg_database_size('hdc_experiments') / 1024 / 1024 as size_mb;
"

# Check for long-running queries
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
SELECT query, state, query_start, now() - query_start as duration
FROM pg_stat_activity 
WHERE state != 'idle' AND query_start < now() - interval '10 minutes';
"
```

#### Weekly Database Maintenance
```bash
# Vacuum and analyze tables
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
VACUUM ANALYZE;
"

# Update table statistics
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
ANALYZE;
"

# Check index usage
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_tup_read = 0;
"
```

### Log Management

#### Log Rotation
```bash
# Configure logrotate for application logs
sudo tee /etc/logrotate.d/hdc-toolkit <<EOF
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    maxsize 100M
}
EOF

# Force log rotation
sudo logrotate -f /etc/logrotate.d/hdc-toolkit
```

#### Log Analysis
```bash
# Analyze error patterns
grep -i error /app/logs/hdc.log | tail -100 | \
jq -r '.timestamp + " " + .level + " " + .message' | \
sort | uniq -c | sort -nr

# Performance log analysis
grep "performance_measurement" /app/logs/hdc.log | tail -1000 | \
jq -r '[.operation, .duration_ms, .memory_usage_mb] | @csv' > performance_data.csv

# Generate log summary report
python scripts/log_analysis.py --input /app/logs/hdc.log --period 24h
```

## Security Maintenance

### Security Scanning

#### Container Security
```bash
# Scan container images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image hd-compute-toolkit:latest

# Check for exposed secrets
docker run --rm -v $(pwd):/code \
  returntocorp/semgrep --config=p/secrets /code

# Security compliance check
docker run --rm -v $(pwd):/code \
  owasp/dependency-check:latest --project hdc-toolkit --scan /code
```

#### Access Control Review
```bash
# Review Kubernetes RBAC
kubectl get clusterrolebindings -o wide | grep hdc
kubectl get rolebindings -n production -o wide | grep hdc

# Check service account permissions
kubectl describe serviceaccount hdc-toolkit -n production

# Review network policies
kubectl get networkpolicies -n production
```

### Certificate Management

#### SSL Certificate Renewal
```bash
# Check certificate expiration
openssl x509 -in /etc/ssl/certs/hdc-toolkit.pem -text -noout | grep -A 2 Validity

# Renew Let's Encrypt certificates
certbot renew --dry-run
certbot renew

# Update Kubernetes secrets with new certificates
kubectl create secret tls hdc-tls-cert \
  --cert=fullchain.pem \
  --key=privkey.pem \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Backup and Recovery

### Backup Procedures

#### Daily Backups
```bash
# Database backup
docker exec hdc-postgres pg_dump -U hdc_user hdc_experiments | \
gzip > backups/hdc_db_$(date +%Y%m%d).sql.gz

# Model and data backup
docker run --rm -v hdc-data:/data -v $(pwd)/backups:/backup \
alpine tar czf /backup/hdc_data_$(date +%Y%m%d).tar.gz /data

# Configuration backup
kubectl get configmap hdc-config -o yaml > backups/hdc_config_$(date +%Y%m%d).yaml
kubectl get secret hdc-secrets -o yaml > backups/hdc_secrets_$(date +%Y%m%d).yaml
```

#### Backup Verification
```bash
# Verify database backup integrity
gunzip -c backups/hdc_db_$(date +%Y%m%d).sql.gz | head -20

# Test data backup restore
docker run --rm -v $(pwd)/backups:/backup -v test-restore:/restore \
alpine tar xzf /backup/hdc_data_$(date +%Y%m%d).tar.gz -C /restore

# Verify configuration backup
kubectl apply --dry-run=client -f backups/hdc_config_$(date +%Y%m%d).yaml
```

### Disaster Recovery Testing

#### Monthly DR Test
```bash
# Simulate complete environment failure
kubectl delete namespace production
docker-compose down -v

# Restore from backup
kubectl create namespace production
kubectl apply -f backups/

# Restore database
docker-compose up -d postgres
gunzip -c backups/hdc_db_latest.sql.gz | \
docker exec -i hdc-postgres psql -U hdc_user hdc_experiments

# Restore application
docker-compose up -d
kubectl apply -f k8s/

# Verify recovery
make test-integration
curl http://localhost:8000/health
```

## Monitoring System Maintenance

### Prometheus Maintenance

#### Configuration Updates
```bash
# Validate Prometheus configuration
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Reload configuration without restart
curl -X POST http://localhost:9090/-/reload

# Check rule files
docker exec prometheus promtool check rules /etc/prometheus/rules/*.yml
```

#### Data Retention
```bash
# Check Prometheus storage usage
du -sh /prometheus/data

# Configure retention policy (30 days)
docker run -p 9090:9090 \
  -v prometheus-data:/prometheus \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.retention.time=30d \
  --storage.tsdb.retention.size=50GB
```

### Grafana Maintenance

#### Dashboard Backups
```bash
# Export all dashboards
for id in $(curl -s "http://admin:admin@localhost:3000/api/search" | jq -r '.[].uid'); do
  curl -s "http://admin:admin@localhost:3000/api/dashboards/uid/${id}" | \
  jq '.dashboard' > "dashboards/${id}.json"
done

# Import dashboards
for file in dashboards/*.json; do
  curl -X POST "http://admin:admin@localhost:3000/api/dashboards/db" \
    -H "Content-Type: application/json" \
    -d @"${file}"
done
```

## Maintenance Windows

### Planned Maintenance

#### Pre-maintenance Checklist
- [ ] Schedule maintenance window (off-peak hours)
- [ ] Notify stakeholders 24 hours in advance
- [ ] Prepare rollback procedures
- [ ] Verify backup completeness
- [ ] Test procedures in staging environment

#### During Maintenance
- [ ] Enable maintenance mode
- [ ] Stop traffic to application
- [ ] Perform maintenance tasks
- [ ] Verify system health
- [ ] Resume traffic gradually

#### Post-maintenance Checklist
- [ ] Monitor system metrics for 2 hours
- [ ] Verify all functionality works
- [ ] Update documentation
- [ ] Notify stakeholders of completion
- [ ] Schedule follow-up review

### Emergency Maintenance

#### Immediate Response
```bash
# Enable emergency maintenance mode
kubectl patch deployment hdc-toolkit -p '{"spec":{"replicas":0}}'

# Display maintenance page
kubectl apply -f maintenance-mode.yaml

# Perform emergency fixes
# ... emergency procedures ...

# Verify fixes
make test-critical

# Resume service
kubectl patch deployment hdc-toolkit -p '{"spec":{"replicas":3}}'
kubectl delete -f maintenance-mode.yaml
```

## Automation Scripts

### Maintenance Automation

#### Daily Maintenance Script
```bash
#!/bin/bash
# daily_maintenance.sh

set -euo pipefail

echo "Starting daily maintenance..."

# Health checks
make test-health || echo "Health check failed"

# Security scan
make security-scan | tee logs/security_$(date +%Y%m%d).log

# Performance baseline
make benchmark --quick | tee logs/performance_$(date +%Y%m%d).log

# Cleanup old logs
find /app/logs -name "*.log" -mtime +30 -delete

# Database maintenance
docker exec hdc-postgres psql -U hdc_user -d hdc_experiments -c "VACUUM ANALYZE;"

# Report generation
python scripts/daily_report.py --date $(date +%Y-%m-%d)

echo "Daily maintenance completed."
```

#### Weekly Maintenance Script
```bash
#!/bin/bash
# weekly_maintenance.sh

set -euo pipefail

echo "Starting weekly maintenance..."

# Update dependencies
make deps-update

# Full test suite
make test-all

# Performance analysis
python scripts/performance_analysis.py --days 7

# Security updates
make security-updates

# Backup verification
bash scripts/verify_backups.sh

echo "Weekly maintenance completed."
```

Remember to:
1. Always test procedures in staging first
2. Have rollback plans ready
3. Monitor system health after changes
4. Document all maintenance activities
5. Keep stakeholders informed of maintenance schedules