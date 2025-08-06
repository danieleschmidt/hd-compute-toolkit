# HD Compute Toolkit - Production Deployment Checklist

This checklist ensures a safe and successful production deployment of the HD Compute Toolkit.

## Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] Kubernetes cluster is running and accessible
- [ ] kubectl is installed and configured
- [ ] Sufficient cluster resources available
  - [ ] At least 6 CPU cores available
  - [ ] At least 8GB RAM available
  - [ ] At least 100GB storage available
- [ ] Container registry is accessible
- [ ] Docker image is built and pushed to registry

### Security Requirements
- [ ] Network policies are configured
- [ ] Secrets are updated with production values (not defaults)
- [ ] TLS certificates are configured for ingress
- [ ] RBAC policies are in place
- [ ] Security scanning completed on container images
- [ ] Firewall rules configured for ingress traffic

### Configuration Requirements
- [ ] ConfigMap values reviewed and updated for production
- [ ] Environment-specific settings configured
- [ ] Database credentials configured
- [ ] Redis configuration reviewed
- [ ] Logging configuration set to appropriate level
- [ ] Monitoring and alerting configured

### Database Setup
- [ ] PostgreSQL persistent volume configured
- [ ] Database credentials secured
- [ ] Database backup strategy implemented
- [ ] Database migrations tested
- [ ] Connection pooling configured

### Monitoring & Observability
- [ ] Prometheus monitoring configured
- [ ] Grafana dashboards imported
- [ ] Alerting rules configured
- [ ] Log aggregation configured
- [ ] Health check endpoints verified
- [ ] Metrics endpoints verified

### Compliance & Security
- [ ] GDPR compliance features enabled
- [ ] Audit logging configured
- [ ] Data retention policies set
- [ ] Security scanning enabled
- [ ] Input validation configured
- [ ] Rate limiting configured

## Deployment Steps

### Step 1: Pre-deployment Validation
```bash
# Verify cluster connectivity
kubectl cluster-info

# Check available resources
kubectl top nodes

# Verify image availability
docker pull hd-compute-toolkit:latest
```

### Step 2: Deploy Infrastructure Components
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Apply storage
kubectl apply -f k8s/pvc.yaml
```

### Step 3: Deploy Database and Cache
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for services to be ready
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=postgres -n hd-compute-toolkit --timeout=300s
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=redis -n hd-compute-toolkit --timeout=300s
```

### Step 4: Deploy Application
```bash
# Deploy main application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for application to be ready
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=hd-compute-toolkit -n hd-compute-toolkit --timeout=600s
```

### Step 5: Configure Networking and Security
```bash
# Apply ingress configuration
kubectl apply -f k8s/ingress.yaml

# Apply network policies
kubectl apply -f k8s/networkpolicy.yaml
```

### Step 6: Enable Autoscaling and Monitoring
```bash
# Configure autoscaling
kubectl apply -f k8s/hpa.yaml

# Configure monitoring (if Prometheus Operator is available)
kubectl apply -f k8s/monitoring.yaml
```

### Step 7: Automated Deployment (Alternative)
```bash
# Use the deployment script for automated deployment
./deploy/deploy.sh --environment production --tag v1.0.0
```

## Post-Deployment Verification

### Application Health
- [ ] All pods are running and ready
- [ ] Health check endpoints respond correctly
- [ ] Application logs show no errors
- [ ] Database connections are working
- [ ] Cache connections are working

### Performance Verification
```bash
# Check resource usage
kubectl top pods -n hd-compute-toolkit

# Verify autoscaling is working
kubectl get hpa -n hd-compute-toolkit

# Test application performance
# (Add your specific performance tests here)
```

### Security Verification
- [ ] Network policies are enforced
- [ ] TLS certificates are valid
- [ ] Security headers are present in responses
- [ ] Audit logs are being generated
- [ ] Secrets are not exposed in logs or configs

### Monitoring Verification
- [ ] Metrics are being collected
- [ ] Dashboards show data
- [ ] Alerts are configured and firing appropriately
- [ ] Log aggregation is working

## Rollback Procedures

### Quick Rollback
```bash
# Rollback to previous deployment
kubectl rollout undo deployment/hdc-toolkit-app -n hd-compute-toolkit

# Check rollback status
kubectl rollout status deployment/hdc-toolkit-app -n hd-compute-toolkit
```

### Complete Environment Teardown
```bash
# Remove entire deployment (DANGEROUS - use with caution)
kubectl delete namespace hd-compute-toolkit

# Or use the cleanup script
./deploy/cleanup.sh --environment production
```

## Production Operations

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment/hdc-toolkit-app --replicas=10 -n hd-compute-toolkit

# Update HPA settings
kubectl patch hpa hdc-toolkit-hpa -n hd-compute-toolkit -p '{"spec":{"maxReplicas":20}}'
```

### Configuration Updates
```bash
# Update ConfigMap
kubectl patch configmap hdc-toolkit-config -n hd-compute-toolkit --patch '{"data":{"HDC_LOG_LEVEL":"DEBUG"}}'

# Restart deployment to pick up changes
kubectl rollout restart deployment/hdc-toolkit-app -n hd-compute-toolkit
```

### Database Operations
```bash
# Access database
kubectl exec -it deployment/postgres -n hd-compute-toolkit -- psql -U hdc_user -d hdc_experiments

# Backup database
kubectl exec deployment/postgres -n hd-compute-toolkit -- pg_dump -U hdc_user hdc_experiments > backup.sql
```

## Maintenance Windows

### Regular Maintenance Tasks
- [ ] Update container images
- [ ] Apply security patches
- [ ] Review and rotate secrets
- [ ] Clean up old data according to retention policies
- [ ] Verify backup procedures
- [ ] Update monitoring configurations

### Monthly Tasks
- [ ] Review resource usage and scaling policies
- [ ] Update certificates if needed
- [ ] Audit security configurations
- [ ] Review compliance reports
- [ ] Update documentation

## Emergency Procedures

### High CPU/Memory Usage
1. Check metrics and identify cause
2. Scale up deployment if needed
3. Investigate resource-intensive operations
4. Apply resource limits if necessary

### Database Issues
1. Check database pod logs
2. Verify connectivity from application pods
3. Check persistent volume status
4. Restore from backup if necessary

### Security Incidents
1. Review audit logs
2. Check for unauthorized access
3. Rotate compromised secrets
4. Apply security patches
5. Update security policies

## Support Information

### Key Contacts
- DevOps Team: devops@company.com
- Security Team: security@company.com
- On-call Engineer: +1-XXX-XXX-XXXX

### Documentation Links
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [API Documentation](../docs/api/)
- [Monitoring Runbooks](../docs/runbooks/)
- [Security Procedures](../docs/security/)

### Useful Commands
```bash
# View application logs
kubectl logs -f deployment/hdc-toolkit-app -n hd-compute-toolkit

# Access application shell
kubectl exec -it deployment/hdc-toolkit-app -n hd-compute-toolkit -- bash

# Port forward for local access
kubectl port-forward svc/hdc-toolkit-service 8080:80 -n hd-compute-toolkit

# Get detailed resource information
kubectl describe deployment hdc-toolkit-app -n hd-compute-toolkit
kubectl describe pod <pod-name> -n hd-compute-toolkit

# Check cluster events
kubectl get events -n hd-compute-toolkit --sort-by='.lastTimestamp'
```

---

**Important:** This checklist should be reviewed and updated regularly to reflect changes in the application, infrastructure, and security requirements.