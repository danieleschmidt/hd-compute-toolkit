# Monitoring and Observability

Comprehensive monitoring setup for HD-Compute-Toolkit in production environments.

## Overview

HD-Compute-Toolkit provides built-in observability features including metrics, logging, tracing, and health checks to ensure reliable operation in production environments.

## Health Checks

### Application Health Endpoint

The toolkit exposes a health check endpoint for monitoring application status:

```python
# Health check implementation
from hd_compute.monitoring import HealthChecker

health_checker = HealthChecker()
health_status = health_checker.check_all()

# Returns:
# {
#   "status": "healthy",
#   "timestamp": "2025-08-02T10:30:00Z",
#   "checks": {
#     "cuda": "available",
#     "memory": "normal",
#     "system": "operational"
#   }
# }
```

### Health Check Configuration

```yaml
# config/monitoring.yaml
health_checks:
  interval: 30s
  timeout: 10s
  endpoints:
    - path: /health
      method: GET
      expected_status: 200
  
  checks:
    cuda:
      enabled: true
      threshold_memory_mb: 1000
    system_memory:
      enabled: true
      threshold_percent: 90
    disk_space:
      enabled: true
      threshold_percent: 85
```

### Docker Health Checks

```dockerfile
# Built into Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hd_compute; print('healthy')" || exit 1
```

## Metrics Collection

### Prometheus Metrics

HD-Compute-Toolkit exposes metrics in Prometheus format:

```python
from hd_compute.monitoring import MetricsCollector

# Initialize metrics collector
metrics = MetricsCollector()

# Core HDC operation metrics
metrics.record_operation_duration('bundle', duration_ms)
metrics.record_memory_usage('hypervector_storage', memory_bytes)
metrics.record_device_utilization('cuda', gpu_utilization_percent)

# Custom business metrics
metrics.increment_counter('operations_total', labels={'operation': 'bundle'})
metrics.set_gauge('active_hypervectors', count)
```

### Available Metrics

#### System Metrics
- `hdc_memory_usage_bytes` - Memory consumption by component
- `hdc_cpu_usage_percent` - CPU utilization
- `hdc_gpu_memory_usage_bytes` - GPU memory usage (if available)
- `hdc_gpu_utilization_percent` - GPU utilization

#### Operation Metrics
- `hdc_operation_duration_seconds` - Operation execution time
- `hdc_operation_total` - Total operations by type
- `hdc_operation_errors_total` - Failed operations by type
- `hdc_hypervector_operations_total` - Hypervector operations count

#### Performance Metrics
- `hdc_throughput_ops_per_second` - Operations throughput
- `hdc_latency_percentiles` - Latency percentiles (50th, 95th, 99th)
- `hdc_batch_size_distribution` - Batch size histogram

### Metrics Exposition

```python
# Expose metrics endpoint
from prometheus_client import start_http_server, generate_latest

# Start metrics server
start_http_server(8080)

# Or integrate with web framework
@app.route('/metrics')
def metrics():
    return generate_latest()
```

## Structured Logging

### Logging Configuration

```python
# hd_compute/logging.py
import structlog
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '%(message)s',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/hdc.log',
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'DEBUG'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Structured Log Messages

```python
import structlog

logger = structlog.get_logger()

# Operation logging
logger.info(
    "hypervector_operation_completed",
    operation="bundle",
    dimension=10000,
    input_count=100,
    duration_ms=15.2,
    device="cuda:0"
)

# Error logging
logger.error(
    "cuda_memory_allocation_failed",
    operation="random_hv",
    dimension=32000,
    requested_memory_mb=128,
    available_memory_mb=64,
    error_code="CUDA_OUT_OF_MEMORY"
)

# Performance logging
logger.debug(
    "performance_measurement",
    operation="bind",
    batch_size=1000,
    throughput_ops_per_sec=15000,
    memory_usage_mb=45
)
```

### Log Correlation

```python
# Add correlation IDs for request tracking
import uuid
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id')

def log_with_correlation(operation: str, **kwargs):
    logger.info(
        operation,
        correlation_id=correlation_id.get(str(uuid.uuid4())),
        **kwargs
    )
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize

# Initialize tracing
tracer = trace.get_tracer(__name__)

# Trace HDC operations
@tracer.start_as_current_span("hdc_bundle_operation")
def bundle_hypervectors(hypervectors):
    span = trace.get_current_span()
    span.set_attribute("hdc.operation", "bundle")
    span.set_attribute("hdc.input_count", len(hypervectors))
    span.set_attribute("hdc.dimension", hypervectors[0].shape[0])
    
    result = perform_bundle(hypervectors)
    
    span.set_attribute("hdc.success", True)
    return result
```

### Jaeger Configuration

```yaml
# docker-compose-monitoring.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "16686:16686"
    - "14268:14268"
  environment:
    - COLLECTOR_ZIPKIN_HTTP_PORT=9411
```

## Performance Monitoring

### Real-time Performance Dashboard

```python
# Performance monitoring dashboard
from hd_compute.monitoring import PerformanceDashboard

dashboard = PerformanceDashboard()

# Monitor key performance indicators
dashboard.add_metric("Throughput (ops/sec)", "hdc_throughput_ops_per_second")
dashboard.add_metric("Latency p95 (ms)", "hdc_latency_percentiles{quantile='0.95'}")
dashboard.add_metric("Memory Usage (MB)", "hdc_memory_usage_bytes / 1024 / 1024")
dashboard.add_metric("GPU Utilization (%)", "hdc_gpu_utilization_percent")

# Real-time alerting
dashboard.add_alert(
    name="high_latency",
    condition="hdc_latency_percentiles{quantile='0.95'} > 100",
    severity="warning"
)
```

### Benchmark Tracking

```python
# Continuous performance monitoring
from hd_compute.benchmarks import ContinuousBenchmark

benchmark = ContinuousBenchmark()

# Run background performance tests
benchmark.schedule_test(
    test_name="bundle_performance",
    interval="1h",
    alert_threshold_degradation=0.1  # Alert if 10% slower
)

# Store results for trend analysis
benchmark.store_results_to_database(connection_string="postgresql://...")
```

## Alerting

### Alert Rules

```yaml
# alerts/hdc-alerts.yml
groups:
  - name: hdc.rules
    rules:
      # High latency alert
      - alert: HDC_HighLatency
        expr: hdc_latency_percentiles{quantile="0.95"} > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "HD-Compute operations experiencing high latency"
          description: "95th percentile latency is {{ $value }}ms"

      # Memory usage alert
      - alert: HDC_HighMemoryUsage
        expr: (hdc_memory_usage_bytes / 1024 / 1024) > 8192
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "HD-Compute high memory usage"
          description: "Memory usage is {{ $value }}MB"

      # GPU utilization alert
      - alert: HDC_LowGPUUtilization
        expr: hdc_gpu_utilization_percent < 20
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "GPU underutilized"
          description: "GPU utilization is only {{ $value }}%"

      # Error rate alert
      - alert: HDC_HighErrorRate
        expr: rate(hdc_operation_errors_total[5m]) / rate(hdc_operation_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in HD-Compute operations"
          description: "Error rate is {{ $value | humanizePercentage }}"
```

### Notification Channels

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'hdc-alerts'

receivers:
  - name: 'hdc-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#hdc-alerts'
        title: 'HD-Compute-Toolkit Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    
    email_configs:
      - to: 'team@example.com'
        subject: 'HD-Compute Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

## Monitoring Stack Deployment

### Docker Compose Monitoring Stack

```yaml
# docker-compose-monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts:/etc/prometheus/alerts
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

volumes:
  grafana-storage:
```

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'hdc-toolkit'
    static_configs:
      - targets: ['hdc-toolkit:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "HD-Compute-Toolkit Dashboard",
    "panels": [
      {
        "title": "Operations per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(hdc_operation_total[1m])",
            "legendFormat": "{{ operation }}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "hdc_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "{{ component }}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "hdc_gpu_utilization_percent",
            "legendFormat": "GPU {{ device }}"
          }
        ]
      }
    ]
  }
}
```

## Log Aggregation

### ELK Stack Integration

```yaml
# docker-compose-logging.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false
  ports:
    - "9200:9200"

logstash:
  image: docker.elastic.co/logstash/logstash:8.0.0
  volumes:
    - ./logging/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  ports:
    - "5044:5044"

kibana:
  image: docker.elastic.co/kibana/kibana:8.0.0
  ports:
    - "5601:5601"
  environment:
    - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

### Log Shipping Configuration

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /app/logs/*.log
    json.keys_under_root: true
    json.message_key: message

output.logstash:
  hosts: ["logstash:5044"]

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

## Best Practices

### Monitoring Guidelines

1. **Golden Signals**: Monitor latency, traffic, errors, and saturation
2. **SLI/SLO**: Define Service Level Indicators and Objectives
3. **Alert Fatigue**: Implement smart alerting to reduce noise
4. **Documentation**: Keep runbooks updated for common issues

### Performance Baselines

```python
# Establish performance baselines
PERFORMANCE_BASELINES = {
    'bundle_10k_vectors': {
        'max_latency_ms': 50,
        'min_throughput_ops_sec': 1000
    },
    'bind_32k_dimension': {
        'max_latency_ms': 10,
        'max_memory_mb': 256
    },
    'similarity_batch_1000': {
        'max_latency_ms': 100,
        'min_throughput_ops_sec': 500
    }
}
```

### Cost Optimization

1. **Resource Right-sizing**: Monitor resource usage and adjust allocations
2. **Auto-scaling**: Implement horizontal pod autoscaling based on metrics
3. **Spot Instances**: Use spot instances for non-critical workloads
4. **Storage Optimization**: Implement log retention policies

This comprehensive monitoring setup ensures HD-Compute-Toolkit operates reliably in production with full observability into performance, errors, and resource utilization.