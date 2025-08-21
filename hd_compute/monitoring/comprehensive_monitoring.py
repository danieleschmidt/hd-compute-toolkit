"""
Comprehensive Monitoring and Telemetry System
============================================

Advanced monitoring, metrics collection, and observability for HDC research operations
with real-time dashboards, alerting, and performance analytics.
"""

import time
import threading
import logging
import json
import os
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    message: str
    active: bool = False
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None


class MetricsCollector:
    """Collects and stores metrics with time-series data."""
    
    def __init__(self, max_points: int = 100000):
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Per metric storage
        self.metric_metadata = {}
        self.lock = threading.RLock()
        
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (monotonically increasing)."""
        self._record_metric(name, value, MetricType.COUNTER, labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (can go up or down)."""
        self._record_metric(name, value, MetricType.GAUGE, labels or {})
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self._record_metric(name, duration, MetricType.TIMER, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str]) -> None:
        """Internal method to record a metric."""
        with self.lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels,
                metric_type=metric_type
            )
            
            self.metrics[name].append(metric_point)
            self.metric_metadata[name] = {
                'type': metric_type,
                'last_updated': metric_point.timestamp,
                'total_points': len(self.metrics[name])
            }
    
    def get_metric_summary(self, name: str, time_window: float = 3600) -> Dict[str, Any]:
        """Get summary statistics for a metric within time window."""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - time_window
            
            recent_points = [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return {'name': name, 'points': 0, 'time_window': time_window}
            
            values = [point.value for point in recent_points]
            
            return {
                'name': name,
                'type': recent_points[0].metric_type.value,
                'points': len(recent_points),
                'time_window': time_window,
                'latest_value': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'sum': sum(values),
                'rate_per_second': len(recent_points) / time_window if time_window > 0 else 0
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all metrics."""
        return {name: self.get_metric_summary(name) for name in self.metrics.keys()}
    
    def timer_context(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels or {})


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)


class AlertManager:
    """Manages alerts and notifications based on metric thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_handlers = []
        self.lock = threading.RLock()
    
    def add_alert(self, alert: Alert) -> None:
        """Add an alert rule."""
        with self.lock:
            self.alerts[alert.name] = alert
            logging.info(f"Added alert rule: {alert.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def check_alerts(self) -> List[Alert]:
        """Check all alert conditions and trigger notifications."""
        triggered_alerts = []
        
        with self.lock:
            for alert_name, alert in self.alerts.items():
                if self._evaluate_alert_condition(alert):
                    if not alert.active:
                        # Alert triggered
                        alert.active = True
                        alert.triggered_at = time.time()
                        triggered_alerts.append(alert)
                        
                        self.alert_history.append({
                            'alert_name': alert_name,
                            'action': 'triggered',
                            'timestamp': alert.triggered_at,
                            'severity': alert.severity.value
                        })
                        
                        # Send notifications
                        for handler in self.notification_handlers:
                            try:
                                handler(alert)
                            except Exception as e:
                                logging.error(f"Notification handler failed: {e}")
                
                elif alert.active:
                    # Alert resolved
                    alert.active = False
                    alert.resolved_at = time.time()
                    
                    self.alert_history.append({
                        'alert_name': alert_name,
                        'action': 'resolved',
                        'timestamp': alert.resolved_at,
                        'severity': alert.severity.value
                    })
        
        return triggered_alerts
    
    def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate alert condition based on metrics."""
        try:
            # Simple threshold-based conditions
            if alert.condition.startswith("metric"):
                parts = alert.condition.split()
                if len(parts) >= 4:  # metric name operator threshold
                    metric_name = parts[1]
                    operator = parts[2]
                    
                    summary = self.metrics_collector.get_metric_summary(metric_name, time_window=300)
                    if not summary or 'latest_value' not in summary:
                        return False
                    
                    current_value = summary['latest_value']
                    
                    if operator == '>':
                        return current_value > alert.threshold
                    elif operator == '<':
                        return current_value < alert.threshold
                    elif operator == '>=':
                        return current_value >= alert.threshold
                    elif operator == '<=':
                        return current_value <= alert.threshold
                    elif operator == '==':
                        return abs(current_value - alert.threshold) < 1e-6
            
            return False
            
        except Exception as e:
            logging.error(f"Error evaluating alert condition for {alert.name}: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.active]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            event for event in self.alert_history
            if event['timestamp'] >= cutoff_time
        ]


class PerformanceProfiler:
    """Profiles performance of HDC operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self.profile_results = defaultdict(list)
        self.lock = threading.RLock()
    
    def start_profile(self, operation_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.active_profiles[profile_id] = {
                'operation': operation_name,
                'start_time': time.time(),
                'context': context or {},
                'memory_start': self._get_memory_usage(),
                'cpu_start': self._get_cpu_usage()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str, result_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """End profiling and record results."""
        with self.lock:
            if profile_id not in self.active_profiles:
                return {}
            
            profile_data = self.active_profiles.pop(profile_id)
            end_time = time.time()
            
            duration = end_time - profile_data['start_time']
            memory_end = self._get_memory_usage()
            cpu_end = self._get_cpu_usage()
            
            profile_result = {
                'operation': profile_data['operation'],
                'duration': duration,
                'memory_delta': memory_end - profile_data['memory_start'],
                'cpu_delta': cpu_end - profile_data['cpu_start'],
                'start_context': profile_data['context'],
                'end_context': result_context or {},
                'timestamp': end_time
            }
            
            self.profile_results[profile_data['operation']].append(profile_result)
            
            # Record metrics
            self.metrics_collector.record_timer(
                f"operation_duration_{profile_data['operation']}", 
                duration,
                {'operation': profile_data['operation']}
            )
            
            self.metrics_collector.record_gauge(
                f"memory_usage_{profile_data['operation']}",
                profile_result['memory_delta'],
                {'operation': profile_data['operation']}
            )
            
            return profile_result
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                profile_id = self.start_profile(operation_name, {
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                
                try:
                    result = func(*args, **kwargs)
                    
                    result_context = {}
                    if hasattr(result, 'shape'):
                        result_context['result_shape'] = result.shape
                    elif hasattr(result, '__len__'):
                        result_context['result_length'] = len(result)
                    
                    self.end_profile(profile_id, result_context)
                    return result
                    
                except Exception as e:
                    self.end_profile(profile_id, {'error': str(e)})
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        with self.lock:
            results = self.profile_results.get(operation_name, [])
            
            if not results:
                return {'operation': operation_name, 'profile_count': 0}
            
            durations = [r['duration'] for r in results]
            memory_deltas = [r['memory_delta'] for r in results]
            
            return {
                'operation': operation_name,
                'profile_count': len(results),
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'avg_memory_delta': np.mean(memory_deltas),
                'total_memory_used': np.sum([max(0, delta) for delta in memory_deltas]),
                'last_run': results[-1]['timestamp']
            }


class SystemMonitor:
    """Monitors system resources and HDC-specific metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector, collection_interval: float = 10.0):
        self.metrics_collector = metrics_collector
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
        self.custom_collectors = []
    
    def add_custom_collector(self, collector_func: Callable[[], Dict[str, float]]) -> None:
        """Add custom metric collector function."""
        self.custom_collectors.append(collector_func)
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.running:
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    self._collect_system_metrics()
                    self._collect_custom_metrics()
                    time.sleep(self.collection_interval)
                except Exception as e:
                    logging.error(f"System monitoring error: {e}")
                    time.sleep(self.collection_interval)
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
        logging.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logging.info("System monitoring stopped")
    
    def _collect_system_metrics(self) -> None:
        """Collect standard system metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_collector.record_gauge('system_cpu_percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_gauge('system_memory_percent', memory.percent)
            self.metrics_collector.record_gauge('system_memory_available_gb', memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_gauge('system_disk_percent', disk.percent)
            self.metrics_collector.record_gauge('system_disk_free_gb', disk.free / (1024**3))
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.record_gauge('process_memory_mb', process.memory_info().rss / (1024**2))
            self.metrics_collector.record_gauge('process_cpu_percent', process.cpu_percent())
            
        except ImportError:
            # Fallback metrics without psutil
            self.metrics_collector.record_gauge('system_cpu_percent', 0.0)
            self.metrics_collector.record_gauge('system_memory_percent', 0.0)
    
    def _collect_custom_metrics(self) -> None:
        """Collect custom metrics from registered collectors."""
        for collector_func in self.custom_collectors:
            try:
                metrics = collector_func()
                for metric_name, value in metrics.items():
                    self.metrics_collector.record_gauge(metric_name, value)
            except Exception as e:
                logging.error(f"Custom metric collector failed: {e}")


class MonitoringDashboard:
    """Provides dashboard data for monitoring visualization."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager,
                 profiler: PerformanceProfiler):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.profiler = profiler
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'timestamp': time.time(),
            'system_overview': self._get_system_overview(),
            'performance_summary': self._get_performance_summary(),
            'alert_summary': self._get_alert_summary(),
            'top_metrics': self._get_top_metrics(),
            'recent_operations': self._get_recent_operations()
        }
    
    def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics."""
        metrics = self.metrics_collector.get_all_metrics()
        
        system_metrics = {
            name: summary for name, summary in metrics.items()
            if name.startswith('system_') or name.startswith('process_')
        }
        
        return {
            'total_metrics': len(metrics),
            'system_health': self._calculate_system_health(system_metrics),
            'uptime_hours': self._get_uptime_hours(),
            'system_metrics': system_metrics
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        operation_stats = {}
        
        for operation_name in self.profiler.profile_results.keys():
            operation_stats[operation_name] = self.profiler.get_operation_stats(operation_name)
        
        return {
            'total_operations': len(operation_stats),
            'operation_stats': operation_stats,
            'slowest_operations': self._get_slowest_operations(operation_stats),
            'memory_intensive_operations': self._get_memory_intensive_operations(operation_stats)
        }
    
    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_history = self.alert_manager.get_alert_history(hours=24)
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'active_alerts_count': len(active_alerts),
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'severity_counts': dict(severity_counts),
            'alerts_24h': len(alert_history),
            'recent_alerts': alert_history[-10:]  # Last 10 alerts
        }
    
    def _get_top_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top metrics by activity."""
        metrics = self.metrics_collector.get_all_metrics()
        
        # Sort by points (activity)
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1].get('points', 0),
            reverse=True
        )
        
        return [
            {'name': name, **summary}
            for name, summary in sorted_metrics[:limit]
        ]
    
    def _get_recent_operations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent operation profiles."""
        all_profiles = []
        
        for operation_name, profiles in self.profiler.profile_results.items():
            for profile in profiles[-5:]:  # Last 5 per operation
                profile_copy = profile.copy()
                profile_copy['operation'] = operation_name
                all_profiles.append(profile_copy)
        
        # Sort by timestamp, most recent first
        all_profiles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_profiles[:limit]
    
    def _calculate_system_health(self, system_metrics: Dict[str, Any]) -> str:
        """Calculate overall system health score."""
        if not system_metrics:
            return 'unknown'
        
        health_issues = 0
        
        # Check CPU usage
        cpu_metric = system_metrics.get('system_cpu_percent', {})
        if cpu_metric.get('latest_value', 0) > 80:
            health_issues += 1
        
        # Check memory usage
        memory_metric = system_metrics.get('system_memory_percent', {})
        if memory_metric.get('latest_value', 0) > 85:
            health_issues += 1
        
        # Check disk usage
        disk_metric = system_metrics.get('system_disk_percent', {})
        if disk_metric.get('latest_value', 0) > 90:
            health_issues += 1
        
        if health_issues == 0:
            return 'healthy'
        elif health_issues == 1:
            return 'warning'
        else:
            return 'critical'
    
    def _get_uptime_hours(self) -> float:
        """Get system uptime in hours (simplified)."""
        # This is a placeholder - in a real system, track actual start time
        return 1.0
    
    def _get_slowest_operations(self, operation_stats: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Get slowest operations by average duration."""
        operations_with_duration = [
            (name, stats) for name, stats in operation_stats.items()
            if 'avg_duration' in stats
        ]
        
        sorted_operations = sorted(
            operations_with_duration,
            key=lambda x: x[1]['avg_duration'],
            reverse=True
        )
        
        return [
            {'operation': name, **stats}
            for name, stats in sorted_operations[:limit]
        ]
    
    def _get_memory_intensive_operations(self, operation_stats: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Get most memory-intensive operations."""
        operations_with_memory = [
            (name, stats) for name, stats in operation_stats.items()
            if 'total_memory_used' in stats
        ]
        
        sorted_operations = sorted(
            operations_with_memory,
            key=lambda x: x[1]['total_memory_used'],
            reverse=True
        )
        
        return [
            {'operation': name, **stats}
            for name, stats in sorted_operations[:limit]
        ]


class ComprehensiveMonitoringSystem:
    """Complete monitoring system combining all components."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.profiler = PerformanceProfiler(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector, collection_interval)
        self.dashboard = MonitoringDashboard(self.metrics_collector, self.alert_manager, self.profiler)
        
        # Initialize default alerts
        self._initialize_default_alerts()
        
        # Add default notification handler
        self.alert_manager.add_notification_handler(self._default_alert_handler)
    
    def _initialize_default_alerts(self) -> None:
        """Initialize default system alerts."""
        alerts = [
            Alert(
                name="high_cpu_usage",
                condition="metric system_cpu_percent > 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                message="CPU usage is above 80%"
            ),
            Alert(
                name="high_memory_usage",
                condition="metric system_memory_percent > 85",
                threshold=85.0,
                severity=AlertSeverity.ERROR,
                message="Memory usage is above 85%"
            ),
            Alert(
                name="low_disk_space",
                condition="metric system_disk_percent > 90",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                message="Disk usage is above 90%"
            )
        ]
        
        for alert in alerts:
            self.alert_manager.add_alert(alert)
    
    def _default_alert_handler(self, alert: Alert) -> None:
        """Default alert notification handler."""
        logging.warning(f"ALERT TRIGGERED: {alert.name} - {alert.message}")
    
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        
        # Start alert checking thread
        def alert_check_loop():
            while self.system_monitor.running:
                try:
                    self.alert_manager.check_alerts()
                    time.sleep(30)  # Check alerts every 30 seconds
                except Exception as e:
                    logging.error(f"Alert checking error: {e}")
                    time.sleep(30)
        
        alert_thread = threading.Thread(target=alert_check_loop, daemon=True)
        alert_thread.start()
        
        logging.info("Comprehensive monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        logging.info("Comprehensive monitoring system stopped")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'dashboard_data': self.dashboard.get_dashboard_data(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'total_metrics': len(self.metrics_collector.get_all_metrics()),
            'monitoring_status': 'active' if self.system_monitor.running else 'stopped'
        }
    
    def add_custom_metric_collector(self, collector_func: Callable[[], Dict[str, float]]) -> None:
        """Add custom metric collector."""
        self.system_monitor.add_custom_collector(collector_func)
    
    def create_operation_monitor(self, operation_name: str):
        """Create monitoring decorator for HDC operations."""
        return self.profiler.profile_operation(operation_name)


# Global monitoring system
global_monitoring = ComprehensiveMonitoringSystem()


# Convenient decorators
def monitor_hdc_operation(operation_name: str):
    """Decorator for monitoring HDC operations."""
    return global_monitoring.create_operation_monitor(operation_name)


def track_metric(metric_name: str, metric_type: MetricType = MetricType.COUNTER):
    """Decorator for tracking metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success metric
                if metric_type == MetricType.TIMER:
                    duration = time.time() - start_time
                    global_monitoring.metrics_collector.record_timer(
                        f"{metric_name}_duration", duration
                    )
                elif metric_type == MetricType.COUNTER:
                    global_monitoring.metrics_collector.record_counter(
                        f"{metric_name}_success"
                    )
                
                return result
                
            except Exception as e:
                # Record error metric
                global_monitoring.metrics_collector.record_counter(
                    f"{metric_name}_error", 1.0, {'error_type': type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize monitoring system
    monitoring = ComprehensiveMonitoringSystem()
    monitoring.start_monitoring()
    
    # Example monitored operation
    @monitor_hdc_operation('example_hdc_computation')
    def example_computation(data_size: int):
        """Example HDC computation for monitoring."""
        import numpy as np
        data = np.random.normal(0, 1, data_size)
        return np.mean(data ** 2)
    
    # Run some operations
    for i in range(5):
        result = example_computation(1000 + i * 500)
        monitoring.metrics_collector.record_gauge('computation_result', result)
        time.sleep(1)
    
    # Get monitoring summary
    summary = monitoring.get_monitoring_summary()
    print(f"Monitoring summary: {summary['total_metrics']} metrics collected")
    print(f"Active alerts: {summary['active_alerts']}")
    
    # Stop monitoring
    monitoring.stop_monitoring()