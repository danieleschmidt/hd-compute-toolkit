#!/usr/bin/env python3
"""
Advanced Performance Optimization System for HD-Compute-Toolkit
==============================================================

This module provides intelligent performance optimization, real-time monitoring,
and adaptive tuning capabilities for hyperdimensional computing operations.

Features:
- Real-time performance monitoring with statistical analysis
- Intelligent caching with LRU and adaptive strategies
- Auto-scaling and load balancing for distributed operations
- Performance prediction and optimization recommendations
- Resource usage optimization and memory management
- Comprehensive benchmarking and profiling tools
"""

import numpy as np
import time
import threading
import queue
import json
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import deque, defaultdict, OrderedDict
from abc import ABC, abstractmethod
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation_type: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    cache_hit_rate: float
    error_rate: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_type': self.operation_type,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'throughput': self.throughput,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate,
            'timestamp': self.timestamp
        }


class IntelligentCache:
    """Intelligent caching system with adaptive replacement policies."""
    
    def __init__(self, max_size: int = 1000, strategy: str = 'adaptive'):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(float)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Adaptive parameters
        self.performance_history = deque(maxlen=100)
        self.adaptive_threshold = 0.7
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent replacement."""
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                
                return value
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
                self.cache[key] = value
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    self._evict_item()
                
                self.cache[key] = value
                self.access_counts[key] = 1
                self.access_times[key] = time.time()
    
    def _evict_item(self) -> None:
        """Evict item based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == 'lru':
            # Least Recently Used
            self.cache.popitem(last=False)
        
        elif self.strategy == 'lfu':
            # Least Frequently Used
            min_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self.cache.pop(min_key)
            del self.access_counts[min_key]
            del self.access_times[min_key]
        
        elif self.strategy == 'adaptive':
            # Adaptive replacement based on performance
            current_hit_rate = self.get_hit_rate()
            
            if current_hit_rate > self.adaptive_threshold:
                # Good hit rate - use LRU
                self.cache.popitem(last=False)
            else:
                # Poor hit rate - use LFU
                min_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
                self.cache.pop(min_key, None)
                self.access_counts.pop(min_key, None)
                self.access_times.pop(min_key, None)
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = self.hit_count + self.miss_count
        return self.hit_count / total_accesses if total_accesses > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hit_rate': self.get_hit_rate(),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'strategy': self.strategy
        }
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class RealTimeMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
        self.alerts = []
        self.thresholds = {
            'execution_time': 1.0,  # seconds
            'memory_usage': 80.0,   # percentage
            'cpu_usage': 90.0,      # percentage
            'error_rate': 0.05      # 5%
        }
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self.lock:
            self.metrics_buffer.append(metrics)
            self._check_alerts(metrics)
    
    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Analyze recent performance
                analysis = self._analyze_recent_performance()
                
                # Check for anomalies
                anomalies = self._detect_anomalies()
                
                if anomalies:
                    self._handle_anomalies(anomalies)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_rss': process.memory_info().rss / (1024 * 1024),  # MB
                'num_threads': process.num_threads(),
                'system_cpu': psutil.cpu_percent(),
                'system_memory': psutil.virtual_memory().percent
            }
        except Exception:
            return {}
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance metrics."""
        if len(self.metrics_buffer) < 10:
            return {}
        
        recent_metrics = list(self.metrics_buffer)[-50:]  # Last 50 operations
        
        # Calculate statistics
        execution_times = [m.execution_time for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        throughput = [m.throughput for m in recent_metrics]
        
        return {
            'avg_execution_time': np.mean(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95),
            'avg_memory_usage': np.mean(memory_usage),
            'avg_throughput': np.mean(throughput),
            'operations_per_second': len(recent_metrics) / (recent_metrics[-1].timestamp - recent_metrics[0].timestamp)
        }
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        if len(self.metrics_buffer) < 20:
            return []
        
        anomalies = []
        recent_metrics = list(self.metrics_buffer)[-20:]
        
        # Statistical anomaly detection
        execution_times = [m.execution_time for m in recent_metrics]
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        
        # Detect outliers (> 3 standard deviations)
        for metrics in recent_metrics[-5:]:  # Check last 5 operations
            if abs(metrics.execution_time - mean_time) > 3 * std_time:
                anomalies.append({
                    'type': 'execution_time_outlier',
                    'value': metrics.execution_time,
                    'expected_range': (mean_time - 2*std_time, mean_time + 2*std_time),
                    'timestamp': metrics.timestamp
                })
        
        return anomalies
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        if metrics.execution_time > self.thresholds['execution_time']:
            alerts.append({
                'type': 'high_execution_time',
                'value': metrics.execution_time,
                'threshold': self.thresholds['execution_time'],
                'timestamp': metrics.timestamp
            })
        
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.thresholds['memory_usage'],
                'timestamp': metrics.timestamp
            })
        
        if metrics.error_rate > self.thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'value': metrics.error_rate,
                'threshold': self.thresholds['error_rate'],
                'timestamp': metrics.timestamp
            })
        
        self.alerts.extend(alerts)
        
        # Keep only recent alerts
        current_time = time.time()
        self.alerts = [a for a in self.alerts if current_time - a['timestamp'] < 3600]  # 1 hour
    
    def _handle_anomalies(self, anomalies: List[Dict[str, Any]]) -> None:
        """Handle detected anomalies."""
        for anomaly in anomalies:
            print(f"⚠️  Anomaly detected: {anomaly['type']} - {anomaly['value']}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_buffer:
            return {'error': 'No performance data available'}
        
        metrics_list = list(self.metrics_buffer)
        
        # Overall statistics
        execution_times = [m.execution_time for m in metrics_list]
        memory_usage = [m.memory_usage for m in metrics_list]
        throughput_values = [m.throughput for m in metrics_list]
        
        # Performance by operation type
        by_operation = defaultdict(list)
        for m in metrics_list:
            by_operation[m.operation_type].append(m)
        
        operation_stats = {}
        for op_type, op_metrics in by_operation.items():
            op_times = [m.execution_time for m in op_metrics]
            operation_stats[op_type] = {
                'count': len(op_metrics),
                'avg_time': np.mean(op_times),
                'p95_time': np.percentile(op_times, 95),
                'total_time': np.sum(op_times)
            }
        
        return {
            'total_operations': len(metrics_list),
            'time_span': metrics_list[-1].timestamp - metrics_list[0].timestamp,
            'overall_stats': {
                'avg_execution_time': np.mean(execution_times),
                'p50_execution_time': np.percentile(execution_times, 50),
                'p95_execution_time': np.percentile(execution_times, 95),
                'p99_execution_time': np.percentile(execution_times, 99),
                'avg_memory_usage': np.mean(memory_usage),
                'avg_throughput': np.mean(throughput_values),
                'total_throughput': np.sum(throughput_values)
            },
            'operation_breakdown': operation_stats,
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'system_health': self._collect_system_metrics()
        }


class AdaptiveOptimizer:
    """Adaptive optimization engine that learns and improves performance."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.optimization_history = deque(maxlen=1000)
        self.learned_optimizations = {}
        self.cache = IntelligentCache(max_size=2000, strategy='adaptive')
        self.monitor = RealTimeMonitor()
        
        # Performance prediction model (simple linear regression)
        self.prediction_model = {
            'weights': defaultdict(float),
            'bias': 0.0,
            'learning_rate': 0.01
        }
        
        # Resource management
        self.memory_threshold = 80.0  # percentage
        self.cpu_threshold = 90.0     # percentage
        
    def optimize_operation(self, operation_func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Optimize operation execution with intelligent caching and resource management."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        # Generate cache key
        cache_key = self._generate_cache_key(operation_func.__name__, args, kwargs)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            execution_time = time.time() - start_time
            metrics = PerformanceMetrics(
                operation_type=operation_func.__name__,
                execution_time=execution_time,
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(),
                throughput=1.0 / execution_time if execution_time > 0 else 0.0,
                cache_hit_rate=self.cache.get_hit_rate(),
                error_rate=0.0,
                timestamp=time.time()
            )
            self.monitor.record_metrics(metrics)
            return cached_result, metrics
        
        # Execute operation with optimization
        try:
            # Apply learned optimizations
            optimized_kwargs = self._apply_optimizations(operation_func.__name__, kwargs)
            
            # Resource-aware execution
            if self._should_throttle():
                time.sleep(0.01)  # Small delay to reduce resource pressure
            
            # Execute operation
            result = operation_func(*args, **optimized_kwargs)
            
            # Cache result if beneficial
            if self._should_cache(operation_func.__name__, execution_time):
                self.cache.put(cache_key, result)
            
            error_rate = 0.0
            
        except Exception as e:
            result = None
            error_rate = 1.0
            print(f"Operation failed: {e}")
        
        # Calculate metrics
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            operation_type=operation_func.__name__,
            execution_time=execution_time,
            memory_usage=max(end_memory - start_memory, 0),
            cpu_usage=max(end_cpu - start_cpu, 0),
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            cache_hit_rate=self.cache.get_hit_rate(),
            error_rate=error_rate,
            timestamp=time.time()
        )
        
        # Record and learn from performance
        self.monitor.record_metrics(metrics)
        self._update_optimization_model(operation_func.__name__, kwargs, metrics)
        
        return result, metrics
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate deterministic cache key."""
        # Create hash of function name and arguments
        key_parts = [func_name]
        
        # Hash numpy arrays and other complex types
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(f"array_{arg.shape}_{hash(arg.tobytes())}")
            else:
                key_parts.append(str(hash(str(arg))))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_parts.append(f"{k}_array_{v.shape}_{hash(v.tobytes())}")
            else:
                key_parts.append(f"{k}_{hash(str(v))}")
        
        return "_".join(key_parts)[:100]  # Limit key length
    
    def _apply_optimizations(self, func_name: str, kwargs: Dict) -> Dict:
        """Apply learned optimizations to operation parameters."""
        optimized_kwargs = kwargs.copy()
        
        if func_name in self.learned_optimizations:
            optimizations = self.learned_optimizations[func_name]
            
            # Apply parameter optimizations
            for param, optimal_value in optimizations.get('parameters', {}).items():
                if param in optimized_kwargs:
                    # Blend current value with optimal value
                    current_value = optimized_kwargs[param]
                    if isinstance(current_value, (int, float)):
                        blend_factor = 0.3  # 30% optimization influence
                        optimized_kwargs[param] = (1 - blend_factor) * current_value + blend_factor * optimal_value
        
        return optimized_kwargs
    
    def _should_cache(self, func_name: str, execution_time: float) -> bool:
        """Determine if result should be cached."""
        # Cache expensive operations or frequently used operations
        return execution_time > 0.1 or self.cache.get_hit_rate() > 0.5
    
    def _should_throttle(self) -> bool:
        """Determine if execution should be throttled due to resource constraints."""
        try:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            return memory_usage > self.memory_threshold or cpu_usage > self.cpu_threshold
        except Exception:
            return False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            return psutil.Process().memory_percent()
        except Exception:
            return 0.0
    
    def _update_optimization_model(self, func_name: str, kwargs: Dict, metrics: PerformanceMetrics) -> None:
        """Update optimization model with new performance data."""
        self.optimization_history.append({
            'func_name': func_name,
            'kwargs': kwargs,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Learn optimal parameters for this function
        if func_name not in self.learned_optimizations:
            self.learned_optimizations[func_name] = {
                'parameters': {},
                'best_performance': float('inf'),
                'performance_history': []
            }
        
        func_opts = self.learned_optimizations[func_name]
        func_opts['performance_history'].append(metrics.execution_time)
        
        # Update best performance and parameters
        if metrics.execution_time < func_opts['best_performance'] and metrics.error_rate == 0:
            func_opts['best_performance'] = metrics.execution_time
            
            # Store optimal parameters
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    func_opts['parameters'][key] = value
    
    def predict_performance(self, func_name: str, estimated_complexity: float = 1.0) -> Dict[str, float]:
        """Predict performance metrics for an operation."""
        if func_name in self.learned_optimizations:
            history = self.learned_optimizations[func_name]['performance_history']
            if len(history) >= 5:
                # Use historical average with complexity scaling
                base_time = np.mean(history[-10:])  # Last 10 operations
                predicted_time = base_time * estimated_complexity
                
                return {
                    'predicted_execution_time': predicted_time,
                    'confidence': min(len(history) / 50.0, 1.0),  # Confidence based on sample size
                    'predicted_throughput': 1.0 / predicted_time if predicted_time > 0 else 0.0
                }
        
        # Default prediction for unknown operations
        return {
            'predicted_execution_time': 0.1 * estimated_complexity,
            'confidence': 0.1,
            'predicted_throughput': 10.0 / estimated_complexity if estimated_complexity > 0 else 0.0
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            'strategy': self.strategy.value,
            'cache_stats': self.cache.get_stats(),
            'learned_optimizations': {
                func: {
                    'best_performance': opt['best_performance'],
                    'parameter_count': len(opt['parameters']),
                    'history_length': len(opt['performance_history'])
                }
                for func, opt in self.learned_optimizations.items()
            },
            'total_operations_optimized': len(self.optimization_history),
            'performance_monitor': self.monitor.get_performance_report(),
            'resource_status': {
                'memory_threshold': self.memory_threshold,
                'cpu_threshold': self.cpu_threshold,
                'current_memory': psutil.virtual_memory().percent,
                'current_cpu': psutil.cpu_percent()
            }
        }


class DistributedOptimizer:
    """Distributed optimization and load balancing system."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, 8))
        
        self.load_balancer = LoadBalancer()
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = deque(maxlen=1000)
        
        self.running = False
        self.worker_threads = []
        
    def start(self) -> None:
        """Start distributed optimization system."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(min(4, self.max_workers)):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def stop(self) -> None:
        """Stop distributed optimization system."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=2.0)
        
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
    
    def submit_task(self, func: Callable, *args, priority: int = 1, use_process: bool = False, **kwargs):
        """Submit task for distributed execution."""
        task_id = f"task_{time.time()}_{np.random.randint(1000)}"
        
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'use_process': use_process,
            'submitted_time': time.time()
        }
        
        self.task_queue.put((priority, time.time(), task))
        
        return task_id
    
    def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            try:
                # Get task from queue with timeout
                priority, submit_time, task = self.task_queue.get(timeout=1.0)
                
                # Execute task
                start_time = time.time()
                
                try:
                    if task['use_process']:
                        # CPU-intensive task - use process pool
                        future = self.process_pool.submit(task['func'], *task['args'], **task['kwargs'])
                        result = future.result(timeout=30.0)
                    else:
                        # I/O or memory-intensive task - use thread pool
                        future = self.thread_pool.submit(task['func'], *task['args'], **task['kwargs'])
                        result = future.result(timeout=30.0)
                    
                    execution_time = time.time() - start_time
                    
                    # Record completed task
                    self.completed_tasks.append({
                        'task_id': task['id'],
                        'worker_id': worker_id,
                        'execution_time': execution_time,
                        'queue_time': start_time - task['submitted_time'],
                        'success': True,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.completed_tasks.append({
                        'task_id': task['id'],
                        'worker_id': worker_id,
                        'execution_time': execution_time,
                        'queue_time': start_time - task['submitted_time'],
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get distributed system performance statistics."""
        if not self.completed_tasks:
            return {'status': 'no_tasks_completed'}
        
        completed = list(self.completed_tasks)
        successful_tasks = [t for t in completed if t['success']]
        
        execution_times = [t['execution_time'] for t in successful_tasks]
        queue_times = [t['queue_time'] for t in successful_tasks]
        
        return {
            'total_tasks': len(completed),
            'successful_tasks': len(successful_tasks),
            'success_rate': len(successful_tasks) / len(completed),
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'avg_queue_time': np.mean(queue_times) if queue_times else 0,
            'throughput': len(successful_tasks) / (completed[-1]['timestamp'] - completed[0]['timestamp']) if len(completed) > 1 else 0,
            'queue_size': self.task_queue.qsize(),
            'active_workers': len(self.worker_threads),
            'thread_pool_size': self.max_workers
        }


class LoadBalancer:
    """Intelligent load balancing for distributed operations."""
    
    def __init__(self):
        self.worker_loads = defaultdict(float)
        self.worker_performance = defaultdict(list)
        self.last_assignment = defaultdict(float)
        
    def select_worker(self, available_workers: List[str], task_complexity: float = 1.0) -> str:
        """Select optimal worker for task based on load and performance."""
        if not available_workers:
            return "default"
        
        if len(available_workers) == 1:
            return available_workers[0]
        
        # Calculate worker scores
        worker_scores = {}
        
        for worker in available_workers:
            # Current load (lower is better)
            load_score = 1.0 / (self.worker_loads[worker] + 1.0)
            
            # Performance history (higher is better)
            perf_history = self.worker_performance[worker]
            if perf_history:
                avg_perf = np.mean(perf_history[-10:])  # Last 10 tasks
                perf_score = 1.0 / (avg_perf + 0.1)
            else:
                perf_score = 1.0  # Neutral for new workers
            
            # Time since last assignment (higher is better for fairness)
            time_since_last = time.time() - self.last_assignment[worker]
            fairness_score = min(time_since_last / 10.0, 1.0)  # Normalize to 10 seconds
            
            # Combined score with weights
            worker_scores[worker] = (0.5 * load_score + 0.3 * perf_score + 0.2 * fairness_score)
        
        # Select worker with highest score
        best_worker = max(worker_scores.keys(), key=lambda w: worker_scores[w])
        
        # Update assignments
        self.last_assignment[best_worker] = time.time()
        self.worker_loads[best_worker] += task_complexity
        
        return best_worker
    
    def report_task_completion(self, worker: str, execution_time: float, task_complexity: float = 1.0) -> None:
        """Report task completion to update worker statistics."""
        # Update performance history
        self.worker_performance[worker].append(execution_time)
        
        # Keep only recent performance data
        if len(self.worker_performance[worker]) > 50:
            self.worker_performance[worker] = self.worker_performance[worker][-50:]
        
        # Reduce worker load
        self.worker_loads[worker] = max(0.0, self.worker_loads[worker] - task_complexity)


# High-level performance management system
class PerformanceManager:
    """High-level performance management and optimization coordinator."""
    
    def __init__(self, optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = optimization_strategy
        self.optimizer = AdaptiveOptimizer(optimization_strategy)
        self.distributed_optimizer = DistributedOptimizer()
        self.monitor = RealTimeMonitor()
        
        # Start systems
        self.monitor.start_monitoring()
        self.distributed_optimizer.start()
        
        self.performance_log = []
        
    def execute_optimized(self, operation_func: Callable, *args, distributed: bool = False, **kwargs):
        """Execute operation with full optimization stack."""
        if distributed and hasattr(operation_func, '__call__'):
            # Submit to distributed system
            task_id = self.distributed_optimizer.submit_task(
                operation_func, *args, **kwargs
            )
            return task_id
        else:
            # Execute with adaptive optimizer
            return self.optimizer.optimize_operation(operation_func, *args, **kwargs)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'strategy': self.strategy.value,
            'adaptive_optimizer': self.optimizer.get_optimization_report(),
            'distributed_system': self.distributed_optimizer.get_performance_stats(),
            'real_time_monitor': self.monitor.get_performance_report(),
            'system_resources': {
                'cpu_count': os.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze current performance
        cache_hit_rate = self.optimizer.cache.get_hit_rate()
        if cache_hit_rate < 0.5:
            recommendations.append("Consider increasing cache size or improving cache key generation")
        
        # Analyze resource usage
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            recommendations.append("High memory usage detected - consider memory optimization or scaling")
        
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 90:
            recommendations.append("High CPU usage detected - consider distributed processing or optimization")
        
        # Analyze distributed performance
        dist_stats = self.distributed_optimizer.get_performance_stats()
        if dist_stats.get('queue_size', 0) > 100:
            recommendations.append("High task queue size - consider adding more workers or optimizing tasks")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Shutdown performance management system."""
        self.monitor.stop_monitoring()
        self.distributed_optimizer.stop()


if __name__ == "__main__":
    print("🚀 Advanced Performance Optimization System")
    print("=" * 50)
    
    # Create performance manager
    manager = PerformanceManager(OptimizationStrategy.ADAPTIVE)
    
    # Example optimized operations
    def dummy_computation(size: int = 1000, complexity: float = 1.0) -> np.ndarray:
        """Dummy computation for testing."""
        result = np.random.random((size, int(size * complexity)))
        return np.mean(result, axis=1)
    
    def dummy_hdc_operation(dim: int = 10000, sparsity: float = 0.5) -> np.ndarray:
        """Dummy HDC operation for testing."""
        return np.random.binomial(1, sparsity, dim).astype(np.float32)
    
    print("\\n🔧 Testing Optimized Operations:")
    
    # Test adaptive optimization
    for i in range(5):
        result, metrics = manager.execute_optimized(dummy_computation, size=1000, complexity=0.5)
        print(f"Operation {i+1}: {metrics.execution_time:.4f}s, Cache hit rate: {metrics.cache_hit_rate:.2f}")
    
    # Test distributed execution
    print("\\n🌐 Testing Distributed Execution:")
    for i in range(3):
        task_id = manager.execute_optimized(dummy_hdc_operation, dim=5000, distributed=True)
        print(f"Submitted distributed task: {task_id}")
    
    # Wait for distributed tasks to complete
    time.sleep(2.0)
    
    # Generate comprehensive report
    print("\\n📊 Performance Report:")
    report = manager.get_comprehensive_report()
    
    print(f"Strategy: {report['strategy']}")
    print(f"Cache hit rate: {report['adaptive_optimizer']['cache_stats']['hit_rate']:.2%}")
    print(f"Distributed tasks completed: {report['distributed_system']['total_tasks']}")
    print(f"System memory usage: {psutil.virtual_memory().percent:.1f}%")
    
    print("\\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Cleanup
    manager.shutdown()
    
    print("\\n✅ Performance System Test Complete!")