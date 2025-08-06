"""High-performance parallel processing for HDC operations."""

import numpy as np
import multiprocessing as mp
import threading
import queue
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque
import concurrent.futures
from functools import partial
import psutil
import logging


@dataclass
class ParallelTask:
    """Task for parallel execution."""
    task_id: str
    operation: Callable
    args: Tuple
    kwargs: Dict
    priority: int = 1
    chunk_id: Optional[int] = None


@dataclass
class ProcessorInfo:
    """Information about processing resources."""
    processor_type: str  # 'cpu', 'gpu', 'tpu'
    processor_id: str
    cores: int
    memory_gb: float
    utilization: float
    capabilities: List[str]
    performance_score: float


class ParallelHDC:
    """Main parallel processing coordinator for HDC operations."""
    
    def __init__(self, max_workers: Optional[int] = None, 
                 gpu_enabled: bool = True,
                 optimization_level: int = 2):
        
        # Determine optimal worker count
        if max_workers is None:
            self.max_workers = min(32, mp.cpu_count() * 2)
        else:
            self.max_workers = max_workers
        
        self.gpu_enabled = gpu_enabled
        self.optimization_level = optimization_level
        
        # Processing engines
        self.mp_engine = MultiProcessingEngine(self.max_workers)
        self.gpu_accelerator = GPUAccelerator() if gpu_enabled else None
        
        # Task scheduling
        self.task_scheduler = TaskScheduler()
        self.load_balancer = ProcessorLoadBalancer()
        
        # Performance monitoring
        self.performance_metrics = deque(maxlen=1000)
        self.operation_times = defaultdict(deque)
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def parallel_operation(self, operation: str, data: Any, 
                         chunk_size: Optional[int] = None,
                         use_gpu: bool = True,
                         **kwargs) -> Any:
        """Execute HDC operation in parallel across available processors."""
        
        # Analyze data and operation for optimal parallelization
        parallel_plan = self._plan_parallelization(operation, data, chunk_size, **kwargs)
        
        if parallel_plan['use_gpu'] and use_gpu and self.gpu_accelerator:
            return self._execute_gpu_parallel(operation, data, parallel_plan, **kwargs)
        elif parallel_plan['chunk_count'] > 1:
            return self._execute_cpu_parallel(operation, data, parallel_plan, **kwargs)
        else:
            return self._execute_sequential(operation, data, **kwargs)
    
    def parallel_batch_processing(self, operations: List[Tuple[str, Any]], 
                                **kwargs) -> List[Any]:
        """Process multiple operations in parallel."""
        
        # Create parallel tasks
        tasks = []
        for i, (operation, data) in enumerate(operations):
            task = ParallelTask(
                task_id=f"batch_{i}",
                operation=partial(self._execute_single_operation, operation),
                args=(data,),
                kwargs=kwargs
            )
            tasks.append(task)
        
        # Execute tasks
        return self._execute_parallel_tasks(tasks)
    
    def adaptive_parallel_processing(self, operation: str, data_stream: List[Any],
                                   initial_chunk_size: int = 100,
                                   adaptation_window: int = 10) -> List[Any]:
        """Adaptive parallel processing with dynamic chunk size adjustment."""
        
        results = []
        current_chunk_size = initial_chunk_size
        performance_window = deque(maxlen=adaptation_window)
        
        i = 0
        while i < len(data_stream):
            # Prepare chunk
            chunk_end = min(i + current_chunk_size, len(data_stream))
            chunk = data_stream[i:chunk_end]
            
            # Process chunk
            start_time = time.time()
            chunk_result = self.parallel_operation(operation, chunk)
            processing_time = time.time() - start_time
            
            # Calculate throughput
            throughput = len(chunk) / processing_time if processing_time > 0 else 0
            performance_window.append(throughput)
            
            results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
            
            # Adapt chunk size based on recent performance
            if len(performance_window) >= 5:
                recent_throughput = np.mean(list(performance_window)[-3:])
                older_throughput = np.mean(list(performance_window)[:-3])
                
                if recent_throughput > older_throughput * 1.2:
                    # Performance improving, increase chunk size
                    current_chunk_size = min(current_chunk_size * 1.5, 1000)
                elif recent_throughput < older_throughput * 0.8:
                    # Performance degrading, decrease chunk size
                    current_chunk_size = max(current_chunk_size * 0.7, 10)
                
                current_chunk_size = int(current_chunk_size)
            
            i = chunk_end
        
        return results
    
    def pipeline_parallel_processing(self, pipeline_stages: List[Tuple[str, Dict]], 
                                   input_data: Any,
                                   buffer_size: int = 10) -> Any:
        """Execute pipeline of operations with parallel stages."""
        
        if len(pipeline_stages) <= 1:
            # Single stage - use regular parallel processing
            operation, kwargs = pipeline_stages[0]
            return self.parallel_operation(operation, input_data, **kwargs)
        
        # Create pipeline with queues between stages
        stage_queues = [queue.Queue(maxsize=buffer_size) for _ in range(len(pipeline_stages))]
        stage_queues.append(queue.Queue())  # Output queue
        
        # Start pipeline stages
        stage_threads = []
        
        for i, (operation, kwargs) in enumerate(pipeline_stages):
            input_queue = stage_queues[i]
            output_queue = stage_queues[i + 1]
            
            thread = threading.Thread(
                target=self._pipeline_stage_worker,
                args=(operation, input_queue, output_queue, kwargs),
                daemon=True
            )
            thread.start()
            stage_threads.append(thread)
        
        # Feed input data
        if isinstance(input_data, list):
            for item in input_data:
                stage_queues[0].put(item)
        else:
            stage_queues[0].put(input_data)
        
        # Signal end of input
        stage_queues[0].put(None)
        
        # Collect results
        results = []
        while True:
            try:
                result = stage_queues[-1].get(timeout=30)
                if result is None:
                    break
                results.append(result)
            except queue.Empty:
                break
        
        # Wait for all stages to complete
        for thread in stage_threads:
            thread.join(timeout=10)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        if not self.performance_metrics:
            return {'message': 'No performance data available'}
        
        recent_metrics = list(self.performance_metrics)[-100:]  # Last 100 operations
        
        # Calculate aggregate statistics
        total_operations = len(recent_metrics)
        avg_execution_time = np.mean([m['execution_time'] for m in recent_metrics])
        throughput = total_operations / sum(m['execution_time'] for m in recent_metrics)
        
        # Resource utilization
        cpu_utilizations = [m.get('cpu_utilization', 0) for m in recent_metrics]
        memory_usages = [m.get('memory_usage', 0) for m in recent_metrics]
        
        # Operation breakdown
        operation_counts = defaultdict(int)
        operation_times = defaultdict(list)
        
        for metric in recent_metrics:
            op = metric['operation']
            operation_counts[op] += 1
            operation_times[op].append(metric['execution_time'])
        
        # GPU metrics if available
        gpu_metrics = {}
        if self.gpu_accelerator:
            gpu_metrics = self.gpu_accelerator.get_performance_metrics()
        
        return {
            'summary': {
                'total_operations': total_operations,
                'avg_execution_time_ms': avg_execution_time * 1000,
                'throughput_ops_per_sec': throughput,
                'avg_cpu_utilization': np.mean(cpu_utilizations),
                'avg_memory_usage_mb': np.mean(memory_usages)
            },
            'operation_breakdown': {
                op: {
                    'count': operation_counts[op],
                    'avg_time_ms': np.mean(times) * 1000,
                    'total_time_ms': np.sum(times) * 1000
                }
                for op, times in operation_times.items()
            },
            'resource_utilization': {
                'max_workers': self.max_workers,
                'gpu_enabled': self.gpu_enabled,
                'multiprocessing_active': self.mp_engine.is_active()
            },
            'gpu_metrics': gpu_metrics,
            'performance_trends': self._calculate_performance_trends()
        }
    
    def _plan_parallelization(self, operation: str, data: Any, 
                            chunk_size: Optional[int], **kwargs) -> Dict[str, Any]:
        """Plan optimal parallelization strategy."""
        
        # Analyze data characteristics
        data_size = len(data) if hasattr(data, '__len__') else 1
        is_large_data = data_size > 1000
        
        # Operation characteristics
        gpu_suitable_ops = [
            'random_hv', 'bundle', 'bind', 'cosine_similarity', 'hamming_distance',
            'fractional_bind', 'quantum_superposition'
        ]
        cpu_intensive_ops = [
            'hierarchical_bind', 'adaptive_threshold', 'temporal_binding'
        ]
        
        # Resource availability
        gpu_available = self.gpu_accelerator and self.gpu_accelerator.is_available()
        cpu_cores = mp.cpu_count()
        
        # Determine strategy
        use_gpu = gpu_available and operation in gpu_suitable_ops and is_large_data
        
        # Calculate optimal chunk size
        if chunk_size is None:
            if use_gpu:
                # GPU prefers larger chunks
                optimal_chunk_size = max(100, data_size // 4)
            else:
                # CPU parallelism prefers smaller chunks
                optimal_chunk_size = max(10, data_size // (cpu_cores * 2))
        else:
            optimal_chunk_size = chunk_size
        
        # Calculate number of chunks
        chunk_count = max(1, data_size // optimal_chunk_size) if is_large_data else 1
        
        return {
            'use_gpu': use_gpu,
            'chunk_size': optimal_chunk_size,
            'chunk_count': chunk_count,
            'data_size': data_size,
            'cpu_cores': cpu_cores,
            'is_cpu_intensive': operation in cpu_intensive_ops
        }
    
    def _execute_gpu_parallel(self, operation: str, data: Any, 
                            plan: Dict[str, Any], **kwargs) -> Any:
        """Execute operation using GPU acceleration."""
        
        start_time = time.time()
        
        try:
            result = self.gpu_accelerator.execute_operation(operation, data, **kwargs)
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_performance(operation, execution_time, 'gpu', plan)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU execution failed for {operation}: {str(e)}, falling back to CPU")
            return self._execute_cpu_parallel(operation, data, plan, **kwargs)
    
    def _execute_cpu_parallel(self, operation: str, data: Any, 
                            plan: Dict[str, Any], **kwargs) -> Any:
        """Execute operation using CPU parallelization."""
        
        start_time = time.time()
        
        # Split data into chunks
        chunks = self._split_data_into_chunks(data, plan['chunk_size'])
        
        # Create parallel tasks
        tasks = []
        for i, chunk in enumerate(chunks):
            task = ParallelTask(
                task_id=f"{operation}_{i}",
                operation=partial(self._execute_single_operation, operation),
                args=(chunk,),
                kwargs=kwargs,
                chunk_id=i
            )
            tasks.append(task)
        
        # Execute tasks in parallel
        chunk_results = self._execute_parallel_tasks(tasks)
        
        # Combine results
        result = self._combine_chunk_results(operation, chunk_results)
        
        # Record performance
        execution_time = time.time() - start_time
        self._record_performance(operation, execution_time, 'cpu_parallel', plan)
        
        return result
    
    def _execute_sequential(self, operation: str, data: Any, **kwargs) -> Any:
        """Execute operation sequentially."""
        
        start_time = time.time()
        result = self._execute_single_operation(operation, data, **kwargs)
        execution_time = time.time() - start_time
        
        self._record_performance(operation, execution_time, 'sequential', {})
        
        return result
    
    def _execute_parallel_tasks(self, tasks: List[ParallelTask]) -> List[Any]:
        """Execute multiple tasks in parallel using multiprocessing."""
        
        if len(tasks) == 1:
            # Single task - execute directly
            task = tasks[0]
            return [task.operation(*task.args, **task.kwargs)]
        
        # Use multiprocessing for multiple tasks
        return self.mp_engine.execute_tasks(tasks)
    
    def _split_data_into_chunks(self, data: Any, chunk_size: int) -> List[Any]:
        """Split data into chunks for parallel processing."""
        
        if not hasattr(data, '__len__'):
            return [data]
        
        if len(data) <= chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _combine_chunk_results(self, operation: str, results: List[Any]) -> Any:
        """Combine results from parallel chunk processing."""
        
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Operation-specific combination
        if operation in ['bundle', 'quantum_superposition']:
            # Element-wise operations that can be combined
            if all(hasattr(r, '__len__') for r in results if r is not None):
                valid_results = [r for r in results if r is not None]
                if valid_results:
                    return np.concatenate(valid_results)
        
        elif operation in ['random_hv']:
            # Concatenate random vectors
            if all(hasattr(r, '__len__') for r in results if r is not None):
                valid_results = [r for r in results if r is not None]
                if valid_results:
                    return np.concatenate(valid_results)
        
        elif operation in ['cosine_similarity', 'hamming_distance']:
            # Average similarity measures
            valid_results = [r for r in results if r is not None and not np.isnan(r)]
            if valid_results:
                return np.mean(valid_results)
        
        # Default: return concatenated or first result
        if isinstance(results[0], (list, np.ndarray)):
            return np.concatenate([r for r in results if r is not None])
        else:
            return results[0]
    
    def _execute_single_operation(self, operation: str, data: Any, **kwargs) -> Any:
        """Execute single HDC operation (placeholder)."""
        
        # This would delegate to actual HDC implementation
        # For now, simulate different operations
        
        if operation == 'random_hv':
            size = len(data) if hasattr(data, '__len__') else data
            return np.random.choice([-1, 1], size=size)
        
        elif operation == 'bundle':
            if hasattr(data, '__len__') and len(data) > 0:
                if hasattr(data[0], '__len__'):  # List of hypervectors
                    return np.sign(np.sum(data, axis=0))
                else:  # Single hypervector
                    return data
            return np.zeros(10000)
        
        elif operation == 'bind':
            if hasattr(data, '__len__') and len(data) >= 2:
                return data[0] * data[1]
            return np.zeros(10000)
        
        elif operation == 'cosine_similarity':
            if hasattr(data, '__len__') and len(data) >= 2:
                hv1, hv2 = data[0], data[1]
                if hasattr(hv1, '__len__') and hasattr(hv2, '__len__'):
                    dot_product = np.dot(hv1, hv2)
                    norm1, norm2 = np.linalg.norm(hv1), np.linalg.norm(hv2)
                    if norm1 > 0 and norm2 > 0:
                        return dot_product / (norm1 * norm2)
            return 0.0
        
        else:
            # Simulate processing time
            time.sleep(0.001)
            return f"result_for_{operation}"
    
    def _pipeline_stage_worker(self, operation: str, input_queue: queue.Queue,
                             output_queue: queue.Queue, kwargs: Dict) -> None:
        """Worker for pipeline stage processing."""
        
        while True:
            try:
                item = input_queue.get(timeout=30)
                if item is None:
                    # End of input signal
                    output_queue.put(None)
                    break
                
                # Process item
                result = self.parallel_operation(operation, item, **kwargs)
                output_queue.put(result)
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Pipeline stage {operation} error: {str(e)}")
                break
    
    def _record_performance(self, operation: str, execution_time: float,
                          execution_mode: str, plan: Dict[str, Any]) -> None:
        """Record performance metrics."""
        
        # Get current system metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metric = {
            'operation': operation,
            'execution_time': execution_time,
            'execution_mode': execution_mode,
            'cpu_utilization': cpu_percent,
            'memory_usage': memory_info.used / (1024 ** 2),  # MB
            'timestamp': time.time(),
            'chunk_count': plan.get('chunk_count', 1),
            'data_size': plan.get('data_size', 0)
        }
        
        self.performance_metrics.append(metric)
        self.operation_times[operation].append(execution_time)
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""
        
        if len(self.performance_metrics) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        recent_metrics = list(self.performance_metrics)[-20:]
        older_metrics = list(self.performance_metrics)[-40:-20] if len(self.performance_metrics) >= 40 else recent_metrics
        
        # Calculate average execution times
        recent_avg = np.mean([m['execution_time'] for m in recent_metrics])
        older_avg = np.mean([m['execution_time'] for m in older_metrics])
        
        # Determine trend
        if recent_avg < older_avg * 0.9:
            execution_trend = 'improving'
        elif recent_avg > older_avg * 1.1:
            execution_trend = 'degrading'
        else:
            execution_trend = 'stable'
        
        # CPU utilization trend
        recent_cpu = np.mean([m['cpu_utilization'] for m in recent_metrics])
        older_cpu = np.mean([m['cpu_utilization'] for m in older_metrics])
        
        if recent_cpu < older_cpu * 0.9:
            cpu_trend = 'decreasing'
        elif recent_cpu > older_cpu * 1.1:
            cpu_trend = 'increasing'
        else:
            cpu_trend = 'stable'
        
        return {
            'execution_time': execution_trend,
            'cpu_utilization': cpu_trend,
            'data_points': len(self.performance_metrics)
        }


class MultiProcessingEngine:
    """Multiprocessing execution engine."""
    
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.process_pool = None
        
    def execute_tasks(self, tasks: List[ParallelTask]) -> List[Any]:
        """Execute tasks using multiprocessing."""
        
        if not tasks:
            return []
        
        try:
            # Create process pool if needed
            if self.process_pool is None:
                self.process_pool = mp.Pool(processes=self.max_workers)
            
            # Submit tasks
            async_results = []
            for task in tasks:
                async_result = self.process_pool.apply_async(
                    task.operation, 
                    task.args, 
                    task.kwargs
                )
                async_results.append(async_result)
            
            # Collect results
            results = []
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=60)  # 60 second timeout
                    results.append(result)
                except mp.TimeoutError:
                    results.append(None)
                except Exception as e:
                    results.append(None)
            
            return results
            
        except Exception as e:
            # Fallback to sequential execution
            results = []
            for task in tasks:
                try:
                    result = task.operation(*task.args, **task.kwargs)
                    results.append(result)
                except Exception:
                    results.append(None)
            return results
    
    def is_active(self) -> bool:
        """Check if process pool is active."""
        return self.process_pool is not None
    
    def shutdown(self) -> None:
        """Shutdown process pool."""
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None


class GPUAccelerator:
    """GPU acceleration for HDC operations."""
    
    def __init__(self):
        self.available = self._check_gpu_availability()
        self.device_info = self._get_device_info()
        self.performance_history = deque(maxlen=100)
        
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.available
    
    def execute_operation(self, operation: str, data: Any, **kwargs) -> Any:
        """Execute operation on GPU."""
        
        if not self.available:
            raise RuntimeError("GPU not available")
        
        start_time = time.time()
        
        try:
            # Convert data to GPU format (simulation)
            gpu_data = self._to_gpu(data)
            
            # Execute operation
            if operation == 'random_hv':
                result = self._gpu_random_hv(gpu_data, **kwargs)
            elif operation == 'bundle':
                result = self._gpu_bundle(gpu_data, **kwargs)
            elif operation == 'bind':
                result = self._gpu_bind(gpu_data, **kwargs)
            elif operation == 'cosine_similarity':
                result = self._gpu_cosine_similarity(gpu_data, **kwargs)
            else:
                raise NotImplementedError(f"GPU operation {operation} not implemented")
            
            # Convert back to CPU format
            cpu_result = self._to_cpu(result)
            
            # Record performance
            execution_time = time.time() - start_time
            self.performance_history.append({
                'operation': operation,
                'execution_time': execution_time,
                'data_size': len(data) if hasattr(data, '__len__') else 1
            })
            
            return cpu_result
            
        except Exception as e:
            raise RuntimeError(f"GPU execution failed: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics."""
        
        if not self.performance_history:
            return {'gpu_available': self.available, 'operations_executed': 0}
        
        recent_ops = list(self.performance_history)
        
        return {
            'gpu_available': self.available,
            'device_info': self.device_info,
            'operations_executed': len(recent_ops),
            'avg_execution_time_ms': np.mean([op['execution_time'] for op in recent_ops]) * 1000,
            'total_execution_time_s': sum(op['execution_time'] for op in recent_ops),
            'operations_per_second': len(recent_ops) / sum(op['execution_time'] for op in recent_ops)
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for computation."""
        try:
            # Try to import CUDA libraries
            import cupy as cp  # noqa
            return True
        except ImportError:
            pass
        
        try:
            # Try PyTorch CUDA
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        return False
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if not self.available:
            return {'error': 'GPU not available'}
        
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'name': torch.cuda.get_device_name(0),
                    'compute_capability': torch.cuda.get_device_capability(0),
                    'memory_total': torch.cuda.get_device_properties(0).total_memory,
                    'multiprocessor_count': torch.cuda.get_device_properties(0).multi_processor_count
                }
        except ImportError:
            pass
        
        return {'type': 'simulated_gpu'}
    
    def _to_gpu(self, data: Any) -> Any:
        """Convert data to GPU format (simulation)."""
        # In real implementation, would convert to GPU arrays
        return data
    
    def _to_cpu(self, data: Any) -> Any:
        """Convert data from GPU format (simulation)."""
        # In real implementation, would convert from GPU arrays
        return data
    
    def _gpu_random_hv(self, data: Any, **kwargs) -> Any:
        """GPU implementation of random hypervector generation."""
        # Simulation - in practice would use GPU random number generation
        size = data if isinstance(data, int) else len(data)
        return np.random.choice([-1, 1], size=size)
    
    def _gpu_bundle(self, data: Any, **kwargs) -> Any:
        """GPU implementation of bundling operation."""
        # Simulation - in practice would use GPU parallel reduction
        if hasattr(data, '__len__') and len(data) > 0:
            if hasattr(data[0], '__len__'):
                return np.sign(np.sum(data, axis=0))
        return np.zeros(10000)
    
    def _gpu_bind(self, data: Any, **kwargs) -> Any:
        """GPU implementation of binding operation."""
        # Simulation - in practice would use GPU element-wise operations
        if hasattr(data, '__len__') and len(data) >= 2:
            return data[0] * data[1]
        return np.zeros(10000)
    
    def _gpu_cosine_similarity(self, data: Any, **kwargs) -> Any:
        """GPU implementation of cosine similarity."""
        # Simulation - in practice would use GPU dot products and norms
        if hasattr(data, '__len__') and len(data) >= 2:
            hv1, hv2 = data[0], data[1]
            if hasattr(hv1, '__len__') and hasattr(hv2, '__len__'):
                dot_product = np.dot(hv1, hv2)
                norm1, norm2 = np.linalg.norm(hv1), np.linalg.norm(hv2)
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
        return 0.0


class TaskScheduler:
    """Intelligent task scheduler for optimal resource utilization."""
    
    def __init__(self):
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = deque(maxlen=1000)
        self.resource_monitor = ResourceMonitor()
        
    def schedule_task(self, task: ParallelTask, 
                     estimated_resources: Optional[Dict[str, float]] = None) -> None:
        """Schedule task for execution."""
        
        # Calculate priority score
        priority_score = self._calculate_priority(task, estimated_resources)
        
        # Add to queue (lower score = higher priority)
        self.task_queue.put((priority_score, task))
    
    def get_next_task(self) -> Optional[ParallelTask]:
        """Get next task for execution."""
        try:
            priority_score, task = self.task_queue.get_nowait()
            return task
        except queue.Empty:
            return None
    
    def _calculate_priority(self, task: ParallelTask, 
                          estimated_resources: Optional[Dict[str, float]]) -> float:
        """Calculate task priority score."""
        
        base_priority = task.priority
        
        # Adjust based on resource requirements
        if estimated_resources:
            memory_factor = estimated_resources.get('memory', 1.0)
            compute_factor = estimated_resources.get('compute', 1.0)
            
            # Higher resource requirements = lower priority (higher score)
            resource_penalty = (memory_factor + compute_factor) * 0.1
            base_priority += resource_penalty
        
        # Adjust based on current system load
        current_load = self.resource_monitor.get_current_load()
        if current_load > 0.8:
            # High load - prioritize smaller tasks
            base_priority *= 0.9 if task.priority <= 2 else 1.1
        
        return base_priority


class ProcessorLoadBalancer:
    """Load balancer for distributing work across processors."""
    
    def __init__(self):
        self.processor_info = self._discover_processors()
        self.load_history = defaultdict(deque)
        
    def select_processor(self, task_requirements: Dict[str, Any]) -> Optional[ProcessorInfo]:
        """Select best processor for task."""
        
        if not self.processor_info:
            return None
        
        # Filter processors by capabilities
        suitable_processors = []
        for processor in self.processor_info:
            if self._processor_meets_requirements(processor, task_requirements):
                suitable_processors.append(processor)
        
        if not suitable_processors:
            return None
        
        # Select least loaded processor
        best_processor = min(suitable_processors, key=lambda p: p.utilization)
        
        return best_processor
    
    def update_processor_load(self, processor_id: str, new_load: float) -> None:
        """Update processor load information."""
        for processor in self.processor_info:
            if processor.processor_id == processor_id:
                processor.utilization = new_load
                self.load_history[processor_id].append(new_load)
                break
    
    def _discover_processors(self) -> List[ProcessorInfo]:
        """Discover available processors."""
        processors = []
        
        # CPU processors
        cpu_count = mp.cpu_count()
        cpu_info = ProcessorInfo(
            processor_type='cpu',
            processor_id='cpu_pool',
            cores=cpu_count,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            utilization=0.0,
            capabilities=['general_compute'],
            performance_score=1.0
        )
        processors.append(cpu_info)
        
        # GPU processors (if available)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info = ProcessorInfo(
                        processor_type='gpu',
                        processor_id=f'cuda_{i}',
                        cores=torch.cuda.get_device_properties(i).multi_processor_count,
                        memory_gb=torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        utilization=0.0,
                        capabilities=['parallel_compute', 'linear_algebra'],
                        performance_score=5.0  # GPUs typically faster for parallel operations
                    )
                    processors.append(gpu_info)
        except ImportError:
            pass
        
        return processors
    
    def _processor_meets_requirements(self, processor: ProcessorInfo, 
                                    requirements: Dict[str, Any]) -> bool:
        """Check if processor meets task requirements."""
        
        # Check memory requirements
        memory_req = requirements.get('memory_gb', 0)
        if processor.memory_gb < memory_req:
            return False
        
        # Check capability requirements
        required_caps = requirements.get('capabilities', [])
        for cap in required_caps:
            if cap not in processor.capabilities:
                return False
        
        # Check utilization threshold
        max_utilization = requirements.get('max_utilization', 0.9)
        if processor.utilization > max_utilization:
            return False
        
        return True


class ResourceMonitor:
    """Monitor system resource utilization."""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_history = deque(maxlen=60)  # Last 60 measurements
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start resource monitoring."""
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
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_current_load(self) -> float:
        """Get current system load (0-1)."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Combined load score
            combined_load = (cpu_percent + memory_percent) / 200.0  # Average and normalize
            return min(1.0, combined_load)
        except Exception:
            return 0.5  # Default moderate load
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        if not self.resource_history:
            return self._get_instant_stats()
        
        cpu_usage = [entry['cpu_percent'] for entry in self.resource_history]
        memory_usage = [entry['memory_percent'] for entry in self.resource_history]
        
        return {
            'current': self._get_instant_stats(),
            'history': {
                'cpu_avg': np.mean(cpu_usage),
                'cpu_max': np.max(cpu_usage),
                'memory_avg': np.mean(memory_usage),
                'memory_max': np.max(memory_usage),
                'samples': len(self.resource_history)
            }
        }
    
    def _monitoring_loop(self, interval: float) -> None:
        """Resource monitoring loop."""
        while self.monitoring_active:
            try:
                stats = self._get_instant_stats()
                self.resource_history.append(stats)
                time.sleep(interval)
            except Exception:
                time.sleep(interval)
    
    def _get_instant_stats(self) -> Dict[str, Any]:
        """Get instantaneous resource statistics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3)
            }
        except Exception:
            return {
                'timestamp': time.time(),
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available_gb': 0.0,
                'memory_used_gb': 0.0
            }