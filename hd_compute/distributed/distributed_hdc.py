"""Distributed hyperdimensional computing across multiple nodes/GPUs."""

import numpy as np
import time
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import socket
import pickle
import concurrent.futures
from pathlib import Path
import logging


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float
    memory_available: int
    gpu_available: bool
    status: str  # 'active', 'busy', 'offline'
    last_heartbeat: float


@dataclass
class Task:
    """Distributed computing task."""
    task_id: str
    operation: str
    args: Tuple
    kwargs: Dict
    priority: int
    timestamp: float
    estimated_time: float
    memory_requirement: int
    node_preference: Optional[str] = None


@dataclass
class TaskResult:
    """Result of distributed task execution."""
    task_id: str
    node_id: str
    result: Any
    execution_time: float
    memory_used: int
    success: bool
    error_message: Optional[str] = None


class DistributedHDC:
    """Main distributed HDC coordinator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.nodes = {}
        self.task_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.load_balancer = LoadBalancer()
        self.cluster_manager = ClusterManager()
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Statistics
        self.task_stats = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        
        # Threading
        self.coordinator_thread = None
        self.heartbeat_thread = None
        self.running = False
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def start_cluster(self) -> None:
        """Start distributed cluster coordination."""
        if self.running:
            return
            
        self.running = True
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordinator_thread.start()
        
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        self.logger.info("Distributed HDC cluster started")
    
    def stop_cluster(self) -> None:
        """Stop distributed cluster."""
        self.running = False
        
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
            
        self.logger.info("Distributed HDC cluster stopped")
    
    def add_node(self, node_info: NodeInfo) -> bool:
        """Add compute node to cluster."""
        try:
            # Test node connectivity
            if self._test_node_connection(node_info):
                self.nodes[node_info.node_id] = node_info
                self.cluster_manager.register_node(node_info)
                self.logger.info(f"Added node {node_info.node_id} to cluster")
                return True
            else:
                self.logger.error(f"Failed to connect to node {node_info.node_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error adding node {node_info.node_id}: {str(e)}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.cluster_manager.unregister_node(node_id)
            self.logger.info(f"Removed node {node_id} from cluster")
            return True
        return False
    
    def distributed_operation(self, operation: str, *args, **kwargs) -> Any:
        """Execute HDC operation across distributed cluster."""
        
        # Analyze operation for distribution strategy
        distribution_plan = self._plan_distribution(operation, args, kwargs)
        
        if distribution_plan['strategy'] == 'single_node':
            return self._execute_single_node(operation, args, kwargs, distribution_plan)
        elif distribution_plan['strategy'] == 'data_parallel':
            return self._execute_data_parallel(operation, args, kwargs, distribution_plan)
        elif distribution_plan['strategy'] == 'model_parallel':
            return self._execute_model_parallel(operation, args, kwargs, distribution_plan)
        else:
            return self._execute_pipeline_parallel(operation, args, kwargs, distribution_plan)
    
    def bulk_operations(self, operations: List[Tuple[str, Tuple, Dict]],
                       max_concurrent: int = 10) -> List[TaskResult]:
        """Execute bulk operations with load balancing."""
        
        # Create tasks
        tasks = []
        for i, (operation, args, kwargs) in enumerate(operations):
            task = Task(
                task_id=f"bulk_{int(time.time())}_{i}",
                operation=operation,
                args=args,
                kwargs=kwargs,
                priority=1,
                timestamp=time.time(),
                estimated_time=self._estimate_task_time(operation, args, kwargs),
                memory_requirement=self._estimate_memory_requirement(operation, args, kwargs)
            )
            tasks.append(task)
        
        # Distribute tasks
        return self._execute_bulk_tasks(tasks, max_concurrent)
    
    def adaptive_batch_processing(self, data_batches: List[Any],
                                 operation: str,
                                 batch_size_adaptation: bool = True) -> List[Any]:
        """Adaptive batch processing with dynamic sizing."""
        
        results = []
        
        if batch_size_adaptation:
            # Start with initial batch size and adapt based on performance
            current_batch_size = min(len(data_batches), 10)
            performance_window = deque(maxlen=5)
        else:
            current_batch_size = len(data_batches)
        
        batch_idx = 0
        while batch_idx < len(data_batches):
            # Get current batch
            end_idx = min(batch_idx + current_batch_size, len(data_batches))
            current_batch = data_batches[batch_idx:end_idx]
            
            # Execute batch
            start_time = time.time()
            batch_result = self.distributed_operation(operation, current_batch)
            execution_time = time.time() - start_time
            
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Adapt batch size if enabled
            if batch_size_adaptation:
                throughput = len(current_batch) / execution_time
                performance_window.append(throughput)
                
                if len(performance_window) >= 3:
                    # Adjust batch size based on throughput trend
                    recent_throughput = np.mean(list(performance_window)[-3:])
                    older_throughput = np.mean(list(performance_window)[:-3]) if len(performance_window) > 3 else recent_throughput
                    
                    if recent_throughput > older_throughput * 1.1:
                        # Performance improving, increase batch size
                        current_batch_size = min(current_batch_size + 2, 50)
                    elif recent_throughput < older_throughput * 0.9:
                        # Performance degrading, decrease batch size
                        current_batch_size = max(current_batch_size - 1, 1)
            
            batch_idx = end_idx
        
        return results
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_nodes = [node for node in self.nodes.values() if node.status == 'active']
        busy_nodes = [node for node in self.nodes.values() if node.status == 'busy']
        offline_nodes = [node for node in self.nodes.values() if node.status == 'offline']
        
        total_memory = sum(node.memory_available for node in active_nodes)
        total_gpus = sum(1 for node in active_nodes if node.gpu_available)
        
        # Performance metrics
        recent_tasks = [task for task in self.performance_history if time.time() - task['timestamp'] < 300]  # Last 5 minutes
        avg_task_time = np.mean([task['execution_time'] for task in recent_tasks]) if recent_tasks else 0
        
        return {
            'cluster_summary': {
                'total_nodes': len(self.nodes),
                'active_nodes': len(active_nodes),
                'busy_nodes': len(busy_nodes),
                'offline_nodes': len(offline_nodes),
                'total_memory_gb': total_memory / (1024**3),
                'total_gpus': total_gpus
            },
            'performance_metrics': {
                'tasks_completed_last_5min': len(recent_tasks),
                'avg_task_execution_time': avg_task_time,
                'cluster_utilization': len(busy_nodes) / len(self.nodes) if self.nodes else 0,
                'queue_size': self.task_queue.qsize()
            },
            'node_details': [
                {
                    'node_id': node.node_id,
                    'status': node.status,
                    'current_load': node.current_load,
                    'memory_available_gb': node.memory_available / (1024**3),
                    'gpu_available': node.gpu_available,
                    'last_heartbeat_age': time.time() - node.last_heartbeat
                }
                for node in self.nodes.values()
            ]
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load cluster configuration."""
        default_config = {
            'heartbeat_interval': 30,
            'node_timeout': 120,
            'max_retries': 3,
            'load_balance_strategy': 'least_loaded',
            'fault_tolerance': True,
            'auto_scaling': False
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        
        return default_config
    
    def _test_node_connection(self, node_info: NodeInfo) -> bool:
        """Test connectivity to a node."""
        try:
            # Simple TCP connection test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node_info.host, node_info.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _coordination_loop(self) -> None:
        """Main coordination loop for distributed processing."""
        while self.running:
            try:
                # Process pending tasks
                if not self.task_queue.empty():
                    try:
                        priority, task = self.task_queue.get(timeout=1)
                        self._dispatch_task(task)
                    except queue.Empty:
                        continue
                
                # Check for completed tasks and node status updates
                self._check_node_status()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {str(e)}")
                time.sleep(1)
    
    def _heartbeat_loop(self) -> None:
        """Monitor node heartbeats."""
        while self.running:
            try:
                current_time = time.time()
                timeout_threshold = self.config['node_timeout']
                
                for node_id, node in list(self.nodes.items()):
                    if current_time - node.last_heartbeat > timeout_threshold:
                        if node.status != 'offline':
                            node.status = 'offline'
                            self.logger.warning(f"Node {node_id} marked offline due to timeout")
                
                time.sleep(self.config['heartbeat_interval'])
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {str(e)}")
                time.sleep(self.config['heartbeat_interval'])
    
    def _plan_distribution(self, operation: str, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        """Plan how to distribute an operation across nodes."""
        
        # Analyze operation characteristics
        is_embarrassingly_parallel = operation in ['random_hv', 'bundle', 'bind']
        requires_synchronization = operation in ['hierarchical_bind', 'quantum_superposition']
        is_memory_intensive = operation in ['adaptive_threshold', 'sequence_prediction']
        
        # Estimate resource requirements
        estimated_memory = self._estimate_memory_requirement(operation, args, kwargs)
        estimated_compute = self._estimate_compute_requirement(operation, args, kwargs)
        
        # Available resources
        available_nodes = [node for node in self.nodes.values() if node.status == 'active']
        total_memory = sum(node.memory_available for node in available_nodes)
        
        # Choose distribution strategy
        if len(available_nodes) <= 1:
            strategy = 'single_node'
        elif is_embarrassingly_parallel and len(args) > 0 and hasattr(args[0], '__len__'):
            # Data can be split across nodes
            strategy = 'data_parallel'
        elif estimated_memory > total_memory * 0.5:
            # Large operation that needs model parallelism
            strategy = 'model_parallel'
        else:
            # Pipeline parallel for complex operations
            strategy = 'pipeline_parallel'
        
        return {
            'strategy': strategy,
            'estimated_memory': estimated_memory,
            'estimated_compute': estimated_compute,
            'available_nodes': len(available_nodes),
            'requires_sync': requires_synchronization
        }
    
    def _execute_single_node(self, operation: str, args: Tuple, kwargs: Dict, plan: Dict) -> Any:
        """Execute operation on single best node."""
        best_node = self.load_balancer.select_node(list(self.nodes.values()), plan)
        
        if not best_node:
            raise RuntimeError("No available nodes for execution")
        
        # Create and dispatch task
        task = Task(
            task_id=f"single_{int(time.time())}",
            operation=operation,
            args=args,
            kwargs=kwargs,
            priority=1,
            timestamp=time.time(),
            estimated_time=plan['estimated_compute'],
            memory_requirement=plan['estimated_memory'],
            node_preference=best_node.node_id
        )
        
        return self._execute_task_on_node(task, best_node)
    
    def _execute_data_parallel(self, operation: str, args: Tuple, kwargs: Dict, plan: Dict) -> Any:
        """Execute operation with data parallelism."""
        available_nodes = [node for node in self.nodes.values() if node.status == 'active']
        
        if not available_nodes:
            raise RuntimeError("No available nodes for parallel execution")
        
        # Split data across nodes
        if len(args) > 0 and hasattr(args[0], '__len__'):
            data = args[0]
            n_chunks = min(len(available_nodes), len(data))
            chunk_size = len(data) // n_chunks
            
            # Create tasks for each chunk
            tasks = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(data)
                chunk_data = data[start_idx:end_idx]
                
                chunk_args = (chunk_data,) + args[1:]  # Replace first arg with chunk
                
                task = Task(
                    task_id=f"parallel_{int(time.time())}_{i}",
                    operation=operation,
                    args=chunk_args,
                    kwargs=kwargs,
                    priority=1,
                    timestamp=time.time(),
                    estimated_time=plan['estimated_compute'] / n_chunks,
                    memory_requirement=plan['estimated_memory'] // n_chunks,
                    node_preference=available_nodes[i % len(available_nodes)].node_id
                )
                tasks.append(task)
            
            # Execute tasks in parallel
            results = self._execute_parallel_tasks(tasks)
            
            # Combine results
            return self._combine_parallel_results(operation, results)
        
        else:
            # Fallback to single node execution
            return self._execute_single_node(operation, args, kwargs, plan)
    
    def _execute_model_parallel(self, operation: str, args: Tuple, kwargs: Dict, plan: Dict) -> Any:
        """Execute operation with model parallelism."""
        # This is a simplified model parallel implementation
        # In practice, this would involve splitting the model/computation across nodes
        
        available_nodes = [node for node in self.nodes.values() if node.status == 'active']
        
        if len(available_nodes) < 2:
            return self._execute_single_node(operation, args, kwargs, plan)
        
        # For demonstration, split operation into sub-operations
        sub_operations = self._decompose_operation(operation, args, kwargs, len(available_nodes))
        
        # Execute sub-operations
        tasks = []
        for i, (sub_op, sub_args, sub_kwargs) in enumerate(sub_operations):
            task = Task(
                task_id=f"model_parallel_{int(time.time())}_{i}",
                operation=sub_op,
                args=sub_args,
                kwargs=sub_kwargs,
                priority=1,
                timestamp=time.time(),
                estimated_time=plan['estimated_compute'] / len(sub_operations),
                memory_requirement=plan['estimated_memory'] // len(sub_operations),
                node_preference=available_nodes[i].node_id
            )
            tasks.append(task)
        
        # Execute tasks
        results = self._execute_parallel_tasks(tasks)
        
        # Compose results
        return self._compose_model_parallel_results(operation, results)
    
    def _execute_pipeline_parallel(self, operation: str, args: Tuple, kwargs: Dict, plan: Dict) -> Any:
        """Execute operation with pipeline parallelism."""
        # Simplified pipeline parallel implementation
        pipeline_stages = self._create_pipeline_stages(operation, args, kwargs)
        
        if len(pipeline_stages) <= 1:
            return self._execute_single_node(operation, args, kwargs, plan)
        
        # Execute pipeline stages sequentially, but with pipelining
        intermediate_results = args
        
        for stage_op, stage_kwargs in pipeline_stages:
            stage_plan = self._plan_distribution(stage_op, intermediate_results, stage_kwargs)
            intermediate_results = (self._execute_single_node(stage_op, intermediate_results, stage_kwargs, stage_plan),)
        
        return intermediate_results[0]
    
    def _execute_bulk_tasks(self, tasks: List[Task], max_concurrent: int) -> List[TaskResult]:
        """Execute multiple tasks with concurrency control."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                # Select node for task
                available_nodes = [node for node in self.nodes.values() if node.status == 'active']
                selected_node = self.load_balancer.select_node(available_nodes, {'memory_requirement': task.memory_requirement})
                
                if selected_node:
                    future = executor.submit(self._execute_task_on_node, task, selected_node)
                    future_to_task[future] = task
                else:
                    # No available node - create failed result
                    results.append(TaskResult(
                        task_id=task.task_id,
                        node_id='none',
                        result=None,
                        execution_time=0.0,
                        memory_used=0,
                        success=False,
                        error_message="No available nodes"
                    ))
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(TaskResult(
                        task_id=task.task_id,
                        node_id='unknown',
                        result=None,
                        execution_time=0.0,
                        memory_used=0,
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _execute_parallel_tasks(self, tasks: List[Task]) -> List[Any]:
        """Execute multiple tasks in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {}
            
            for task in tasks:
                if task.node_preference and task.node_preference in self.nodes:
                    selected_node = self.nodes[task.node_preference]
                else:
                    available_nodes = [node for node in self.nodes.values() if node.status == 'active']
                    selected_node = self.load_balancer.select_node(available_nodes, {})
                
                if selected_node:
                    future = executor.submit(self._execute_task_on_node, task, selected_node)
                    future_to_task[future] = task
            
            # Collect results in order
            task_results = {}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    task_results[task.task_id] = result.result
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {str(e)}")
                    task_results[task.task_id] = None
            
            # Return results in original order
            results = [task_results.get(task.task_id) for task in tasks]
        
        return results
    
    def _execute_task_on_node(self, task: Task, node: NodeInfo) -> TaskResult:
        """Execute single task on specific node."""
        start_time = time.time()
        
        try:
            # In a real implementation, this would send the task to the remote node
            # For simulation, we'll execute locally
            result = self._simulate_remote_execution(task, node)
            
            execution_time = time.time() - start_time
            
            # Record performance
            self.performance_history.append({
                'task_id': task.task_id,
                'node_id': node.node_id,
                'operation': task.operation,
                'execution_time': execution_time,
                'timestamp': time.time()
            })
            
            return TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                result=result,
                execution_time=execution_time,
                memory_used=task.memory_requirement,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                result=None,
                execution_time=execution_time,
                memory_used=0,
                success=False,
                error_message=str(e)
            )
    
    def _simulate_remote_execution(self, task: Task, node: NodeInfo) -> Any:
        """Simulate remote task execution (placeholder)."""
        # In real implementation, this would serialize the task and send to remote node
        # For now, simulate with local execution and delay
        
        # Simulate network and processing delay
        base_delay = 0.01  # 10ms base network delay
        processing_delay = task.estimated_time
        
        time.sleep(base_delay + processing_delay)
        
        # Simulate different operations
        if task.operation == 'random_hv':
            dim = task.args[0] if task.args else 10000
            return np.random.choice([-1, 1], size=dim)
        elif task.operation == 'bundle':
            hvs = task.args[0] if task.args else []
            if hvs:
                return np.sign(np.sum(hvs, axis=0))
            else:
                return np.zeros(10000)
        elif task.operation == 'bind':
            if len(task.args) >= 2:
                return task.args[0] * task.args[1]
            else:
                return np.zeros(10000)
        else:
            return f"result_for_{task.operation}"
    
    def _estimate_task_time(self, operation: str, args: Tuple, kwargs: Dict) -> float:
        """Estimate execution time for task."""
        base_times = {
            'random_hv': 0.001,
            'bundle': 0.002,
            'bind': 0.001,
            'cosine_similarity': 0.001,
            'hamming_distance': 0.001,
            'fractional_bind': 0.005,
            'quantum_superposition': 0.01
        }
        
        base_time = base_times.get(operation, 0.005)
        
        # Scale by data size if applicable
        if args and hasattr(args[0], '__len__'):
            data_size_factor = len(args[0]) / 1000
            base_time *= max(1.0, data_size_factor)
        
        return base_time
    
    def _estimate_memory_requirement(self, operation: str, args: Tuple, kwargs: Dict) -> int:
        """Estimate memory requirement in bytes."""
        base_memory = {
            'random_hv': 40000,  # 10k floats
            'bundle': 80000,
            'bind': 40000,
            'cosine_similarity': 8,
            'hamming_distance': 8,
            'fractional_bind': 120000,
            'quantum_superposition': 200000
        }
        
        base_mem = base_memory.get(operation, 40000)
        
        # Scale by data size
        if args and hasattr(args[0], '__len__'):
            data_size_factor = len(args[0]) / 1000
            base_mem = int(base_mem * max(1.0, data_size_factor))
        
        return base_mem
    
    def _estimate_compute_requirement(self, operation: str, args: Tuple, kwargs: Dict) -> float:
        """Estimate compute requirement (arbitrary units)."""
        compute_weights = {
            'random_hv': 1.0,
            'bundle': 2.0,
            'bind': 1.0,
            'cosine_similarity': 1.5,
            'hamming_distance': 1.0,
            'fractional_bind': 5.0,
            'quantum_superposition': 10.0
        }
        
        return compute_weights.get(operation, 2.0)
    
    def _dispatch_task(self, task: Task) -> None:
        """Dispatch task to appropriate node."""
        # This is called by coordination loop
        available_nodes = [node for node in self.nodes.values() if node.status == 'active']
        
        if not available_nodes:
            # Re-queue task
            self.task_queue.put((task.priority, task))
            return
        
        # Select best node
        selected_node = self.load_balancer.select_node(available_nodes, {
            'memory_requirement': task.memory_requirement,
            'estimated_time': task.estimated_time
        })
        
        if selected_node:
            # Execute task (this would be async in real implementation)
            threading.Thread(
                target=self._execute_task_on_node, 
                args=(task, selected_node),
                daemon=True
            ).start()
        else:
            # Re-queue
            self.task_queue.put((task.priority, task))
    
    def _check_node_status(self) -> None:
        """Check and update node status."""
        # This would check for completed tasks, node health, etc.
        # Placeholder implementation
        pass
    
    def _combine_parallel_results(self, operation: str, results: List[Any]) -> Any:
        """Combine results from parallel execution."""
        if not results:
            return None
        
        if operation in ['bundle', 'quantum_superposition']:
            # Combine by bundling all results
            valid_results = [r for r in results if r is not None]
            if valid_results:
                return np.sign(np.sum(valid_results, axis=0))
        elif operation == 'random_hv':
            # Concatenate random vectors
            valid_results = [r for r in results if r is not None]
            if valid_results:
                return np.concatenate(valid_results)
        
        # Default: return first valid result
        return next((r for r in results if r is not None), None)
    
    def _decompose_operation(self, operation: str, args: Tuple, kwargs: Dict, n_parts: int) -> List[Tuple[str, Tuple, Dict]]:
        """Decompose operation for model parallelism."""
        # Simplified decomposition - in practice this would be operation-specific
        sub_operations = []
        
        for i in range(n_parts):
            sub_op = f"sub_{operation}_{i}"
            sub_args = args  # Would modify based on decomposition strategy
            sub_kwargs = kwargs.copy()
            sub_kwargs['part_id'] = i
            sub_kwargs['total_parts'] = n_parts
            
            sub_operations.append((sub_op, sub_args, sub_kwargs))
        
        return sub_operations
    
    def _compose_model_parallel_results(self, operation: str, results: List[Any]) -> Any:
        """Compose results from model parallel execution."""
        # Placeholder - would be operation-specific
        return results[0] if results else None
    
    def _create_pipeline_stages(self, operation: str, args: Tuple, kwargs: Dict) -> List[Tuple[str, Dict]]:
        """Create pipeline stages for operation."""
        # Simplified pipeline creation
        if operation == 'hierarchical_bind':
            return [
                ('bind', {}),
                ('bundle', {}),
                ('normalize', {})
            ]
        elif operation == 'sequence_prediction':
            return [
                ('encode_sequence', {}),
                ('predict_next', {}),
                ('decode_result', {})
            ]
        else:
            return [(operation, kwargs)]


class ClusterManager:
    """Manages cluster nodes and their lifecycle."""
    
    def __init__(self):
        self.registered_nodes = {}
        self.node_capabilities = defaultdict(dict)
        
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node."""
        self.registered_nodes[node_info.node_id] = node_info
        self.node_capabilities[node_info.node_id] = node_info.capabilities
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node."""
        if node_id in self.registered_nodes:
            del self.registered_nodes[node_id]
            del self.node_capabilities[node_id]
            return True
        return False
    
    def update_node_status(self, node_id: str, status_update: Dict[str, Any]) -> bool:
        """Update node status information."""
        if node_id in self.registered_nodes:
            node = self.registered_nodes[node_id]
            
            # Update relevant fields
            if 'current_load' in status_update:
                node.current_load = status_update['current_load']
            if 'memory_available' in status_update:
                node.memory_available = status_update['memory_available']
            if 'status' in status_update:
                node.status = status_update['status']
            
            node.last_heartbeat = time.time()
            return True
        
        return False
    
    def get_cluster_capacity(self) -> Dict[str, Any]:
        """Get total cluster capacity."""
        active_nodes = [node for node in self.registered_nodes.values() if node.status == 'active']
        
        return {
            'total_nodes': len(self.registered_nodes),
            'active_nodes': len(active_nodes),
            'total_memory': sum(node.memory_available for node in active_nodes),
            'total_gpus': sum(1 for node in active_nodes if node.gpu_available),
            'avg_load': np.mean([node.current_load for node in active_nodes]) if active_nodes else 0
        }


class LoadBalancer:
    """Intelligent load balancer for distributing tasks."""
    
    def __init__(self, strategy: str = 'least_loaded'):
        self.strategy = strategy
        self.node_performance_history = defaultdict(deque)
        
    def select_node(self, available_nodes: List[NodeInfo], 
                   task_requirements: Dict[str, Any]) -> Optional[NodeInfo]:
        """Select best node for task based on current strategy."""
        
        if not available_nodes:
            return None
        
        if self.strategy == 'least_loaded':
            return self._least_loaded_selection(available_nodes, task_requirements)
        elif self.strategy == 'round_robin':
            return self._round_robin_selection(available_nodes)
        elif self.strategy == 'performance_based':
            return self._performance_based_selection(available_nodes, task_requirements)
        elif self.strategy == 'resource_aware':
            return self._resource_aware_selection(available_nodes, task_requirements)
        else:
            return available_nodes[0]  # Fallback to first available
    
    def _least_loaded_selection(self, nodes: List[NodeInfo], 
                               requirements: Dict[str, Any]) -> NodeInfo:
        """Select node with lowest current load."""
        return min(nodes, key=lambda node: node.current_load)
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Simple round-robin selection."""
        # In practice, would maintain round-robin state
        return nodes[int(time.time()) % len(nodes)]
    
    def _performance_based_selection(self, nodes: List[NodeInfo], 
                                   requirements: Dict[str, Any]) -> NodeInfo:
        """Select based on historical performance."""
        best_node = nodes[0]
        best_score = float('inf')
        
        for node in nodes:
            if node.node_id in self.node_performance_history:
                recent_times = list(self.node_performance_history[node.node_id])[-10:]  # Last 10 tasks
                if recent_times:
                    avg_time = np.mean(recent_times)
                    score = avg_time * (1 + node.current_load)  # Factor in current load
                    
                    if score < best_score:
                        best_score = score
                        best_node = node
            else:
                # New node - give it a chance
                if node.current_load < 0.5:  # Only if not heavily loaded
                    return node
        
        return best_node
    
    def _resource_aware_selection(self, nodes: List[NodeInfo], 
                                requirements: Dict[str, Any]) -> NodeInfo:
        """Select based on resource requirements and availability."""
        memory_req = requirements.get('memory_requirement', 0)
        
        # Filter nodes with sufficient resources
        suitable_nodes = [
            node for node in nodes 
            if node.memory_available >= memory_req
        ]
        
        if not suitable_nodes:
            # Fallback to least loaded if no node meets requirements exactly
            return self._least_loaded_selection(nodes, requirements)
        
        # Among suitable nodes, pick the one with best resource utilization
        best_node = suitable_nodes[0]
        best_utilization_score = float('inf')
        
        for node in suitable_nodes:
            # Calculate utilization score (lower is better)
            memory_utilization = memory_req / max(node.memory_available, 1)
            load_factor = node.current_load
            
            utilization_score = memory_utilization + load_factor
            
            if utilization_score < best_utilization_score:
                best_utilization_score = utilization_score
                best_node = node
        
        return best_node
    
    def record_task_performance(self, node_id: str, execution_time: float) -> None:
        """Record task performance for future load balancing decisions."""
        self.node_performance_history[node_id].append(execution_time)
        
        # Keep only recent history
        if len(self.node_performance_history[node_id]) > 100:
            self.node_performance_history[node_id].popleft()