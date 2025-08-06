"""Novel HDC algorithms: Temporal, Causal, Attention, and Meta-Learning approaches."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import networkx as nx


class TemporalHDC(ABC):
    """Temporal hyperdimensional computing with memory traces and prediction."""
    
    def __init__(self, dim: int, memory_length: int = 100, decay_rate: float = 0.95):
        self.dim = dim
        self.memory_length = memory_length
        self.decay_rate = decay_rate
        self.temporal_buffer = deque(maxlen=memory_length)
        self.temporal_weights = []
        
    @abstractmethod
    def temporal_binding(self, current_hv: Any, context_hvs: List[Any], 
                        temporal_positions: List[int]) -> Any:
        """Bind hypervector with temporal context."""
        pass
    
    @abstractmethod
    def sequence_prediction(self, sequence: List[Any], prediction_horizon: int = 1) -> List[Any]:
        """Predict future hypervectors in sequence."""
        pass
    
    @abstractmethod
    def temporal_interpolation(self, hv1: Any, hv2: Any, time_ratio: float) -> Any:
        """Interpolate between hypervectors based on temporal distance."""
        pass
    
    def add_temporal_experience(self, hv: Any, timestamp: Optional[float] = None) -> None:
        """Add experience to temporal buffer with timestamp."""
        if timestamp is None:
            timestamp = len(self.temporal_buffer)
        
        self.temporal_buffer.append((hv, timestamp))
        
        # Update temporal weights with exponential decay
        self.temporal_weights = [w * self.decay_rate for w in self.temporal_weights]
        self.temporal_weights.append(1.0)
        
        if len(self.temporal_weights) > self.memory_length:
            self.temporal_weights.pop(0)
    
    def temporal_similarity(self, query_hv: Any, time_window: int = 10) -> List[Tuple[Any, float, float]]:
        """Find similar hypervectors within time window."""
        if len(self.temporal_buffer) == 0:
            return []
        
        results = []
        current_time = self.temporal_buffer[-1][1] if self.temporal_buffer else 0
        
        for i, (hv, timestamp) in enumerate(self.temporal_buffer):
            # Check if within time window
            if current_time - timestamp <= time_window:
                similarity = self.cosine_similarity(query_hv, hv)
                temporal_weight = self.temporal_weights[i] if i < len(self.temporal_weights) else 0.0
                
                weighted_similarity = similarity * temporal_weight
                results.append((hv, similarity, weighted_similarity))
        
        # Sort by weighted similarity
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def create_temporal_trace(self, sequence: List[Any], trace_length: Optional[int] = None) -> Any:
        """Create compressed temporal trace of sequence."""
        if not sequence:
            return self._zero_hypervector()
        
        if trace_length is None:
            trace_length = min(len(sequence), 50)
        
        # Sample sequence if too long
        if len(sequence) > trace_length:
            indices = np.linspace(0, len(sequence) - 1, trace_length, dtype=int)
            sampled_sequence = [sequence[i] for i in indices]
        else:
            sampled_sequence = sequence
        
        # Create temporal trace with position encodings
        trace = self._zero_hypervector()
        
        for i, hv in enumerate(sampled_sequence):
            position_weight = self._temporal_position_weight(i, len(sampled_sequence))
            temporal_hv = self.temporal_binding(hv, [], [i])
            trace = self._add_weighted(trace, temporal_hv, position_weight)
        
        return self._normalize(trace)
    
    def temporal_clustering(self, similarity_threshold: float = 0.7, 
                          time_threshold: float = 10.0) -> List[List[int]]:
        """Cluster temporal experiences by similarity and temporal proximity."""
        if len(self.temporal_buffer) < 2:
            return []
        
        # Build similarity and temporal adjacency matrices
        n = len(self.temporal_buffer)
        similarity_matrix = np.zeros((n, n))
        temporal_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                hv_i, time_i = self.temporal_buffer[i]
                hv_j, time_j = self.temporal_buffer[j]
                
                # Similarity
                sim = self.cosine_similarity(hv_i, hv_j)
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim
                
                # Temporal proximity
                time_diff = abs(time_j - time_i)
                temporal_proximity = np.exp(-time_diff / time_threshold)
                temporal_matrix[i, j] = temporal_matrix[j, i] = temporal_proximity
        
        # Combined adjacency matrix
        combined_matrix = similarity_matrix * temporal_matrix
        
        # Find connected components above threshold
        adjacency = combined_matrix > similarity_threshold
        
        # Use graph-based clustering
        graph = nx.from_numpy_array(adjacency)
        clusters = list(nx.connected_components(graph))
        
        return [list(cluster) for cluster in clusters]
    
    @abstractmethod
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity."""
        pass
    
    @abstractmethod
    def _zero_hypervector(self) -> Any:
        """Create zero hypervector."""
        pass
    
    @abstractmethod
    def _add_weighted(self, hv1: Any, hv2: Any, weight: float) -> Any:
        """Add weighted hypervectors."""
        pass
    
    @abstractmethod
    def _normalize(self, hv: Any) -> Any:
        """Normalize hypervector."""
        pass
    
    def _temporal_position_weight(self, position: int, total_length: int) -> float:
        """Calculate weight for temporal position (recent positions weighted higher)."""
        # Exponential decay from recent to distant past
        relative_position = position / total_length
        return np.exp(-2 * (1 - relative_position))


class CausalHDC(ABC):
    """Causal hyperdimensional computing with intervention and counterfactual reasoning."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.causal_graph = nx.DiGraph()
        self.intervention_effects = {}
        self.confounders = set()
        
    @abstractmethod
    def causal_binding(self, cause_hv: Any, effect_hv: Any, strength: float = 1.0) -> Any:
        """Bind cause and effect with causal strength."""
        pass
    
    @abstractmethod
    def intervention(self, target_var: str, intervention_value: Any, 
                    causal_model: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intervention on causal model."""
        pass
    
    @abstractmethod
    def counterfactual_reasoning(self, observed_outcome: Any, 
                               alternative_cause: Any,
                               causal_chain: List[Tuple[str, Any]]) -> Any:
        """Perform counterfactual reasoning."""
        pass
    
    def learn_causal_structure(self, observations: List[Dict[str, Any]], 
                             significance_threshold: float = 0.05) -> nx.DiGraph:
        """Learn causal structure from observational data."""
        variables = list(observations[0].keys()) if observations else []
        
        # Convert observations to hypervector representations
        hv_observations = {}
        for var in variables:
            hv_observations[var] = [self._encode_value(obs[var]) for obs in observations]
        
        # Test all possible causal relationships
        causal_edges = []
        
        for cause in variables:
            for effect in variables:
                if cause != effect:
                    # Test causal relationship using conditional independence
                    causal_strength = self._test_causal_relationship(
                        hv_observations[cause], 
                        hv_observations[effect],
                        hv_observations
                    )
                    
                    if causal_strength > significance_threshold:
                        causal_edges.append((cause, effect, causal_strength))
        
        # Build causal graph
        self.causal_graph.clear()
        self.causal_graph.add_nodes_from(variables)
        
        for cause, effect, strength in causal_edges:
            self.causal_graph.add_edge(cause, effect, weight=strength)
        
        return self.causal_graph
    
    def identify_confounders(self, cause: str, effect: str) -> List[str]:
        """Identify confounding variables for cause-effect relationship."""
        if not self.causal_graph.has_node(cause) or not self.causal_graph.has_node(effect):
            return []
        
        confounders = []
        
        for node in self.causal_graph.nodes():
            if node != cause and node != effect:
                # Check if node is a confounder (affects both cause and effect)
                affects_cause = (self.causal_graph.has_edge(node, cause) or 
                               nx.has_path(self.causal_graph, node, cause))
                affects_effect = (self.causal_graph.has_edge(node, effect) or 
                                nx.has_path(self.causal_graph, node, effect))
                
                if affects_cause and affects_effect:
                    confounders.append(node)
        
        return confounders
    
    def causal_effect_estimation(self, cause: str, effect: str, 
                               observations: List[Dict[str, Any]],
                               adjustment_set: Optional[List[str]] = None) -> float:
        """Estimate causal effect using adjustment formula."""
        if adjustment_set is None:
            adjustment_set = self.identify_confounders(cause, effect)
        
        # Convert to hypervectors
        cause_hvs = [self._encode_value(obs[cause]) for obs in observations]
        effect_hvs = [self._encode_value(obs[effect]) for obs in observations]
        
        if adjustment_set:
            # Stratify by adjustment set
            strata = defaultdict(list)
            
            for i, obs in enumerate(observations):
                # Create stratum key
                stratum_key = tuple(obs[var] for var in adjustment_set)
                strata[stratum_key].append(i)
            
            # Calculate weighted average of stratum-specific effects
            total_effect = 0.0
            total_weight = 0.0
            
            for stratum_indices in strata.values():
                if len(stratum_indices) < 2:
                    continue
                
                stratum_cause_hvs = [cause_hvs[i] for i in stratum_indices]
                stratum_effect_hvs = [effect_hvs[i] for i in stratum_indices]
                
                # Calculate effect in this stratum
                stratum_effect = self._calculate_stratum_effect(stratum_cause_hvs, stratum_effect_hvs)
                
                weight = len(stratum_indices)
                total_effect += stratum_effect * weight
                total_weight += weight
            
            return total_effect / total_weight if total_weight > 0 else 0.0
        
        else:
            # Direct effect calculation (no confounders)
            return self._calculate_direct_effect(cause_hvs, effect_hvs)
    
    def mediation_analysis(self, cause: str, mediator: str, effect: str,
                          observations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform mediation analysis to decompose causal effects."""
        # Total effect (cause -> effect)
        total_effect = self.causal_effect_estimation(cause, effect, observations)
        
        # Direct effect (cause -> effect, controlling for mediator)
        direct_effect = self.causal_effect_estimation(
            cause, effect, observations, adjustment_set=[mediator]
        )
        
        # Indirect effect through mediator
        indirect_effect = total_effect - direct_effect
        
        # Mediation proportion
        mediation_proportion = (indirect_effect / total_effect) if total_effect != 0 else 0.0
        
        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediation_proportion': mediation_proportion
        }
    
    @abstractmethod
    def _encode_value(self, value: Any) -> Any:
        """Encode value as hypervector."""
        pass
    
    def _test_causal_relationship(self, cause_hvs: List[Any], effect_hvs: List[Any],
                                all_hvs: Dict[str, List[Any]]) -> float:
        """Test causal relationship strength between variables."""
        if len(cause_hvs) != len(effect_hvs):
            return 0.0
        
        # Simple correlation-based test (in practice, would use more sophisticated methods)
        similarities = []
        
        for i in range(len(cause_hvs)):
            sim = self.cosine_similarity(cause_hvs[i], effect_hvs[i])
            similarities.append(sim)
        
        # Return average similarity as proxy for causal strength
        return np.mean(similarities)
    
    def _calculate_stratum_effect(self, cause_hvs: List[Any], effect_hvs: List[Any]) -> float:
        """Calculate causal effect within a stratum."""
        if len(cause_hvs) != len(effect_hvs) or len(cause_hvs) == 0:
            return 0.0
        
        # Simplified effect calculation
        similarities = [self.cosine_similarity(c, e) for c, e in zip(cause_hvs, effect_hvs)]
        return np.mean(similarities)
    
    def _calculate_direct_effect(self, cause_hvs: List[Any], effect_hvs: List[Any]) -> float:
        """Calculate direct causal effect."""
        return self._calculate_stratum_effect(cause_hvs, effect_hvs)


class AttentionHDC(ABC):
    """Attention-based hyperdimensional computing with focus and context modulation."""
    
    def __init__(self, dim: int, num_attention_heads: int = 8):
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.attention_weights = {}
        self.context_stack = []
        
    @abstractmethod
    def multi_head_attention(self, query_hv: Any, key_hvs: List[Any], 
                           value_hvs: List[Any], mask: Optional[List[bool]] = None) -> Any:
        """Multi-head attention mechanism for hypervectors."""
        pass
    
    @abstractmethod
    def self_attention(self, hvs: List[Any], position_encodings: Optional[List[Any]] = None) -> List[Any]:
        """Self-attention over sequence of hypervectors."""
        pass
    
    @abstractmethod
    def cross_attention(self, queries: List[Any], keys: List[Any], 
                       values: List[Any]) -> List[Any]:
        """Cross-attention between two sequences."""
        pass
    
    def contextual_retrieval(self, query_hv: Any, memory_hvs: List[Any], 
                           context_hv: Optional[Any] = None,
                           top_k: int = 5) -> List[Tuple[Any, float]]:
        """Retrieve hypervectors with contextual attention weighting."""
        if not memory_hvs:
            return []
        
        # Compute attention weights
        attention_scores = []
        
        for memory_hv in memory_hvs:
            # Base similarity
            base_score = self.cosine_similarity(query_hv, memory_hv)
            
            # Context modulation
            if context_hv is not None:
                context_score = self.cosine_similarity(memory_hv, context_hv)
                # Combine base and context scores
                final_score = base_score * (1.0 + 0.5 * context_score)
            else:
                final_score = base_score
            
            attention_scores.append(final_score)
        
        # Apply softmax normalization
        attention_scores = np.array(attention_scores)
        attention_weights = self._softmax(attention_scores)
        
        # Create ranked results
        results = list(zip(memory_hvs, attention_weights))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def hierarchical_attention(self, input_hvs: List[Any], 
                             hierarchy_levels: int = 3) -> List[Any]:
        """Apply hierarchical attention across multiple levels."""
        current_hvs = input_hvs
        
        for level in range(hierarchy_levels):
            # Group hypervectors for this level
            group_size = max(1, len(current_hvs) // (2 ** level))
            groups = [current_hvs[i:i + group_size] 
                     for i in range(0, len(current_hvs), group_size)]
            
            # Apply attention within each group
            next_level_hvs = []
            
            for group in groups:
                if len(group) == 1:
                    next_level_hvs.extend(group)
                else:
                    # Self-attention within group
                    attended_group = self.self_attention(group)
                    
                    # Aggregate group with attention pooling
                    group_representation = self._attention_pooling(attended_group)
                    next_level_hvs.append(group_representation)
            
            current_hvs = next_level_hvs
        
        return current_hvs
    
    def attention_guided_composition(self, components: List[Any], 
                                   composition_query: Any) -> Any:
        """Compose hypervectors guided by attention to query."""
        if not components:
            return self._zero_hypervector()
        
        # Calculate attention weights for each component
        attention_weights = []
        
        for component in components:
            weight = self.cosine_similarity(composition_query, component)
            attention_weights.append(weight)
        
        # Normalize weights
        attention_weights = self._softmax(np.array(attention_weights))
        
        # Weighted composition
        result = self._zero_hypervector()
        
        for component, weight in zip(components, attention_weights):
            result = self._add_weighted(result, component, weight)
        
        return self._normalize(result)
    
    def temporal_attention(self, sequence: List[Any], query_time: int,
                          attention_window: int = 10) -> List[float]:
        """Calculate temporal attention weights for sequence elements."""
        sequence_length = len(sequence)
        attention_weights = np.zeros(sequence_length)
        
        for i in range(sequence_length):
            # Temporal distance
            temporal_distance = abs(i - query_time)
            
            if temporal_distance <= attention_window:
                # Attention based on temporal proximity and content similarity
                if query_time < sequence_length:
                    content_similarity = self.cosine_similarity(sequence[i], sequence[query_time])
                else:
                    content_similarity = 0.5  # Default similarity
                
                # Combine temporal and content attention
                temporal_weight = np.exp(-temporal_distance / attention_window)
                attention_weights[i] = temporal_weight * (0.5 + 0.5 * content_similarity)
        
        # Normalize
        if np.sum(attention_weights) > 0:
            attention_weights = attention_weights / np.sum(attention_weights)
        
        return attention_weights.tolist()
    
    def push_context(self, context_hv: Any, context_name: str = "default") -> None:
        """Push context onto context stack for hierarchical attention."""
        self.context_stack.append((context_hv, context_name))
    
    def pop_context(self) -> Optional[Tuple[Any, str]]:
        """Pop context from context stack."""
        return self.context_stack.pop() if self.context_stack else None
    
    def get_current_context(self) -> Optional[Any]:
        """Get current context from top of stack."""
        return self.context_stack[-1][0] if self.context_stack else None
    
    @abstractmethod
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity."""
        pass
    
    @abstractmethod
    def _zero_hypervector(self) -> Any:
        """Create zero hypervector."""
        pass
    
    @abstractmethod
    def _add_weighted(self, hv1: Any, hv2: Any, weight: float) -> Any:
        """Add weighted hypervectors."""
        pass
    
    @abstractmethod
    def _normalize(self, hv: Any) -> Any:
        """Normalize hypervector."""
        pass
    
    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply softmax normalization with temperature."""
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Numerical stability
        return exp_scores / np.sum(exp_scores)
    
    def _attention_pooling(self, hvs: List[Any]) -> Any:
        """Pool hypervectors using attention mechanism."""
        if not hvs:
            return self._zero_hypervector()
        
        if len(hvs) == 1:
            return hvs[0]
        
        # Calculate self-attention scores
        scores = []
        for i, hv_i in enumerate(hvs):
            score = 0.0
            for j, hv_j in enumerate(hvs):
                if i != j:
                    score += self.cosine_similarity(hv_i, hv_j)
            scores.append(score)
        
        # Normalize scores
        weights = self._softmax(np.array(scores))
        
        # Weighted sum
        result = self._zero_hypervector()
        for hv, weight in zip(hvs, weights):
            result = self._add_weighted(result, hv, weight)
        
        return self._normalize(result)


class MetaLearningHDC(ABC):
    """Meta-learning hyperdimensional computing for few-shot learning and adaptation."""
    
    def __init__(self, dim: int, meta_memory_size: int = 1000):
        self.dim = dim
        self.meta_memory_size = meta_memory_size
        self.task_memory = {}
        self.meta_parameters = {}
        self.adaptation_history = []
        
    @abstractmethod
    def meta_bind(self, task_hv: Any, data_hv: Any, meta_context: Any) -> Any:
        """Meta-learning binding operation."""
        pass
    
    @abstractmethod
    def few_shot_adaptation(self, support_set: List[Tuple[Any, Any]], 
                           query_set: List[Any],
                           num_adaptation_steps: int = 5) -> List[Any]:
        """Perform few-shot adaptation given support and query sets."""
        pass
    
    @abstractmethod
    def meta_gradient_update(self, task_gradients: List[Any], 
                           learning_rate: float = 0.01) -> Any:
        """Update meta-parameters using gradient information."""
        pass
    
    def learn_task_representation(self, task_data: List[Tuple[Any, Any]], 
                                task_id: str) -> Any:
        """Learn hypervector representation for a task."""
        if not task_data:
            return self._zero_hypervector()
        
        # Extract task-specific patterns
        input_hvs = [data for data, _ in task_data]
        output_hvs = [label for _, label in task_data]
        
        # Create task representation through statistical moments
        task_features = []
        
        # Mean representations
        mean_input = self._mean_hypervector(input_hvs)
        mean_output = self._mean_hypervector(output_hvs)
        task_features.extend([mean_input, mean_output])
        
        # Variance representations (approximate)
        if len(input_hvs) > 1:
            input_variance = self._variance_hypervector(input_hvs, mean_input)
            output_variance = self._variance_hypervector(output_hvs, mean_output)
            task_features.extend([input_variance, output_variance])
        
        # Cross-correlations
        cross_correlations = []
        for inp, out in task_data:
            cross_corr = self._correlation_hypervector(inp, out)
            cross_correlations.append(cross_corr)
        
        if cross_correlations:
            mean_cross_corr = self._mean_hypervector(cross_correlations)
            task_features.append(mean_cross_corr)
        
        # Combine all features
        task_representation = self._combine_task_features(task_features)
        
        # Store in task memory
        self.task_memory[task_id] = {
            'representation': task_representation,
            'data_size': len(task_data),
            'feature_count': len(task_features)
        }
        
        return task_representation
    
    def transfer_learning(self, source_task_id: str, target_task_data: List[Tuple[Any, Any]],
                         transfer_strength: float = 0.5) -> Any:
        """Transfer knowledge from source task to target task."""
        if source_task_id not in self.task_memory:
            # No source task available, learn from scratch
            return self.learn_task_representation(target_task_data, f"target_{len(self.task_memory)}")
        
        source_repr = self.task_memory[source_task_id]['representation']
        
        # Learn target task representation
        target_repr = self.learn_task_representation(target_task_data, f"target_{len(self.task_memory)}")
        
        # Transfer learning through weighted combination
        transferred_repr = self._transfer_combination(source_repr, target_repr, transfer_strength)
        
        return transferred_repr
    
    def continual_learning_update(self, new_task_data: List[Tuple[Any, Any]], 
                                task_id: str,
                                forgetting_prevention: float = 0.8) -> None:
        """Update model with new task while preventing catastrophic forgetting."""
        # Learn new task representation
        new_task_repr = self.learn_task_representation(new_task_data, task_id)
        
        # Update existing task representations to prevent forgetting
        for existing_task_id in self.task_memory:
            if existing_task_id != task_id:
                existing_repr = self.task_memory[existing_task_id]['representation']
                
                # Apply forgetting prevention
                consolidated_repr = self._consolidate_representations(
                    existing_repr, new_task_repr, forgetting_prevention
                )
                
                self.task_memory[existing_task_id]['representation'] = consolidated_repr
        
        # Record adaptation
        self.adaptation_history.append({
            'task_id': task_id,
            'data_size': len(new_task_data),
            'existing_tasks': len(self.task_memory)
        })
    
    def meta_optimization_step(self, task_batch: List[Dict[str, Any]],
                             meta_learning_rate: float = 0.01) -> Dict[str, float]:
        """Perform one step of meta-optimization across task batch."""
        meta_gradients = []
        task_losses = []
        
        for task_info in task_batch:
            task_id = task_info['id']
            support_set = task_info['support']
            query_set = task_info['query']
            
            # Adapt to task using current meta-parameters
            adapted_predictions = self.few_shot_adaptation(support_set, query_set)
            
            # Calculate task-specific loss/gradient
            task_loss = self._calculate_task_loss(query_set, adapted_predictions)
            task_losses.append(task_loss)
            
            # Approximate gradient for this task
            task_gradient = self._approximate_task_gradient(support_set, query_set, adapted_predictions)
            meta_gradients.append(task_gradient)
        
        # Update meta-parameters
        if meta_gradients:
            combined_gradient = self._combine_gradients(meta_gradients)
            self.meta_parameters = self._apply_meta_update(self.meta_parameters, combined_gradient, meta_learning_rate)
        
        return {
            'average_task_loss': np.mean(task_losses),
            'gradient_norm': self._gradient_norm(meta_gradients),
            'num_tasks': len(task_batch)
        }
    
    def similarity_based_task_selection(self, current_task_repr: Any, 
                                      num_similar_tasks: int = 3) -> List[str]:
        """Select most similar tasks for meta-learning."""
        if not self.task_memory:
            return []
        
        similarities = []
        for task_id, task_info in self.task_memory.items():
            task_repr = task_info['representation']
            similarity = self.cosine_similarity(current_task_repr, task_repr)
            similarities.append((task_id, similarity))
        
        # Sort by similarity and return top tasks
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in similarities[:num_similar_tasks]]
    
    def meta_memory_consolidation(self, consolidation_threshold: float = 0.7) -> None:
        """Consolidate meta-memory by merging similar task representations."""
        if len(self.task_memory) < 2:
            return
        
        task_ids = list(self.task_memory.keys())
        merged_tasks = []
        
        # Find pairs of similar tasks
        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                task_i = self.task_memory[task_ids[i]]
                task_j = self.task_memory[task_ids[j]]
                
                similarity = self.cosine_similarity(task_i['representation'], task_j['representation'])
                
                if similarity > consolidation_threshold:
                    # Merge tasks
                    merged_repr = self._merge_task_representations(
                        task_i['representation'], 
                        task_j['representation'],
                        task_i['data_size'],
                        task_j['data_size']
                    )
                    
                    merged_id = f"merged_{task_ids[i]}_{task_ids[j]}"
                    merged_tasks.append((merged_id, merged_repr, task_i['data_size'] + task_j['data_size']))
                    
                    # Mark for removal
                    task_i['to_remove'] = True
                    task_j['to_remove'] = True
        
        # Remove merged tasks and add consolidated ones
        for task_id in list(self.task_memory.keys()):
            if self.task_memory[task_id].get('to_remove', False):
                del self.task_memory[task_id]
        
        for merged_id, merged_repr, merged_size in merged_tasks:
            self.task_memory[merged_id] = {
                'representation': merged_repr,
                'data_size': merged_size,
                'feature_count': 0  # Will be recalculated if needed
            }
    
    # Abstract methods for backend implementation
    
    @abstractmethod
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity."""
        pass
    
    @abstractmethod
    def _zero_hypervector(self) -> Any:
        """Create zero hypervector."""
        pass
    
    @abstractmethod
    def _mean_hypervector(self, hvs: List[Any]) -> Any:
        """Calculate mean of hypervectors."""
        pass
    
    @abstractmethod
    def _variance_hypervector(self, hvs: List[Any], mean_hv: Any) -> Any:
        """Calculate variance hypervector."""
        pass
    
    @abstractmethod
    def _correlation_hypervector(self, hv1: Any, hv2: Any) -> Any:
        """Calculate correlation hypervector."""
        pass
    
    @abstractmethod
    def _combine_task_features(self, features: List[Any]) -> Any:
        """Combine task features into single representation."""
        pass
    
    @abstractmethod
    def _transfer_combination(self, source_hv: Any, target_hv: Any, strength: float) -> Any:
        """Combine source and target representations for transfer."""
        pass
    
    @abstractmethod
    def _consolidate_representations(self, existing_hv: Any, new_hv: Any, prevention_strength: float) -> Any:
        """Consolidate representations to prevent forgetting."""
        pass
    
    @abstractmethod
    def _calculate_task_loss(self, query_set: List[Any], predictions: List[Any]) -> float:
        """Calculate task-specific loss."""
        pass
    
    @abstractmethod
    def _approximate_task_gradient(self, support_set: List[Tuple[Any, Any]], 
                                 query_set: List[Any], predictions: List[Any]) -> Any:
        """Approximate gradient for task."""
        pass
    
    @abstractmethod
    def _combine_gradients(self, gradients: List[Any]) -> Any:
        """Combine gradients from multiple tasks."""
        pass
    
    @abstractmethod
    def _apply_meta_update(self, params: Dict[str, Any], gradient: Any, lr: float) -> Dict[str, Any]:
        """Apply meta-parameter update."""
        pass
    
    def _gradient_norm(self, gradients: List[Any]) -> float:
        """Calculate norm of gradient list (simplified)."""
        return float(len(gradients))  # Placeholder implementation
    
    def _merge_task_representations(self, repr1: Any, repr2: Any, size1: int, size2: int) -> Any:
        """Merge two task representations with size weighting."""
        total_size = size1 + size2
        weight1 = size1 / total_size
        weight2 = size2 / total_size
        
        # Weighted combination (backend-specific implementation needed)
        return self._transfer_combination(repr1, repr2, weight2)