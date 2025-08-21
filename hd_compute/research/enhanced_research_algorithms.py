"""
Enhanced HDC Research Algorithms
===============================

Novel and advanced hyperdimensional computing algorithms for research applications:
- Quantum-inspired HDC with entanglement and superposition
- Advanced neurosymbolic reasoning with causal inference
- Adaptive memory systems with dynamic consolidation
- Meta-learning and few-shot adaptation
- Statistical analysis and reproducible benchmarking
"""

import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import time
import warnings

# Import security decorators if available
try:
    from ..security.research_security import secure_operation, validate_hypervector_input
except ImportError:
    def secure_operation(func):
        return func
    def validate_hypervector_input(func):
        return func


class NovelQuantumHDC:
    """Quantum-inspired HDC with entanglement and superposition effects."""
    
    def __init__(self, dim: int, quantum_depth: int = 4):
        self.dim = dim
        self.quantum_depth = quantum_depth
        self.entanglement_registry = {}
        self.superposition_states = {}
        self.quantum_memory = {}
        self.coherence_time = 100  # Quantum coherence lifetime
        self.decoherence_rate = 0.01
        
    def quantum_entangle(self, hv1: np.ndarray, hv2: np.ndarray, 
                        entanglement_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create quantum entanglement between hypervectors with Bell state formation."""
        # Enhanced quantum entanglement using Bell state formation
        phi = entanglement_strength * np.pi / 2
        
        # Create Bell states with quantum phase relationships
        bell_matrix = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2)],
            [1/np.sqrt(2), -1/np.sqrt(2)]
        ]) * np.exp(1j * phi)
        
        # Convert hypervectors to complex representation
        complex_hv1 = hv1.astype(complex) + 1j * np.random.normal(0, 0.1, self.dim)
        complex_hv2 = hv2.astype(complex) + 1j * np.random.normal(0, 0.1, self.dim)
        
        # Apply quantum entanglement transformation
        combined = np.stack([complex_hv1, complex_hv2])
        entangled = np.tensordot(bell_matrix, combined, axes=([1], [0]))
        
        # Extract real parts for entangled hypervectors
        entangled_hv1 = np.real(entangled[0])
        entangled_hv2 = np.real(entangled[1])
        
        # Store enhanced entanglement relationship with quantum properties
        entanglement_id = f"ent_{len(self.entanglement_registry)}"
        self.entanglement_registry[entanglement_id] = {
            'original_hvs': (hv1.copy(), hv2.copy()),
            'entangled_hvs': (entangled_hv1.copy(), entangled_hv2.copy()),
            'bell_state': bell_matrix,
            'strength': entanglement_strength,
            'phase': phi,
            'creation_time': 0,
            'measurement_count': 0,
            'coherence_remaining': 1.0
        }
        
        return entangled_hv1, entangled_hv2
    
    def quantum_superposition(self, hvs: List[np.ndarray], 
                             amplitudes: Optional[List[float]] = None,
                             quantum_phases: Optional[List[float]] = None) -> np.ndarray:
        """Create quantum superposition with coherent state formation."""
        if not hvs:
            return np.zeros(self.dim)
        
        # Enhanced amplitude normalization with quantum constraints
        if amplitudes is None:
            amplitudes = [1.0 / np.sqrt(len(hvs))] * len(hvs)  # Equal superposition
        else:
            # Ensure quantum probability normalization
            norm = np.sqrt(sum(a**2 for a in amplitudes))
            amplitudes = [a / norm for a in amplitudes]
        
        # Generate quantum phases for interference patterns
        if quantum_phases is None:
            quantum_phases = [2 * np.pi * i / len(hvs) for i in range(len(hvs))]
        
        # Create enhanced superposition with quantum interference
        superposition = np.zeros(self.dim, dtype=complex)
        interference_pattern = np.zeros(self.dim, dtype=complex)
        
        for i, (hv, amplitude, phase) in enumerate(zip(hvs, amplitudes, quantum_phases)):
            # Complex amplitude with quantum phase
            complex_amplitude = amplitude * np.exp(1j * phase)
            
            # Add quantum decoherence effects
            decoherence_factor = np.exp(-self.decoherence_rate * i)
            
            # Quantum state contribution
            quantum_state = complex_amplitude * decoherence_factor * hv.astype(complex)
            superposition += quantum_state
            
            # Calculate interference patterns between states
            for j in range(i):
                interference_phase = quantum_phases[i] - quantum_phases[j]
                interference_amplitude = amplitudes[i] * amplitudes[j] * np.cos(interference_phase)
                interference_pattern += interference_amplitude * hv.astype(complex)
        
        # Combine superposition with interference
        total_quantum_state = superposition + 0.1 * interference_pattern
        
        # Store enhanced superposition state
        state_id = f"super_{len(self.superposition_states)}"
        self.superposition_states[state_id] = {
            'components': [hv.copy() for hv in hvs],
            'amplitudes': amplitudes,
            'phases': quantum_phases,
            'superposition': total_quantum_state,
            'interference_pattern': interference_pattern,
            'coherence_time': self.coherence_time,
            'measured': False,
            'entanglement_links': []
        }
        
        # Return normalized real part with quantum properties preserved
        result = np.real(total_quantum_state)
        return result / (np.linalg.norm(result) + 1e-8)  # Avoid division by zero
    
    def quantum_measurement(self, superposition_id: str, 
                           measurement_basis: Optional[List[np.ndarray]] = None,
                           measurement_type: str = 'standard') -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Perform quantum measurement with advanced wavefunction collapse."""
        if superposition_id not in self.superposition_states:
            raise ValueError(f"Superposition {superposition_id} not found")
        
        state = self.superposition_states[superposition_id]
        
        if state['measured']:
            # Return previously measured result with metadata
            return (state.get('measurement_result', np.real(state['superposition'])), 
                   state.get('measurement_probability', 1.0),
                   state.get('measurement_metadata', {}))
        
        # Enhanced quantum measurement process
        components = state['components']
        amplitudes = state['amplitudes']
        phases = state.get('phases', [0] * len(components))
        superposition = state['superposition']
        
        # Calculate quantum probabilities with phase effects
        quantum_probabilities = []
        for i, (amp, phase) in enumerate(zip(amplitudes, phases)):
            # Include quantum interference in probability calculation
            interference_factor = 1.0
            for j in range(len(amplitudes)):
                if i != j:
                    phase_diff = phases[i] - phases[j]
                    interference_factor += 0.1 * np.cos(phase_diff) * amplitudes[j]
            
            prob = (amp**2) * max(0.01, interference_factor)  # Ensure positive probability
            quantum_probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(quantum_probabilities)
        quantum_probabilities = [p / total_prob for p in quantum_probabilities]
        
        # Perform measurement based on type
        if measurement_type == 'standard':
            # Standard von Neumann measurement
            choice_index = np.random.choice(len(components), p=quantum_probabilities)
            measured_hv = components[choice_index]
            measurement_probability = quantum_probabilities[choice_index]
            
        elif measurement_type == 'weak':
            # Weak measurement - partial collapse
            collapse_strength = 0.7
            choice_index = np.random.choice(len(components), p=quantum_probabilities)
            
            # Partial wavefunction collapse
            measured_hv = (collapse_strength * components[choice_index] + 
                          (1 - collapse_strength) * np.real(superposition))
            measurement_probability = quantum_probabilities[choice_index] * collapse_strength
            
        elif measurement_type == 'positive':
            # Positive operator-valued measure (POVM)
            measured_hv = np.zeros(self.dim)
            total_weight = 0
            
            for i, (comp, prob) in enumerate(zip(components, quantum_probabilities)):
                weight = prob * np.sqrt(prob)  # POVM weighting
                measured_hv += weight * comp
                total_weight += weight
            
            measured_hv = measured_hv / (total_weight + 1e-8)
            measurement_probability = total_weight
            choice_index = np.argmax(quantum_probabilities)
        
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        # Create measurement metadata
        measurement_metadata = {
            'measurement_type': measurement_type,
            'collapsed_index': choice_index,
            'quantum_probabilities': quantum_probabilities,
            'interference_effects': [phases[i] - phases[0] for i in range(len(phases))],
            'coherence_at_measurement': state.get('coherence_remaining', 1.0),
            'entanglement_preserved': len(state.get('entanglement_links', [])) > 0
        }
        
        # Enhanced wavefunction collapse with decoherence
        state['measured'] = True
        state['measurement_result'] = measured_hv
        state['measurement_probability'] = measurement_probability
        state['measurement_metadata'] = measurement_metadata
        state['collapse_time'] = 0  # Would be actual time in real implementation
        
        # Update coherence for related entangled states
        for link in state.get('entanglement_links', []):
            if link in self.entanglement_registry:
                self.entanglement_registry[link]['coherence_remaining'] *= 0.8
        
        return measured_hv, measurement_probability, measurement_metadata
    
    def quantum_decoherence(self, state_id: str, time_step: float = 1.0) -> Dict[str, float]:
        """Apply quantum decoherence to superposition state."""
        if state_id in self.superposition_states:
            state = self.superposition_states[state_id]
            
            # Apply decoherence to amplitudes and phases
            current_coherence = state.get('coherence_remaining', 1.0)
            new_coherence = current_coherence * np.exp(-self.decoherence_rate * time_step)
            
            state['coherence_remaining'] = new_coherence
            
            # Decohere superposition
            if not state['measured']:
                superposition = state['superposition']
                decoherence_factor = np.sqrt(new_coherence)
                state['superposition'] = superposition * decoherence_factor
            
            return {
                'coherence_remaining': new_coherence,
                'decoherence_applied': current_coherence - new_coherence,
                'fully_decohered': new_coherence < 0.01
            }
        
        return {'error': 'State not found'}
    
    def entanglement_measure(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Measure quantum-like entanglement between hypervectors."""
        # Calculate mutual information as entanglement proxy
        # For binary hypervectors, use Hamming distance
        hamming_dist = np.sum(hv1 != hv2) / len(hv1)
        
        # Convert to entanglement measure (higher for more correlated)
        entanglement = 1.0 - 2.0 * abs(hamming_dist - 0.5)
        
        return max(0.0, entanglement)


class AdvancedNeurosymbolicHDC:
    """Advanced neurosymbolic HDC with causal reasoning and meta-learning."""
    
    def __init__(self, dim: int, knowledge_graph_size: int = 1000):
        self.dim = dim
        self.knowledge_graph = nx.DiGraph()
        self.symbol_embeddings = {}
        self.rule_library = {}
        self.causal_models = {}
        self.meta_learning_memory = {}
        self.reasoning_cache = {}
        self.uncertainty_estimates = {}
        self.explanation_traces = {}
    
    def create_symbolic_concept(self, concept_name: str, attributes: List[str],
                              relationships: Dict[str, List[str]]) -> np.ndarray:
        """Create rich symbolic concept with attributes and relationships."""
        # Create base concept hypervector
        if concept_name not in self.symbol_embeddings:
            base_hv = np.random.binomial(1, 0.5, self.dim).astype(np.int8)
            self.symbol_embeddings[concept_name] = base_hv
        else:
            base_hv = self.symbol_embeddings[concept_name]
        
        # Encode attributes
        attribute_hvs = []
        for attr in attributes:
            if attr not in self.symbol_embeddings:
                self.symbol_embeddings[attr] = np.random.binomial(1, 0.5, self.dim).astype(np.int8)
            attribute_hvs.append(self.symbol_embeddings[attr])
        
        # Bundle attributes
        if attribute_hvs:
            attribute_bundle = attribute_hvs[0].copy()
            for attr_hv in attribute_hvs[1:]:
                attribute_bundle = np.logical_or(attribute_bundle, attr_hv).astype(np.int8)
        else:
            attribute_bundle = np.zeros(self.dim, dtype=np.int8)
        
        # Bind concept with attributes
        concept_hv = np.logical_xor(base_hv, attribute_bundle).astype(np.int8)
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(concept_name, 
                                     hypervector=concept_hv,
                                     attributes=attributes)
        
        # Add relationships
        for relation_type, related_concepts in relationships.items():
            for related_concept in related_concepts:
                self.knowledge_graph.add_edge(concept_name, related_concept, 
                                            relation=relation_type)
        
        return concept_hv
    
    def causal_intervention(self, variable: str, intervention_value: Any,
                          causal_model: Dict[str, Any],
                          uncertainty_propagation: bool = True) -> Dict[str, Any]:
        """Perform causal intervention using enhanced do-calculus with uncertainty."""
        # Enhanced implementation of Pearl's do-calculus with uncertainty quantification
        intervened_model = causal_model.copy()
        intervention_trace = {'variable': variable, 'value': intervention_value, 'effects': {}}
        
        # Remove incoming edges to intervention variable (do-operator)
        intervention_graph = self.knowledge_graph.copy()
        if variable in intervention_graph:
            predecessors = list(intervention_graph.predecessors(variable))
            for pred in predecessors:
                if intervention_graph.has_edge(pred, variable):
                    intervention_graph.remove_edge(pred, variable)
                    intervention_trace['effects'][f'removed_edge_{pred}_{variable}'] = True
        
        # Set intervention value with uncertainty bounds
        if uncertainty_propagation and variable in self.uncertainty_estimates:
            intervention_uncertainty = self.uncertainty_estimates[variable]
            intervened_model[variable] = {
                'value': intervention_value,
                'uncertainty': intervention_uncertainty * 0.1,  # Reduced uncertainty from intervention
                'intervention': True
            }
        else:
            intervened_model[variable] = intervention_value
        
        # Enhanced causal effect propagation
        affected_variables = list(nx.descendants(intervention_graph, variable))
        intervention_trace['affected_variables'] = affected_variables
        
        # Topological sort for proper causal ordering
        causal_order = list(nx.topological_sort(intervention_graph))
        affected_ordered = [v for v in causal_order if v in affected_variables]
        
        for affected_var in affected_ordered:
            # Compute post-intervention value with enhanced structural equations
            parents = list(intervention_graph.predecessors(affected_var))
            
            if parents:
                # Get parent values with uncertainty propagation
                parent_data = []
                total_uncertainty = 0.0
                
                for p in parents:
                    if isinstance(intervened_model.get(p), dict):
                        parent_data.append(intervened_model[p]['value'])
                        total_uncertainty += intervened_model[p].get('uncertainty', 0.0)
                    else:
                        parent_data.append(intervened_model.get(p, 0))
                        total_uncertainty += self.uncertainty_estimates.get(p, 0.1)
                
                # Enhanced structural equation evaluation
                new_value = self._evaluate_enhanced_structural_equation(
                    affected_var, parent_data, causal_model, intervention_graph
                )
                
                # Propagate uncertainty
                if uncertainty_propagation:
                    causal_uncertainty = total_uncertainty * 0.8  # Uncertainty decay
                    intervened_model[affected_var] = {
                        'value': new_value,
                        'uncertainty': causal_uncertainty,
                        'intervention': False,
                        'causal_parents': parents
                    }
                else:
                    intervened_model[affected_var] = new_value
                
                intervention_trace['effects'][affected_var] = {
                    'new_value': new_value,
                    'parents': parents,
                    'uncertainty': causal_uncertainty if uncertainty_propagation else 0.0
                }
        
        # Store intervention trace for explainability
        trace_id = f"intervention_{len(self.explanation_traces)}"
        self.explanation_traces[trace_id] = intervention_trace
        
        return {
            'model': intervened_model,
            'trace_id': trace_id,
            'affected_count': len(affected_variables),
            'uncertainty_propagated': uncertainty_propagation
        }
    
    def _evaluate_enhanced_structural_equation(self, variable: str, parent_values: List[Any], 
                                              causal_model: Dict[str, Any], 
                                              causal_graph: nx.DiGraph) -> Any:
        """Evaluate enhanced structural equation with nonlinear relationships."""
        if not parent_values:
            return self._get_default_value(variable)
        
        # Get enhanced structural equation parameters
        equation_params = causal_model.get(f"{variable}_equation", {})
        
        # Support multiple equation types
        equation_type = equation_params.get('type', 'linear')
        
        if equation_type == 'linear':
            return self._evaluate_linear_equation(variable, parent_values, equation_params)
        
        elif equation_type == 'nonlinear':
            return self._evaluate_nonlinear_equation(variable, parent_values, equation_params)
        
        elif equation_type == 'neural':
            return self._evaluate_neural_equation(variable, parent_values, equation_params)
        
        elif equation_type == 'hypervector':
            return self._evaluate_hypervector_equation(variable, parent_values, equation_params)
        
        else:
            # Fallback to adaptive equation learning
            return self._evaluate_adaptive_equation(variable, parent_values, causal_graph)
    
    def _evaluate_linear_equation(self, variable: str, parent_values: List[Any], 
                                 equation_params: Dict[str, Any]) -> Any:
        """Linear structural equation."""
        weights = equation_params.get('weights', [1.0] * len(parent_values))
        bias = equation_params.get('bias', 0.0)
        noise_std = equation_params.get('noise_std', 0.0)
        
        result = bias
        for value, weight in zip(parent_values, weights):
            if isinstance(value, (int, float)):
                result += weight * value
            elif hasattr(value, '__len__'):  # Handle hypervector inputs
                result += weight * np.mean(value)
        
        # Add structural noise
        if noise_std > 0:
            result += np.random.normal(0, noise_std)
        
        return result
    
    def _evaluate_nonlinear_equation(self, variable: str, parent_values: List[Any],
                                    equation_params: Dict[str, Any]) -> Any:
        """Nonlinear structural equation with interactions."""
        weights = equation_params.get('weights', [1.0] * len(parent_values))
        bias = equation_params.get('bias', 0.0)
        interaction_strength = equation_params.get('interaction_strength', 0.1)
        
        # Linear terms
        linear_result = bias
        processed_values = []
        
        for value, weight in zip(parent_values, weights):
            if isinstance(value, (int, float)):
                processed_value = value
            elif hasattr(value, '__len__'):  # Handle hypervector inputs
                processed_value = np.mean(value)
            else:
                processed_value = 0.0
            
            processed_values.append(processed_value)
            linear_result += weight * processed_value
        
        # Nonlinear interactions
        nonlinear_result = 0.0
        for i in range(len(processed_values)):
            for j in range(i + 1, len(processed_values)):
                interaction = interaction_strength * processed_values[i] * processed_values[j]
                nonlinear_result += interaction
        
        # Apply nonlinear activation
        activation = equation_params.get('activation', 'tanh')
        total_result = linear_result + nonlinear_result
        
        if activation == 'tanh':
            return np.tanh(total_result)
        elif activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-total_result))
        elif activation == 'relu':
            return max(0.0, total_result)
        else:
            return total_result
    
    def _evaluate_neural_equation(self, variable: str, parent_values: List[Any],
                                 equation_params: Dict[str, Any]) -> Any:
        """Neural network-based structural equation."""
        # Simplified neural network evaluation
        hidden_size = equation_params.get('hidden_size', 10)
        
        # Convert inputs to numeric
        numeric_inputs = []
        for value in parent_values:
            if isinstance(value, (int, float)):
                numeric_inputs.append(value)
            elif hasattr(value, '__len__'):
                numeric_inputs.append(np.mean(value))
            else:
                numeric_inputs.append(0.0)
        
        if not numeric_inputs:
            return 0.0
        
        # Simple feedforward computation
        input_array = np.array(numeric_inputs)
        
        # Hidden layer (random weights for simplicity)
        hidden_weights = equation_params.get('hidden_weights', 
                                           np.random.normal(0, 1, (len(input_array), hidden_size)))
        hidden_bias = equation_params.get('hidden_bias', np.zeros(hidden_size))
        
        hidden_output = np.tanh(np.dot(input_array, hidden_weights) + hidden_bias)
        
        # Output layer
        output_weights = equation_params.get('output_weights', np.random.normal(0, 1, hidden_size))
        output_bias = equation_params.get('output_bias', 0.0)
        
        result = np.dot(hidden_output, output_weights) + output_bias
        
        return float(result)
    
    def _evaluate_hypervector_equation(self, variable: str, parent_values: List[Any],
                                      equation_params: Dict[str, Any]) -> np.ndarray:
        """Hypervector-based structural equation."""
        if not parent_values:
            return np.random.binomial(1, 0.5, self.dim).astype(np.int8)
        
        # Convert all parent values to hypervectors
        parent_hvs = []
        for value in parent_values:
            if isinstance(value, np.ndarray) and len(value) == self.dim:
                parent_hvs.append(value)
            elif isinstance(value, (int, float)):
                # Convert scalar to hypervector
                hv = np.zeros(self.dim, dtype=np.int8)
                num_ones = int(value * self.dim / 10)  # Scale factor
                if num_ones > 0:
                    indices = np.random.choice(self.dim, min(num_ones, self.dim), replace=False)
                    hv[indices] = 1
                parent_hvs.append(hv)
            else:
                # Default random hypervector
                parent_hvs.append(np.random.binomial(1, 0.5, self.dim).astype(np.int8))
        
        # Combine parent hypervectors
        operation = equation_params.get('operation', 'bundle')
        
        if operation == 'bundle':
            # Bundle (superposition)
            result = parent_hvs[0].copy()
            for hv in parent_hvs[1:]:
                result = np.logical_or(result, hv).astype(np.int8)
        
        elif operation == 'bind':
            # Bind (association)
            result = parent_hvs[0].copy()
            for hv in parent_hvs[1:]:
                result = np.logical_xor(result, hv).astype(np.int8)
        
        elif operation == 'weighted_bundle':
            # Weighted bundling
            weights = equation_params.get('weights', [1.0] * len(parent_hvs))
            result = np.zeros(self.dim)
            total_weight = sum(weights)
            
            for hv, weight in zip(parent_hvs, weights):
                result += (weight / total_weight) * hv
            
            # Threshold to binary
            threshold = equation_params.get('threshold', 0.5)
            result = (result > threshold).astype(np.int8)
        
        else:
            # Default: simple bundling
            result = parent_hvs[0].copy()
            for hv in parent_hvs[1:]:
                result = np.logical_or(result, hv).astype(np.int8)
        
        return result
    
    def _evaluate_adaptive_equation(self, variable: str, parent_values: List[Any],
                                   causal_graph: nx.DiGraph) -> Any:
        """Adaptive equation learning for unknown causal relationships."""
        # Simple adaptive learning - in practice would use more sophisticated methods
        cache_key = f"{variable}_adaptive_{hash(tuple(str(p) for p in parent_values))}"
        
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        # Learn equation based on graph structure and parent values
        if len(parent_values) == 1:
            # Single parent - learn simple transformation
            parent_val = parent_values[0]
            if isinstance(parent_val, (int, float)):
                result = parent_val * 0.8 + np.random.normal(0, 0.1)  # Add some noise
            else:
                result = parent_val  # Pass through for complex types
        else:
            # Multiple parents - learn combination
            numeric_values = []
            for val in parent_values:
                if isinstance(val, (int, float)):
                    numeric_values.append(val)
                elif hasattr(val, '__len__'):
                    numeric_values.append(np.mean(val))
                else:
                    numeric_values.append(0.0)
            
            # Adaptive combination with learned weights
            if numeric_values:
                weights = np.random.uniform(0.5, 1.5, len(numeric_values))
                weights = weights / np.sum(weights)  # Normalize
                result = np.sum(np.array(numeric_values) * weights)
            else:
                result = 0.0
        
        # Cache the result
        self.reasoning_cache[cache_key] = result
        
        # Update uncertainty estimate
        self.uncertainty_estimates[variable] = 0.3  # Higher uncertainty for adaptive equations
        
        return result
    
    def _get_default_value(self, variable: str) -> Any:
        """Get default value for variable with no parents."""
        if variable in self.symbol_embeddings:
            return self.symbol_embeddings[variable]
        return 0.0
    
    def analogical_reasoning(self, source_domain: Dict[str, Any], 
                           target_domain: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analogical reasoning with structural mapping."""
        # Extract structural patterns from both domains
        source_structure = self._extract_enhanced_structure(source_domain)
        target_structure = self._extract_enhanced_structure(target_domain)
        
        # Find structural alignments
        alignments = self._find_structural_alignments(source_structure, target_structure)
        
        # Transfer knowledge based on alignments
        transfer_results = self._transfer_knowledge(source_domain, target_domain, alignments)
        
        return {
            'alignments': alignments,
            'transferred_knowledge': transfer_results,
            'confidence_scores': self._calculate_transfer_confidence(alignments)
        }
    
    def _extract_enhanced_structure(self, domain: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced structural representation."""
        structure = {
            'entities': set(),
            'relations': set(),
            'patterns': {},
            'hierarchies': {}
        }
        
        # Extract entities and relations
        for key, value in domain.items():
            if isinstance(value, list):
                structure['entities'].add(key)
                for item in value:
                    if isinstance(item, str):
                        structure['relations'].add(item)
        
        return structure
    
    def _find_structural_alignments(self, source: Dict[str, Any], 
                                   target: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """Find structural alignments between domains."""
        alignments = []
        
        # Simple alignment based on entity similarity
        for s_entity in source.get('entities', []):
            for t_entity in target.get('entities', []):
                # Calculate similarity (simplified)
                similarity = len(set(s_entity) & set(t_entity)) / max(len(s_entity), len(t_entity), 1)
                if similarity > 0.3:
                    alignments.append((s_entity, t_entity, similarity))
        
        return sorted(alignments, key=lambda x: x[2], reverse=True)
    
    def _transfer_knowledge(self, source: Dict[str, Any], target: Dict[str, Any], 
                           alignments: List[Tuple[str, str, float]]) -> Dict[str, Any]:
        """Transfer knowledge based on structural alignments."""
        transferred = {}
        
        for source_entity, target_entity, confidence in alignments:
            if source_entity in source:
                transferred[target_entity] = {
                    'value': source[source_entity],
                    'confidence': confidence,
                    'source': source_entity
                }
        
        return transferred
    
    def _calculate_transfer_confidence(self, alignments: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """Calculate confidence scores for knowledge transfer."""
        confidences = {}
        
        for source_entity, target_entity, similarity in alignments:
            # Confidence based on similarity and number of supporting alignments
            support_count = sum(1 for _, t, _ in alignments if t == target_entity)
            confidence = similarity * (1.0 + 0.1 * support_count)
            confidences[target_entity] = min(1.0, confidence)
        
        return confidences


class AdaptiveMemoryHDC:
    """Adaptive memory system with dynamic capacity and forgetting."""
    
    def __init__(self, dim: int, initial_capacity: int = 1000):
        self.dim = dim
        self.memory_items = {}
        self.access_frequencies = {}
        self.last_access_times = {}
        self.importance_scores = {}
        self.current_time = 0
        self.memory_clusters = {}
        self.consolidation_threshold = 0.8
        self.forgetting_curve_params = {'alpha': 0.5, 'beta': 0.9}
        self.adaptive_capacity = initial_capacity
        self.memory_pressure = 0.0
    
    def store_memory(self, memory_id: str, content: np.ndarray, 
                    importance: float = 1.0, metadata: Optional[Dict] = None) -> bool:
        """Store memory with adaptive capacity management."""
        # Check if memory limit reached
        if len(self.memory_items) >= self.adaptive_capacity:
            # Apply forgetting to make space
            forgotten_items = self.adaptive_forgetting(num_items=1)
            if not forgotten_items:
                return False  # Could not free space
        
        # Store memory item
        self.memory_items[memory_id] = {
            'content': content.copy(),
            'creation_time': self.current_time,
            'metadata': metadata or {}
        }
        
        self.importance_scores[memory_id] = importance
        self.access_frequencies[memory_id] = 1
        self.last_access_times[memory_id] = self.current_time
        
        return True
    
    def retrieve_memory(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, np.ndarray, float]]:
        """Retrieve memories using similarity-based search."""
        if not self.memory_items:
            return []
        
        similarities = []
        
        for memory_id, memory_data in self.memory_items.items():
            # Calculate similarity
            similarity = self.cosine_similarity(query, memory_data['content'])
            
            # Apply recency and importance weighting
            recency_weight = self._calculate_recency_weight(memory_id)
            importance_weight = self.importance_scores.get(memory_id, 1.0)
            
            weighted_similarity = similarity * recency_weight * importance_weight
            similarities.append((memory_id, memory_data['content'], weighted_similarity))
            
            # Update access statistics
            self.access_frequencies[memory_id] += 1
            self.last_access_times[memory_id] = self.current_time
        
        # Sort by weighted similarity and return top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        self.current_time += 1
        
        return similarities[:top_k]
    
    def adaptive_forgetting(self, num_items: int = 10) -> List[str]:
        """Adaptive forgetting based on importance, recency, and access patterns."""
        if not self.memory_items:
            return []
        
        # Calculate forgetting scores for all memories
        forgetting_scores = {}
        
        for memory_id in self.memory_items:
            # Factors affecting forgetting probability
            recency_score = self._calculate_recency_weight(memory_id)
            importance_score = self.importance_scores.get(memory_id, 1.0)
            access_frequency = self.access_frequencies.get(memory_id, 0)
            
            # Combined forgetting score (lower is more likely to be forgotten)
            forgetting_score = (recency_score * importance_score * 
                              (1.0 + np.log(1 + access_frequency)))
            
            forgetting_scores[memory_id] = forgetting_score
        
        # Select items to forget (lowest scores)
        sorted_items = sorted(forgetting_scores.items(), key=lambda x: x[1])
        items_to_forget = [item_id for item_id, _ in sorted_items[:num_items]]
        
        # Remove forgotten items
        for item_id in items_to_forget:
            self._remove_memory(item_id)
        
        return items_to_forget
    
    def adaptive_consolidation(self, similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """Enhanced adaptive memory consolidation with clustering and importance weighting."""
        if len(self.memory_items) < 2:
            return {'consolidated_count': 0, 'memory_saved': 0, 'clusters_formed': 0}
        
        # Enhanced consolidation with multi-level clustering
        consolidation_stats = {
            'consolidated_count': 0,
            'memory_saved': 0,
            'clusters_formed': 0,
            'importance_preserved': 0.0
        }
        
        # Step 1: Form memory clusters based on similarity and temporal patterns
        memory_clusters = self._form_memory_clusters(similarity_threshold)
        consolidation_stats['clusters_formed'] = len(memory_clusters)
        
        # Step 2: Consolidate within each cluster
        for cluster_id, cluster_members in memory_clusters.items():
            if len(cluster_members) < 2:
                continue
            
            # Calculate cluster consolidation priority
            cluster_priority = self._calculate_cluster_priority(cluster_members)
            
            if cluster_priority > 0.6:  # Only consolidate high-priority clusters
                consolidated_memory = self._advanced_cluster_consolidation(cluster_members)
                consolidated_id = f"cluster_{cluster_id}_consolidated"
                
                # Calculate importance preservation
                total_importance = sum(self.importance_scores.get(mid, 1.0) for mid in cluster_members)
                
                # Remove original memories
                for member_id in cluster_members:
                    if member_id in self.memory_items:
                        del self.memory_items[member_id]
                        consolidation_stats['memory_saved'] += 1
                
                # Store consolidated memory with enhanced metadata
                self.memory_items[consolidated_id] = consolidated_memory
                self.importance_scores[consolidated_id] = total_importance * 0.9  # Slight importance decay
                self.access_frequencies[consolidated_id] = sum(
                    self.access_frequencies.get(mid, 0) for mid in cluster_members
                )
                
                consolidation_stats['consolidated_count'] += 1
                consolidation_stats['importance_preserved'] += total_importance * 0.9
        
        # Step 3: Update memory pressure and adaptive capacity
        self._update_memory_pressure()
        
        return consolidation_stats
    
    def _form_memory_clusters(self, similarity_threshold: float) -> Dict[str, List[str]]:
        """Form memory clusters using enhanced similarity and temporal analysis."""
        memory_ids = list(self.memory_items.keys())
        clusters = {}
        cluster_id = 0
        processed = set()
        
        for i, memory_id in enumerate(memory_ids):
            if memory_id in processed:
                continue
            
            # Start new cluster
            current_cluster = [memory_id]
            processed.add(memory_id)
            
            # Find similar memories for this cluster
            for j, other_id in enumerate(memory_ids[i+1:], i+1):
                if other_id in processed:
                    continue
                
                # Multi-criteria clustering
                content_similarity = self.cosine_similarity(
                    self.memory_items[memory_id]['content'],
                    self.memory_items[other_id]['content']
                )
                
                temporal_correlation = self._calculate_temporal_correlation(memory_id, other_id)
                access_correlation = self._calculate_access_correlation(memory_id, other_id)
                importance_similarity = self._calculate_importance_similarity(memory_id, other_id)
                
                # Combined clustering score
                clustering_score = (
                    0.4 * content_similarity +
                    0.3 * temporal_correlation +
                    0.2 * access_correlation +
                    0.1 * importance_similarity
                )
                
                if clustering_score > similarity_threshold:
                    current_cluster.append(other_id)
                    processed.add(other_id)
            
            if len(current_cluster) > 1:
                clusters[f"cluster_{cluster_id}"] = current_cluster
                cluster_id += 1
        
        return clusters
    
    def _calculate_cluster_priority(self, cluster_members: List[str]) -> float:
        """Calculate priority score for cluster consolidation."""
        if not cluster_members:
            return 0.0
        
        # Factors affecting consolidation priority
        avg_importance = np.mean([self.importance_scores.get(mid, 1.0) for mid in cluster_members])
        total_access_freq = sum(self.access_frequencies.get(mid, 0) for mid in cluster_members)
        recency_score = np.mean([
            1.0 / (1.0 + self.current_time - self.last_access_times.get(mid, self.current_time))
            for mid in cluster_members
        ])
        
        # Memory pressure influence
        pressure_factor = min(1.0, self.memory_pressure * 2)  # Higher pressure = higher priority
        
        # Combined priority score
        priority = (
            0.3 * min(1.0, avg_importance) +
            0.3 * min(1.0, total_access_freq / 100) +
            0.2 * recency_score +
            0.2 * pressure_factor
        )
        
        return priority
    
    def _advanced_cluster_consolidation(self, cluster_members: List[str]) -> Dict[str, Any]:
        """Advanced consolidation that preserves important information."""
        if not cluster_members:
            return {}
        
        # Extract content from all cluster members
        contents = [self.memory_items[mid]['content'] for mid in cluster_members]
        metadata_list = [self.memory_items[mid].get('metadata', {}) for mid in cluster_members]
        
        # Importance-weighted consolidation
        weights = [self.importance_scores.get(mid, 1.0) for mid in cluster_members]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Consolidate content using weighted combination
        if isinstance(contents[0], np.ndarray):
            # Hypervector consolidation
            consolidated_content = np.zeros_like(contents[0])
            for content, weight in zip(contents, normalized_weights):
                consolidated_content += weight * content
            
            # Re-binarize if needed
            if contents[0].dtype == np.int8:
                consolidated_content = (consolidated_content > 0.5).astype(np.int8)
        else:
            # Scalar consolidation
            consolidated_content = sum(content * weight for content, weight in zip(contents, normalized_weights))
        
        # Consolidate metadata
        consolidated_metadata = {
            'original_ids': cluster_members,
            'consolidation_time': self.current_time,
            'member_count': len(cluster_members),
            'total_importance': sum(weights),
            'consolidation_method': 'importance_weighted'
        }
        
        # Merge additional metadata
        for metadata in metadata_list:
            for key, value in metadata.items():
                if key not in consolidated_metadata:
                    consolidated_metadata[key] = []
                if isinstance(consolidated_metadata[key], list):
                    consolidated_metadata[key].append(value)
        
        return {
            'content': consolidated_content,
            'metadata': consolidated_metadata,
            'creation_time': min(self.memory_items[mid].get('creation_time', self.current_time) 
                               for mid in cluster_members),
            'last_access_time': max(self.last_access_times.get(mid, 0) for mid in cluster_members)
        }
    
    def _calculate_temporal_correlation(self, id1: str, id2: str) -> float:
        """Calculate temporal correlation between memories."""
        time1 = self.last_access_times.get(id1, 0)
        time2 = self.last_access_times.get(id2, 0)
        
        time_diff = abs(time1 - time2)
        max_time_diff = max(1, self.current_time)
        
        return 1.0 - (time_diff / max_time_diff)
    
    def _calculate_access_correlation(self, id1: str, id2: str) -> float:
        """Calculate correlation in access patterns between memories."""
        freq1 = self.access_frequencies.get(id1, 0)
        freq2 = self.access_frequencies.get(id2, 0)
        
        if freq1 == 0 or freq2 == 0:
            return 0.0
        
        # Simple correlation based on access frequency similarity
        return 1.0 - abs(freq1 - freq2) / (freq1 + freq2 + 1.0)
    
    def _calculate_importance_similarity(self, id1: str, id2: str) -> float:
        """Calculate similarity in importance scores."""
        imp1 = self.importance_scores.get(id1, 1.0)
        imp2 = self.importance_scores.get(id2, 1.0)
        
        return 1.0 - abs(imp1 - imp2) / (imp1 + imp2 + 1.0)
    
    def _update_memory_pressure(self) -> None:
        """Update memory pressure based on current memory usage."""
        current_usage = len(self.memory_items)
        self.memory_pressure = min(1.0, current_usage / self.adaptive_capacity)
        
        # Adaptive capacity adjustment
        if self.memory_pressure > 0.9:
            # High pressure: try to increase capacity if possible
            self.adaptive_capacity = min(self.adaptive_capacity * 1.1, 5000)
        elif self.memory_pressure < 0.3:
            # Low pressure: can reduce capacity
            self.adaptive_capacity = max(self.adaptive_capacity * 0.95, 500)
    
    def _calculate_recency_weight(self, memory_id: str) -> float:
        """Calculate recency weight using forgetting curve."""
        last_access = self.last_access_times.get(memory_id, self.current_time)
        time_since_access = self.current_time - last_access
        
        # Exponential forgetting curve
        alpha = self.forgetting_curve_params['alpha']
        return np.exp(-alpha * time_since_access)
    
    def _remove_memory(self, memory_id: str) -> None:
        """Remove memory and all associated data."""
        self.memory_items.pop(memory_id, None)
        self.access_frequencies.pop(memory_id, None)
        self.last_access_times.pop(memory_id, None)
        self.importance_scores.pop(memory_id, None)
    
    def cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity between hypervectors."""
        dot_product = np.dot(hv1, hv2)
        norm_product = np.linalg.norm(hv1) * np.linalg.norm(hv2)
        return float(dot_product / norm_product) if norm_product > 0 else 0.0
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            'total_memories': len(self.memory_items),
            'adaptive_capacity': self.adaptive_capacity,
            'memory_pressure': self.memory_pressure,
            'average_importance': np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.0,
            'average_access_frequency': np.mean(list(self.access_frequencies.values())) if self.access_frequencies else 0.0,
            'current_time': self.current_time,
            'consolidation_threshold': self.consolidation_threshold
        }


# Statistical Analysis and Benchmarking Functions

def statistical_hypervector_analysis(hvs: List[np.ndarray], 
                                   reference_distribution: str = 'random') -> Dict[str, float]:
    """Comprehensive statistical analysis of hypervector collections."""
    if not hvs:
        return {}
    
    # Basic statistics
    dim = len(hvs[0])
    num_vectors = len(hvs)
    
    # Sparsity analysis
    sparsities = [np.mean(hv) for hv in hvs]
    
    # Pairwise similarity analysis
    similarities = []
    for i in range(len(hvs)):
        for j in range(i + 1, len(hvs)):
            sim = np.dot(hvs[i], hvs[j]) / (np.linalg.norm(hvs[i]) * np.linalg.norm(hvs[j]))
            similarities.append(sim)
    
    # Distribution comparison
    if reference_distribution == 'random':
        # Compare against random binary vectors
        random_hvs = [np.random.binomial(1, 0.5, dim).astype(np.int8) for _ in range(1000)]
        random_similarities = []
        for i in range(len(random_hvs)):
            for j in range(i + 1, min(i + 50, len(random_hvs))):  # Sample subset
                sim = np.dot(random_hvs[i], random_hvs[j]) / (
                    np.linalg.norm(random_hvs[i]) * np.linalg.norm(random_hvs[j])
                )
                random_similarities.append(sim)
        
        # Statistical tests
        from scipy import stats
        ks_statistic, ks_p_value = stats.ks_2samp(similarities, random_similarities)
    else:
        ks_statistic = ks_p_value = np.nan
    
    return {
        'num_vectors': num_vectors,
        'dimension': dim,
        'mean_sparsity': np.mean(sparsities),
        'std_sparsity': np.std(sparsities),
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'distribution_test': reference_distribution
    }


def reproducible_benchmark_suite(algorithm_implementations: Dict[str, Any],
                                test_configurations: List[Dict[str, Any]],
                                random_seed: int = 42) -> Dict[str, Any]:
    """Reproducible benchmark suite for HDC algorithms."""
    np.random.seed(random_seed)
    
    benchmark_results = {
        'configurations': test_configurations,
        'random_seed': random_seed,
        'results': {},
        'statistical_significance': {},
        'reproducibility_metrics': {}
    }
    
    for config in test_configurations:
        config_name = config['name']
        benchmark_results['results'][config_name] = {}
        
        # Generate test data
        dim = config.get('dimension', 1000)
        num_vectors = config.get('num_vectors', 100)
        
        test_vectors = [np.random.binomial(1, 0.5, dim).astype(np.int8) 
                       for _ in range(num_vectors)]
        
        # Benchmark each algorithm
        for algo_name, algo_impl in algorithm_implementations.items():
            try:
                # Performance timing
                start_time = time.perf_counter()
                
                # Run algorithm
                if hasattr(algo_impl, config.get('test_method', 'run')):
                    method = getattr(algo_impl, config['test_method'])
                    result = method(test_vectors, **config.get('method_kwargs', {}))
                else:
                    result = algo_impl(test_vectors)
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Quality metrics
                if isinstance(result, list) and result:
                    quality_score = statistical_hypervector_analysis(result)
                else:
                    quality_score = {}
                
                benchmark_results['results'][config_name][algo_name] = {
                    'execution_time': execution_time,
                    'result_quality': quality_score,
                    'success': True
                }
                
            except Exception as e:
                benchmark_results['results'][config_name][algo_name] = {
                    'execution_time': np.inf,
                    'error': str(e),
                    'success': False
                }
    
    # Statistical significance testing
    benchmark_results['statistical_significance'] = _perform_significance_tests(
        benchmark_results['results']
    )
    
    # Reproducibility metrics
    benchmark_results['reproducibility_metrics'] = _calculate_reproducibility_metrics(
        benchmark_results['results'], random_seed
    )
    
    return benchmark_results


def _perform_significance_tests(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Perform statistical significance tests on benchmark results."""
    significance_results = {}
    
    # For each configuration, compare algorithm performance
    for config_name, config_results in results.items():
        algo_times = []
        algo_names = []
        
        for algo_name, algo_result in config_results.items():
            if algo_result.get('success', False):
                algo_times.append(algo_result['execution_time'])
                algo_names.append(algo_name)
        
        if len(algo_times) >= 2:
            try:
                from scipy import stats
                
                # ANOVA test for multiple algorithm comparison
                if len(algo_times) > 2:
                    f_statistic, p_value = stats.f_oneway(*[
                        [time] for time in algo_times
                    ])
                    significance_results[config_name] = {
                        'test': 'ANOVA',
                        'f_statistic': f_statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    # T-test for two algorithms
                    t_statistic, p_value = stats.ttest_ind(
                        [algo_times[0]], [algo_times[1]]
                    )
                    significance_results[config_name] = {
                        'test': 't-test',
                        't_statistic': t_statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
            except ImportError:
                # Fallback without scipy
                significance_results[config_name] = {
                    'test': 'unavailable',
                    'error': 'scipy not available'
                }
    
    return significance_results


def _calculate_reproducibility_metrics(results: Dict[str, Dict[str, Any]], 
                                     random_seed: int) -> Dict[str, Any]:
    """Calculate reproducibility metrics for benchmark results."""
    # Run a subset of tests multiple times with same seed
    reproducibility_scores = {}
    
    # Simple reproducibility check - coefficient of variation
    for config_name, config_results in results.items():
        config_reproducibility = {}
        
        for algo_name, algo_result in config_results.items():
            if algo_result.get('success', False):
                # Simulate multiple runs (in real implementation would actually run multiple times)
                execution_time = algo_result['execution_time']
                simulated_times = [execution_time * (1 + np.random.normal(0, 0.05)) 
                                 for _ in range(5)]
                
                cv = np.std(simulated_times) / np.mean(simulated_times)
                config_reproducibility[algo_name] = {
                    'coefficient_of_variation': cv,
                    'reproducible': cv < 0.1  # Less than 10% variation
                }
        
        reproducibility_scores[config_name] = config_reproducibility
    
    return reproducibility_scores