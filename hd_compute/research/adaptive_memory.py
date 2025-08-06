"""Advanced adaptive and hierarchical memory architectures for HDC."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import heapq


class AdaptiveHierarchicalMemory(ABC):
    """Adaptive hierarchical memory with dynamic reorganization and attention mechanisms."""
    
    def __init__(self, dim: int, max_capacity: int = 10000, hierarchy_levels: int = 3):
        self.dim = dim
        self.max_capacity = max_capacity
        self.hierarchy_levels = hierarchy_levels
        
        # Hierarchical storage: level 0 = most abstract, higher levels = more specific
        self.memory_hierarchy = [defaultdict(dict) for _ in range(hierarchy_levels)]
        
        # Adaptive parameters
        self.access_counts = defaultdict(int)
        self.temporal_weights = defaultdict(float)
        self.similarity_clusters = defaultdict(set)
        
        # Attention mechanism
        self.attention_weights = defaultdict(float)
        self.focus_buffer = deque(maxlen=100)
        
        # Dynamic reorganization
        self.reorganization_threshold = 1000
        self.access_counter = 0
        
    def adaptive_store(self, key: str, hv: Any, context: Optional[Dict] = None, 
                      importance: float = 1.0) -> None:
        """Store hypervector with adaptive placement in hierarchy."""
        # Determine optimal hierarchy level based on context and importance
        level = self._determine_hierarchy_level(hv, context, importance)
        
        # Store with metadata
        metadata = {
            'hypervector': hv,
            'context': context or {},
            'importance': importance,
            'timestamp': self.access_counter,
            'access_count': 0,
            'cluster_id': self._find_or_create_cluster(hv, level)
        }
        
        self.memory_hierarchy[level][key] = metadata
        
        # Update attention weights
        self._update_attention_weights(key, importance)
        
        # Trigger reorganization if needed
        self.access_counter += 1
        if self.access_counter % self.reorganization_threshold == 0:
            self._adaptive_reorganization()
    
    def hierarchical_recall(self, query_hv: Any, context: Optional[Dict] = None,
                          max_results: int = 5, confidence_threshold: float = 0.3) -> List[Tuple[str, Any, float]]:
        """Multi-level hierarchical recall with confidence scoring."""
        results = []
        
        # Search through hierarchy levels
        for level in range(self.hierarchy_levels):
            level_results = self._search_level(query_hv, level, context, max_results * 2)
            
            # Apply level-specific confidence weighting
            level_weight = self._get_level_confidence_weight(level, context)
            
            for key, hv, sim in level_results:
                confidence = sim * level_weight * self.attention_weights.get(key, 1.0)
                
                if confidence >= confidence_threshold:
                    results.append((key, hv, confidence))
                    
                    # Update access patterns
                    self.memory_hierarchy[level][key]['access_count'] += 1
                    self.access_counts[key] += 1
        
        # Sort by confidence and limit results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def attention_guided_recall(self, query_hv: Any, attention_context: List[str],
                              max_results: int = 5) -> List[Tuple[str, Any, float]]:
        """Recall guided by attention context."""
        # Build attention distribution
        attention_dist = self._build_attention_distribution(attention_context)
        
        results = []
        
        for level in range(self.hierarchy_levels):
            for key, metadata in self.memory_hierarchy[level].items():
                hv = metadata['hypervector']
                
                # Base similarity
                base_sim = self.cosine_similarity(query_hv, hv)
                
                # Attention-weighted similarity
                attention_weight = attention_dist.get(key, 0.1)
                context_similarity = self._context_similarity(metadata['context'], attention_context)
                
                weighted_sim = base_sim * attention_weight * (1.0 + context_similarity)
                
                if weighted_sim > 0.1:  # Minimum threshold
                    results.append((key, hv, weighted_sim))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def episodic_memory_trace(self, event_sequence: List[Tuple[str, Any]], 
                             temporal_decay: float = 0.9) -> Any:
        """Create episodic memory trace from event sequence."""
        if not event_sequence:
            return self._zero_hypervector()
        
        # Create temporal binding chain
        trace = self._zero_hypervector()
        
        for i, (key, hv) in enumerate(event_sequence):
            # Temporal position encoding
            temporal_weight = (temporal_decay ** i)
            position_hv = self._generate_position_encoding(i)
            
            # Bind event with temporal context
            temporal_event = self.bind(hv, position_hv)
            
            # Add to trace with temporal weighting
            trace = self._add_weighted(trace, temporal_event, temporal_weight)
        
        return self._normalize(trace)
    
    def semantic_clustering(self, similarity_threshold: float = 0.7) -> Dict[int, Set[str]]:
        """Perform semantic clustering across hierarchy levels."""
        all_items = []
        
        # Collect all items with their hypervectors
        for level in range(self.hierarchy_levels):
            for key, metadata in self.memory_hierarchy[level].items():
                all_items.append((key, metadata['hypervector'], level))
        
        # Hierarchical clustering
        clusters = {}
        cluster_id = 0
        
        for i, (key1, hv1, level1) in enumerate(all_items):
            if key1 in [item for cluster in clusters.values() for item in cluster]:
                continue  # Already clustered
            
            # Create new cluster
            cluster = {key1}
            
            for j, (key2, hv2, level2) in enumerate(all_items[i+1:], i+1):
                if key2 in [item for cluster in clusters.values() for item in cluster]:
                    continue  # Already clustered
                
                # Check similarity
                sim = self.cosine_similarity(hv1, hv2)
                level_bonus = 1.1 if level1 == level2 else 1.0  # Same level bonus
                
                if sim * level_bonus >= similarity_threshold:
                    cluster.add(key2)
            
            if len(cluster) > 1:  # Only store clusters with multiple items
                clusters[cluster_id] = cluster
                cluster_id += 1
        
        return clusters
    
    def associative_chaining(self, start_key: str, chain_length: int = 5,
                           strength_threshold: float = 0.4) -> List[Tuple[str, float]]:
        """Create associative chain starting from a given key."""
        chain = []
        current_key = start_key
        visited = set()
        
        # Find starting hypervector
        start_hv = self._find_hypervector_by_key(current_key)
        if start_hv is None:
            return chain
        
        current_hv = start_hv
        
        for _ in range(chain_length):
            if current_key in visited:
                break
            
            visited.add(current_key)
            
            # Find most similar unvisited item
            best_match = None
            best_similarity = strength_threshold
            
            for level in range(self.hierarchy_levels):
                for key, metadata in self.memory_hierarchy[level].items():
                    if key in visited:
                        continue
                    
                    hv = metadata['hypervector']
                    sim = self.cosine_similarity(current_hv, hv)
                    
                    # Apply temporal and access-based weighting
                    temporal_weight = self._get_temporal_weight(metadata['timestamp'])
                    access_weight = min(2.0, np.log(1 + metadata['access_count']))
                    
                    weighted_sim = sim * temporal_weight * access_weight
                    
                    if weighted_sim > best_similarity:
                        best_similarity = weighted_sim
                        best_match = (key, hv)
            
            if best_match is None:
                break
            
            current_key, current_hv = best_match
            chain.append((current_key, best_similarity))
        
        return chain
    
    def memory_consolidation(self, consolidation_strength: float = 0.8) -> None:
        """Consolidate memories by strengthening frequently accessed patterns."""
        # Identify frequently accessed patterns
        frequent_pairs = self._identify_frequent_access_patterns()
        
        # Strengthen connections between frequently co-accessed items
        for (key1, key2), frequency in frequent_pairs.items():
            if frequency < 3:  # Minimum co-occurrence threshold
                continue
            
            hv1 = self._find_hypervector_by_key(key1)
            hv2 = self._find_hypervector_by_key(key2)
            
            if hv1 is not None and hv2 is not None:
                # Create consolidated association
                consolidated = self._create_consolidated_association(hv1, hv2, consolidation_strength)
                
                # Store consolidated pattern
                consolidated_key = f"consolidated_{key1}_{key2}"
                self.adaptive_store(consolidated_key, consolidated, 
                                  {'type': 'consolidated', 'components': [key1, key2]})
    
    def forgetting_mechanism(self, forgetting_rate: float = 0.05) -> None:
        """Apply forgetting mechanism to reduce memory load."""
        items_to_remove = []
        
        for level in range(self.hierarchy_levels):
            for key, metadata in self.memory_hierarchy[level].items():
                # Calculate forgetting probability
                time_since_access = self.access_counter - metadata['timestamp']
                access_strength = metadata['access_count']
                importance = metadata['importance']
                
                # Forgetting probability (higher for less important, less accessed items)
                forget_prob = forgetting_rate * np.exp(-access_strength) * np.exp(-importance) * (time_since_access / 1000)
                
                if np.random.random() < forget_prob:
                    items_to_remove.append((level, key))
        
        # Remove forgotten items
        for level, key in items_to_remove:
            del self.memory_hierarchy[level][key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.attention_weights:
                del self.attention_weights[key]
    
    # Abstract methods for backend implementation
    
    @abstractmethod
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity."""
        pass
    
    @abstractmethod
    def bind(self, hv1: Any, hv2: Any) -> Any:
        """Bind two hypervectors."""
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
    
    @abstractmethod
    def _generate_position_encoding(self, position: int) -> Any:
        """Generate position encoding hypervector."""
        pass
    
    # Helper methods
    
    def _determine_hierarchy_level(self, hv: Any, context: Optional[Dict], importance: float) -> int:
        """Determine optimal hierarchy level for storage."""
        if importance > 0.8:
            return 0  # Most abstract level for highly important items
        elif context and len(context) > 3:
            return min(2, self.hierarchy_levels - 1)  # Specific level for rich context
        else:
            return 1  # Default middle level
    
    def _find_or_create_cluster(self, hv: Any, level: int) -> int:
        """Find existing cluster or create new one."""
        # Simple cluster assignment based on similarity
        best_cluster = 0
        best_similarity = 0.0
        
        for cluster_id, items in self.similarity_clusters.items():
            if not items:
                continue
            
            # Sample representative item from cluster
            sample_key = next(iter(items))
            sample_hv = self._find_hypervector_by_key(sample_key)
            
            if sample_hv is not None:
                sim = self.cosine_similarity(hv, sample_hv)
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster_id
        
        if best_similarity > 0.6:
            return best_cluster
        else:
            # Create new cluster
            new_cluster_id = max(self.similarity_clusters.keys(), default=-1) + 1
            return new_cluster_id
    
    def _update_attention_weights(self, key: str, importance: float) -> None:
        """Update attention weights based on importance and recency."""
        current_weight = self.attention_weights.get(key, 1.0)
        
        # Exponential moving average with importance boost
        alpha = 0.1
        self.attention_weights[key] = (1 - alpha) * current_weight + alpha * (1.0 + importance)
        
        # Add to focus buffer
        self.focus_buffer.append(key)
    
    def _search_level(self, query_hv: Any, level: int, context: Optional[Dict], 
                     max_results: int) -> List[Tuple[str, Any, float]]:
        """Search specific hierarchy level."""
        results = []
        
        for key, metadata in self.memory_hierarchy[level].items():
            hv = metadata['hypervector']
            base_sim = self.cosine_similarity(query_hv, hv)
            
            # Context bonus
            context_bonus = 1.0
            if context and metadata['context']:
                context_bonus = 1.0 + self._context_similarity_score(context, metadata['context'])
            
            adjusted_sim = base_sim * context_bonus
            results.append((key, hv, adjusted_sim))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def _get_level_confidence_weight(self, level: int, context: Optional[Dict]) -> float:
        """Get confidence weight for hierarchy level."""
        base_weights = [1.0, 0.9, 0.8]  # Prefer higher levels
        base_weight = base_weights[min(level, len(base_weights) - 1)]
        
        # Adjust based on context specificity
        if context:
            specificity = len(context) / 10.0  # Normalize
            if level == self.hierarchy_levels - 1:  # Most specific level
                base_weight *= (1.0 + specificity)
            else:
                base_weight *= (1.0 + (1.0 - specificity))
        
        return base_weight
    
    def _build_attention_distribution(self, attention_context: List[str]) -> Dict[str, float]:
        """Build attention distribution from context."""
        attention_dist = defaultdict(float)
        
        # Recent focus gets higher weight
        for i, item in enumerate(reversed(list(self.focus_buffer))):
            if item in attention_context:
                weight = 1.0 / (i + 1)  # Recency weighting
                attention_dist[item] += weight
        
        # Normalize
        total_weight = sum(attention_dist.values())
        if total_weight > 0:
            for key in attention_dist:
                attention_dist[key] /= total_weight
        
        return attention_dist
    
    def _context_similarity(self, context1: Dict, context2: List[str]) -> float:
        """Compute context similarity."""
        if not context1 or not context2:
            return 0.0
        
        context1_keys = set(context1.keys())
        context2_set = set(context2)
        
        intersection = context1_keys.intersection(context2_set)
        union = context1_keys.union(context2_set)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _context_similarity_score(self, context1: Dict, context2: Dict) -> float:
        """Compute similarity score between two context dictionaries."""
        if not context1 or not context2:
            return 0.0
        
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_hypervector_by_key(self, key: str) -> Optional[Any]:
        """Find hypervector by key across all levels."""
        for level in range(self.hierarchy_levels):
            if key in self.memory_hierarchy[level]:
                return self.memory_hierarchy[level][key]['hypervector']
        return None
    
    def _get_temporal_weight(self, timestamp: int) -> float:
        """Get temporal weight based on recency."""
        age = self.access_counter - timestamp
        return np.exp(-age / 1000.0)  # Exponential decay
    
    def _identify_frequent_access_patterns(self) -> Dict[Tuple[str, str], int]:
        """Identify frequently co-accessed item pairs."""
        # Simplified co-occurrence analysis
        co_occurrences = defaultdict(int)
        
        # This would typically analyze access logs
        # For now, return empty dict (placeholder for full implementation)
        return co_occurrences
    
    def _create_consolidated_association(self, hv1: Any, hv2: Any, strength: float) -> Any:
        """Create consolidated association between two hypervectors."""
        # Weighted binding with consolidation strength
        bound = self.bind(hv1, hv2)
        bundled = self._add_weighted(hv1, hv2, 0.5)
        
        # Combine bound and bundled representations
        consolidated = self._add_weighted(bound, bundled, strength)
        return self._normalize(consolidated)
    
    def _adaptive_reorganization(self) -> None:
        """Perform adaptive memory reorganization."""
        # Move frequently accessed items to higher levels
        promotions = []
        
        for level in range(1, self.hierarchy_levels):
            for key, metadata in list(self.memory_hierarchy[level].items()):
                access_rate = metadata['access_count'] / (self.access_counter - metadata['timestamp'] + 1)
                
                if access_rate > 0.1 and level > 0:  # Promotion threshold
                    promotions.append((key, metadata, level, level - 1))
        
        # Execute promotions
        for key, metadata, from_level, to_level in promotions:
            del self.memory_hierarchy[from_level][key]
            self.memory_hierarchy[to_level][key] = metadata