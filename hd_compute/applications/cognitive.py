"""Cognitive computing applications using hyperdimensional computing."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..memory.item_memory import ItemMemory
from ..memory.associative_memory import AssociativeMemory


class SemanticMemory:
    """Semantic memory for storing and querying conceptual knowledge."""
    
    def __init__(self, hdc_backend: Any, dim: int = 32000):
        """Initialize semantic memory.
        
        Args:
            hdc_backend: HDCompute backend instance
            dim: Hypervector dimensionality
        """
        self.hdc = hdc_backend
        self.dim = dim
        
        # Memory structures
        self.concept_memory = ItemMemory(hdc_backend)
        self.attribute_memory = ItemMemory(hdc_backend)
        self.semantic_associations = AssociativeMemory(hdc_backend, capacity=5000)
        
        # Semantic relations
        self.relations = {}  # relation_name -> hypervector
        self._initialize_relations()
    
    def _initialize_relations(self):
        """Initialize basic semantic relations."""
        relation_names = [
            "is_a", "has_property", "part_of", "used_for", 
            "made_of", "located_at", "causes", "similar_to"
        ]
        
        for relation in relation_names:
            self.relations[relation] = self.hdc.random_hv()
    
    def store(self, concept: str, attributes: List[str], relations: Optional[Dict[str, List[str]]] = None):
        """Store a concept with its attributes and relations.
        
        Args:
            concept: Concept name
            attributes: List of attributes
            relations: Dictionary of relation_type -> [related_concepts]
        """
        # Add concept and attributes to memories
        self.concept_memory.add_items([concept])
        self.attribute_memory.add_items(attributes)
        
        # Get concept hypervector
        concept_hv = self.concept_memory.get_hv(concept)
        
        # Bundle attributes
        if attributes:
            attribute_hvs = [self.attribute_memory.get_hv(attr) for attr in attributes]
            bundled_attributes = self.hdc.bundle(attribute_hvs)
            
            # Bind concept with attributes
            concept_with_attrs = self.hdc.bind(concept_hv, bundled_attributes)
            
            # Store in associative memory
            self.semantic_associations.store(concept_with_attrs, f"concept_{concept}")
        
        # Handle relations if provided
        if relations:
            for relation_type, related_concepts in relations.items():
                if relation_type not in self.relations:
                    self.relations[relation_type] = self.hdc.random_hv()
                
                # Add related concepts to memory
                self.concept_memory.add_items(related_concepts)
                
                # Create relation encoding
                relation_hv = self.relations[relation_type]
                for related_concept in related_concepts:
                    related_hv = self.concept_memory.get_hv(related_concept)
                    
                    # Encode: concept + relation + related_concept
                    relation_encoding = self.hdc.bundle([
                        self.hdc.bind(concept_hv, relation_hv),
                        related_hv
                    ])
                    
                    self.semantic_associations.store(
                        relation_encoding, 
                        f"relation_{concept}_{relation_type}_{related_concept}"
                    )
    
    def query(self, attributes: List[str], threshold: float = 0.3) -> List[str]:
        """Query concepts by attributes.
        
        Args:
            attributes: List of query attributes
            threshold: Similarity threshold for matches
            
        Returns:
            List of matching concept names
        """
        if not attributes:
            return []
        
        # Bundle query attributes
        try:
            attribute_hvs = [self.attribute_memory.get_hv(attr) for attr in attributes]
            query_hv = self.hdc.bundle(attribute_hvs)
            
            # Search for similar concepts
            recalls = self.semantic_associations.recall(query_hv, k=10, threshold=threshold)
            
            # Extract concept names
            concepts = []
            for label, similarity in recalls:
                if label.startswith("concept_"):
                    concept_name = label.replace("concept_", "")
                    concepts.append(concept_name)
            
            return concepts
        
        except KeyError as e:
            # Some attributes not in memory
            return []
    
    def find_relations(self, concept: str, relation_type: str) -> List[Tuple[str, float]]:
        """Find concepts related to a given concept by a specific relation.
        
        Args:
            concept: Source concept
            relation_type: Type of relation to search
            
        Returns:
            List of (related_concept, similarity) tuples
        """
        if concept not in self.concept_memory.item_to_index:
            return []
        
        if relation_type not in self.relations:
            return []
        
        # Create query: concept + relation
        concept_hv = self.concept_memory.get_hv(concept)
        relation_hv = self.relations[relation_type]
        query_hv = self.hdc.bind(concept_hv, relation_hv)
        
        # Search for relations
        recalls = self.semantic_associations.recall(query_hv, k=20, threshold=0.2)
        
        # Extract related concepts
        related = []
        prefix = f"relation_{concept}_{relation_type}_"
        
        for label, similarity in recalls:
            if label.startswith(prefix):
                related_concept = label.replace(prefix, "")
                related.append((related_concept, similarity))
        
        return related
    
    def analogy(self, concept_a: str, concept_b: str, concept_c: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform analogical reasoning: A is to B as C is to ?
        
        Args:
            concept_a: First concept in analogy
            concept_b: Second concept in analogy  
            concept_c: Third concept in analogy
            k: Number of results to return
            
        Returns:
            List of (concept, similarity) tuples for concept D
        """
        try:
            # Get hypervectors
            hv_a = self.concept_memory.get_hv(concept_a)
            hv_b = self.concept_memory.get_hv(concept_b)
            hv_c = self.concept_memory.get_hv(concept_c)
            
            # Compute analogy: D = C + (B - A)
            # In HDC: bind C with (bind B with inverse of A)
            inverse_a = hv_a  # In binary HDC, inverse is identity
            relation_ab = self.hdc.bind(hv_b, inverse_a)
            query_d = self.hdc.bind(hv_c, relation_ab)
            
            # Find similar concepts
            results = []
            for concept in self.concept_memory.items:
                if concept not in [concept_a, concept_b, concept_c]:
                    concept_hv = self.concept_memory.get_hv(concept)
                    similarity = self.hdc.cosine_similarity(query_d, concept_hv)
                    results.append((concept, similarity))
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        
        except KeyError:
            return []
    
    def get_concept_profile(self, concept: str) -> Dict[str, Any]:
        """Get comprehensive profile of a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Dictionary with concept information
        """
        if concept not in self.concept_memory.item_to_index:
            return {}
        
        profile = {"concept": concept}
        
        # Find associated attributes (approximate)
        concept_hv = self.concept_memory.get_hv(concept)
        attribute_similarities = []
        
        for attr in self.attribute_memory.items:
            attr_hv = self.attribute_memory.get_hv(attr)
            similarity = self.hdc.cosine_similarity(concept_hv, attr_hv)
            if similarity > 0.1:  # Threshold for association
                attribute_similarities.append((attr, similarity))
        
        profile["likely_attributes"] = sorted(attribute_similarities, key=lambda x: x[1], reverse=True)[:10]
        
        # Find relations
        profile["relations"] = {}
        for relation_type in self.relations.keys():
            related = self.find_relations(concept, relation_type)
            if related:
                profile["relations"][relation_type] = related[:5]  # Top 5
        
        return profile
    
    def memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "concepts": self.concept_memory.size(),
            "attributes": self.attribute_memory.size(),
            "associations": self.semantic_associations.size(),
            "relations": len(self.relations),
            "dimension": self.dim
        }