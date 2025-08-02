"""Example tests demonstrating HD-Compute-Toolkit usage patterns."""

import numpy as np
import pytest


class TestExamples:
    """Example tests showing typical HDC usage patterns."""
    
    def test_basic_hypervector_creation(self, mock_hdc_backend):
        """Example: Creating and manipulating hypervectors."""
        # Create random hypervectors
        apple = mock_hdc_backend.random_hv()
        fruit = mock_hdc_backend.random_hv()
        red = mock_hdc_backend.random_hv()
        
        # Basic properties
        assert len(apple) == mock_hdc_backend.dim
        assert apple.dtype == bool
        
        # Create concepts through binding
        red_apple = mock_hdc_backend.bind(apple, red)
        fruit_concept = mock_hdc_backend.bind(apple, fruit)
        
        # Both should be different from original apple
        apple_red_sim = mock_hdc_backend.cosine_similarity(apple, red_apple)
        apple_fruit_sim = mock_hdc_backend.cosine_similarity(apple, fruit_concept)
        
        # Bound vectors should be dissimilar to originals
        assert abs(apple_red_sim) < 0.8
        assert abs(apple_fruit_sim) < 0.8
    
    def test_category_learning_example(self, mock_hdc_backend):
        """Example: Learning categories through bundling."""
        # Create item hypervectors
        items = {
            'apple': mock_hdc_backend.random_hv(),
            'banana': mock_hdc_backend.random_hv(),
            'orange': mock_hdc_backend.random_hv(),
            'cherry': mock_hdc_backend.random_hv(),
            'grape': mock_hdc_backend.random_hv(),
        }
        
        # Define categories
        fruits = ['apple', 'banana', 'orange', 'cherry', 'grape']
        red_fruits = ['apple', 'cherry']
        yellow_fruits = ['banana']
        
        # Create category prototypes by bundling
        fruit_prototype = mock_hdc_backend.bundle([items[name] for name in fruits])
        red_prototype = mock_hdc_backend.bundle([items[name] for name in red_fruits])
        
        # Test category membership
        apple_to_fruit = mock_hdc_backend.cosine_similarity(items['apple'], fruit_prototype)
        apple_to_red = mock_hdc_backend.cosine_similarity(items['apple'], red_prototype)
        
        # Apple should be similar to both fruit and red categories
        assert apple_to_fruit > 0.3
        assert apple_to_red > 0.3
        
        # Banana should be similar to fruit but not red
        banana_to_fruit = mock_hdc_backend.cosine_similarity(items['banana'], fruit_prototype)
        banana_to_red = mock_hdc_backend.cosine_similarity(items['banana'], red_prototype)
        
        assert banana_to_fruit > 0.3
        assert banana_to_red < apple_to_red  # Less similar to red than apple
    
    def test_sequence_encoding_example(self, mock_hdc_backend):
        """Example: Encoding sequences with positional information."""
        # Create vocabulary
        vocab = {
            'I': mock_hdc_backend.random_hv(),
            'love': mock_hdc_backend.random_hv(),
            'HDC': mock_hdc_backend.random_hv(),
            'computing': mock_hdc_backend.random_hv(),
        }
        
        # Create position vectors
        positions = [mock_hdc_backend.random_hv() for _ in range(10)]
        
        # Encode sentence: "I love HDC computing"
        sentence = ['I', 'love', 'HDC', 'computing']
        encoded_words = []
        
        for i, word in enumerate(sentence):
            word_hv = vocab[word]
            pos_hv = positions[i]
            # Bind word with its position
            positioned_word = mock_hdc_backend.bind(word_hv, pos_hv)
            encoded_words.append(positioned_word)
        
        # Bundle all positioned words to create sentence representation
        sentence_hv = mock_hdc_backend.bundle(encoded_words)
        
        # Test decoding: what word is at position 2?
        query_pos2 = mock_hdc_backend.bind(sentence_hv, positions[2])
        
        # Compare with all vocabulary words
        similarities = {}
        for word, word_hv in vocab.items():
            sim = mock_hdc_backend.cosine_similarity(query_pos2, word_hv)
            similarities[word] = sim
        
        # 'HDC' should be most similar (it's at position 2)
        best_word = max(similarities.keys(), key=lambda w: similarities[w])
        assert best_word == 'HDC'
    
    def test_associative_memory_example(self, mock_hdc_backend):
        """Example: Implementing associative memory."""
        # Create key-value pairs
        keys = [mock_hdc_backend.random_hv() for _ in range(10)]
        values = [mock_hdc_backend.random_hv() for _ in range(10)]
        
        # Store associations by bundling bound pairs
        memory_items = []
        for key, value in zip(keys, values):
            # Bind key with value
            bound_pair = mock_hdc_backend.bind(key, value)
            memory_items.append(bound_pair)
        
        # Create memory by bundling all associations
        associative_memory = mock_hdc_backend.bundle(memory_items)
        
        # Retrieve value for first key
        query_key = keys[0]
        retrieved = mock_hdc_backend.bind(associative_memory, query_key)
        
        # Should be most similar to the corresponding value
        similarities = []
        for value in values:
            sim = mock_hdc_backend.cosine_similarity(retrieved, value)
            similarities.append(sim)
        
        best_match_idx = np.argmax(similarities)
        assert best_match_idx == 0  # Should match the first value
    
    def test_similarity_search_example(self, mock_hdc_backend):
        """Example: Similarity-based search and retrieval."""
        # Create a database of items
        database = {}
        n_items = 20
        
        for i in range(n_items):
            item_name = f"item_{i}"
            database[item_name] = mock_hdc_backend.random_hv()
        
        # Create a query that's similar to item_5
        target_name = "item_5"
        target_hv = database[target_name]
        
        # Add some noise to create a noisy query
        noise = mock_hdc_backend.random_hv()
        # Combine target with small amount of noise
        noisy_query = mock_hdc_backend.bundle([target_hv, target_hv, target_hv, noise])
        
        # Search for most similar item
        similarities = {}
        for name, hv in database.items():
            sim = mock_hdc_backend.cosine_similarity(noisy_query, hv)
            similarities[name] = sim
        
        # Sort by similarity
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Target should be among top results (though not necessarily first due to noise)
        top_5_names = [name for name, _ in sorted_items[:5]]
        assert target_name in top_5_names
    
    def test_classification_example(self, mock_hdc_backend):
        """Example: Simple classification using HDC."""
        # Create training data
        n_features = 50
        n_classes = 3
        n_samples_per_class = 10
        
        # Feature hypervectors
        feature_hvs = [mock_hdc_backend.random_hv() for _ in range(n_features)]
        
        # Generate synthetic training data
        training_data = []
        for class_id in range(n_classes):
            for sample in range(n_samples_per_class):
                # Random feature activation pattern
                np.random.seed(class_id * 100 + sample)  # Reproducible
                active_features = np.random.choice(n_features, size=10, replace=False)
                
                # Bundle active features
                sample_hvs = [feature_hvs[i] for i in active_features]
                sample_hv = mock_hdc_backend.bundle(sample_hvs)
                
                training_data.append((sample_hv, class_id))
        
        # Create class prototypes
        class_prototypes = {}
        for class_id in range(n_classes):
            class_samples = [hv for hv, label in training_data if label == class_id]
            class_prototypes[class_id] = mock_hdc_backend.bundle(class_samples)
        
        # Test classification
        test_sample_hv, true_label = training_data[0]  # Use first training sample
        
        # Find most similar class
        similarities = {}
        for class_id, prototype in class_prototypes.items():
            sim = mock_hdc_backend.cosine_similarity(test_sample_hv, prototype)
            similarities[class_id] = sim
        
        predicted_label = max(similarities.keys(), key=lambda c: similarities[c])
        
        # Should classify correctly (perfect accuracy on training data expected)
        assert predicted_label == true_label
    
    @pytest.mark.slow
    def test_large_scale_example(self, large_dimension):
        """Example: Large-scale HDC operations."""
        from tests.conftest import MockHDCBackend
        
        # Create large-dimension backend
        large_backend = MockHDCBackend(dim=large_dimension, seed=42)
        
        # Create many hypervectors
        n_vectors = 100
        hvs = [large_backend.random_hv() for _ in range(n_vectors)]
        
        # Bundle all vectors
        super_bundle = large_backend.bundle(hvs)
        
        # Should still be valid hypervector
        assert len(super_bundle) == large_dimension
        assert super_bundle.dtype == bool
        
        # Should be similar to constituent vectors
        similarities = [large_backend.cosine_similarity(super_bundle, hv) for hv in hvs[:5]]
        
        # All similarities should be positive (similar to bundle)
        for sim in similarities:
            assert sim > 0.0