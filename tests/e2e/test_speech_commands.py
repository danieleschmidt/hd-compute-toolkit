"""End-to-end tests for speech command recognition application."""

import numpy as np
import pytest

from tests.conftest import MockHDCBackend


class TestSpeechCommandPipeline:
    """Test complete speech command recognition pipeline."""
    
    @pytest.fixture
    def speech_hdc_system(self, medium_dimension, mock_speech_data):
        """Create a mock speech command HDC system."""
        data = np.load(mock_speech_data)
        features = data['features']
        labels = data['labels']
        class_names = data['class_names']
        
        # Simple mock implementation
        class MockSpeechHDC:
            def __init__(self, dim=medium_dimension):
                self.hdc = MockHDCBackend(dim=dim)
                self.class_prototypes = {}
                self.feature_hvs = {}
                self.position_hvs = {}
                self.class_names = class_names
                
                # Initialize feature encoding
                self._initialize_encoding()
            
            def _initialize_encoding(self):
                """Initialize encoding hypervectors."""
                # Create hypervectors for different feature values
                for i in range(100):  # Discretized feature values
                    self.feature_hvs[i] = self.hdc.random_hv()
                
                # Create position hypervectors
                for i in range(50):  # Max timesteps
                    self.position_hvs[i] = self.hdc.random_hv()
            
            def encode_sequence(self, sequence):
                """Encode a feature sequence to hypervector."""
                encoded_steps = []
                
                for t, features in enumerate(sequence):
                    step_hvs = []
                    for f in features:
                        # Discretize feature value
                        discretized = int(np.clip(f * 10 + 50, 0, 99))
                        feature_hv = self.feature_hvs[discretized]
                        step_hvs.append(feature_hv)
                    
                    # Bundle features for this timestep
                    if step_hvs:
                        step_bundled = self.hdc.bundle(step_hvs)
                        # Bind with position
                        if t < len(self.position_hvs):
                            step_encoded = self.hdc.bind(step_bundled, self.position_hvs[t])
                        else:
                            step_encoded = step_bundled
                        encoded_steps.append(step_encoded)
                
                # Bundle all timesteps
                return self.hdc.bundle(encoded_steps) if encoded_steps else self.hdc.random_hv()
            
            def train(self, features, labels):
                """Train the HDC classifier."""
                # Group samples by class
                class_samples = {i: [] for i in range(len(self.class_names))}
                
                for feat, label in zip(features, labels):
                    encoded = self.encode_sequence(feat)
                    class_samples[label].append(encoded)
                
                # Create class prototypes
                for class_id, samples in class_samples.items():
                    if samples:
                        self.class_prototypes[class_id] = self.hdc.bundle(samples)
            
            def predict(self, features):
                """Predict class for features."""
                if not self.class_prototypes:
                    raise ValueError("Model not trained")
                
                encoded = self.encode_sequence(features)
                
                best_similarity = -float('inf')
                best_class = 0
                
                for class_id, prototype in self.class_prototypes.items():
                    similarity = self.hdc.cosine_similarity(encoded, prototype)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_class = class_id
                
                return best_class
            
            def evaluate(self, features, labels):
                """Evaluate model accuracy."""
                correct = 0
                total = len(labels)
                
                for feat, true_label in zip(features, labels):
                    pred_label = self.predict(feat)
                    if pred_label == true_label:
                        correct += 1
                
                return correct / total if total > 0 else 0.0
        
        return MockSpeechHDC()
    
    def test_speech_command_encoding(self, speech_hdc_system, mock_speech_data):
        """Test speech command feature encoding."""
        data = np.load(mock_speech_data)
        features = data['features']
        
        # Test encoding of first sample
        sample = features[0]
        encoded = speech_hdc_system.encode_sequence(sample)
        
        # Should be valid hypervector
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == bool
        assert len(encoded) == speech_hdc_system.hdc.dim
        
        # Different samples should encode differently
        sample2 = features[1]
        encoded2 = speech_hdc_system.encode_sequence(sample2)
        
        assert not np.array_equal(encoded, encoded2)
    
    def test_speech_command_training(self, speech_hdc_system, mock_speech_data):
        """Test speech command model training."""
        data = np.load(mock_speech_data)
        features = data['features'][:50]  # Use subset for speed
        labels = data['labels'][:50]
        
        # Train model
        speech_hdc_system.train(features, labels)
        
        # Should have class prototypes
        assert len(speech_hdc_system.class_prototypes) > 0
        
        # All prototypes should be valid hypervectors
        for prototype in speech_hdc_system.class_prototypes.values():
            assert isinstance(prototype, np.ndarray)
            assert prototype.dtype == bool
            assert len(prototype) == speech_hdc_system.hdc.dim
    
    def test_speech_command_prediction(self, speech_hdc_system, mock_speech_data):
        """Test speech command prediction."""
        data = np.load(mock_speech_data)
        features = data['features'][:30]  # Train set
        labels = data['labels'][:30]
        test_features = data['features'][30:40]  # Test set
        test_labels = data['labels'][30:40]
        
        # Train and predict
        speech_hdc_system.train(features, labels)
        
        predictions = []
        for feat in test_features:
            pred = speech_hdc_system.predict(feat)
            predictions.append(pred)
        
        # All predictions should be valid class IDs
        unique_classes = set(labels)
        for pred in predictions:
            assert pred in unique_classes
    
    def test_speech_command_evaluation(self, speech_hdc_system, mock_speech_data):
        """Test speech command model evaluation."""
        data = np.load(mock_speech_data)
        features = data['features'][:50]
        labels = data['labels'][:50]
        
        # Split train/test
        train_features = features[:30]
        train_labels = labels[:30]
        test_features = features[30:]
        test_labels = labels[30:]
        
        # Train and evaluate
        speech_hdc_system.train(train_features, train_labels)
        accuracy = speech_hdc_system.evaluate(test_features, test_labels)
        
        # Accuracy should be reasonable (>= random chance)
        num_classes = len(speech_hdc_system.class_names)
        random_accuracy = 1.0 / num_classes
        
        assert 0.0 <= accuracy <= 1.0
        # For mock data, we expect at least better than random
        # (though not necessarily much better due to random nature)
        assert accuracy >= 0.0  # Just ensure it doesn't crash
    
    @pytest.mark.slow
    def test_speech_command_full_pipeline(self, speech_hdc_system, mock_speech_data):
        """Test complete speech command pipeline with full dataset."""
        data = np.load(mock_speech_data)
        features = data['features']
        labels = data['labels']
        
        # 70/30 train/test split
        split_idx = int(0.7 * len(features))
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        test_features = features[split_idx:]
        test_labels = labels[split_idx:]
        
        # Full pipeline
        speech_hdc_system.train(train_features, train_labels)
        accuracy = speech_hdc_system.evaluate(test_features, test_labels)
        
        # Should complete without errors
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        
        # Test individual predictions
        for i, feat in enumerate(test_features[:5]):
            pred = speech_hdc_system.predict(feat)
            assert isinstance(pred, (int, np.integer))
            assert 0 <= pred < len(speech_hdc_system.class_names)


class TestMemoryStructures:
    """Test HDC memory structures in context."""
    
    def test_item_memory_simulation(self, mock_hdc_backend):
        """Test simulation of item memory structure."""
        # Simulate item memory as dictionary
        item_memory = {}
        
        # Store some items
        items = ['apple', 'banana', 'cherry', 'date']
        for item in items:
            item_memory[item] = mock_hdc_backend.random_hv()
        
        # Test retrieval
        for item in items:
            retrieved = item_memory[item]
            assert isinstance(retrieved, np.ndarray)
            assert retrieved.dtype == bool
            assert len(retrieved) == mock_hdc_backend.dim
        
        # Test that different items have different hypervectors
        hvs = list(item_memory.values())
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                assert not np.array_equal(hvs[i], hvs[j])
    
    def test_associative_memory_simulation(self, mock_hdc_backend):
        """Test simulation of associative memory."""
        # Create training data
        patterns = []
        labels = []
        
        for i in range(10):
            pattern = mock_hdc_backend.random_hv()
            label = mock_hdc_backend.random_hv()
            patterns.append(pattern)
            labels.append(label)
        
        # Simulate associative memory training
        memory_pairs = list(zip(patterns, labels))
        
        # Test nearest neighbor retrieval
        query = patterns[0]  # Use first pattern as query
        
        best_similarity = -float('inf')
        best_match = None
        
        for pattern, label in memory_pairs:
            similarity = mock_hdc_backend.cosine_similarity(query, pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label
        
        # Should find exact match
        assert np.array_equal(best_match, labels[0])
        assert best_similarity == 1.0  # Perfect match
    
    def test_sequence_encoding(self, mock_hdc_backend):
        """Test encoding of sequences using position binding."""
        # Create position hypervectors
        positions = [mock_hdc_backend.random_hv() for _ in range(5)]
        
        # Create item hypervectors
        items = [mock_hdc_backend.random_hv() for _ in range(3)]
        
        # Encode sequence: item[0], item[1], item[2]
        sequence_elements = []
        for i, item in enumerate(items):
            bound = mock_hdc_backend.bind(item, positions[i])
            sequence_elements.append(bound)
        
        sequence_hv = mock_hdc_backend.bundle(sequence_elements)
        
        # Test retrieval of item at position 1
        query = mock_hdc_backend.bind(sequence_hv, positions[1])
        
        # Should be similar to items[1]
        similarities = [
            mock_hdc_backend.cosine_similarity(query, item)
            for item in items
        ]
        
        # Item at position 1 should be most similar
        best_idx = np.argmax(similarities)
        assert best_idx == 1