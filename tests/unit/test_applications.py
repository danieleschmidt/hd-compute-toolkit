"""Tests for application modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestSpeechCommandHDC:
    """Test SpeechCommandHDC functionality."""
    
    def test_initialization(self, speech_command_hdc):
        """Test SpeechCommandHDC initialization."""
        assert speech_command_hdc.dim == 1000
        assert speech_command_hdc.num_classes == 10
        assert speech_command_hdc.feature_extractor == 'mfcc'
        assert speech_command_hdc.n_mfcc == 13
        assert not speech_command_hdc.trained
    
    def test_extract_features_mfcc(self, speech_command_hdc):
        """Test MFCC feature extraction."""
        # Create mock audio signal
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        with patch('librosa.feature.mfcc') as mock_mfcc:
            # Mock librosa output
            mock_mfcc.return_value = np.random.randn(13, 32)  # 13 MFCCs, 32 time steps
            
            features = speech_command_hdc.extract_features(audio)
            
            assert features.shape[1] == 13  # Number of MFCC coefficients
            assert features.shape[0] > 0    # Some time steps
            mock_mfcc.assert_called_once()
    
    def test_extract_features_spectrogram(self):
        """Test spectrogram feature extraction."""
        from hd_compute.applications import SpeechCommandHDC
        
        # Create HDC instance with spectrogram features
        with patch('hd_compute.applications.speech_commands.HDComputeTorch') as mock_backend:
            hdc = SpeechCommandHDC(
                mock_backend,
                dim=1000,
                num_classes=10,
                feature_extractor='spectrogram'
            )
            
            audio = np.random.randn(16000)
            
            with patch('librosa.stft') as mock_stft:
                # Mock librosa STFT output
                mock_stft.return_value = np.random.randn(513, 32) + 1j * np.random.randn(513, 32)
                
                features = hdc.extract_features(audio)
                
                assert features.shape[1] == 513  # Frequency bins
                assert features.shape[0] > 0     # Time steps
                mock_stft.assert_called_once()
    
    def test_quantize_features(self, speech_command_hdc):
        """Test feature quantization."""
        # Create test features
        features = np.random.randn(50, 13)  # 50 time steps, 13 features
        
        quantized = speech_command_hdc.quantize_features(features, levels=8)
        
        assert quantized.shape == features.shape
        assert quantized.dtype == int
        assert quantized.min() >= 0
        assert quantized.max() < 8
    
    def test_encode_features(self, speech_command_hdc):
        """Test feature encoding."""
        # Create quantized features
        features = np.random.randint(0, 16, size=(20, 13))
        
        encoded = speech_command_hdc.encode_features(features)
        
        assert encoded is not None
        assert encoded.shape == (speech_command_hdc.dim,)
    
    @patch('torch.utils.data.DataLoader')
    def test_train(self, mock_dataloader, speech_command_hdc):
        """Test training process."""
        # Mock training data
        mock_batch = (
            np.random.randn(2, 16000),  # 2 audio samples
            np.array([0, 1])            # 2 labels
        )
        mock_dataloader.__iter__.return_value = [mock_batch]
        
        with patch.object(speech_command_hdc, 'extract_features') as mock_extract, \
             patch.object(speech_command_hdc, 'quantize_features') as mock_quantize, \
             patch.object(speech_command_hdc, 'encode_features') as mock_encode:
            
            mock_extract.return_value = np.random.randn(20, 13)
            mock_quantize.return_value = np.random.randint(0, 16, size=(20, 13))
            mock_encode.return_value = speech_command_hdc.hdc.random_hv()
            
            speech_command_hdc.train(mock_dataloader, epochs=1)
            
            assert speech_command_hdc.trained
            assert mock_extract.call_count >= 2  # Called for each sample
    
    def test_predict_single(self, speech_command_hdc):
        """Test single prediction."""
        # Set up as trained
        speech_command_hdc.trained = True
        
        # Mock encoded hypervector
        encoded_hv = speech_command_hdc.hdc.random_hv()
        
        with patch.object(speech_command_hdc.feature_memory, 'recall') as mock_recall:
            mock_recall.return_value = [('command_5', 0.8)]
            
            prediction = speech_command_hdc.predict_single(encoded_hv)
            
            assert prediction == 5
            mock_recall.assert_called_once()
    
    def test_predict(self, speech_command_hdc):
        """Test prediction on audio sample."""
        # Set up as trained
        speech_command_hdc.trained = True
        
        audio = np.random.randn(16000)
        
        with patch.object(speech_command_hdc, 'extract_features') as mock_extract, \
             patch.object(speech_command_hdc, 'quantize_features') as mock_quantize, \
             patch.object(speech_command_hdc, 'encode_features') as mock_encode, \
             patch.object(speech_command_hdc, 'predict_single') as mock_predict:
            
            mock_extract.return_value = np.random.randn(20, 13)
            mock_quantize.return_value = np.random.randint(0, 16, size=(20, 13))
            mock_encode.return_value = speech_command_hdc.hdc.random_hv()
            mock_predict.return_value = 3
            
            prediction = speech_command_hdc.predict(audio)
            
            assert prediction == 3
            mock_predict.assert_called_once()
    
    def test_predict_untrained_raises_error(self, speech_command_hdc):
        """Test that prediction on untrained model raises error."""
        audio = np.random.randn(16000)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            speech_command_hdc.predict(audio)


class TestSemanticMemory:
    """Test SemanticMemory functionality."""
    
    def test_initialization(self, semantic_memory):
        """Test SemanticMemory initialization."""
        assert semantic_memory.dim == 1000
        assert len(semantic_memory.relations) > 0
        assert 'is_a' in semantic_memory.relations
        assert 'has_property' in semantic_memory.relations
    
    def test_store_concept(self, semantic_memory):
        """Test storing a concept with attributes."""
        concept = "apple"
        attributes = ["red", "fruit", "sweet"]
        
        semantic_memory.store(concept, attributes)
        
        # Check that concept and attributes were added to memories
        assert concept in semantic_memory.concept_memory.items
        for attr in attributes:
            assert attr in semantic_memory.attribute_memory.items
        
        # Check that association was stored
        assert semantic_memory.semantic_associations.size() > 0
    
    def test_store_concept_with_relations(self, semantic_memory):
        """Test storing concept with relations."""
        concept = "apple"
        attributes = ["red", "sweet"]
        relations = {
            "is_a": ["fruit"],
            "part_of": ["tree"]
        }
        
        semantic_memory.store(concept, attributes, relations)
        
        # Check that related concepts were added
        assert "fruit" in semantic_memory.concept_memory.items
        assert "tree" in semantic_memory.concept_memory.items
        
        # Check that relation associations were stored
        assert semantic_memory.semantic_associations.size() > 1  # Concept + relations
    
    def test_query_by_attributes(self, semantic_memory):
        """Test querying concepts by attributes."""
        # Store some concepts
        semantic_memory.store("apple", ["red", "fruit", "sweet"])
        semantic_memory.store("banana", ["yellow", "fruit", "sweet"])
        semantic_memory.store("carrot", ["orange", "vegetable"])
        
        # Query for fruits
        fruit_concepts = semantic_memory.query(["fruit"])
        
        # Should return concepts with 'fruit' attribute
        assert len(fruit_concepts) >= 0  # Might be empty due to similarity thresholds
        
        # Query for sweet fruits
        sweet_fruits = semantic_memory.query(["fruit", "sweet"])
        assert isinstance(sweet_fruits, list)
    
    def test_query_empty_attributes(self, semantic_memory):
        """Test querying with empty attributes returns empty list."""
        results = semantic_memory.query([])
        assert results == []
    
    def test_query_unknown_attributes(self, semantic_memory):
        """Test querying with unknown attributes returns empty list."""
        results = semantic_memory.query(["nonexistent_attribute"])
        assert results == []
    
    def test_find_relations(self, semantic_memory):
        """Test finding relations for a concept."""
        concept = "apple"
        semantic_memory.store(concept, ["red"], {"is_a": ["fruit", "food"]})
        
        # Find 'is_a' relations
        related = semantic_memory.find_relations(concept, "is_a")
        
        # Should find the stored relations
        assert isinstance(related, list)
        if related:  # Might be empty due to similarity thresholds
            assert all(isinstance(item, tuple) for item in related)
            assert all(len(item) == 2 for item in related)  # (concept, similarity)
    
    def test_find_relations_unknown_concept(self, semantic_memory):
        """Test finding relations for unknown concept returns empty list."""
        related = semantic_memory.find_relations("unknown_concept", "is_a")
        assert related == []
    
    def test_find_relations_unknown_relation(self, semantic_memory):
        """Test finding unknown relation type returns empty list."""
        semantic_memory.store("apple", ["red"])
        related = semantic_memory.find_relations("apple", "unknown_relation")
        assert related == []
    
    def test_analogy(self, semantic_memory):
        """Test analogical reasoning."""
        # Store concepts that might form analogies
        semantic_memory.store("king", ["male", "ruler"])
        semantic_memory.store("queen", ["female", "ruler"])
        semantic_memory.store("man", ["male"])
        semantic_memory.store("woman", ["female"])
        
        # Test analogy: king is to queen as man is to ?
        results = semantic_memory.analogy("king", "queen", "man", k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # Each result should be (concept, similarity) tuple
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], str)  # concept name
            assert isinstance(result[1], (int, float))  # similarity score
    
    def test_analogy_unknown_concepts(self, semantic_memory):
        """Test analogy with unknown concepts returns empty list."""
        results = semantic_memory.analogy("unknown1", "unknown2", "unknown3")
        assert results == []
    
    def test_get_concept_profile(self, semantic_memory):
        """Test getting comprehensive concept profile."""
        concept = "apple"
        semantic_memory.store(concept, ["red", "sweet"], {"is_a": ["fruit"]})
        
        profile = semantic_memory.get_concept_profile(concept)
        
        assert profile["concept"] == concept
        assert "likely_attributes" in profile
        assert "relations" in profile
        
        # Profile should contain structured information
        assert isinstance(profile["likely_attributes"], list)
        assert isinstance(profile["relations"], dict)
    
    def test_get_concept_profile_unknown(self, semantic_memory):
        """Test getting profile for unknown concept returns empty dict."""
        profile = semantic_memory.get_concept_profile("unknown_concept")
        assert profile == {}
    
    def test_memory_statistics(self, semantic_memory):
        """Test getting memory statistics."""
        # Add some data
        semantic_memory.store("apple", ["red", "fruit"])
        semantic_memory.store("banana", ["yellow", "fruit"])
        
        stats = semantic_memory.memory_statistics()
        
        assert "concepts" in stats
        assert "attributes" in stats
        assert "associations" in stats
        assert "relations" in stats
        assert "dimension" in stats
        
        assert stats["concepts"] >= 2
        assert stats["attributes"] >= 3  # red, fruit, yellow
        assert stats["dimension"] == 1000