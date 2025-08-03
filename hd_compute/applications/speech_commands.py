"""Speech command recognition using hyperdimensional computing."""

import numpy as np
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
from ..memory.item_memory import ItemMemory
from ..memory.associative_memory import AssociativeMemory


class SpeechCommandHDC:
    """HDC-based speech command recognition system.
    
    Reproduces Qualcomm's 2025 HDC speech command demonstration.
    """
    
    def __init__(
        self,
        hdc_backend: Any,
        dim: int = 16000,
        num_classes: int = 35,
        feature_extractor: str = 'mfcc',
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """Initialize speech command HDC system.
        
        Args:
            hdc_backend: HDCompute backend instance
            dim: Hypervector dimensionality
            num_classes: Number of command classes
            feature_extractor: Feature extraction method ('mfcc', 'spectrogram')
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.hdc = hdc_backend
        self.dim = dim
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize memories
        self.command_memory = ItemMemory(hdc_backend)
        self.feature_memory = AssociativeMemory(hdc_backend, capacity=10000)
        
        # Position hypervectors for temporal encoding
        self.position_hvs = None
        self.trained = False
    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract audio features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Feature matrix [time_steps, feature_dim]
        """
        if self.feature_extractor == 'mfcc':
            mfccs = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            return mfccs.T  # [time_steps, n_mfcc]
        
        elif self.feature_extractor == 'spectrogram':
            spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(spec)
            return magnitude.T  # [time_steps, freq_bins]
        
        else:
            raise ValueError(f"Unknown feature extractor: {self.feature_extractor}")
    
    def quantize_features(self, features: np.ndarray, levels: int = 16) -> np.ndarray:
        """Quantize features for HDC encoding.
        
        Args:
            features: Feature matrix [time_steps, feature_dim]
            levels: Number of quantization levels
            
        Returns:
            Quantized features
        """
        # Min-max normalization
        feat_min = features.min(axis=0, keepdims=True)
        feat_max = features.max(axis=0, keepdims=True)
        normalized = (features - feat_min) / (feat_max - feat_min + 1e-8)
        
        # Quantize to discrete levels
        quantized = np.floor(normalized * (levels - 1)).astype(int)
        return np.clip(quantized, 0, levels - 1)
    
    def encode_features(self, features: np.ndarray) -> Any:
        """Encode quantized features as hypervectors.
        
        Args:
            features: Quantized feature matrix [time_steps, feature_dim]
            
        Returns:
            Encoded hypervector representing the audio
        """
        time_steps, feature_dim = features.shape
        
        # Create feature value hypervectors if not exists
        if not hasattr(self, 'feature_hvs'):
            # Create hypervectors for each feature value combination
            self.feature_hvs = {}
            max_levels = features.max() + 1
            
            for i in range(feature_dim):
                for level in range(max_levels):
                    key = f"feat_{i}_level_{level}"
                    self.feature_hvs[key] = self.hdc.random_hv()
        
        # Create position hypervectors if not exists
        if self.position_hvs is None or self.position_hvs.shape[0] < time_steps:
            self.position_hvs = self.hdc.random_hv(batch_size=time_steps)
        
        # Encode each time step
        time_step_hvs = []
        for t in range(time_steps):
            # Bundle feature hypervectors for this time step
            step_features = []
            for f in range(feature_dim):
                level = features[t, f]
                key = f"feat_{f}_level_{level}"
                if key in self.feature_hvs:
                    step_features.append(self.feature_hvs[key])
            
            if step_features:
                bundled_features = self.hdc.bundle(step_features)
                # Bind with position information
                positioned = self.hdc.bind(bundled_features, self.position_hvs[t])
                time_step_hvs.append(positioned)
        
        # Bundle all time steps
        if time_step_hvs:
            return self.hdc.bundle(time_step_hvs)
        else:
            return self.hdc.random_hv()  # Fallback
    
    def train(self, train_loader: DataLoader, epochs: int = 10, learning_rate: float = 0.01):
        """Train the HDC model.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for prototype updates
        """
        print(f"Training HDC model for {epochs} epochs...")
        
        # Initialize command hypervectors
        all_commands = set()
        for batch in train_loader:
            _, labels = batch
            for label in labels:
                all_commands.add(label.item() if hasattr(label, 'item') else label)
        
        command_list = sorted(list(all_commands))
        self.command_memory.add_items([f"command_{cmd}" for cmd in command_list])
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (audio_batch, label_batch) in enumerate(train_loader):
                batch_size = audio_batch.shape[0]
                
                for i in range(batch_size):
                    audio = audio_batch[i].numpy()
                    label = label_batch[i].item() if hasattr(label_batch[i], 'item') else label_batch[i]
                    
                    # Extract and encode features
                    features = self.extract_features(audio)
                    quantized = self.quantize_features(features)
                    encoded_hv = self.encode_features(quantized)
                    
                    # Store in associative memory
                    command_label = f"command_{label}"
                    self.feature_memory.store(encoded_hv, command_label)
                    
                    # Prediction for accuracy calculation
                    prediction = self.predict_single(encoded_hv)
                    if prediction == label:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
        
        self.trained = True
        print("Training completed!")
    
    def predict_single(self, encoded_hv: Any) -> int:
        """Predict command for a single encoded hypervector.
        
        Args:
            encoded_hv: Encoded audio hypervector
            
        Returns:
            Predicted command class
        """
        recalls = self.feature_memory.recall(encoded_hv, k=1)
        if recalls:
            command_label = recalls[0][0]
            # Extract command number from label
            try:
                command_num = int(command_label.split('_')[1])
                return command_num
            except:
                return 0
        return 0
    
    def predict(self, audio_sample: np.ndarray) -> int:
        """Predict command for audio sample.
        
        Args:
            audio_sample: Audio signal
            
        Returns:
            Predicted command class
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self.extract_features(audio_sample)
        quantized = self.quantize_features(features)
        encoded_hv = self.encode_features(quantized)
        
        return self.predict_single(encoded_hv)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        correct = 0
        total = 0
        
        for audio_batch, label_batch in test_loader:
            batch_size = audio_batch.shape[0]
            
            for i in range(batch_size):
                audio = audio_batch[i].numpy()
                true_label = label_batch[i].item() if hasattr(label_batch[i], 'item') else label_batch[i]
                
                predicted_label = self.predict(audio)
                
                if predicted_label == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }