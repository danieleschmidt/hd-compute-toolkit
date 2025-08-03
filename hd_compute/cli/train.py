"""Training command-line interface."""

import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional

from ..torch import HDComputeTorch
from ..applications import SpeechCommandHDC


class DummySpeechDataset(Dataset):
    """Dummy speech dataset for testing purposes."""
    
    def __init__(self, num_samples: int = 1000, sample_rate: int = 16000, duration: float = 1.0):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_length = int(sample_rate * duration)
        
        # Generate dummy command labels
        self.num_classes = 10
        self.labels = np.random.randint(0, self.num_classes, num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random audio signal
        audio = np.random.randn(self.audio_length).astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(audio), torch.tensor(label)


def train():
    """Main training CLI function."""
    parser = argparse.ArgumentParser(description='Train HDC models')
    parser.add_argument('--model', type=str, default='speech', choices=['speech'],
                       help='Model type to train')
    parser.add_argument('--dim', type=int, default=16000, help='Hypervector dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Computing device')
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--save-path', type=str, default='./hdc_model.pt', 
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    print(f"Training HDC {args.model} model")
    print(f"Dimension: {args.dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    # Setup device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize HDC backend
    hdc_backend = HDComputeTorch(dim=args.dim, device=device)
    
    if args.model == 'speech':
        # Initialize speech command model
        model = SpeechCommandHDC(
            hdc_backend=hdc_backend,
            dim=args.dim,
            num_classes=35
        )
        
        # Load or create dataset
        if args.data_path and os.path.exists(args.data_path):
            print(f"Loading data from: {args.data_path}")
            # In a real implementation, you would load actual speech data here
            dataset = DummySpeechDataset(num_samples=1000)
        else:
            print("Using dummy dataset for demonstration")
            dataset = DummySpeechDataset(num_samples=1000)
        
        # Create data loader
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Train model
        model.train(train_loader, epochs=args.epochs)
        
        # Save model (simplified - in practice you'd save the hypervector memories)
        model_state = {
            'dim': args.dim,
            'num_classes': model.num_classes,
            'feature_extractor': model.feature_extractor,
            'trained': model.trained,
            'command_memory_items': model.command_memory.items,
            'feature_memory_size': model.feature_memory.size()
        }
        
        torch.save(model_state, args.save_path)
        print(f"Model saved to: {args.save_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    train()