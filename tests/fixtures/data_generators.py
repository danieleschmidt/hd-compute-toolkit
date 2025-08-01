"""Test data generators for HD-Compute-Toolkit tests."""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import tempfile
import json


class HypervectorGenerator:
    """Generator for test hypervectors with various properties."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with fixed seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(seed)
    
    def binary_hypervector(self, dim: int, batch_size: int = 1) -> np.ndarray:
        """Generate binary hypervectors (0/1 encoding)."""
        if batch_size == 1:
            return self.rng.choice([0, 1], size=dim).astype(np.int8)
        return self.rng.choice([0, 1], size=(batch_size, dim)).astype(np.int8)
    
    def bipolar_hypervector(self, dim: int, batch_size: int = 1) -> np.ndarray:
        """Generate bipolar hypervectors (-1/+1 encoding)."""
        if batch_size == 1:
            return self.rng.choice([-1, 1], size=dim).astype(np.int8)
        return self.rng.choice([-1, 1], size=(batch_size, dim)).astype(np.int8)
    
    def sparse_hypervector(self, dim: int, sparsity: float = 0.1, batch_size: int = 1) -> np.ndarray:
        """Generate sparse hypervectors with controlled sparsity."""
        if batch_size == 1:
            hv = np.zeros(dim, dtype=np.int8)
            num_ones = int(dim * sparsity)
            indices = self.rng.choice(dim, size=num_ones, replace=False)
            hv[indices] = 1
            return hv
        else:
            hvs = np.zeros((batch_size, dim), dtype=np.int8)
            for i in range(batch_size):
                num_ones = int(dim * sparsity)
                indices = self.rng.choice(dim, size=num_ones, replace=False)
                hvs[i, indices] = 1
            return hvs
    
    def orthogonal_hypervectors(self, dim: int, num_vectors: int) -> List[np.ndarray]:
        """Generate approximately orthogonal hypervectors."""
        hvs = []
        for _ in range(num_vectors):
            hv = self.bipolar_hypervector(dim)
            # Gram-Schmidt-like orthogonalization (simplified)
            for existing_hv in hvs:
                correlation = np.dot(hv, existing_hv) / dim
                if abs(correlation) > 0.1:  # If too similar, regenerate
                    hv = self.bipolar_hypervector(dim)
            hvs.append(hv)
        return hvs
    
    def torch_hypervector(self, dim: int, device: str = "cpu", dtype: torch.dtype = torch.int8) -> torch.Tensor:
        """Generate PyTorch tensor hypervector."""
        hv = torch.randint(0, 2, (dim,), generator=self.torch_gen, dtype=dtype)
        return hv.to(device)
    
    def similar_hypervectors(self, dim: int, base_hv: Optional[np.ndarray] = None, 
                           similarity: float = 0.8, num_vectors: int = 1) -> List[np.ndarray]:
        """Generate hypervectors with controlled similarity to a base vector."""
        if base_hv is None:
            base_hv = self.bipolar_hypervector(dim)
        
        similar_hvs = []
        for _ in range(num_vectors):
            # Start with base vector
            hv = base_hv.copy()
            
            # Flip bits to achieve desired similarity
            num_flips = int(dim * (1 - similarity) / 2)
            flip_indices = self.rng.choice(dim, size=num_flips, replace=False)
            hv[flip_indices] *= -1
            
            similar_hvs.append(hv)
        
        return similar_hvs


class DatasetGenerator:
    """Generator for test datasets and sequences."""
    
    def __init__(self, hv_gen: HypervectorGenerator):
        """Initialize with hypervector generator."""
        self.hv_gen = hv_gen
    
    def symbol_mapping(self, symbols: List[str], dim: int) -> Dict[str, np.ndarray]:
        """Generate mapping from symbols to hypervectors."""
        mapping = {}
        for symbol in symbols:
            mapping[symbol] = self.hv_gen.binary_hypervector(dim)
        return mapping
    
    def sequence_dataset(self, sequences: List[List[str]], symbol_mapping: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Generate encoded sequences using symbol mapping."""
        encoded_sequences = []
        for sequence in sequences:
            # Bundle all symbols in sequence
            sequence_hvs = [symbol_mapping[symbol] for symbol in sequence]
            # Simple bundling (would use proper bundling function in real implementation)
            encoded_seq = np.mean(sequence_hvs, axis=0) > 0.5
            encoded_sequences.append(encoded_seq.astype(np.int8))
        return encoded_sequences
    
    def associative_memory_data(self, num_pairs: int, dim: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate key-value pairs for associative memory testing."""
        pairs = []
        for _ in range(num_pairs):
            key = self.hv_gen.binary_hypervector(dim)
            value = self.hv_gen.binary_hypervector(dim)
            pairs.append((key, value))
        return pairs
    
    def classification_dataset(self, num_classes: int, samples_per_class: int, 
                             dim: int, noise_level: float = 0.1) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic classification dataset."""
        # Create prototype for each class
        prototypes = [self.hv_gen.bipolar_hypervector(dim) for _ in range(num_classes)]
        
        samples = []
        labels = []
        
        for class_id, prototype in enumerate(prototypes):
            for _ in range(samples_per_class):
                # Add noise to prototype
                noisy_sample = prototype.copy()
                num_flips = int(dim * noise_level)
                flip_indices = self.hv_gen.rng.choice(dim, size=num_flips, replace=False)
                noisy_sample[flip_indices] *= -1
                
                samples.append(noisy_sample)
                labels.append(class_id)
        
        return samples, labels


class BenchmarkDataGenerator:
    """Generator for benchmark and performance test data."""
    
    def __init__(self, seed: int = 42):
        """Initialize benchmark data generator."""
        self.hv_gen = HypervectorGenerator(seed)
    
    def bundle_benchmark_data(self, dimensions: List[int], batch_sizes: List[int]) -> Dict[str, Any]:
        """Generate data for bundling operation benchmarks."""
        benchmark_data = {
            "dimensions": dimensions,
            "batch_sizes": batch_sizes,
            "test_cases": []
        }
        
        for dim in dimensions:
            for batch_size in batch_sizes:
                hvs = [self.hv_gen.binary_hypervector(dim) for _ in range(batch_size)]
                benchmark_data["test_cases"].append({
                    "dim": dim,
                    "batch_size": batch_size,
                    "hypervectors": hvs
                })
        
        return benchmark_data
    
    def similarity_benchmark_data(self, dimensions: List[int], num_comparisons: int = 1000) -> Dict[str, Any]:
        """Generate data for similarity computation benchmarks."""
        benchmark_data = {
            "dimensions": dimensions,
            "num_comparisons": num_comparisons,
            "test_cases": []
        }
        
        for dim in dimensions:
            hvs1 = [self.hv_gen.bipolar_hypervector(dim) for _ in range(num_comparisons)]
            hvs2 = [self.hv_gen.bipolar_hypervector(dim) for _ in range(num_comparisons)]
            
            benchmark_data["test_cases"].append({
                "dim": dim,
                "hvs1": hvs1,
                "hvs2": hvs2
            })
        
        return benchmark_data
    
    def memory_benchmark_data(self, max_dim: int = 32000, num_vectors: int = 1000) -> Dict[str, Any]:
        """Generate data for memory usage benchmarks."""
        return {
            "max_dim": max_dim,
            "num_vectors": num_vectors,
            "dimensions": [1000, 5000, 10000, 16000, 32000],
            "vector_counts": [10, 50, 100, 500, 1000]
        }


class FixtureManager:
    """Manager for creating and saving test fixtures."""
    
    def __init__(self, fixture_dir: Optional[Path] = None):
        """Initialize fixture manager."""
        self.fixture_dir = fixture_dir or Path(__file__).parent
        self.hv_gen = HypervectorGenerator()
        self.dataset_gen = DatasetGenerator(self.hv_gen)
        self.benchmark_gen = BenchmarkDataGenerator()
    
    def create_standard_fixtures(self) -> Dict[str, Any]:
        """Create standard test fixtures."""
        fixtures = {
            "small_binary_hv": self.hv_gen.binary_hypervector(100),
            "medium_binary_hv": self.hv_gen.binary_hypervector(1000),
            "large_binary_hv": self.hv_gen.binary_hypervector(10000),
            "small_bipolar_hv": self.hv_gen.bipolar_hypervector(100),
            "medium_bipolar_hv": self.hv_gen.bipolar_hypervector(1000),
            "sparse_hv": self.hv_gen.sparse_hypervector(1000, sparsity=0.1),
            "orthogonal_hvs": self.hv_gen.orthogonal_hypervectors(1000, 5),
            "similar_hvs": self.hv_gen.similar_hypervectors(1000, similarity=0.8, num_vectors=3)
        }
        return fixtures
    
    def save_fixtures(self, fixtures: Dict[str, Any], filename: str = "test_fixtures.npz"):
        """Save fixtures to file."""
        filepath = self.fixture_dir / filename
        np.savez_compressed(filepath, **fixtures)
        return filepath
    
    def load_fixtures(self, filename: str = "test_fixtures.npz") -> Dict[str, np.ndarray]:
        """Load fixtures from file."""
        filepath = self.fixture_dir / filename
        if filepath.exists():
            return dict(np.load(filepath))
        else:
            # Create and save if doesn't exist
            fixtures = self.create_standard_fixtures()
            self.save_fixtures(fixtures, filename)
            return fixtures
    
    def create_performance_fixtures(self) -> Dict[str, Any]:
        """Create fixtures for performance testing."""
        return {
            "bundle_data": self.benchmark_gen.bundle_benchmark_data([1000, 10000], [10, 100]),
            "similarity_data": self.benchmark_gen.similarity_benchmark_data([1000, 10000]),
            "memory_data": self.benchmark_gen.memory_benchmark_data()
        }


# Factory functions for easy fixture creation
def create_test_hypervectors(seed: int = 42) -> Dict[str, np.ndarray]:
    """Factory function to create standard test hypervectors."""
    gen = HypervectorGenerator(seed)
    return {
        "binary_100": gen.binary_hypervector(100),
        "binary_1000": gen.binary_hypervector(1000),
        "binary_10000": gen.binary_hypervector(10000),
        "bipolar_100": gen.bipolar_hypervector(100),
        "bipolar_1000": gen.bipolar_hypervector(1000),
        "bipolar_10000": gen.bipolar_hypervector(10000),
        "sparse_1000": gen.sparse_hypervector(1000, 0.1),
        "batch_binary": gen.binary_hypervector(1000, batch_size=32),
        "batch_bipolar": gen.bipolar_hypervector(1000, batch_size=32)
    }


def create_test_datasets(seed: int = 42) -> Dict[str, Any]:
    """Factory function to create test datasets."""
    hv_gen = HypervectorGenerator(seed)
    dataset_gen = DatasetGenerator(hv_gen)
    
    symbols = ["A", "B", "C", "D", "E"]
    symbol_map = dataset_gen.symbol_mapping(symbols, 1000)
    sequences = [["A", "B", "C"], ["B", "C", "D"], ["A", "D", "E"]]
    
    return {
        "symbol_mapping": symbol_map,
        "sequences": sequences,
        "encoded_sequences": dataset_gen.sequence_dataset(sequences, symbol_map),
        "associative_pairs": dataset_gen.associative_memory_data(50, 1000),
        "classification_data": dataset_gen.classification_dataset(5, 20, 1000)
    }


# Global fixture manager instance
_fixture_manager = None

def get_fixture_manager() -> FixtureManager:
    """Get global fixture manager instance."""
    global _fixture_manager
    if _fixture_manager is None:
        _fixture_manager = FixtureManager()
    return _fixture_manager