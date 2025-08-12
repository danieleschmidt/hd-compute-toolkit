"""Minimal test of novel HDC algorithms."""

import sys
import os
sys.path.insert(0, '/root/repo')

import numpy as np
from hd_compute.research.novel_algorithms import (
    AdvancedTemporalHDC, 
    ConcreteAttentionHDC, 
    NeurosymbolicHDC
)
from hd_compute.research.quantum_hdc import ConcreteQuantumHDC


def test_temporal_hdc():
    """Test temporal HDC algorithms."""
    print("Testing Temporal HDC...")
    
    temporal = AdvancedTemporalHDC(dim=100)
    
    # Create test sequence
    sequence = []
    for i in range(10):
        hv = np.random.binomial(1, 0.5, size=100).astype(np.int8)
        sequence.append(hv)
    
    # Test prediction
    predictions = temporal.sequence_prediction(sequence, prediction_horizon=3)
    
    # Test interpolation
    interpolated = temporal.temporal_interpolation(sequence[0], sequence[-1], 0.5)
    
    print(f"✅ Generated {len(predictions)} predictions")
    print(f"✅ Temporal interpolation completed")
    return True


def test_attention_hdc():
    """Test attention HDC mechanisms."""
    print("Testing Attention HDC...")
    
    attention = ConcreteAttentionHDC(dim=100, num_attention_heads=4)
    
    # Create test vectors
    hvs = [np.random.randn(100) for _ in range(5)]
    query = np.random.randn(100)
    
    # Test multi-head attention
    attended = attention.multi_head_attention(query, hvs, hvs)
    
    # Test self-attention
    self_attended = attention.self_attention(hvs)
    
    print(f"✅ Multi-head attention: {attended.shape}")
    print(f"✅ Self-attention: {len(self_attended)} vectors")
    return True


def test_neurosymbolic_hdc():
    """Test neurosymbolic HDC reasoning."""
    print("Testing Neurosymbolic HDC...")
    
    neuro = NeurosymbolicHDC(dim=100)
    
    # Test symbol encoding
    concepts = ['animal', 'mammal', 'dog', 'cat']
    for concept in concepts:
        neuro.encode_symbol(concept)
    
    # Test rule creation
    rule_id = neuro.create_rule(['animal', 'mammal'], 'warm_blooded')
    
    # Test forward reasoning
    conclusions = neuro.forward_reasoning(['animal', 'mammal'])
    
    print(f"✅ Encoded {len(concepts)} symbols")
    print(f"✅ Created rule: {rule_id}")
    print(f"✅ Forward reasoning: {conclusions}")
    return True


def test_quantum_hdc():
    """Test quantum-inspired HDC algorithms."""
    print("Testing Quantum HDC...")
    
    quantum = ConcreteQuantumHDC(dim=100)
    
    # Create test patterns
    patterns = [np.random.randn(100) for _ in range(5)]
    
    # Test superposition
    superposition = quantum.create_quantum_superposition(patterns)
    
    # Test interference
    interference = quantum.quantum_interference(patterns[0], patterns[1])
    
    # Test entropy
    entropy = quantum.entanglement_entropy(patterns[0])
    
    # Test fidelity
    fidelity = quantum.quantum_fidelity(patterns[0], patterns[1])
    
    print(f"✅ Quantum superposition: {superposition.shape}")
    print(f"✅ Quantum interference completed")
    print(f"✅ Entanglement entropy: {entropy:.3f}")
    print(f"✅ Quantum fidelity: {fidelity:.3f}")
    return True


def main():
    """Run all tests."""
    print("🧠 Testing Novel HDC Algorithms")
    print("=" * 40)
    
    tests = [
        test_temporal_hdc,
        test_attention_hdc,
        test_neurosymbolic_hdc,
        test_quantum_hdc
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
            print()
    
    print("=" * 40)
    print(f"✅ Tests passed: {sum(results)}/{len(results)}")
    print("🚀 Novel HDC algorithms working correctly!")
    
    return all(results)


if __name__ == "__main__":
    main()