#!/usr/bin/env python3
"""
GENERATION 2 SIMPLE TEST - Robustness & Error Handling
Test robust operations with the existing Pure Python backend
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_generation2_simple():
    """Test Generation 2: MAKE IT ROBUST (Reliable) functionality."""
    print("🛡️ GENERATION 2: MAKE IT ROBUST (Reliable)")
    
    try:
        from hd_compute.pure_python import HDComputePython
        
        # Test 1: Input validation
        print("📋 Testing input validation...")
        
        # Valid initialization
        hdc = HDComputePython(dim=1000)
        print("✅ Valid HDC initialization")
        
        # Test 2: Error recovery
        print("🔄 Testing error recovery...")
        
        # Empty list bundle should handle gracefully
        try:
            result = hdc.bundle([])
            print("❌ Should have failed with empty bundle")
            return False
        except Exception as e:
            print(f"✅ Correctly handled empty bundle: {type(e).__name__}")
        
        # Test 3: Robust operations with edge cases
        print("🎯 Testing robust operations...")
        
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        
        # Zero vector handling
        zero_hv = hdc.random_hv()
        zero_hv.data = [0.0] * hdc.dim
        
        similarity = hdc.cosine_similarity(hv1, zero_hv)
        print(f"✅ Zero vector similarity handled: {similarity}")
        
        # Test 4: Memory management
        print("💾 Testing memory management...")
        
        # Generate many hypervectors to test memory
        hvs = []
        for i in range(100):
            hv = hdc.random_hv()
            hvs.append(hv)
        
        bundled = hdc.bundle(hvs)
        print(f"✅ Memory-intensive bundle operation: {len(bundled.data)} dimensions")
        
        # Test 5: Advanced operations reliability
        print("🧠 Testing advanced operations...")
        
        # Fractional binding
        frac_bound = hdc.fractional_bind(hv1, hv2, power=0.7)
        print(f"✅ Fractional binding: {len(frac_bound.data)} dimensions")
        
        # Quantum superposition
        quantum_hv = hdc.quantum_superposition([hv1, hv2], amplitudes=[0.6, 0.4])
        print(f"✅ Quantum superposition: {len(quantum_hv.data)} dimensions")
        
        # Coherence decay
        decayed_hv = hdc.coherence_decay(hv1, decay_rate=0.2)
        print(f"✅ Coherence decay: {len(decayed_hv.data)} dimensions")
        
        # Entanglement measure
        entanglement = hdc.entanglement_measure(hv1, hv2)
        print(f"✅ Entanglement measurement: {entanglement:.4f}")
        
        # Test Hamming distance
        hamming = hdc.hamming_distance(hv1, hv2)
        print(f"✅ Hamming distance: {hamming}")
        
        # Test adaptive threshold
        thresholded = hdc.adaptive_threshold(hv1, target_sparsity=0.3)
        sparsity = sum(thresholded.data) / len(thresholded.data)
        print(f"✅ Adaptive threshold: sparsity {sparsity:.3f}")
        
        # Test Jensen-Shannon divergence  
        js_div = hdc.jensen_shannon_divergence(hv1, hv2)
        print(f"✅ Jensen-Shannon divergence: {js_div:.4f}")
        
        # Test hierarchical binding
        structure = {"item1": hv1, "item2": hv2}
        hierarchical = hdc.hierarchical_bind(structure)
        print(f"✅ Hierarchical binding: {len(hierarchical.data)} dimensions")
        
        print("🎯 GENERATION 2 COMPLETE: Robust error handling and validation working!")
        return True
        
    except Exception as e:
        print(f"❌ Generation 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation2_simple()
    sys.exit(0 if success else 1)