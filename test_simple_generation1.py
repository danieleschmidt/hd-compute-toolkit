#!/usr/bin/env python3
"""
GENERATION 1 SIMPLE TEST - Basic HDC Functionality
Test core HDC operations with Pure Python backend
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_generation1_simple():
    """Test Generation 1: MAKE IT WORK (Simple) functionality."""
    print("🚀 GENERATION 1: MAKE IT WORK (Simple)")
    
    # Test Pure Python backend
    try:
        from hd_compute.pure_python import HDComputePython
        
        print("✅ HDComputePython import successful")
        
        # Initialize HDC context
        hdc = HDComputePython(dim=1000)
        print(f"✅ HDC initialized with dim={hdc.dim}")
        
        # Generate random hypervectors
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        print(f"✅ Random hypervectors generated: {len(hv1)} dimensions")
        
        # Bundle operation
        bundled = hdc.bundle([hv1, hv2])
        print(f"✅ Bundle operation: {len(bundled)} dimensions")
        
        # Bind operation  
        bound = hdc.bind(hv1, hv2)
        print(f"✅ Bind operation: {len(bound)} dimensions")
        
        # Similarity computation
        similarity = hdc.cosine_similarity(hv1, hv2)
        print(f"✅ Cosine similarity: {similarity:.4f}")
        
        print("🎯 GENERATION 1 COMPLETE: Core HDC operations working!")
        return True
        
    except Exception as e:
        print(f"❌ Generation 1 failed: {e}")
        return False

if __name__ == "__main__":
    success = test_generation1_simple()
    sys.exit(0 if success else 1)