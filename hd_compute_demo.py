#!/usr/bin/env python3
"""
Generation 1 HDC Demo: Basic Functionality Validation
Test core hyperdimensional computing operations.
"""

import sys
import traceback

def test_basic_imports():
    """Test that core modules import successfully."""
    try:
        from hd_compute.core.hdc import HDCompute
        from hd_compute.pure_python.hdc_python import HDComputePython
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_operations():
    """Test basic HDC operations."""
    try:
        from hd_compute.pure_python.hdc_python import HDComputePython
        
        # Initialize HDC context
        hdc = HDComputePython(dim=1000)
        print(f"✅ HDC initialized with dimension {hdc.dim}")
        
        # Generate random hypervectors
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        print(f"✅ Generated random hypervectors of length {len(hv1.data)}")
        
        # Test bundling (superposition)
        bundled = hdc.bundle([hv1, hv2])
        print("✅ Bundle operation successful")
        
        # Test binding (association) 
        bound = hdc.bind(hv1, hv2)
        print("✅ Bind operation successful")
        
        # Test similarity
        similarity = hdc.cosine_similarity(hv1, hv2)
        print(f"✅ Cosine similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        traceback.print_exc()
        return False

def test_research_operations():
    """Test advanced research operations."""
    try:
        from hd_compute.pure_python.hdc_python import HDComputePython
        
        hdc = HDComputePython(dim=1000)
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        
        # Test fractional binding
        frac_bound = hdc.fractional_bind(hv1, hv2, power=0.5)
        print("✅ Fractional binding successful")
        
        # Test quantum superposition
        quantum_sup = hdc.quantum_superposition([hv1, hv2], amplitudes=[0.6, 0.8])
        print("✅ Quantum superposition successful")
        
        # Test entanglement measure
        entanglement = hdc.entanglement_measure(hv1, hv2)
        print(f"✅ Entanglement measure: {entanglement:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research operations failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run Generation 1 validation tests."""
    print("🚀 GENERATION 1: BASIC HDC FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Operations", test_basic_operations), 
        ("Research Operations", test_research_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 GENERATION 1 RESULTS:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Generation 1 COMPLETE - Basic functionality working!")
        return True
    else:
        print("⚠️  Generation 1 needs fixes")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)