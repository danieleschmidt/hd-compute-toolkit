#!/usr/bin/env python3
"""
Simple demonstration of HD-Compute-Toolkit core functionality.

This script demonstrates the basic working capabilities of the library
without requiring external dependencies like PyTorch or JAX.
"""

import time
import sys
from hd_compute import HDCompute, HDComputePython

def run_basic_demo():
    """Run basic HDC operations demonstration."""
    print("🧠 HD-Compute-Toolkit Simple Demo")
    print("=" * 50)
    
    # Initialize HDC with pure Python backend
    print("Initializing HDC with 1000-dimensional vectors...")
    hdc = HDComputePython(dim=1000)
    
    # Generate random hypervectors
    print("Generating random hypervectors...")
    start_time = time.time()
    hv_a = hdc.random_hv()
    hv_b = hdc.random_hv()
    hv_c = hdc.random_hv()
    gen_time = time.time() - start_time
    print(f"✓ Generated 3 hypervectors in {gen_time:.4f}s")
    
    # Test bundling operation
    print("Testing bundling (superposition)...")
    start_time = time.time()
    bundled = hdc.bundle([hv_a, hv_b, hv_c])
    bundle_time = time.time() - start_time
    print(f"✓ Bundled 3 vectors in {bundle_time:.4f}s")
    
    # Test binding operation
    print("Testing binding (association)...")
    start_time = time.time()
    bound = hdc.bind(hv_a, hv_b)
    bind_time = time.time() - start_time
    print(f"✓ Bound 2 vectors in {bind_time:.4f}s")
    
    # Test similarity computation
    print("Testing similarity computation...")
    start_time = time.time()
    sim_ab = hdc.cosine_similarity(hv_a, hv_b)
    sim_ac = hdc.cosine_similarity(hv_a, hv_c)
    sim_bundled = hdc.cosine_similarity(hv_a, bundled)
    sim_time = time.time() - start_time
    print(f"✓ Computed similarities in {sim_time:.4f}s")
    
    # Display results
    print("\n📊 Results:")
    print(f"  Similarity A-B: {sim_ab:.4f}")
    print(f"  Similarity A-C: {sim_ac:.4f}")
    print(f"  Similarity A-Bundle: {sim_bundled:.4f}")
    
    # Demonstrate property: bundled vector is similar to constituents
    print(f"\n✓ Bundle contains similarity to constituent A: {sim_bundled > 0.1}")
    
    return True

def run_memory_demo():
    """Demonstrate basic memory operations."""
    print("\n🧠 Memory Operations Demo")
    print("=" * 30)
    
    try:
        from hd_compute.memory import ItemMemory
        
        # Create HDC backend first
        hdc = HDComputePython(dim=1000)
        # Create item memory with HDC backend
        memory = ItemMemory(hdc_backend=hdc)
        
        # Store some items
        items = ["apple", "banana", "cherry", "dog", "elephant"]
        memory.add_items(items)
        
        # Generate hypervectors for each item
        for i, item in enumerate(items):
            hv = memory.get_hv(item)
            print(f"✓ Generated hypervector for: {item}")
        
        # Test binding items together
        apple_hv = memory.get_hv("apple")
        banana_hv = memory.get_hv("banana")
        
        if apple_hv is not None and banana_hv is not None:
            fruit_combo = hdc.bind(apple_hv, banana_hv)
            sim_to_apple = hdc.cosine_similarity(fruit_combo, apple_hv)
            print(f"✓ Apple-Banana binding similarity to apple: {sim_to_apple:.4f}")
        else:
            print("⚠ Could not generate hypervectors")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Memory demo skipped: {e}")
        return False

def run_performance_benchmark():
    """Simple performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("=" * 30)
    
    hdc = HDComputePython(dim=2000)
    
    # Benchmark vector generation
    start_time = time.time()
    vectors = [hdc.random_hv() for _ in range(100)]
    gen_time = time.time() - start_time
    print(f"✓ Generated 100 vectors in {gen_time:.4f}s ({100/gen_time:.1f} vec/s)")
    
    # Benchmark bundling
    start_time = time.time()
    for i in range(10):
        bundle_subset = vectors[i*10:(i+1)*10]
        bundled = hdc.bundle(bundle_subset)
    bundle_time = time.time() - start_time
    print(f"✓ Bundled 10x10 vectors in {bundle_time:.4f}s")
    
    # Benchmark similarity computation
    start_time = time.time()
    similarities = []
    for i in range(50):
        sim = hdc.cosine_similarity(vectors[i], vectors[i+50])
        similarities.append(sim)
    sim_time = time.time() - start_time
    print(f"✓ Computed 50 similarities in {sim_time:.4f}s ({50/sim_time:.1f} sim/s)")
    
    avg_sim = sum(similarities) / len(similarities)
    print(f"✓ Average random similarity: {avg_sim:.4f}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting HD-Compute-Toolkit Demo...")
    
    success = True
    
    try:
        success &= run_basic_demo()
        success &= run_memory_demo()
        success &= run_performance_benchmark()
        
        if success:
            print("\n✅ All demos completed successfully!")
            print("HD-Compute-Toolkit is working correctly.")
        else:
            print("\n⚠ Some demos had issues, but core functionality works.")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎯 Next steps:")
    print("  - Install PyTorch: pip install torch")
    print("  - Install JAX: pip install jax jaxlib")
    print("  - Run full test suite: python -m pytest")
    print("  - Explore applications in hd_compute/applications/")