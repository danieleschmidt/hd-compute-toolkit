"""Simple scaling and performance test."""

import time
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, '/root/repo')

try:
    from hd_compute.performance.advanced_optimization import (
        AdaptiveCache, VectorizedOperations, HypervectorMemoryPool
    )
    from hd_compute.research.optimized_algorithms import (
        OptimizedTemporalHDC, create_optimized_suite
    )
    print("✅ Successfully imported optimization modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_caching():
    """Test caching performance."""
    print("\n💾 Testing Caching...")
    
    cache = AdaptiveCache(max_size=50, ttl_seconds=60)
    
    # Test cache operations
    for i in range(30):
        key = f"test_{i}"
        value = np.random.randn(100)
        cache.put(key, value)
    
    # Test cache hits
    hits = 0
    for i in range(20):
        key = f"test_{i}"
        if cache.get(key) is not None:
            hits += 1
    
    hit_rate = hits / 20
    stats = cache.get_stats()
    
    print(f"  Cache hit rate: {hit_rate:.1%}")
    print(f"  Cache utilization: {stats['utilization']:.1%}")
    return hit_rate > 0.8


def test_vectorized_ops():
    """Test vectorized operations."""
    print("\n⚡ Testing Vectorized Operations...")
    
    ops = VectorizedOperations()
    query = np.random.randn(500)
    vectors = np.random.randn(100, 500)
    
    # Test batch similarity
    start_time = time.time()
    similarities = ops.batch_cosine_similarity(query, vectors)
    batch_time = time.time() - start_time
    
    # Test individual (for comparison)
    start_time = time.time()
    individual_sims = []
    for vector in vectors[:10]:  # Test subset
        sim = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector))
        individual_sims.append(sim)
    individual_time = time.time() - start_time
    
    # Scale individual time to full batch
    scaled_individual_time = individual_time * 10
    speedup = scaled_individual_time / batch_time if batch_time > 0 else 1
    
    print(f"  Vectorization speedup: {speedup:.1f}x")
    print(f"  Batch processing time: {batch_time:.4f}s")
    return speedup > 1.5


def test_memory_pool():
    """Test memory pool efficiency."""
    print("\n🧠 Testing Memory Pool...")
    
    pool = HypervectorMemoryPool(dim=500, initial_size=20, max_size=100)
    
    # Allocate vectors
    vectors = []
    start_time = time.time()
    for _ in range(50):
        vector = pool.get_vector()
        vectors.append(vector)
    alloc_time = time.time() - start_time
    
    # Return vectors
    start_time = time.time()
    for vector in vectors:
        pool.return_vector(vector)
    return_time = time.time() - start_time
    
    stats = pool.get_stats()
    
    print(f"  Allocation time: {alloc_time:.4f}s")
    print(f"  Return time: {return_time:.4f}s")
    print(f"  Pool utilization: {stats['utilization']:.1%}")
    return alloc_time < 0.1 and return_time < 0.1


def test_optimized_algorithms():
    """Test optimized algorithm performance."""
    print("\n🚀 Testing Optimized Algorithms...")
    
    temporal = OptimizedTemporalHDC(dim=300)
    
    # Test sequence prediction
    sequence = [np.random.randn(300) for _ in range(15)]
    
    start_time = time.time()
    predictions = temporal.sequence_prediction(sequence, prediction_horizon=3)
    prediction_time = time.time() - start_time
    
    # Test temporal binding
    current_hv = np.random.randn(300)
    context_hvs = [np.random.randn(300) for _ in range(3)]
    
    start_time = time.time()
    bound_result = temporal.temporal_binding(current_hv, context_hvs, [1, 2, 3])
    binding_time = time.time() - start_time
    
    stats = temporal.get_performance_stats()
    
    print(f"  Prediction time: {prediction_time:.4f}s")
    print(f"  Binding time: {binding_time:.4f}s")
    print(f"  Cache utilization: {stats['cache_stats']['utilization']:.1%}")
    print(f"  Memory pool available: {stats['memory_pool_stats']['available']}")
    
    return len(predictions) == 3 and bound_result.shape == (300,)


def test_suite_creation():
    """Test optimized suite creation."""
    print("\n🎯 Testing Suite Creation...")
    
    suite = create_optimized_suite(dim=200)
    
    required_components = ['temporal', 'attention', 'quantum']
    all_present = all(comp in suite for comp in required_components)
    
    # Test basic operations
    temporal = suite['temporal']
    attention = suite['attention']
    quantum = suite['quantum']
    
    # Simple functionality test
    test_vectors = [np.random.randn(200) for _ in range(5)]
    query = np.random.randn(200)
    
    try:
        # Temporal test
        temporal_result = temporal.sequence_prediction(test_vectors, 1)
        
        # Attention test
        attention_result = attention.multi_head_attention_optimized(
            query, test_vectors, test_vectors
        )
        
        # Quantum test
        quantum_result = quantum.create_quantum_superposition_optimized(test_vectors[:3])
        
        operations_work = (
            len(temporal_result) == 1 and
            attention_result.shape == (200,) and
            quantum_result.shape == (200,)
        )
        
    except Exception as e:
        print(f"  Operation error: {e}")
        operations_work = False
    
    print(f"  All components present: {all_present}")
    print(f"  Operations working: {operations_work}")
    
    return all_present and operations_work


def main():
    """Run simple scaling tests."""
    print("🚀 SIMPLE SCALING & PERFORMANCE TEST")
    print("=" * 40)
    
    tests = [
        ("Caching Performance", test_caching),
        ("Vectorized Operations", test_vectorized_ops),
        ("Memory Pool", test_memory_pool),
        ("Optimized Algorithms", test_optimized_algorithms),
        ("Suite Creation", test_suite_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📊 SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    success_rate = passed / total * 100
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🚀 EXCELLENT: Scaling optimizations working!")
    elif success_rate >= 60:
        print("✅ GOOD: Most optimizations functional")
    else:
        print("⚠️ NEEDS WORK: Some optimizations failing")
    
    return results


if __name__ == "__main__":
    main()