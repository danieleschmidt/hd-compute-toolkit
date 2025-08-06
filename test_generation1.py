#!/usr/bin/env python3
"""
Generation 1 Test Suite: Verify core HDC functionality works across all backends
"""

import sys
import traceback
import numpy as np
from typing import Dict, List

# Test HDC backends
from hd_compute.numpy import HDComputeNumPy
from hd_compute.torch import HDComputeTorch

def test_backend_core_operations(backend_class, backend_name: str) -> Dict[str, bool]:
    """Test core operations for a specific backend."""
    results = {}
    print(f"\n=== Testing {backend_name} Backend ===")
    
    try:
        # Initialize backend
        hdc = backend_class(dim=1000)
        results['initialization'] = True
        print(f"✅ {backend_name} initialization successful")
        
        # Test basic operations
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv() 
        results['random_hv'] = True
        print(f"✅ {backend_name} random_hv successful")
        
        bundled = hdc.bundle([hv1, hv2])
        results['bundle'] = True
        print(f"✅ {backend_name} bundle successful")
        
        bound = hdc.bind(hv1, hv2)
        results['bind'] = True
        print(f"✅ {backend_name} bind successful")
        
        similarity = hdc.cosine_similarity(hv1, hv2)
        results['cosine_similarity'] = True
        print(f"✅ {backend_name} cosine_similarity: {similarity:.3f}")
        
        # Test advanced similarity metrics (NEW in Generation 1)
        try:
            js_div = hdc.jensen_shannon_divergence(hv1, hv2)
            results['jensen_shannon_divergence'] = True
            print(f"✅ {backend_name} jensen_shannon_divergence: {js_div:.3f}")
        except Exception as e:
            results['jensen_shannon_divergence'] = False
            print(f"❌ {backend_name} jensen_shannon_divergence failed: {e}")
        
        try:
            wass_dist = hdc.wasserstein_distance(hv1, hv2)
            results['wasserstein_distance'] = True
            print(f"✅ {backend_name} wasserstein_distance: {wass_dist:.3f}")
        except Exception as e:
            results['wasserstein_distance'] = False
            print(f"❌ {backend_name} wasserstein_distance failed: {e}")
        
        # Test novel research operations (NEW in Generation 1)
        try:
            frac_bound = hdc.fractional_bind(hv1, hv2, power=0.3)
            results['fractional_bind'] = True
            print(f"✅ {backend_name} fractional_bind successful")
        except Exception as e:
            results['fractional_bind'] = False
            print(f"❌ {backend_name} fractional_bind failed: {e}")
        
        try:
            quantum_sup = hdc.quantum_superposition([hv1, hv2], [0.6, 0.4])
            results['quantum_superposition'] = True
            print(f"✅ {backend_name} quantum_superposition successful")
        except Exception as e:
            results['quantum_superposition'] = False
            print(f"❌ {backend_name} quantum_superposition failed: {e}")
        
        try:
            entanglement = hdc.entanglement_measure(hv1, hv2)
            results['entanglement_measure'] = True
            print(f"✅ {backend_name} entanglement_measure: {entanglement:.3f}")
        except Exception as e:
            results['entanglement_measure'] = False
            print(f"❌ {backend_name} entanglement_measure failed: {e}")
        
        try:
            decayed = hdc.coherence_decay(hv1, decay_rate=0.05)
            results['coherence_decay'] = True
            print(f"✅ {backend_name} coherence_decay successful")
        except Exception as e:
            results['coherence_decay'] = False
            print(f"❌ {backend_name} coherence_decay failed: {e}")
        
        try:
            thresholded = hdc.adaptive_threshold(hv1, target_sparsity=0.6)
            results['adaptive_threshold'] = True
            print(f"✅ {backend_name} adaptive_threshold successful")
        except Exception as e:
            results['adaptive_threshold'] = False
            print(f"❌ {backend_name} adaptive_threshold failed: {e}")
        
        # Test hierarchical operations (NEW in Generation 1)
        try:
            structure = {"key1": hv1, "key2": {"nested": hv2}}
            hierarchical = hdc.hierarchical_bind(structure)
            results['hierarchical_bind'] = True
            print(f"✅ {backend_name} hierarchical_bind successful")
        except Exception as e:
            results['hierarchical_bind'] = False
            print(f"❌ {backend_name} hierarchical_bind failed: {e}")
        
        try:
            basis_hvs = [hdc.random_hv() for _ in range(3)]
            projection = hdc.semantic_projection(hv1, basis_hvs)
            results['semantic_projection'] = True
            print(f"✅ {backend_name} semantic_projection successful: {len(projection)} coeffs")
        except Exception as e:
            results['semantic_projection'] = False
            print(f"❌ {backend_name} semantic_projection failed: {e}")
            
    except Exception as e:
        print(f"❌ {backend_name} failed during initialization: {e}")
        traceback.print_exc()
        for key in ['initialization', 'random_hv', 'bundle', 'bind', 'cosine_similarity', 
                   'jensen_shannon_divergence', 'wasserstein_distance', 'fractional_bind',
                   'quantum_superposition', 'entanglement_measure', 'coherence_decay', 
                   'adaptive_threshold', 'hierarchical_bind', 'semantic_projection']:
            results[key] = False
    
    return results

def test_research_functionality():
    """Test advanced research functionality across backends."""
    print("\n🧠 GENERATION 1 RESEARCH FUNCTIONALITY TEST")
    
    # Test NumPy backend research operations
    hdc_numpy = HDComputeNumPy(dim=500)
    
    # Create test hypervectors
    concept_animal = hdc_numpy.random_hv()
    concept_dog = hdc_numpy.random_hv() 
    concept_color = hdc_numpy.random_hv()
    concept_brown = hdc_numpy.random_hv()
    
    print("\n🔬 Testing Advanced HDC Research Operations:")
    
    # Test fractional binding for gradual association
    partial_association = hdc_numpy.fractional_bind(concept_dog, concept_animal, power=0.7)
    full_association = hdc_numpy.fractional_bind(concept_dog, concept_animal, power=1.0)
    
    partial_sim = hdc_numpy.cosine_similarity(partial_association, concept_animal)
    full_sim = hdc_numpy.cosine_similarity(full_association, concept_animal)
    
    print(f"✅ Fractional binding: partial={partial_sim:.3f}, full={full_sim:.3f}")
    
    # Test quantum superposition with different amplitudes
    quantum_state = hdc_numpy.quantum_superposition(
        [concept_dog, concept_brown], 
        amplitudes=[0.8, 0.2]
    )
    
    dog_similarity = hdc_numpy.cosine_similarity(quantum_state, concept_dog)
    brown_similarity = hdc_numpy.cosine_similarity(quantum_state, concept_brown)
    
    print(f"✅ Quantum superposition: dog_sim={dog_similarity:.3f}, brown_sim={brown_similarity:.3f}")
    
    # Test entanglement measure
    entanglement = hdc_numpy.entanglement_measure(concept_dog, concept_animal)
    print(f"✅ Entanglement measure: {entanglement:.3f}")
    
    # Test coherence decay simulation
    original = hdc_numpy.random_hv()
    decayed_light = hdc_numpy.coherence_decay(original, decay_rate=0.1)
    decayed_heavy = hdc_numpy.coherence_decay(original, decay_rate=0.5)
    
    light_sim = hdc_numpy.cosine_similarity(original, decayed_light)
    heavy_sim = hdc_numpy.cosine_similarity(original, decayed_heavy)
    
    print(f"✅ Coherence decay: light={light_sim:.3f}, heavy={heavy_sim:.3f}")
    
    # Test hierarchical binding with complex structure
    knowledge_structure = {
        "animals": {
            "mammals": [concept_dog],
            "birds": [hdc_numpy.random_hv()]
        },
        "properties": {
            "colors": [concept_brown, concept_color]
        }
    }
    
    hierarchical_encoding = hdc_numpy.hierarchical_bind(knowledge_structure)
    print(f"✅ Hierarchical binding: encoded complex knowledge structure")
    
    # Test semantic projection
    semantic_basis = [concept_animal, concept_color, hdc_numpy.random_hv()]
    brown_dog = hdc_numpy.bind(concept_dog, concept_brown)
    projection_coeffs = hdc_numpy.semantic_projection(brown_dog, semantic_basis)
    
    print(f"✅ Semantic projection: coefficients={[f'{c:.3f}' for c in projection_coeffs]}")

def main():
    """Run Generation 1 test suite."""
    print("🚀 HD-COMPUTE GENERATION 1 TEST SUITE")
    print("=" * 50)
    
    # Test all available backends
    backends = [
        (HDComputeNumPy, "NumPy"),
        (HDComputeTorch, "PyTorch")
    ]
    
    all_results = {}
    
    for backend_class, backend_name in backends:
        try:
            results = test_backend_core_operations(backend_class, backend_name)
            all_results[backend_name] = results
        except ImportError as e:
            print(f"⚠️  {backend_name} backend not available: {e}")
            all_results[backend_name] = {}
    
    # Test research functionality
    test_research_functionality()
    
    # Summary
    print("\n📊 GENERATION 1 TEST SUMMARY")
    print("=" * 50)
    
    for backend_name, results in all_results.items():
        if not results:
            continue
            
        total_tests = len(results)
        passed_tests = sum(results.values())
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"{backend_name}: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = [test for test, result in results.items() if not result]
        if failed_tests:
            print(f"  Failed: {', '.join(failed_tests)}")
    
    # Overall Generation 1 status
    total_backends_tested = len([r for r in all_results.values() if r])
    if total_backends_tested > 0:
        overall_success = all(
            sum(results.values()) == len(results) 
            for results in all_results.values() if results
        )
        
        print(f"\n🎯 GENERATION 1 STATUS: {'✅ COMPLETE' if overall_success else '⚠️ PARTIAL'}")
        if overall_success:
            print("All core HDC operations and research features are working!")
        else:
            print("Some operations need attention before proceeding to Generation 2.")
    else:
        print("\n❌ GENERATION 1 STATUS: No backends available for testing")
    
    return all_results

if __name__ == "__main__":
    main()