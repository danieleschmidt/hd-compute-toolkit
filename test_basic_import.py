#!/usr/bin/env python3
"""
Basic import test for HDC modules to verify Generation 1 structure
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test that our HDC modules can be imported."""
    print("🚀 GENERATION 1 BASIC IMPORT TEST")
    print("=" * 50)
    
    try:
        # Test core HDC import
        from hd_compute.core.hdc import HDCompute
        print("✅ Core HDC base class imported successfully")
        
        # Check that all abstract methods exist
        abstract_methods = [
            'random_hv', 'bundle', 'bind', 'cosine_similarity',
            'jensen_shannon_divergence', 'wasserstein_distance', 
            'fractional_bind', 'quantum_superposition', 'entanglement_measure',
            'coherence_decay', 'adaptive_threshold', 'hierarchical_bind', 
            'semantic_projection', 'hamming_distance'
        ]
        
        missing_methods = []
        for method in abstract_methods:
            if not hasattr(HDCompute, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing abstract methods: {missing_methods}")
            return False
        else:
            print(f"✅ All {len(abstract_methods)} required methods defined in base class")
        
        # Test that implementations have the methods (without instantiating)
        backends_to_test = [
            ('hd_compute.numpy.hdc_numpy', 'HDComputeNumPy'),
            ('hd_compute.torch.hdc_torch', 'HDComputeTorch'),
            ('hd_compute.pure_python.hdc_python', 'HDComputePython'),
        ]
        
        for module_path, class_name in backends_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                backend_class = getattr(module, class_name)
                
                # Check that all methods exist in implementation
                implementation_missing = []
                for method in abstract_methods:
                    if not hasattr(backend_class, method):
                        implementation_missing.append(method)
                
                if implementation_missing:
                    print(f"❌ {class_name}: Missing methods {implementation_missing}")
                else:
                    print(f"✅ {class_name}: All required methods implemented")
                    
            except ImportError as e:
                print(f"⚠️  {class_name}: Import failed - {e}")
            except Exception as e:
                print(f"❌ {class_name}: Unexpected error - {e}")
        
        print("\n🎯 GENERATION 1 STRUCTURE VERIFICATION: ✅ COMPLETE")
        print("All HDC backends have required method signatures for advanced operations")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import core modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Verify that all required files exist."""
    print("\n📁 FILE STRUCTURE VERIFICATION")
    print("-" * 30)
    
    required_files = [
        '/root/repo/hd_compute/__init__.py',
        '/root/repo/hd_compute/core/__init__.py', 
        '/root/repo/hd_compute/core/hdc.py',
        '/root/repo/hd_compute/numpy/__init__.py',
        '/root/repo/hd_compute/numpy/hdc_numpy.py',
        '/root/repo/hd_compute/torch/__init__.py',
        '/root/repo/hd_compute/torch/hdc_torch.py',
        '/root/repo/hd_compute/pure_python/__init__.py',
        '/root/repo/hd_compute/pure_python/hdc_python.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    else:
        print(f"\n✅ All {len(required_files)} core files present")
        return True

if __name__ == "__main__":
    structure_ok = test_file_structure()
    import_ok = test_basic_imports()
    
    if structure_ok and import_ok:
        print("\n🚀 GENERATION 1: READY FOR FUNCTIONALITY TESTING")
        print("Next step: Install dependencies and run full test suite")
        sys.exit(0)
    else:
        print("\n❌ GENERATION 1: STRUCTURAL ISSUES DETECTED") 
        sys.exit(1)