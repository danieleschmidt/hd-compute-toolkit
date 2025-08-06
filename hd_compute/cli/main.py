"""Main CLI interface for HD-Compute-Toolkit."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..utils.config import get_config
from ..utils.logging_config import setup_logging
from ..utils.environment import EnvironmentManager
from ..core.hdc import HDCompute
from ..pure_python.hdc_python import HDComputePython


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='hdc',
        description='HD-Compute-Toolkit - Hyperdimensional Computing Library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], 
                       default='auto', help='Computing device')
    parser.add_argument('--dimension', type=int, default=10000, 
                       help='Default hypervector dimension')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Environment command
    env_parser = subparsers.add_parser('env', help='Environment management')
    env_parser.add_argument('--check', action='store_true', 
                           help='Check environment setup')
    env_parser.add_argument('--report', action='store_true', 
                           help='Generate environment report')
    env_parser.add_argument('--setup', action='store_true', 
                           help='Setup environment')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--operations', nargs='+', 
                             choices=['random', 'bundle', 'bind', 'similarity'],
                             default=['random', 'bundle', 'bind', 'similarity'],
                             help='Operations to benchmark')
    bench_parser.add_argument('--dimensions', nargs='+', type=int,
                             default=[1000, 5000, 10000],
                             help='Dimensions to test')
    bench_parser.add_argument('--iterations', type=int, default=100,
                             help='Number of iterations per test')
    bench_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--example', choices=['basic', 'memory', 'sequence'],
                            default='basic', help='Demo example to run')
    
    return parser


def handle_env_command(args) -> int:
    """Handle environment management command."""
    env_manager = EnvironmentManager()
    
    if args.check or args.report:
        validation_results = env_manager.validate_environment()
        
        if args.report:
            report = env_manager.generate_setup_report()
            print(report)
        else:
            status = validation_results['overall_status']
            print(f"Environment status: {status.upper()}")
            
            if validation_results['errors']:
                print("\nErrors found:")
                for error in validation_results['errors']:
                    print(f"  ✗ {error}")
            
            if validation_results['warnings']:
                print("\nWarnings:")
                for warning in validation_results['warnings']:
                    print(f"  ⚠ {warning}")
        
        return 0 if validation_results['overall_status'] == 'pass' else 1
    
    elif args.setup:
        success = env_manager.setup_environment()
        if success:
            print("✓ Environment setup completed successfully")
            return 0
        else:
            print("✗ Environment setup failed")
            return 1
    
    else:
        print("Please specify --check, --report, or --setup")
        return 1


def handle_benchmark_command(args) -> int:
    """Handle benchmark command."""
    import time
    
    try:
        hdc = HDComputePython(dim=args.dimensions[0])
        print(f"Running benchmarks with HDComputePython backend")
        
        results = []
        
        for dim in args.dimensions:
            print(f"\nBenchmarking dimension {dim}:")
            hdc_dim = HDComputePython(dim=dim)
            
            for operation in args.operations:
                print(f"  Testing {operation}...", end=" ")
                
                if operation == 'random':
                    start_time = time.time()
                    for _ in range(args.iterations):
                        hv = hdc_dim.random_hv()
                    end_time = time.time()
                
                elif operation == 'bundle':
                    hv1 = hdc_dim.random_hv()
                    hv2 = hdc_dim.random_hv()
                    start_time = time.time()
                    for _ in range(args.iterations):
                        bundled = hdc_dim.bundle([hv1, hv2])
                    end_time = time.time()
                
                elif operation == 'bind':
                    hv1 = hdc_dim.random_hv()
                    hv2 = hdc_dim.random_hv()
                    start_time = time.time()
                    for _ in range(args.iterations):
                        bound = hdc_dim.bind(hv1, hv2)
                    end_time = time.time()
                
                elif operation == 'similarity':
                    hv1 = hdc_dim.random_hv()
                    hv2 = hdc_dim.random_hv()
                    start_time = time.time()
                    for _ in range(args.iterations):
                        sim = hdc_dim.cosine_similarity(hv1, hv2)
                    end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = (total_time / args.iterations) * 1000  # Convert to ms
                
                results.append({
                    'operation': operation,
                    'dimension': dim,
                    'iterations': args.iterations,
                    'total_time_s': total_time,
                    'avg_time_ms': avg_time,
                    'backend': 'python'
                })
                
                print(f"{avg_time:.3f}ms avg")
        
        # Print summary
        print("\nBenchmark Results:")
        print(f"{'Operation':<10} {'Dimension':<10} {'Avg Time (ms)':<15}")
        print("-" * 40)
        for result in results:
            print(f"{result['operation']:<10} {result['dimension']:<10} {result['avg_time_ms']:<15.3f}")
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


def handle_demo_command(args) -> int:
    """Handle demonstration command."""
    try:
        if args.example == 'basic':
            print("=== Basic HDC Operations Demo ===")
            hdc = HDComputePython(dim=1000)
            
            print("1. Generating random hypervectors...")
            hv1 = hdc.random_hv()
            hv2 = hdc.random_hv()
            print(f"   Generated HVs with dimension {len(hv1.data)}")
            
            print("2. Bundle operation...")
            bundled = hdc.bundle([hv1, hv2])
            print(f"   Bundled HV dimension: {len(bundled.data)}")
            
            print("3. Bind operation...")
            bound = hdc.bind(hv1, hv2)
            print(f"   Bound HV dimension: {len(bound.data)}")
            
            print("4. Similarity calculation...")
            sim1 = hdc.cosine_similarity(hv1, hv2)
            sim2 = hdc.cosine_similarity(hv1, bundled)
            sim3 = hdc.cosine_similarity(hv1, bound)
            
            print(f"   Similarity(hv1, hv2): {sim1:.3f}")
            print(f"   Similarity(hv1, bundled): {sim2:.3f}")
            print(f"   Similarity(hv1, bound): {sim3:.3f}")
            
        elif args.example == 'memory':
            print("=== Memory Systems Demo ===")
            from ..memory import ItemMemory, AssociativeMemory
            
            hdc = HDComputePython(dim=500)
            
            print("1. Creating item memory...")
            item_memory = ItemMemory(hdc, ['apple', 'banana', 'cherry'])
            print(f"   Stored {item_memory.size()} items")
            
            print("2. Creating associative memory...")
            assoc_memory = AssociativeMemory(hdc)
            
            # Store some patterns
            for fruit in ['apple', 'banana', 'cherry']:
                fruit_hv = item_memory.get_hv(fruit)
                assoc_memory.store(fruit_hv, f"fruit_{fruit}")
            
            print(f"   Stored {assoc_memory.size()} patterns")
            
            print("3. Testing memory retrieval...")
            apple_hv = item_memory.get_hv('apple')
            recalls = assoc_memory.recall(apple_hv, k=2)
            
            print("   Retrieved patterns:")
            for label, similarity in recalls:
                print(f"     {label}: {similarity:.3f}")
            
        elif args.example == 'sequence':
            print("=== Sequence Encoding Demo ===")
            from ..memory import ItemMemory
            
            hdc = HDComputePython(dim=800)
            
            print("1. Creating vocabulary...")
            vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
            item_memory = ItemMemory(hdc, vocab)
            
            print("2. Encoding sequences...")
            seq1 = ['the', 'quick', 'brown', 'fox']
            seq2 = ['the', 'lazy', 'dog']
            
            encoded_seq1 = item_memory.encode_sequence(seq1)
            encoded_seq2 = item_memory.encode_sequence(seq2)
            
            print(f"   Encoded sequence 1: {seq1}")
            print(f"   Encoded sequence 2: {seq2}")
            
            print("3. Comparing sequences...")
            similarity = hdc.cosine_similarity(encoded_seq1, encoded_seq2)
            print(f"   Sequence similarity: {similarity:.3f}")
        
        print("\n✓ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    
    # Handle case where no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Handle commands
    try:
        if args.command == 'env':
            return handle_env_command(args)
        elif args.command == 'benchmark':
            return handle_benchmark_command(args)
        elif args.command == 'demo':
            return handle_demo_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())