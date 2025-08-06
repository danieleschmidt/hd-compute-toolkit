# HD-Compute-Toolkit Performance Report

**Generated**: 2025-08-06 12:38:07

## Executive Summary
- **Total Operations Tested**: 8
- **Fastest Operation**: random_hv_generation (2.92ms avg)
- **Slowest Operation**: sequence_encoding (34.09ms avg)

## Performance Profiles

### random_hv_generation
- **Average Time**: 2.92ms
- **Throughput**: 342 ops/sec
- **P95 Latency**: 5.80ms
- **P99 Latency**: 10.78ms
- **Scalability Score**: 0.98

### bundle_operation
- **Average Time**: 17.89ms
- **Throughput**: 56 ops/sec
- **P95 Latency**: 33.89ms
- **P99 Latency**: 43.64ms
- **Scalability Score**: 0.97

### bind_operation
- **Average Time**: 5.68ms
- **Throughput**: 176 ops/sec
- **P95 Latency**: 10.96ms
- **P99 Latency**: 11.24ms
- **Scalability Score**: 0.97

### cosine_similarity
- **Average Time**: 6.23ms
- **Throughput**: 160 ops/sec
- **P95 Latency**: 11.87ms
- **P99 Latency**: 12.25ms
- **Scalability Score**: 0.97

### hamming_distance
- **Average Time**: 5.90ms
- **Throughput**: 169 ops/sec
- **P95 Latency**: 11.61ms
- **P99 Latency**: 13.06ms
- **Scalability Score**: 0.98

### fractional_bind
- **Average Time**: 6.89ms
- **Throughput**: 145 ops/sec
- **P95 Latency**: 13.00ms
- **P99 Latency**: 15.58ms
- **Scalability Score**: 0.97

### sequence_encoding
- **Average Time**: 34.09ms
- **Throughput**: 29 ops/sec
- **P95 Latency**: 65.52ms
- **P99 Latency**: 69.66ms
- **Scalability Score**: 0.98

### memory_operations
- **Average Time**: 3.06ms
- **Throughput**: 327 ops/sec
- **P95 Latency**: 5.90ms
- **P99 Latency**: 6.24ms
- **Scalability Score**: 0.98

## Performance Recommendations

- **bundle_operation**: Consider optimization (avg: 17.9ms)
- **sequence_encoding**: Consider optimization (avg: 34.1ms)