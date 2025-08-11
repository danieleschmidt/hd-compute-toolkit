# HD-Compute-Toolkit Performance Report

**Generated**: 2025-08-11 08:36:28

## Executive Summary
- **Total Operations Tested**: 8
- **Fastest Operation**: random_hv_generation (2.74ms avg)
- **Slowest Operation**: sequence_encoding (34.31ms avg)

## Performance Profiles

### random_hv_generation
- **Average Time**: 2.74ms
- **Throughput**: 365 ops/sec
- **P95 Latency**: 5.23ms
- **P99 Latency**: 5.53ms
- **Scalability Score**: 0.97

### bundle_operation
- **Average Time**: 17.55ms
- **Throughput**: 57 ops/sec
- **P95 Latency**: 33.03ms
- **P99 Latency**: 33.80ms
- **Scalability Score**: 0.97

### bind_operation
- **Average Time**: 5.73ms
- **Throughput**: 174 ops/sec
- **P95 Latency**: 10.93ms
- **P99 Latency**: 11.25ms
- **Scalability Score**: 0.98

### cosine_similarity
- **Average Time**: 6.42ms
- **Throughput**: 156 ops/sec
- **P95 Latency**: 12.29ms
- **P99 Latency**: 12.49ms
- **Scalability Score**: 0.97

### hamming_distance
- **Average Time**: 5.97ms
- **Throughput**: 168 ops/sec
- **P95 Latency**: 11.29ms
- **P99 Latency**: 11.61ms
- **Scalability Score**: 0.97

### fractional_bind
- **Average Time**: 6.79ms
- **Throughput**: 147 ops/sec
- **P95 Latency**: 12.82ms
- **P99 Latency**: 13.51ms
- **Scalability Score**: 0.97

### sequence_encoding
- **Average Time**: 34.31ms
- **Throughput**: 29 ops/sec
- **P95 Latency**: 65.65ms
- **P99 Latency**: 67.54ms
- **Scalability Score**: 0.97

### memory_operations
- **Average Time**: 3.08ms
- **Throughput**: 324 ops/sec
- **P95 Latency**: 5.87ms
- **P99 Latency**: 6.03ms
- **Scalability Score**: 0.97

## Performance Recommendations

- **bundle_operation**: Consider optimization (avg: 17.6ms)
- **sequence_encoding**: Consider optimization (avg: 34.3ms)