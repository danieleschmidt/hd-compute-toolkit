# HD-Compute-Toolkit Performance Report

**Generated**: 2025-08-14 14:49:58

## Executive Summary
- **Total Operations Tested**: 8
- **Fastest Operation**: random_hv_generation (2.75ms avg)
- **Slowest Operation**: sequence_encoding (34.17ms avg)

## Performance Profiles

### random_hv_generation
- **Average Time**: 2.75ms
- **Throughput**: 363 ops/sec
- **P95 Latency**: 5.24ms
- **P99 Latency**: 6.00ms
- **Scalability Score**: 0.98

### bundle_operation
- **Average Time**: 17.36ms
- **Throughput**: 58 ops/sec
- **P95 Latency**: 33.13ms
- **P99 Latency**: 35.56ms
- **Scalability Score**: 0.98

### bind_operation
- **Average Time**: 5.76ms
- **Throughput**: 174 ops/sec
- **P95 Latency**: 10.85ms
- **P99 Latency**: 11.49ms
- **Scalability Score**: 0.97

### cosine_similarity
- **Average Time**: 6.45ms
- **Throughput**: 155 ops/sec
- **P95 Latency**: 12.15ms
- **P99 Latency**: 17.03ms
- **Scalability Score**: 0.98

### hamming_distance
- **Average Time**: 6.03ms
- **Throughput**: 166 ops/sec
- **P95 Latency**: 11.67ms
- **P99 Latency**: 14.62ms
- **Scalability Score**: 0.98

### fractional_bind
- **Average Time**: 6.85ms
- **Throughput**: 146 ops/sec
- **P95 Latency**: 12.99ms
- **P99 Latency**: 13.71ms
- **Scalability Score**: 0.97

### sequence_encoding
- **Average Time**: 34.17ms
- **Throughput**: 29 ops/sec
- **P95 Latency**: 64.46ms
- **P99 Latency**: 67.34ms
- **Scalability Score**: 0.97

### memory_operations
- **Average Time**: 3.05ms
- **Throughput**: 328 ops/sec
- **P95 Latency**: 5.71ms
- **P99 Latency**: 5.77ms
- **Scalability Score**: 0.98

## Performance Recommendations

- **bundle_operation**: Consider optimization (avg: 17.4ms)
- **sequence_encoding**: Consider optimization (avg: 34.2ms)