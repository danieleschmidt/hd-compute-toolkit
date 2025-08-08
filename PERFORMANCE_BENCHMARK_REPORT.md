# HD-Compute-Toolkit Performance Report

**Generated**: 2025-08-08 16:34:23

## Executive Summary
- **Total Operations Tested**: 8
- **Fastest Operation**: random_hv_generation (2.76ms avg)
- **Slowest Operation**: sequence_encoding (35.07ms avg)

## Performance Profiles

### random_hv_generation
- **Average Time**: 2.76ms
- **Throughput**: 362 ops/sec
- **P95 Latency**: 5.26ms
- **P99 Latency**: 5.35ms
- **Scalability Score**: 0.97

### bundle_operation
- **Average Time**: 18.12ms
- **Throughput**: 55 ops/sec
- **P95 Latency**: 33.60ms
- **P99 Latency**: 54.71ms
- **Scalability Score**: 0.98

### bind_operation
- **Average Time**: 5.87ms
- **Throughput**: 170 ops/sec
- **P95 Latency**: 11.02ms
- **P99 Latency**: 12.24ms
- **Scalability Score**: 0.97

### cosine_similarity
- **Average Time**: 6.52ms
- **Throughput**: 153 ops/sec
- **P95 Latency**: 12.31ms
- **P99 Latency**: 12.77ms
- **Scalability Score**: 0.97

### hamming_distance
- **Average Time**: 6.13ms
- **Throughput**: 163 ops/sec
- **P95 Latency**: 11.74ms
- **P99 Latency**: 13.75ms
- **Scalability Score**: 0.97

### fractional_bind
- **Average Time**: 7.02ms
- **Throughput**: 142 ops/sec
- **P95 Latency**: 13.19ms
- **P99 Latency**: 16.57ms
- **Scalability Score**: 0.97

### sequence_encoding
- **Average Time**: 35.07ms
- **Throughput**: 29 ops/sec
- **P95 Latency**: 68.07ms
- **P99 Latency**: 69.92ms
- **Scalability Score**: 0.97

### memory_operations
- **Average Time**: 3.15ms
- **Throughput**: 317 ops/sec
- **P95 Latency**: 6.13ms
- **P99 Latency**: 7.58ms
- **Scalability Score**: 0.98

## Performance Recommendations

- **bundle_operation**: Consider optimization (avg: 18.1ms)
- **sequence_encoding**: Consider optimization (avg: 35.1ms)