#!/bin/bash
# HD-Compute-Toolkit Quantum Task Planner - Health Check Script
# Comprehensive health verification for production deployments

set -e

# Configuration
API_PORT="${API_PORT:-8080}"
CLUSTER_PORT="${CLUSTER_PORT:-8081}"
METRICS_PORT="${METRICS_PORT:-9090}"
NODE_ROLE="${NODE_ROLE:-coordinator}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"

# Health check endpoints
API_HEALTH_URL="http://localhost:${API_PORT}/health"
CLUSTER_HEALTH_URL="http://localhost:${CLUSTER_PORT}/cluster/health"
METRICS_URL="http://localhost:${METRICS_PORT}/metrics"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

success() {
    echo -e "${GREEN}✅ $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" >&2
}

error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

# Check if service is responding
check_http_endpoint() {
    local url=$1
    local description=$2
    local timeout=${3:-$HEALTH_TIMEOUT}
    
    if timeout "$timeout" curl -sf "$url" >/dev/null 2>&1; then
        success "$description is responding"
        return 0
    else
        error "$description is not responding"
        return 1
    fi
}

# Check if port is open
check_port() {
    local port=$1
    local description=$2
    local timeout=${3:-5}
    
    if timeout "$timeout" bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
        success "$description port $port is open"
        return 0
    else
        error "$description port $port is not accessible"
        return 1
    fi
}

# Parse JSON response safely
parse_json_value() {
    local json=$1
    local key=$2
    echo "$json" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('$key', 'N/A'))" 2>/dev/null || echo "N/A"
}

# Check quantum coherence
check_quantum_coherence() {
    log "Checking quantum coherence..."
    
    local response
    if response=$(timeout "$HEALTH_TIMEOUT" curl -sf "$API_HEALTH_URL" 2>/dev/null); then
        local coherence
        coherence=$(parse_json_value "$response" "quantum_coherence")
        
        if [ "$coherence" != "N/A" ]; then
            # Use awk for float comparison since bash doesn't support it natively
            if awk "BEGIN {exit !($coherence >= 0.3)}"; then
                success "Quantum coherence: $coherence"
                return 0
            else
                warning "Quantum coherence low: $coherence"
                return 1
            fi
        else
            warning "Quantum coherence data not available"
            return 1
        fi
    else
        error "Cannot retrieve quantum coherence data"
        return 1
    fi
}

# Check memory usage
check_memory_usage() {
    log "Checking memory usage..."
    
    if command -v free >/dev/null 2>&1; then
        local memory_usage
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100.0}')
        
        if awk "BEGIN {exit !($memory_usage < 90)}"; then
            success "Memory usage: ${memory_usage}%"
            return 0
        else
            warning "High memory usage: ${memory_usage}%"
            return 1
        fi
    else
        warning "Cannot check memory usage (free command not available)"
        return 1
    fi
}

# Check disk space
check_disk_space() {
    log "Checking disk space..."
    
    local disk_usage
    disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -lt 85 ]; then
        success "Disk usage: ${disk_usage}%"
        return 0
    else
        warning "High disk usage: ${disk_usage}%"
        return 1
    fi
}

# Check process status
check_process_status() {
    log "Checking process status..."
    
    local python_processes
    python_processes=$(pgrep -f "python.*quantum" | wc -l)
    
    if [ "$python_processes" -gt 0 ]; then
        success "Quantum planner processes running: $python_processes"
        return 0
    else
        error "No quantum planner processes found"
        return 1
    fi
}

# Check log files
check_log_files() {
    log "Checking log files..."
    
    local log_file="/app/logs/quantum-planner.log"
    local error_count=0
    
    if [ -f "$log_file" ]; then
        # Check for recent errors (last 100 lines)
        if command -v tail >/dev/null 2>&1; then
            error_count=$(tail -100 "$log_file" | grep -i "error\|exception\|critical" | wc -l)
        fi
        
        if [ "$error_count" -eq 0 ]; then
            success "No recent errors in logs"
            return 0
        else
            warning "Found $error_count recent errors in logs"
            return 1
        fi
    else
        warning "Log file not found: $log_file"
        return 1
    fi
}

# Check cluster connectivity (for distributed nodes)
check_cluster_connectivity() {
    if [ "$NODE_ROLE" != "coordinator" ] && [ -n "$COORDINATOR_ENDPOINT" ]; then
        log "Checking cluster connectivity..."
        
        local coordinator_host
        local coordinator_port
        coordinator_host=$(echo "$COORDINATOR_ENDPOINT" | cut -d':' -f1)
        coordinator_port=$(echo "$COORDINATOR_ENDPOINT" | cut -d':' -f2)
        
        if timeout 5 bash -c "</dev/tcp/$coordinator_host/$coordinator_port" 2>/dev/null; then
            success "Coordinator connectivity OK"
            return 0
        else
            error "Cannot connect to coordinator: $COORDINATOR_ENDPOINT"
            return 1
        fi
    else
        success "Cluster connectivity check skipped (coordinator node or no endpoint)"
        return 0
    fi
}

# Check metrics availability
check_metrics() {
    log "Checking metrics availability..."
    
    if timeout "$HEALTH_TIMEOUT" curl -sf "$METRICS_URL" >/dev/null 2>&1; then
        local metrics_response
        if metrics_response=$(timeout "$HEALTH_TIMEOUT" curl -sf "$METRICS_URL" 2>/dev/null); then
            local metric_count
            metric_count=$(echo "$metrics_response" | grep -c "^[a-zA-Z]" || echo "0")
            
            if [ "$metric_count" -gt 0 ]; then
                success "Metrics endpoint responding with $metric_count metrics"
                return 0
            else
                warning "Metrics endpoint responding but no metrics found"
                return 1
            fi
        else
            error "Metrics endpoint not responding properly"
            return 1
        fi
    else
        error "Metrics endpoint not accessible"
        return 1
    fi
}

# Main health check function
perform_health_check() {
    log "Starting comprehensive health check for node role: $NODE_ROLE"
    log "Checking endpoints: API($API_PORT), Cluster($CLUSTER_PORT), Metrics($METRICS_PORT)"
    
    local checks_passed=0
    local total_checks=0
    local critical_failed=0
    
    # Core service checks (critical)
    log "=== Core Service Checks ==="
    
    if check_port "$API_PORT" "API"; then
        ((checks_passed++))
    else
        ((critical_failed++))
    fi
    ((total_checks++))
    
    if check_http_endpoint "$API_HEALTH_URL" "API Health"; then
        ((checks_passed++))
    else
        ((critical_failed++))
    fi
    ((total_checks++))
    
    if check_process_status; then
        ((checks_passed++))
    else
        ((critical_failed++))
    fi
    ((total_checks++))
    
    # Role-specific checks
    log "=== Role-Specific Checks ==="
    
    case "$NODE_ROLE" in
        coordinator)
            if check_port "$CLUSTER_PORT" "Cluster"; then
                ((checks_passed++))
            fi
            ((total_checks++))
            
            if check_http_endpoint "$CLUSTER_HEALTH_URL" "Cluster Health"; then
                ((checks_passed++))
            fi
            ((total_checks++))
            ;;
        worker|planner|executor)
            if check_cluster_connectivity; then
                ((checks_passed++))
            else
                ((critical_failed++))
            fi
            ((total_checks++))
            ;;
    esac
    
    # Performance and resource checks (non-critical)
    log "=== Performance and Resource Checks ==="
    
    if check_quantum_coherence; then
        ((checks_passed++))
    fi
    ((total_checks++))
    
    if check_memory_usage; then
        ((checks_passed++))
    fi
    ((total_checks++))
    
    if check_disk_space; then
        ((checks_passed++))
    fi
    ((total_checks++))
    
    if check_metrics; then
        ((checks_passed++))
    fi
    ((total_checks++))
    
    if check_log_files; then
        ((checks_passed++))
    fi
    ((total_checks++))
    
    # Summary
    log "=== Health Check Summary ==="
    log "Checks passed: $checks_passed/$total_checks"
    log "Critical failures: $critical_failed"
    
    local success_rate
    success_rate=$(awk "BEGIN {printf \"%.1f\", ($checks_passed/$total_checks)*100}")
    
    if [ "$critical_failed" -eq 0 ] && awk "BEGIN {exit !($success_rate >= 70)}"; then
        success "Health check PASSED (${success_rate}% success rate)"
        echo "healthy"
        exit 0
    elif [ "$critical_failed" -eq 0 ]; then
        warning "Health check DEGRADED (${success_rate}% success rate)"
        echo "degraded"
        exit 0
    else
        error "Health check FAILED ($critical_failed critical failures, ${success_rate}% success rate)"
        echo "unhealthy"
        exit 1
    fi
}

# Quick health check (for rapid probes)
quick_health_check() {
    if check_port "$API_PORT" "API" 2 && \
       timeout 2 curl -sf "$API_HEALTH_URL" >/dev/null 2>&1; then
        echo "healthy"
        exit 0
    else
        echo "unhealthy"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-full}" in
    quick|fast)
        quick_health_check
        ;;
    full|comprehensive|"")
        perform_health_check
        ;;
    *)
        error "Unknown health check type: $1"
        error "Usage: $0 [quick|full]"
        exit 1
        ;;
esac