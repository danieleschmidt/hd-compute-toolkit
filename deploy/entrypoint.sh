#!/bin/bash
set -e

# HD-Compute-Toolkit Quantum Task Planner - Container Entrypoint
# Production-ready startup script with comprehensive initialization

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Default configuration
DEFAULT_NODE_ROLE="coordinator"
DEFAULT_PORT="8080"
DEFAULT_CLUSTER_PORT="8081"
DEFAULT_METRICS_PORT="9090"
DEFAULT_QUANTUM_DIMENSION="10000"
DEFAULT_LOG_LEVEL="INFO"

# Set configuration from environment variables or defaults
NODE_ROLE="${NODE_ROLE:-$DEFAULT_NODE_ROLE}"
API_PORT="${API_PORT:-$DEFAULT_PORT}"
CLUSTER_PORT="${CLUSTER_PORT:-$DEFAULT_CLUSTER_PORT}"
METRICS_PORT="${METRICS_PORT:-$DEFAULT_METRICS_PORT}"
QUANTUM_DIMENSION="${QUANTUM_DIMENSION:-$DEFAULT_QUANTUM_DIMENSION}"
LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
WORKER_THREADS="${WORKER_THREADS:-4}"
CACHE_SIZE_MB="${CACHE_SIZE_MB:-1024}"

log "Starting HD-Compute-Toolkit Quantum Task Planner"
log "Configuration:"
log "  Node Role: $NODE_ROLE"
log "  API Port: $API_PORT"
log "  Cluster Port: $CLUSTER_PORT"
log "  Metrics Port: $METRICS_PORT"
log "  Quantum Dimension: $QUANTUM_DIMENSION"
log "  Log Level: $LOG_LEVEL"
log "  Worker Threads: $WORKER_THREADS"
log "  Cache Size: ${CACHE_SIZE_MB}MB"

# Validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check required environment variables
    if [[ -z "$NODE_ROLE" ]]; then
        log_error "NODE_ROLE is not set"
        exit 1
    fi
    
    # Validate node role
    case "$NODE_ROLE" in
        coordinator|worker|planner|executor|cache|monitor)
            log_success "Valid node role: $NODE_ROLE"
            ;;
        *)
            log_error "Invalid node role: $NODE_ROLE"
            log "Valid roles: coordinator, worker, planner, executor, cache, monitor"
            exit 1
            ;;
    esac
    
    # Validate quantum dimension
    if ! [[ "$QUANTUM_DIMENSION" =~ ^[0-9]+$ ]] || [ "$QUANTUM_DIMENSION" -lt 100 ]; then
        log_error "Invalid quantum dimension: $QUANTUM_DIMENSION (must be >= 100)"
        exit 1
    fi
    
    # Validate ports
    for port in $API_PORT $CLUSTER_PORT $METRICS_PORT; do
        if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
            log_error "Invalid port: $port (must be 1024-65535)"
            exit 1
        fi
    done
    
    log_success "Configuration validation passed"
}

# System checks
system_checks() {
    log "Performing system checks..."
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    log "Python version: $python_version"
    
    # Check available memory
    available_memory=$(free -m | grep '^Mem:' | awk '{print $7}')
    if [ "$available_memory" -lt 512 ]; then
        log_warning "Low available memory: ${available_memory}MB (recommend > 512MB)"
    else
        log_success "Available memory: ${available_memory}MB"
    fi
    
    # Check disk space
    available_disk=$(df /app | tail -1 | awk '{print $4}')
    if [ "$available_disk" -lt 1048576 ]; then  # 1GB in KB
        log_warning "Low disk space: $(($available_disk/1024))MB (recommend > 1GB)"
    else
        log_success "Available disk space: $(($available_disk/1024/1024))GB"
    fi
    
    # Check CPU cores
    cpu_cores=$(nproc)
    log "CPU cores available: $cpu_cores"
    
    log_success "System checks completed"
}

# Initialize directories and permissions
init_directories() {
    log "Initializing directories..."
    
    directories=("/app/logs" "/app/cache" "/app/data" "/app/tmp")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
        
        # Ensure correct ownership (may not work in some container environments)
        if [ -w "$dir" ]; then
            log_success "Directory accessible: $dir"
        else
            log_warning "Directory not writable: $dir"
        fi
    done
    
    log_success "Directory initialization completed"
}

# Wait for dependencies
wait_for_dependencies() {
    log "Checking dependencies..."
    
    # If this is a worker node, wait for coordinator
    if [ "$NODE_ROLE" = "worker" ] && [ -n "$COORDINATOR_ENDPOINT" ]; then
        log "Waiting for coordinator at $COORDINATOR_ENDPOINT..."
        
        coordinator_host=$(echo "$COORDINATOR_ENDPOINT" | cut -d':' -f1)
        coordinator_port=$(echo "$COORDINATOR_ENDPOINT" | cut -d':' -f2)
        
        max_attempts=30
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if timeout 5 bash -c "</dev/tcp/$coordinator_host/$coordinator_port" 2>/dev/null; then
                log_success "Coordinator is reachable"
                break
            fi
            
            log "Attempt $attempt/$max_attempts: Coordinator not reachable, waiting 10s..."
            sleep 10
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            log_error "Failed to connect to coordinator after $max_attempts attempts"
            exit 1
        fi
    fi
    
    log_success "Dependency checks completed"
}

# Initialize quantum components
init_quantum_components() {
    log "Initializing quantum components..."
    
    # Set quantum-specific environment variables
    export QUANTUM_DIMENSION
    export QUANTUM_COHERENCE_THRESHOLD="${QUANTUM_COHERENCE_THRESHOLD:-0.7}"
    export MAX_SUPERPOSITION_STATES="${MAX_SUPERPOSITION_STATES:-100}"
    export ENABLE_QUANTUM_ENCRYPTION="${ENABLE_QUANTUM_ENCRYPTION:-true}"
    
    # Initialize random seed for reproducibility in development
    if [ "$ENVIRONMENT" = "development" ]; then
        export PYTHONHASHSEED=42
        log "Development mode: Using fixed random seed for reproducibility"
    fi
    
    # GPU detection
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L | wc -l)
        log "NVIDIA GPUs detected: $gpu_count"
        export ENABLE_GPU_ACCELERATION="true"
        export GPU_COUNT="$gpu_count"
    else
        log "No GPU acceleration available (CPU only)"
        export ENABLE_GPU_ACCELERATION="false"
        export GPU_COUNT="0"
    fi
    
    log_success "Quantum components initialized"
}

# Setup monitoring and metrics
setup_monitoring() {
    log "Setting up monitoring and metrics..."
    
    # Export metrics port for Prometheus
    export METRICS_PORT
    export PROMETHEUS_MULTIPROC_DIR="/app/tmp/prometheus"
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
    
    # Setup logging configuration
    export LOG_LEVEL
    export LOG_FORMAT="${LOG_FORMAT:-json}"
    export LOG_FILE="/app/logs/quantum-planner.log"
    
    # Ensure log file exists
    touch "$LOG_FILE"
    
    log_success "Monitoring setup completed"
}

# Start the appropriate service based on node role
start_service() {
    log "Starting $NODE_ROLE service..."
    
    case "$NODE_ROLE" in
        coordinator)
            log "Starting Quantum Task Planner Coordinator"
            exec python3 -m hd_compute.applications.task_planning_server \
                --role coordinator \
                --port "$API_PORT" \
                --cluster-port "$CLUSTER_PORT" \
                --metrics-port "$METRICS_PORT" \
                --dimension "$QUANTUM_DIMENSION" \
                --workers "$WORKER_THREADS" \
                --cache-size "$CACHE_SIZE_MB" \
                --log-level "$LOG_LEVEL"
            ;;
            
        worker)
            log "Starting Quantum Task Planner Worker"
            exec python3 -m hd_compute.applications.task_planning_server \
                --role worker \
                --port "$API_PORT" \
                --coordinator "$COORDINATOR_ENDPOINT" \
                --metrics-port "$METRICS_PORT" \
                --dimension "$QUANTUM_DIMENSION" \
                --workers "$WORKER_THREADS" \
                --cache-size "$CACHE_SIZE_MB" \
                --log-level "$LOG_LEVEL"
            ;;
            
        planner)
            log "Starting Specialized Planning Node"
            exec python3 -m hd_compute.distributed.quantum_task_distribution \
                --role planner \
                --port "$API_PORT" \
                --coordinator "$COORDINATOR_ENDPOINT" \
                --metrics-port "$METRICS_PORT" \
                --dimension "$QUANTUM_DIMENSION" \
                --log-level "$LOG_LEVEL"
            ;;
            
        executor)
            log "Starting Specialized Execution Node"
            exec python3 -m hd_compute.distributed.quantum_task_distribution \
                --role executor \
                --port "$API_PORT" \
                --coordinator "$COORDINATOR_ENDPOINT" \
                --metrics-port "$METRICS_PORT" \
                --dimension "$QUANTUM_DIMENSION" \
                --log-level "$LOG_LEVEL"
            ;;
            
        cache)
            log "Starting Cache Node"
            exec python3 -m hd_compute.cache.distributed_cache_server \
                --port "$API_PORT" \
                --coordinator "$COORDINATOR_ENDPOINT" \
                --metrics-port "$METRICS_PORT" \
                --cache-size "$CACHE_SIZE_MB" \
                --log-level "$LOG_LEVEL"
            ;;
            
        monitor)
            log "Starting Monitoring Node"
            exec python3 -m hd_compute.monitoring.monitoring_server \
                --port "$API_PORT" \
                --coordinator "$COORDINATOR_ENDPOINT" \
                --metrics-port "$METRICS_PORT" \
                --log-level "$LOG_LEVEL"
            ;;
            
        *)
            log_error "Unknown node role: $NODE_ROLE"
            exit 1
            ;;
    esac
}

# Graceful shutdown handler
shutdown_handler() {
    log "Received shutdown signal, shutting down gracefully..."
    
    # Send shutdown signal to child processes
    if [ -n "$service_pid" ]; then
        log "Stopping service (PID: $service_pid)..."
        kill -TERM "$service_pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$service_pid" 2>/dev/null; then
                log_success "Service stopped gracefully"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$service_pid" 2>/dev/null; then
            log_warning "Forcing service shutdown..."
            kill -KILL "$service_pid" 2>/dev/null || true
        fi
    fi
    
    # Cleanup
    log "Performing cleanup..."
    rm -rf /app/tmp/prometheus/* 2>/dev/null || true
    
    log_success "Shutdown completed"
    exit 0
}

# Register shutdown handler
trap shutdown_handler SIGTERM SIGINT

# Main execution flow
main() {
    log "=" "50"
    log "HD-COMPUTE-TOOLKIT QUANTUM TASK PLANNER"
    log "Autonomous SDLC Implementation - Production Ready"
    log "=" "50"
    
    # Initialization steps
    validate_config
    system_checks
    init_directories
    wait_for_dependencies
    init_quantum_components
    setup_monitoring
    
    log_success "Initialization completed successfully"
    log "=" "50"
    
    # Start the service
    start_service &
    service_pid=$!
    
    log "Service started with PID: $service_pid"
    log "Quantum Task Planner is running..."
    
    # Wait for the service to complete
    wait "$service_pid"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Service exited successfully"
    else
        log_error "Service exited with code: $exit_code"
    fi
    
    exit $exit_code
}

# Handle direct command execution
if [ $# -gt 0 ]; then
    case "$1" in
        bash|sh)
            log "Starting interactive shell..."
            exec "$@"
            ;;
        python|python3)
            log "Starting Python interpreter..."
            exec "$@"
            ;;
        coordinator|worker|planner|executor|cache|monitor)
            NODE_ROLE="$1"
            shift
            main "$@"
            ;;
        *)
            log "Executing custom command: $*"
            exec "$@"
            ;;
    esac
else
    # Default execution
    main
fi