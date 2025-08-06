#!/bin/bash

# HD-Compute-Toolkit Cleanup Script
# Safely removes the application from Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="hd-compute-toolkit"
FORCE_DELETE="${FORCE_DELETE:-false}"
KEEP_DATA="${KEEP_DATA:-true}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Confirmation function
confirm_deletion() {
    local resource_type="$1"
    
    if [[ "$FORCE_DELETE" != "true" ]]; then
        log_warning "You are about to delete $resource_type in namespace: $NAMESPACE"
        log_warning "Environment: $ENVIRONMENT"
        
        if [[ "$ENVIRONMENT" == "production" ]]; then
            log_error "WARNING: This is a PRODUCTION environment!"
        fi
        
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
}

# Check if namespace exists
check_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE does not exist, nothing to clean up"
        exit 0
    fi
}

# Backup data before cleanup (if requested)
backup_data() {
    if [[ "$KEEP_DATA" == "true" ]]; then
        log_info "Backing up data before cleanup..."
        
        local backup_dir="/tmp/hdc-toolkit-backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup database
        log_info "Backing up PostgreSQL database..."
        if kubectl get pod -l app.kubernetes.io/name=postgres -n "$NAMESPACE" &> /dev/null; then
            kubectl exec deployment/postgres -n "$NAMESPACE" -- pg_dump -U hdc_user hdc_experiments > "$backup_dir/database.sql" 2>/dev/null || log_warning "Database backup failed"
        fi
        
        # Backup ConfigMaps and Secrets (but not the secrets themselves)
        log_info "Backing up configurations..."
        kubectl get configmap hdc-toolkit-config -n "$NAMESPACE" -o yaml > "$backup_dir/configmap.yaml" 2>/dev/null || log_warning "ConfigMap backup failed"
        
        # Backup PVC information (not the data itself)
        kubectl get pvc -n "$NAMESPACE" -o yaml > "$backup_dir/pvc.yaml" 2>/dev/null || log_warning "PVC info backup failed"
        
        log_success "Backup completed: $backup_dir"
        echo "Restore with: kubectl apply -f $backup_dir/"
    fi
}

# Scale down deployments gracefully
scale_down_deployments() {
    log_info "Scaling down deployments gracefully..."
    
    # Scale application to 0 replicas
    if kubectl get deployment hdc-toolkit-app -n "$NAMESPACE" &> /dev/null; then
        kubectl scale deployment hdc-toolkit-app --replicas=0 -n "$NAMESPACE"
        kubectl wait --for=condition=complete --timeout=300s job --all -n "$NAMESPACE" 2>/dev/null || true
        log_success "Application scaled down"
    fi
    
    # Wait a moment for graceful shutdown
    sleep 10
}

# Remove monitoring resources
remove_monitoring() {
    log_info "Removing monitoring resources..."
    
    # Remove ServiceMonitor and PrometheusRule if they exist
    kubectl delete servicemonitor hdc-toolkit-metrics -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete prometheusrule hdc-toolkit-alerts -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete configmap hdc-toolkit-dashboard -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Monitoring resources removed"
}

# Remove autoscaling resources
remove_autoscaling() {
    log_info "Removing autoscaling resources..."
    
    kubectl delete hpa hdc-toolkit-hpa -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete vpa hdc-toolkit-vpa -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Autoscaling resources removed"
}

# Remove networking resources
remove_networking() {
    log_info "Removing networking resources..."
    
    kubectl delete ingress --all -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete networkpolicy --all -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Networking resources removed"
}

# Remove application resources
remove_application() {
    log_info "Removing application resources..."
    
    kubectl delete deployment hdc-toolkit-app -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service hdc-toolkit-service hdc-toolkit-headless -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Application resources removed"
}

# Remove database and cache
remove_services() {
    log_info "Removing database and cache services..."
    
    # Remove deployments
    kubectl delete deployment postgres redis -n "$NAMESPACE" --ignore-not-found=true
    
    # Remove services
    kubectl delete service postgres-service redis-service -n "$NAMESPACE" --ignore-not-found=true
    
    # Remove configmaps
    kubectl delete configmap redis-config -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Database and cache services removed"
}

# Remove configurations
remove_configurations() {
    log_info "Removing configurations..."
    
    kubectl delete configmap hdc-toolkit-config -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete secret hdc-toolkit-secrets -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Configurations removed"
}

# Remove persistent volumes (if not keeping data)
remove_storage() {
    if [[ "$KEEP_DATA" != "true" ]]; then
        log_warning "Removing persistent storage (data will be lost)..."
        
        kubectl delete pvc --all -n "$NAMESPACE" --ignore-not-found=true
        
        log_success "Persistent storage removed"
    else
        log_info "Keeping persistent storage (KEEP_DATA=true)"
        log_info "PVCs remain: $(kubectl get pvc -n "$NAMESPACE" --no-headers | wc -l)"
    fi
}

# Remove namespace
remove_namespace() {
    if [[ "$KEEP_DATA" != "true" ]]; then
        log_info "Removing namespace..."
        
        # Force delete if stuck
        kubectl delete namespace "$NAMESPACE" --timeout=300s || {
            log_warning "Namespace deletion timeout, forcing deletion..."
            kubectl patch namespace "$NAMESPACE" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
        }
        
        log_success "Namespace removed"
    else
        log_info "Keeping namespace (contains persistent data)"
    fi
}

# Verify cleanup
verify_cleanup() {
    log_info "Verifying cleanup..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Remaining resources in namespace $NAMESPACE:"
        kubectl get all -n "$NAMESPACE" 2>/dev/null || log_info "No resources remaining"
        
        if [[ "$KEEP_DATA" == "true" ]]; then
            log_info "Persistent storage preserved:"
            kubectl get pvc -n "$NAMESPACE" 2>/dev/null || log_info "No PVCs remaining"
        fi
    else
        log_success "Namespace $NAMESPACE has been completely removed"
    fi
}

# Main cleanup function
main() {
    log_info "Starting HD Compute Toolkit cleanup..."
    log_info "Namespace: $NAMESPACE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Keep Data: $KEEP_DATA"
    log_info "Force Delete: $FORCE_DELETE"
    
    check_namespace
    confirm_deletion "all resources"
    backup_data
    scale_down_deployments
    remove_monitoring
    remove_autoscaling
    remove_networking
    remove_application
    remove_services
    remove_configurations
    remove_storage
    remove_namespace
    verify_cleanup
    
    log_success "Cleanup completed successfully!"
    
    if [[ "$KEEP_DATA" == "true" ]]; then
        log_info "Data has been preserved. To completely remove everything, run:"
        log_info "  $0 --no-keep-data --force"
    fi
}

# Script help
show_help() {
    echo "HD Compute Toolkit Cleanup Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -e, --environment       Environment (development|staging|production) [default: production]"
    echo "  -n, --namespace         Kubernetes namespace [default: hd-compute-toolkit]"
    echo "  -f, --force             Force deletion without confirmation"
    echo "  --no-keep-data          Remove persistent data (DANGEROUS)"
    echo "  --keep-data             Keep persistent data [default: true]"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT             Deployment environment"
    echo "  FORCE_DELETE            Skip confirmation prompts (true/false)"
    echo "  KEEP_DATA               Preserve persistent data (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Safe cleanup (keeps data)"
    echo "  $0 --environment staging --force"
    echo "  $0 --no-keep-data --force                   # Complete removal"
    echo "  KEEP_DATA=false FORCE_DELETE=true $0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_DELETE="true"
            shift
            ;;
        --no-keep-data)
            KEEP_DATA="false"
            shift
            ;;
        --keep-data)
            KEEP_DATA="true"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    log_error "Valid environments: development, staging, production"
    exit 1
fi

# Check kubectl availability
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed"
    exit 1
fi

# Check kubernetes connection
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

# Run main function
main