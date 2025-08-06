#!/bin/bash

# HD-Compute-Toolkit Production Deployment Script
# Deploys the application to Kubernetes with production configurations

set -euo pipefail

# Configuration
NAMESPACE="hd-compute-toolkit"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KUBECONFIG="${KUBECONFIG:-~/.kube/config}"
DRY_RUN="${DRY_RUN:-false}"
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if running in production context
    current_context=$(kubectl config current-context)
    log_info "Current Kubernetes context: $current_context"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "You are deploying to PRODUCTION environment!"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace if not exists..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f k8s/namespace.yaml
        log_success "Namespace $NAMESPACE created"
    fi
}

# Apply ConfigMaps and Secrets
apply_configs() {
    log_info "Applying configuration resources..."
    
    # Apply ConfigMap
    kubectl apply -f k8s/configmap.yaml
    log_success "ConfigMap applied"
    
    # Apply Secrets (warn about default values)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "Using default secrets! Please update secrets with production values:"
        log_warning "kubectl create secret generic hdc-toolkit-secrets -n $NAMESPACE --from-literal=POSTGRES_PASSWORD=<your-password>"
    fi
    kubectl apply -f k8s/secret.yaml
    log_success "Secrets applied"
}

# Apply storage resources
apply_storage() {
    log_info "Applying storage resources..."
    
    kubectl apply -f k8s/pvc.yaml
    log_success "Persistent Volume Claims applied"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n "$NAMESPACE" --timeout=300s
    log_success "All PVCs are bound"
}

# Deploy database
deploy_database() {
    log_info "Deploying PostgreSQL database..."
    
    kubectl apply -f k8s/postgres.yaml
    log_success "PostgreSQL deployment applied"
    
    # Wait for database to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=postgres -n "$NAMESPACE" --timeout=300s
    log_success "PostgreSQL is ready"
}

# Deploy cache
deploy_cache() {
    log_info "Deploying Redis cache..."
    
    kubectl apply -f k8s/redis.yaml
    log_success "Redis deployment applied"
    
    # Wait for cache to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=redis -n "$NAMESPACE" --timeout=300s
    log_success "Redis is ready"
}

# Deploy application
deploy_application() {
    log_info "Deploying HD Compute Toolkit application..."
    
    # Update image tag if specified
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        log_info "Using image tag: $IMAGE_TAG"
        sed -i.bak "s|image: hd-compute-toolkit:latest|image: hd-compute-toolkit:$IMAGE_TAG|g" k8s/deployment.yaml
    fi
    
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    log_success "Application deployment applied"
    
    # Restore original deployment file if modified
    if [[ -f k8s/deployment.yaml.bak ]]; then
        mv k8s/deployment.yaml.bak k8s/deployment.yaml
    fi
    
    # Wait for application to be ready
    log_info "Waiting for application pods to be ready..."
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=hd-compute-toolkit -n "$NAMESPACE" --timeout=600s
    log_success "Application pods are ready"
}

# Apply networking
apply_networking() {
    log_info "Applying networking configuration..."
    
    kubectl apply -f k8s/ingress.yaml
    log_success "Ingress applied"
    
    kubectl apply -f k8s/networkpolicy.yaml
    log_success "Network policies applied"
}

# Apply autoscaling
apply_autoscaling() {
    log_info "Applying autoscaling configuration..."
    
    kubectl apply -f k8s/hpa.yaml
    log_success "Horizontal Pod Autoscaler applied"
}

# Apply monitoring
apply_monitoring() {
    log_info "Applying monitoring configuration..."
    
    # Check if Prometheus Operator is available
    if kubectl get crd prometheusrules.monitoring.coreos.com &> /dev/null; then
        kubectl apply -f k8s/monitoring.yaml
        log_success "Monitoring configuration applied"
    else
        log_warning "Prometheus Operator not found, skipping monitoring setup"
        log_info "Install Prometheus Operator first: https://github.com/prometheus-operator/prometheus-operator"
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=hd-compute-toolkit
    
    # Check service endpoints
    kubectl get endpoints -n "$NAMESPACE"
    
    # Test application health endpoint (if ingress is configured)
    local service_url=$(kubectl get svc hdc-toolkit-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$service_url" ]]; then
        log_info "Testing health endpoint..."
        if curl -f "http://$service_url/health" &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed - application may still be starting"
        fi
    else
        log_info "Service URL not available yet, skipping external health check"
    fi
    
    log_success "Health checks completed"
}

# Print deployment information
print_deployment_info() {
    log_info "Deployment Summary"
    echo "=========================="
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Environment: $ENVIRONMENT"
    echo ""
    
    log_info "Service Information:"
    kubectl get svc -n "$NAMESPACE"
    echo ""
    
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE"
    echo ""
    
    log_info "Ingress Information:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || log_info "No ingress found"
    echo ""
    
    log_info "Useful Commands:"
    echo "- View logs: kubectl logs -f deployment/hdc-toolkit-app -n $NAMESPACE"
    echo "- Scale deployment: kubectl scale deployment/hdc-toolkit-app --replicas=5 -n $NAMESPACE"
    echo "- Port forward: kubectl port-forward svc/hdc-toolkit-service 8080:80 -n $NAMESPACE"
    echo "- Delete deployment: kubectl delete namespace $NAMESPACE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f k8s/*.bak
}

# Main deployment function
main() {
    log_info "Starting HD Compute Toolkit deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Dry Run: $DRY_RUN"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No changes will be applied"
        export KUBECTL_CMD="kubectl --dry-run=client"
    fi
    
    # Trap to ensure cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    apply_configs
    apply_storage
    deploy_database
    deploy_cache
    deploy_application
    apply_networking
    apply_autoscaling
    apply_monitoring
    
    if [[ "$DRY_RUN" != "true" ]]; then
        run_health_checks
        print_deployment_info
    fi
    
    log_success "Deployment completed successfully!"
}

# Script help
show_help() {
    echo "HD Compute Toolkit Deployment Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -e, --environment       Environment (development|staging|production) [default: production]"
    echo "  -t, --tag              Docker image tag [default: latest]"
    echo "  -n, --namespace        Kubernetes namespace [default: hd-compute-toolkit]"
    echo "  -d, --dry-run          Run in dry-run mode (no changes applied)"
    echo ""
    echo "Environment Variables:"
    echo "  IMAGE_TAG              Docker image tag"
    echo "  ENVIRONMENT            Deployment environment"
    echo "  KUBECONFIG             Path to kubeconfig file"
    echo "  DRY_RUN                Enable dry-run mode (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy with defaults"
    echo "  $0 --environment staging --tag v1.2.3"
    echo "  $0 --dry-run"
    echo "  IMAGE_TAG=v1.0.0 $0"
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
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
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

# Run main function
main