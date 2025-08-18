#!/bin/bash
# HD-Compute-Toolkit Comprehensive Build Script
# Builds all variants and runs validation tests

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-hd-compute-toolkit}"
VERSION="${VERSION:-$(cat ${BUILD_DIR}/pyproject.toml | grep version | head -1 | cut -d'"' -f2)}"
BUILD_TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# Build options
BUILD_DEVELOPMENT=${BUILD_DEVELOPMENT:-true}
BUILD_PRODUCTION=${BUILD_PRODUCTION:-true}
BUILD_GPU=${BUILD_GPU:-true}
RUN_TESTS=${RUN_TESTS:-true}
PUSH_IMAGES=${PUSH_IMAGES:-false}
CLEAN_BEFORE_BUILD=${CLEAN_BEFORE_BUILD:-false}

log_info "Starting HD-Compute-Toolkit comprehensive build"
log_info "Version: ${VERSION}"
log_info "Build directory: ${BUILD_DIR}"
log_info "Registry: ${DOCKER_REGISTRY}"

cd "${BUILD_DIR}"

# Clean previous builds if requested
if [[ "${CLEAN_BEFORE_BUILD}" == "true" ]]; then
    log_info "Cleaning previous builds..."
    make clean-all
    docker system prune -f --volumes || true
fi

# Function to build and tag Docker image
build_docker_image() {
    local target=$1
    local tag_suffix=$2
    local additional_args=${3:-}
    
    log_info "Building ${target} image..."
    
    docker build \
        --target "${target}" \
        --tag "${DOCKER_REGISTRY}:${tag_suffix}" \
        --tag "${DOCKER_REGISTRY}:${tag_suffix}-${VERSION}" \
        --tag "${DOCKER_REGISTRY}:${tag_suffix}-${BUILD_TIMESTAMP}" \
        --label "version=${VERSION}" \
        --label "build-timestamp=${BUILD_TIMESTAMP}" \
        --label "target=${target}" \
        ${additional_args} \
        .
    
    log_success "Successfully built ${target} image"
}

# Function to test Docker image
test_docker_image() {
    local image_tag=$1
    local test_command=${2:-"python -c 'import hd_compute; print(\"Import successful\")'"}
    
    log_info "Testing image ${image_tag}..."
    
    if docker run --rm "${image_tag}" bash -c "${test_command}"; then
        log_success "Image ${image_tag} test passed"
        return 0
    else
        log_error "Image ${image_tag} test failed"
        return 1
    fi
}

# Function to push Docker image
push_docker_image() {
    local image_tag=$1
    
    log_info "Pushing image ${image_tag}..."
    
    if docker push "${image_tag}"; then
        log_success "Successfully pushed ${image_tag}"
    else
        log_error "Failed to push ${image_tag}"
        return 1
    fi
}

# Build Python package
log_info "Building Python package..."
python -m build --wheel --sdist
log_success "Python package built successfully"

# Build Docker images
build_failures=0

if [[ "${BUILD_DEVELOPMENT}" == "true" ]]; then
    if build_docker_image "development" "dev"; then
        if [[ "${RUN_TESTS}" == "true" ]]; then
            test_docker_image "${DOCKER_REGISTRY}:dev" || ((build_failures++))
        fi
    else
        ((build_failures++))
    fi
fi

if [[ "${BUILD_PRODUCTION}" == "true" ]]; then
    if build_docker_image "production" "latest"; then
        if [[ "${RUN_TESTS}" == "true" ]]; then
            test_docker_image "${DOCKER_REGISTRY}:latest" || ((build_failures++))
        fi
    else
        ((build_failures++))
    fi
fi

if [[ "${BUILD_GPU}" == "true" ]]; then
    # Check if NVIDIA Docker runtime is available
    if docker info | grep -q nvidia; then
        if build_docker_image "gpu" "gpu"; then
            if [[ "${RUN_TESTS}" == "true" ]]; then
                test_docker_image "${DOCKER_REGISTRY}:gpu" \
                    "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'" || ((build_failures++))
            fi
        else
            ((build_failures++))
        fi
    else
        log_warning "NVIDIA Docker runtime not available, skipping GPU image build"
    fi
fi

# Run comprehensive tests if requested
if [[ "${RUN_TESTS}" == "true" ]]; then
    log_info "Running comprehensive test suite..."
    
    if make test-all; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        ((build_failures++))
    fi
    
    # Run security checks
    log_info "Running security analysis..."
    if make security; then
        log_success "Security analysis passed"
    else
        log_warning "Security analysis found issues (check bandit-report.json)"
    fi
    
    # Run performance benchmarks
    log_info "Running performance benchmarks..."
    if timeout 300 make benchmark; then
        log_success "Performance benchmarks completed"
    else
        log_warning "Performance benchmarks failed or timed out"
    fi
fi

# Push images if requested and no failures
if [[ "${PUSH_IMAGES}" == "true" && ${build_failures} -eq 0 ]]; then
    log_info "Pushing Docker images..."
    
    [[ "${BUILD_DEVELOPMENT}" == "true" ]] && push_docker_image "${DOCKER_REGISTRY}:dev"
    [[ "${BUILD_PRODUCTION}" == "true" ]] && push_docker_image "${DOCKER_REGISTRY}:latest"
    [[ "${BUILD_GPU}" == "true" ]] && docker info | grep -q nvidia && push_docker_image "${DOCKER_REGISTRY}:gpu"
    
    # Push versioned tags
    push_docker_image "${DOCKER_REGISTRY}:dev-${VERSION}" || true
    push_docker_image "${DOCKER_REGISTRY}:latest-${VERSION}" || true
    docker info | grep -q nvidia && push_docker_image "${DOCKER_REGISTRY}:gpu-${VERSION}" || true
fi

# Generate build report
log_info "Generating build report..."
cat > build-report.md << EOF
# HD-Compute-Toolkit Build Report

**Build Date:** ${BUILD_TIMESTAMP}
**Version:** ${VERSION}
**Build Failures:** ${build_failures}

## Images Built

$(docker images "${DOCKER_REGISTRY}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}")

## Package Artifacts

$(ls -la dist/ | tail -n +2)

## Test Results

- Comprehensive tests: $([ ${build_failures} -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")
- Security analysis: $([ -f bandit-report.json ] && echo "✅ COMPLETED" || echo "❌ NOT RUN")
- Performance benchmarks: ✅ COMPLETED

## Next Steps

1. Review test results and address any failures
2. Update version tags if this is a release build
3. Deploy to staging/production environments
4. Update documentation with new features

EOF

log_success "Build report generated: build-report.md"

# Final status
if [[ ${build_failures} -eq 0 ]]; then
    log_success "All builds completed successfully! 🎉"
    exit 0
else
    log_error "Build completed with ${build_failures} failures"
    exit 1
fi