#!/bin/bash
# Create GCP Compute Engine Instance for Training
# SQL Codegen SLM - Module 2.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_FILE="$PROJECT_DIR/training/configs/gcp_compute.yaml"

echo "🚀 Creating GCP Training Instance"
echo "=================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found"
    echo ""
    echo "Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list 2>&1 | grep -q "ACTIVE"; then
    echo "❌ Not authenticated with GCP"
    echo ""
    echo "Please run: gcloud auth login"
    exit 1
fi

# Configuration (update these or read from YAML)
PROJECT_ID="${GCP_PROJECT_ID:-YOUR_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-sql-codegen-training}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-highmem-8}"
GPU_TYPE="${GCP_GPU_TYPE:-nvidia-tesla-a100}"
GPU_COUNT="${GCP_GPU_COUNT:-1}"
BOOT_DISK_SIZE="${GCP_BOOT_DISK_SIZE:-200}"
IMAGE_FAMILY="common-cu121-ubuntu-2204"
IMAGE_PROJECT="deeplearning-platform-release"

# Check if project ID is set
if [[ "$PROJECT_ID" == "YOUR_PROJECT_ID" ]]; then
    echo "❌ GCP Project ID not set"
    echo ""
    echo "Please set your project ID:"
    echo "  export GCP_PROJECT_ID=your-project-id"
    echo ""
    echo "Or update training/configs/gcp_compute.yaml"
    exit 1
fi

echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Zone: $ZONE"
echo "   Instance: $INSTANCE_NAME"
echo "   Machine: $MACHINE_TYPE"
echo "   GPU: $GPU_TYPE x $GPU_COUNT"
echo "   Disk: ${BOOT_DISK_SIZE}GB SSD"
echo ""

# Confirm creation
read -p "Create instance? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🔧 Creating instance..."

# Create the instance
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="${BOOT_DISK_SIZE}GB" \
    --boot-disk-type="pd-ssd" \
    --maintenance-policy="TERMINATE" \
    --metadata="install-nvidia-driver=True" \
    --scopes="https://www.googleapis.com/auth/cloud-platform"

echo ""
echo "✅ Instance created successfully!"
echo ""

# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
sleep 30

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ GCP Instance Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Instance: $INSTANCE_NAME"
echo "IP: $INSTANCE_IP"
echo ""
echo "Next steps:"
echo "  1. Sync data: ./scripts/gcp/sync_data_to_gcp.sh"
echo "  2. Connect: ./scripts/gcp/connect_to_instance.sh"
echo ""
echo "⚠️  Remember to stop the instance when not in use!"
echo "   ./scripts/gcp/stop_instance.sh"
echo ""
echo "💰 Estimated cost: ~\$4.41/hour"
