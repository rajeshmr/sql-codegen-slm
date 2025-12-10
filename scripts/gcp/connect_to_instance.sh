#!/bin/bash
# Connect to GCP Training Instance via SSH
# SQL Codegen SLM - Module 2.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔌 Connecting to GCP Training Instance"
echo "======================================="
echo ""

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-YOUR_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-sql-codegen-training}"

# Check if project ID is set
if [[ "$PROJECT_ID" == "YOUR_PROJECT_ID" ]]; then
    echo "❌ GCP Project ID not set"
    echo "   export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Check instance status
echo "🔍 Checking instance status..."
INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format="get(status)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$INSTANCE_STATUS" == "NOT_FOUND" ]]; then
    echo "❌ Instance '$INSTANCE_NAME' not found"
    echo "   Run: ./scripts/gcp/create_training_instance.sh"
    exit 1
fi

if [[ "$INSTANCE_STATUS" != "RUNNING" ]]; then
    echo "⚠️  Instance is $INSTANCE_STATUS"
    read -p "Start instance? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Starting instance..."
        gcloud compute instances start "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE"
        echo "   Waiting for instance to start..."
        sleep 30
    else
        echo "Cancelled."
        exit 0
    fi
fi

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ Instance is running"
echo "   IP: $INSTANCE_IP"
echo ""
echo "📡 Connecting with port forwarding..."
echo "   - TensorBoard: http://localhost:6006"
echo "   - Jupyter: http://localhost:8888"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Connect with port forwarding
gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    -- -L 6006:localhost:6006 -L 8888:localhost:8888
