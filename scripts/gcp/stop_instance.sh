#!/bin/bash
# Stop GCP Training Instance (to save costs)
# SQL Codegen SLM - Module 2.1

set -e

echo "🛑 Stopping GCP Training Instance"
echo "=================================="
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
    exit 1
fi

if [[ "$INSTANCE_STATUS" == "TERMINATED" ]] || [[ "$INSTANCE_STATUS" == "STOPPED" ]]; then
    echo "✅ Instance is already stopped"
    exit 0
fi

echo "Instance: $INSTANCE_NAME"
echo "Status: $INSTANCE_STATUS"
echo ""

# Confirm stop
read -p "Stop instance? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🛑 Stopping instance..."
gcloud compute instances stop "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Instance Stopped!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💰 Cost savings:"
echo "   - A100 GPU: \$3.67/hour saved"
echo "   - Machine: \$0.47/hour saved"
echo "   - Total: ~\$4.14/hour saved"
echo ""
echo "Note: Disk storage still incurs charges (~\$0.27/hour)"
echo ""
echo "To restart: ./scripts/gcp/connect_to_instance.sh"
echo "To delete completely: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
