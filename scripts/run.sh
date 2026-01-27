#!/bin/bash
# =============================================================================
# ONE-COMMAND TRAINING LAUNCHER
# =============================================================================
#
# This is THE ONLY script you need to run on a fresh RunPod instance.
# It handles EVERYTHING: deps, auth, verification, training.
#
# Usage (from RunPod web terminal):
#   git clone https://github.com/stanleyngugi/global-chess-challenge.git
#   cd global-chess-challenge
#   HF_TOKEN=your_token ./scripts/run.sh
#
# Or with token inline:
#   HF_TOKEN=hf_xxx ./scripts/run.sh
#
# =============================================================================

set -e

echo "=============================================================="
echo "Chess AI Training - One Command Setup"
echo "=============================================================="
echo ""

# =============================================================================
# PHASE 0: Pre-flight (NO PYTHON - pure bash)
# =============================================================================

echo "[Phase 0] Pre-flight checks (bash only)..."

# Check we're on Linux with GPU
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "ERROR: This script is for Linux (RunPod). You're on: $OSTYPE"
    exit 1
fi

# Check nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "  GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    # Try to read from cached location
    if [ -f "/root/.cache/huggingface/token" ]; then
        export HF_TOKEN=$(cat /root/.cache/huggingface/token)
        echo "  HF token: loaded from cache"
    else
        echo ""
        echo "ERROR: HuggingFace token not provided!"
        echo ""
        echo "Usage:"
        echo "  HF_TOKEN=hf_your_token ./scripts/run.sh"
        echo ""
        echo "Get your token from: https://huggingface.co/settings/tokens"
        echo "Make sure you have access to meta-llama/Llama-3.1-8B-Instruct"
        exit 1
    fi
else
    echo "  HF token: provided via environment"
fi

# Verify token is not empty
if [ ${#HF_TOKEN} -lt 10 ]; then
    echo "ERROR: HF_TOKEN looks invalid (too short)"
    exit 1
fi

# Training data paths (will check/download after deps installed)
TRAIN_FILE="src/division2/data/train.jsonl"
VAL_FILE="src/division2/data/val.jsonl"

# Quick check if data exists (detailed download handled in Phase 1.5)
if [ -f "$TRAIN_FILE" ]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    echo "  Training data: $TRAIN_COUNT samples (local)"
else
    echo "  Training data: will download from HuggingFace"
fi

echo ""
echo "[Phase 0] Pre-flight OK!"
echo ""

# =============================================================================
# PHASE 1: Install Dependencies (before ANY Python)
# =============================================================================

echo "[Phase 1] Installing dependencies..."
echo ""

# Set cache locations FIRST
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
export HF_TOKEN="$HF_TOKEN"
mkdir -p "$HF_HOME"

# Upgrade pip silently
pip install --upgrade pip -q

# Install NumPy FIRST (must be <2.0 for Trainium)
echo "  Installing NumPy 1.26.4..."
pip install "numpy==1.26.4" -q

# Install PyTorch with CUDA
echo "  Installing PyTorch 2.4.1+cu124..."
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 -q

# Install requirements
echo "  Installing training requirements..."
pip install -r requirements.txt -q

# Install Flash Attention (optional but recommended)
echo "  Installing Flash Attention 2..."
pip install flash-attn==2.6.3 --no-build-isolation -q 2>/dev/null || {
    echo "  (Flash Attention wheel not available, using SDPA fallback)"
}

echo ""
echo "[Phase 1] Dependencies installed!"
echo ""

# =============================================================================
# PHASE 1.5: Download Training Data (if not present)
# =============================================================================

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[Phase 1.5] Downloading training data from HuggingFace..."
    echo ""
    echo "  Dataset: stan4u/global_chess_trainining"
    echo "  This may take a few minutes (617MB)..."
    echo ""
    
    python3 << 'DOWNLOAD_DATA'
import os
from datasets import load_dataset

# HuggingFace dataset location
DATASET_NAME = "stan4u/global_chess_trainining"

print(f"  Loading dataset: {DATASET_NAME}")

# Load dataset from HuggingFace
ds = load_dataset(DATASET_NAME)

# Create output directory
os.makedirs("src/division2/data", exist_ok=True)

# Save training data
if "train" in ds:
    print(f"  Saving train split: {len(ds['train'])} samples")
    ds["train"].to_json("src/division2/data/train.jsonl", lines=True)
else:
    # If no splits, use the entire dataset
    print(f"  Saving dataset: {len(ds)} samples")
    ds.to_json("src/division2/data/train.jsonl", lines=True)

# Save validation data if present
if "validation" in ds:
    print(f"  Saving validation split: {len(ds['validation'])} samples")
    ds["validation"].to_json("src/division2/data/val.jsonl", lines=True)
elif "test" in ds:
    print(f"  Saving test split as validation: {len(ds['test'])} samples")
    ds["test"].to_json("src/division2/data/val.jsonl", lines=True)

print("")
print("  Data download complete!")
DOWNLOAD_DATA

    # Verify download succeeded
    if [ ! -f "$TRAIN_FILE" ]; then
        echo ""
        echo "ERROR: Failed to download training data!"
        echo "Please check your HuggingFace token and dataset access."
        exit 1
    fi
    
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    echo "  Downloaded: $TRAIN_COUNT training samples"
    echo ""
else
    echo "[Phase 1.5] Training data already present, skipping download."
    echo ""
fi

# Check for weight field (incompatible with packing) - now that we have data
if [ -f "$TRAIN_FILE" ] && head -1 "$TRAIN_FILE" | grep -q '"weight"'; then
    echo "  Stripping 'weight' field from training data (required for packing)..."
    python3 << 'STRIP_WEIGHTS'
import json

# Read, strip weights, write back
with open("src/division2/data/train.jsonl", "r") as f:
    lines = f.readlines()

with open("src/division2/data/train.jsonl", "w") as f:
    for line in lines:
        data = json.loads(line)
        data.pop("weight", None)
        f.write(json.dumps(data) + "\n")

print("  Weights stripped successfully!")
STRIP_WEIGHTS
fi

# =============================================================================
# PHASE 2: Verify Environment (NOW we can use Python)
# =============================================================================

echo "[Phase 2] Verifying environment..."
echo ""

python3 << 'VERIFY_SCRIPT'
import sys

def check(name, condition, message=""):
    status = "OK" if condition else "FAIL"
    print(f"  {name}: {status} {message}")
    if not condition:
        sys.exit(1)

# Core imports
import torch
import transformers
import peft
import numpy as np
from packaging import version

# Version checks
check("NumPy", version.parse(np.__version__) < version.parse("2.0"), f"({np.__version__})")
check("Transformers", version.parse(transformers.__version__) >= version.parse("4.43.0"), f"({transformers.__version__})")
check("PyTorch", True, f"({torch.__version__})")
check("PEFT", True, f"({peft.__version__})")

# Hardware checks
check("CUDA", torch.cuda.is_available())
check("BF16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

# HuggingFace auth
import os
token = os.environ.get("HF_TOKEN", "")
check("HF_TOKEN", len(token) > 10, "(token set)")

print("")
print("Environment verified!")
VERIFY_SCRIPT

echo ""
echo "[Phase 2] Environment OK!"
echo ""

# =============================================================================
# PHASE 3: HuggingFace Authentication
# =============================================================================

echo "[Phase 3] Authenticating with HuggingFace..."
echo ""

# Write token to cache so transformers can find it
mkdir -p /root/.cache/huggingface
echo -n "$HF_TOKEN" > /root/.cache/huggingface/token

# Verify we can access the model
python3 << 'AUTH_CHECK'
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
api = HfApi()

try:
    # Try to get model info (will fail if no access)
    info = api.model_info("meta-llama/Llama-3.1-8B-Instruct", token=token)
    print(f"  Model access: OK (Llama-3.1-8B-Instruct)")
except Exception as e:
    print(f"  ERROR: Cannot access Llama-3.1-8B-Instruct")
    print(f"  {e}")
    print("")
    print("  Make sure you:")
    print("  1. Accepted the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("  2. Have a valid token with read access")
    exit(1)
AUTH_CHECK

echo ""
echo "[Phase 3] Authentication OK!"
echo ""

# =============================================================================
# PHASE 4: Setup Output Directory
# =============================================================================

echo "[Phase 4] Setting up training..."
echo ""

OUTPUT_DIR="/workspace/output/chess-lora"
mkdir -p "$OUTPUT_DIR"
echo "  Output: $OUTPUT_DIR"

# Calculate training plan
TRAIN_SAMPLES=$(wc -l < "$TRAIN_FILE")
# With packing, ~10 samples per 2048 sequence
# batch=4, grad_accum=4, packing=10x => ~160 samples/step
SAMPLES_PER_STEP=160
TOTAL_STEPS=$((TRAIN_SAMPLES / SAMPLES_PER_STEP))
HOURS_ESTIMATE=$((TOTAL_STEPS * 10 / 3600))  # ~10 sec/step

echo ""
echo "  Training Plan:"
echo "    Samples: $TRAIN_SAMPLES"
echo "    Steps: ~$TOTAL_STEPS"
echo "    Estimated time: ~${HOURS_ESTIMATE} hours"
echo ""

# =============================================================================
# PHASE 5: Launch Training
# =============================================================================

echo "[Phase 5] Launching training..."
echo ""
echo "=============================================================="
echo "TRAINING STARTED"
echo "=============================================================="
echo ""
echo "Logs: $OUTPUT_DIR/training.log"
echo ""
echo "To monitor:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "This will take ~12 hours. The process continues even if you"
echo "close this terminal (nohup mode)."
echo ""
echo "=============================================================="
echo ""

# Run training with nohup so it survives terminal close
# Also tee to both console and log file
python -m src.division2.training.train_lora \
    --train_file "$TRAIN_FILE" \
    --val_file "src/division2/data/val.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_seq_length 2048 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# =============================================================================
# PHASE 6: Post-Training
# =============================================================================

echo ""
echo "=============================================================="
echo "TRAINING COMPLETE!"
echo "=============================================================="
echo ""
echo "Adapter saved to: $OUTPUT_DIR/final_adapter"
echo ""
echo "Next steps:"
echo ""
echo "  1. Merge adapter:"
echo "     python -m src.division2.training.merge_adapters \\"
echo "         --adapter_path $OUTPUT_DIR/final_adapter \\"
echo "         --output_path /workspace/output/chess-merged \\"
echo "         --verify"
echo ""
echo "  2. Upload to HuggingFace:"
echo "     huggingface-cli upload stanleyngugi/chess-llama /workspace/output/chess-merged"
echo ""
