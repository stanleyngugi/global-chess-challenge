#!/bin/bash
# =============================================================================
# RunPod A40/A100 Training Script - Production Version
# =============================================================================
# 
# RESEARCH-OPTIMIZED CONFIGURATION:
# - 1 epoch (756K samples is sufficient for SFT)
# - Learning rate 1e-4 (safer for r=64)
# - Packing enabled (10x speedup)
# - Flash Attention 2 (2-3x speedup)
# - All paths point to /workspace (persistent storage)
#
# Prerequisites:
#   1. GPU instance with 40-80GB VRAM (A100 or A40)
#   2. HuggingFace token with Llama-3.1-8B-Instruct access
#   3. Training data uploaded to /workspace/data/
#
# Estimated time: ~12 hours for 756K samples, 1 epoch with packing
# =============================================================================

set -e  # Exit on error

echo "=============================================================="
echo "Chess LoRA Training - Production Configuration"
echo "=============================================================="
echo ""
echo "RESEARCH-OPTIMIZED SETTINGS:"
echo "  - 1 epoch (prevents overfitting on 756K samples)"
echo "  - Learning rate: 1e-4 (safe for r=64)"
echo "  - Packing: ENABLED (10x speedup)"
echo "  - Flash Attention 2: AUTO"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Data files (use weight-free versions for packing compatibility)
TRAIN_FILE="${TRAIN_FILE:-src/division2/data/train.jsonl}"
VAL_FILE="${VAL_FILE:-src/division2/data/val.jsonl}"

# Output directory - MUST be on /workspace for RunPod persistence
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/chess-lora}"

# Model
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

# Training hyperparameters - RESEARCH OPTIMIZED
NUM_EPOCHS=1          # CRITICAL: 1 epoch for 756K samples
LEARNING_RATE=1e-4    # CRITICAL: Safer for r=64
BATCH_SIZE=4          # With packing, effective ~40 samples/batch
GRAD_ACCUM=4          # Effective batch = 4 * 4 * ~10 = ~160 samples

# LoRA configuration
LORA_R=64
LORA_ALPHA=128

# =============================================================================
# Environment Setup
# =============================================================================

echo "Setting up environment..."

# Set HuggingFace cache to persistent storage
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p "$HF_HOME"

# Enable TF32 for free speedup on Ampere
export NVIDIA_TF32_OVERRIDE=1

# =============================================================================
# System Checks (bash only - NO Python yet)
# =============================================================================

echo ""
echo "System Information:"
echo "-------------------"

# Check GPU (bash only)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "WARNING: nvidia-smi not found - are you on a GPU instance?"
fi

echo ""

# NOTE: Python checks moved to AFTER dependency installation (line ~200)

# =============================================================================
# Data Validation (bash only)
# =============================================================================

echo "Checking training data..."

# If training file doesn't exist, download after deps are installed
DATA_NEEDS_DOWNLOAD=false
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Training data not found - will download after installing dependencies"
    DATA_NEEDS_DOWNLOAD=true
else
    # Count samples
    TRAIN_SAMPLES=$(wc -l < "$TRAIN_FILE")
    echo "Training samples: $TRAIN_SAMPLES"
    
    if [ -f "$VAL_FILE" ]; then
        VAL_SAMPLES=$(wc -l < "$VAL_FILE")
        echo "Validation samples: $VAL_SAMPLES"
    fi
    
    # Verify data format (no weights)
    FIRST_LINE=$(head -1 "$TRAIN_FILE")
    if echo "$FIRST_LINE" | grep -q '"weight"'; then
        echo ""
        echo "WARNING: Training data contains 'weight' field."
        echo "This will be stripped after dependencies are installed."
    fi
fi

echo ""

# =============================================================================
# Training Plan
# =============================================================================

# With packing (~10 samples per sequence), effective samples per step
# = batch_size * grad_accum * 10
SAMPLES_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM * 10))
STEPS_PER_EPOCH=$((TRAIN_SAMPLES / SAMPLES_PER_STEP))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))

echo "Training Plan:"
echo "--------------"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "  Packing: ENABLED (~10 samples per 2048 sequence)"
echo "  Effective samples/step: ~$SAMPLES_PER_STEP"
echo "  Steps per epoch: ~$STEPS_PER_EPOCH"
echo "  Total steps: ~$TOTAL_STEPS"
echo "  Estimated time: ~12 hours"
echo ""

# =============================================================================
# Dependencies
# =============================================================================

echo "Installing dependencies..."

# Upgrade pip
pip install --upgrade pip -q

# Install NumPy first (CRITICAL: must be <2.0)
pip install "numpy==1.26.4" -q

# Install PyTorch with CUDA 12.4
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 -q

# Install main requirements
pip install -r requirements.txt -q

# Install Flash Attention 2 (pre-built wheel for speed)
echo "Installing Flash Attention 2..."
pip install flash-attn==2.6.3 --no-build-isolation -q 2>/dev/null || {
    echo "Flash Attention wheel not available, attempting source build..."
    pip install flash-attn --no-build-isolation -q 2>/dev/null || {
        echo "Flash Attention installation failed (will use SDPA fallback)"
    }
}

echo ""

# =============================================================================
# Data Download (if needed)
# =============================================================================

if [ "$DATA_NEEDS_DOWNLOAD" = true ]; then
    echo "Downloading training data from HuggingFace..."
    echo ""
    echo "  Dataset: stan4u/global_chess_trainining"
    echo "  This may take a few minutes (617MB)..."
    echo ""
    
    python3 << 'DOWNLOAD_DATA'
import os
from datasets import load_dataset

DATASET_NAME = "stan4u/global_chess_trainining"
print(f"  Loading dataset: {DATASET_NAME}")

ds = load_dataset(DATASET_NAME)
os.makedirs("src/division2/data", exist_ok=True)

if "train" in ds:
    print(f"  Saving train split: {len(ds['train'])} samples")
    ds["train"].to_json("src/division2/data/train.jsonl", lines=True)
else:
    print(f"  Saving dataset: {len(ds)} samples")
    ds.to_json("src/division2/data/train.jsonl", lines=True)

if "validation" in ds:
    print(f"  Saving validation split: {len(ds['validation'])} samples")
    ds["validation"].to_json("src/division2/data/val.jsonl", lines=True)
elif "test" in ds:
    print(f"  Saving test split as validation: {len(ds['test'])} samples")
    ds["test"].to_json("src/division2/data/val.jsonl", lines=True)

print("")
print("  Data download complete!")
DOWNLOAD_DATA

    if [ ! -f "$TRAIN_FILE" ]; then
        echo "ERROR: Failed to download training data!"
        exit 1
    fi
    
    TRAIN_SAMPLES=$(wc -l < "$TRAIN_FILE")
    echo "Downloaded: $TRAIN_SAMPLES training samples"
    echo ""
fi

# Strip weights if present (required for packing)
if [ -f "$TRAIN_FILE" ] && head -1 "$TRAIN_FILE" | grep -q '"weight"'; then
    echo "Stripping 'weight' field from training data..."
    python3 << 'STRIP_WEIGHTS'
import json
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
# HuggingFace Authentication
# =============================================================================

# Check if HF_TOKEN is provided via environment
if [ -n "$HF_TOKEN" ]; then
    echo "HuggingFace token provided via HF_TOKEN environment variable"
    mkdir -p /root/.cache/huggingface
    echo -n "$HF_TOKEN" > /root/.cache/huggingface/token
elif [ -f "/root/.cache/huggingface/token" ]; then
    echo "Using cached HuggingFace token"
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
elif ! huggingface-cli whoami &>/dev/null; then
    echo "HuggingFace authentication required for Llama-3.1-8B-Instruct"
    echo ""
    echo "Option 1: Provide token via environment:"
    echo "  HF_TOKEN=hf_xxx bash scripts/train_a100.sh"
    echo ""
    echo "Option 2: Login interactively:"
    huggingface-cli login
fi

# =============================================================================
# Environment Verification
# =============================================================================

echo ""
echo "Running environment verification..."
python3 -c "
import sys
import torch
import transformers
import peft
import numpy as np

print('Environment Check:')
print(f'  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'  NumPy: {np.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  BF16: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else \"N/A\"}')

# Verify NumPy < 2.0
from packaging import version
if version.parse(np.__version__) >= version.parse('2.0.0'):
    print('CRITICAL: NumPy 2.0+ detected - AWS Neuron incompatibility!')
    sys.exit(1)

# Verify transformers version
if version.parse(transformers.__version__) < version.parse('4.43.0'):
    print('CRITICAL: transformers < 4.43 - Llama 3.1 RoPE not supported!')
    sys.exit(1)

print('')
print('Environment OK!')
"

# =============================================================================
# Create Output Directory
# =============================================================================

mkdir -p "$OUTPUT_DIR"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# Start Training
# =============================================================================

echo "=============================================================="
echo "Starting training..."
echo "=============================================================="
echo ""
echo "Monitor with: tail -f $OUTPUT_DIR/training.log"
echo ""

# Run training
python -m src.division2.training.train_lora \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --max_seq_length 2048 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# =============================================================================
# Post-Training
# =============================================================================

echo ""
echo "=============================================================="
echo "Training complete!"
echo "=============================================================="
echo ""
echo "Adapter saved to: $OUTPUT_DIR/final_adapter"
echo ""
echo "Next steps:"
echo ""
echo "  1. Merge adapter into base model:"
echo "     python -m src.division2.training.merge_adapters \\"
echo "         --adapter_path $OUTPUT_DIR/final_adapter \\"
echo "         --output_path /workspace/output/chess-merged \\"
echo "         --verify"
echo ""
echo "  2. Upload to HuggingFace Hub:"
echo "     huggingface-cli upload YOUR_USERNAME/chess-llama /workspace/output/chess-merged"
echo ""
echo "  3. Submit to AIcrowd:"
echo "     aicrowd submit --challenge global-chess-challenge-2025 \\"
echo "         --hf-repo YOUR_USERNAME/chess-llama"
echo ""
