#!/bin/bash
# =============================================================================
# TMUX Training Launcher - Survives SSH Disconnection
# =============================================================================
#
# This script starts training inside a tmux session that persists even if
# your SSH connection drops. This is CRITICAL for long training runs.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/start_training_tmux.sh
#
# Or if already logged in:
#   bash scripts/start_training_tmux.sh
#
# To reconnect after disconnect:
#   tmux attach -t chess-training
#
# To check if training is running:
#   tmux ls
#
# =============================================================================

set -e

SESSION_NAME="chess-training"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Get HF_TOKEN from environment or cache
if [ -z "$HF_TOKEN" ] && [ -f "/root/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set and no cached token found."
    echo ""
    echo "Usage:"
    echo "  HF_TOKEN=hf_xxx bash scripts/start_training_tmux.sh"
    echo ""
    echo "Or login first:"
    echo "  huggingface-cli login"
    exit 1
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "=============================================================="
    echo "Training session '$SESSION_NAME' already exists!"
    echo "=============================================================="
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session:"
    echo "     tmux attach -t $SESSION_NAME"
    echo ""
    echo "  2. Kill existing session and start new:"
    echo "     tmux kill-session -t $SESSION_NAME"
    echo "     HF_TOKEN=$HF_TOKEN bash scripts/start_training_tmux.sh"
    echo ""
    exit 1
fi

echo "=============================================================="
echo "Starting Training in TMUX Session"
echo "=============================================================="
echo ""
echo "Session name: $SESSION_NAME"
echo ""
echo "IMPORTANT: Your training will continue even if SSH disconnects!"
echo ""
echo "To reconnect: tmux attach -t $SESSION_NAME"
echo "To detach:    Ctrl+B, then D"
echo ""

# Create tmux session and run training
# CRITICAL: Pass HF_TOKEN into the tmux environment
tmux new-session -d -s $SESSION_NAME

# Set environment variables inside tmux
tmux send-keys -t $SESSION_NAME "export HF_TOKEN='$HF_TOKEN'" C-m
tmux send-keys -t $SESSION_NAME "export HF_HOME='/workspace/hf_cache'" C-m
tmux send-keys -t $SESSION_NAME "export TRANSFORMERS_CACHE='/workspace/hf_cache'" C-m

# Navigate and run
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "bash scripts/train_a100.sh 2>&1 | tee /workspace/output/training_$(date +%Y%m%d_%H%M%S).log" C-m

echo "Training started in background!"
echo ""
echo "Commands:"
echo "  View training:    tmux attach -t $SESSION_NAME"
echo "  Detach (in tmux): Ctrl+B, then D"
echo "  Check status:     tmux ls"
echo "  View logs:        tail -f /workspace/output/chess-lora/training.log"
echo ""

# Attach to the session
echo "Attaching to session..."
sleep 2
tmux attach -t $SESSION_NAME
