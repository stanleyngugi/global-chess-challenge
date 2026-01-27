#!/bin/bash
# =============================================================================
# RECOVERY GUIDE - What to do after disconnect/crash
# =============================================================================
#
# Run this script to see recovery options:
#   bash scripts/recovery_guide.sh
#
# =============================================================================

echo ""
echo "=============================================================="
echo "TRAINING RECOVERY GUIDE"
echo "=============================================================="
echo ""

# Check if we're on RunPod (has /workspace)
if [ -d "/workspace" ]; then
    OUTPUT_DIR="/workspace/output/chess-lora"
    echo "Detected RunPod environment"
else
    OUTPUT_DIR="output/chess-lora"
    echo "Local environment (not RunPod)"
fi

echo "Output directory: $OUTPUT_DIR"
echo ""

# Check for tmux session
echo "1. CHECKING FOR RUNNING TRAINING SESSION"
echo "   ----------------------------------------"
if command -v tmux &> /dev/null; then
    if tmux has-session -t chess-training 2>/dev/null; then
        echo "   GOOD NEWS! Training session 'chess-training' is RUNNING!"
        echo ""
        echo "   To reconnect:"
        echo "   $ tmux attach -t chess-training"
        echo ""
    else
        echo "   No active tmux session found."
    fi
else
    echo "   tmux not installed"
fi
echo ""

# Check for checkpoints
echo "2. CHECKING FOR CHECKPOINTS"
echo "   ----------------------------------------"
if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINTS=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | wc -l)
    if [ "$CHECKPOINTS" -gt 0 ]; then
        echo "   Found $CHECKPOINTS checkpoint(s):"
        ls -la ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | tail -5
        echo ""
        LATEST=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        echo "   Latest checkpoint: $LATEST"
        
        if [ -f "$LATEST/trainer_state.json" ]; then
            STEP=$(python3 -c "import json; print(json.load(open('$LATEST/trainer_state.json'))['global_step'])" 2>/dev/null)
            EPOCH=$(python3 -c "import json; print(f\"{json.load(open('$LATEST/trainer_state.json'))['epoch']:.3f}\")" 2>/dev/null)
            echo "   Step: $STEP, Epoch: $EPOCH"
        fi
    else
        echo "   No checkpoints found in $OUTPUT_DIR"
    fi
else
    echo "   Output directory does not exist: $OUTPUT_DIR"
fi
echo ""

# Check for final adapter
echo "3. CHECKING FOR COMPLETED TRAINING"
echo "   ----------------------------------------"
if [ -d "$OUTPUT_DIR/final_adapter" ]; then
    echo "   TRAINING COMPLETED! Final adapter found at:"
    echo "   $OUTPUT_DIR/final_adapter"
    echo ""
    echo "   Next step: Merge the adapter"
    echo "   $ python -m src.division2.training.merge_adapters \\"
    echo "       --adapter_path $OUTPUT_DIR/final_adapter \\"
    echo "       --output_path /workspace/output/chess-merged \\"
    echo "       --verify"
else
    echo "   Training not yet completed."
fi
echo ""

# Recovery options
echo "=============================================================="
echo "RECOVERY OPTIONS"
echo "=============================================================="
echo ""
echo "OPTION A: Reconnect to running training"
echo "   $ tmux attach -t chess-training"
echo ""
echo "OPTION B: Resume from checkpoint"
echo "   $ python scripts/resume_training.py"
echo ""
echo "OPTION C: Resume from specific checkpoint"
echo "   $ python scripts/resume_training.py --checkpoint $OUTPUT_DIR/checkpoint-XXXX"
echo ""
echo "OPTION D: Inspect checkpoints first"
echo "   $ python scripts/inspect_checkpoints.py --output_dir $OUTPUT_DIR"
echo ""
echo "OPTION E: Start fresh (only if no valid checkpoints)"
echo "   $ bash scripts/start_training_tmux.sh"
echo ""
echo "=============================================================="
echo ""

# Quick command summary
echo "QUICK COMMANDS:"
echo "---------------"
echo "# View training logs"
echo "tail -f $OUTPUT_DIR/training.log"
echo ""
echo "# Check GPU usage"  
echo "nvidia-smi"
echo ""
echo "# Monitor training process"
echo "watch -n 5 nvidia-smi"
echo ""
