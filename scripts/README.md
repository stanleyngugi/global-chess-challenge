# Training Scripts Reference

Complete guide to all training and recovery scripts.

## Quick Start

```bash
# 1. Verify environment
python scripts/verify_environment.py

# 2. Pre-flight check
python scripts/preflight_check.py

# 3. Start training (in tmux - survives disconnect)
bash scripts/start_training_tmux.sh
```

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_a100.sh` | Main training script | Starting fresh training |
| `start_training_tmux.sh` | Launch in tmux | **Recommended** - survives SSH disconnect |
| `resume_training.py` | Resume from checkpoint | After crash/disconnect |
| `inspect_checkpoints.py` | View checkpoint status | Before resuming |
| `recovery_guide.sh` | Show recovery options | After any interruption |
| `verify_environment.py` | Validate dependencies | Before training |
| `preflight_check.py` | Validate data/config | Before training |
| `remove_weights.py` | Strip weights from data | One-time data prep |

---

## Checkpoint Recovery

### How Checkpoints Work

Training saves checkpoints every **500 steps** to `/workspace/output/chess-lora/`:

```
/workspace/output/chess-lora/
├── checkpoint-500/
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── adapter_config.json        # LoRA config
│   ├── trainer_state.json         # Training progress
│   ├── optimizer.pt               # Optimizer state
│   ├── scheduler.pt               # LR scheduler
│   └── rng_state.pth             # Random state
├── checkpoint-1000/
├── checkpoint-1500/
└── ... (keeps last 3)
```

### After Disconnect/Crash

```bash
# Step 1: Check what happened
bash scripts/recovery_guide.sh

# Step 2: Inspect checkpoints
python scripts/inspect_checkpoints.py

# Step 3: Resume from latest checkpoint
python scripts/resume_training.py

# Or resume from specific checkpoint
python scripts/resume_training.py --checkpoint /workspace/output/chess-lora/checkpoint-2500
```

---

## Training with tmux (Recommended)

**ALWAYS use tmux** to prevent losing training progress on SSH disconnect.

### Start Training
```bash
bash scripts/start_training_tmux.sh
```

### Detach from Session
Press `Ctrl+B`, then `D`

### Reconnect After Disconnect
```bash
tmux attach -t chess-training
```

### Check if Training is Running
```bash
tmux ls
```

### Kill Session (if needed)
```bash
tmux kill-session -t chess-training
```

---

## Training Timeline

Estimated for 756K samples, 1 epoch with packing:

| Step | Time | Progress |
|------|------|----------|
| 0 | 0h | Start |
| 500 | ~1.3h | First checkpoint |
| 1000 | ~2.5h | 21% |
| 2000 | ~5h | 43% |
| 3000 | ~7.5h | 64% |
| 4000 | ~10h | 85% |
| ~4700 | ~12h | Complete |

**Checkpoints are saved at:** 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500

---

## Troubleshooting

### Training crashed before first checkpoint (step < 500)
```bash
# No checkpoint to resume from - start fresh
bash scripts/start_training_tmux.sh
```

### SSH disconnected but training should be running
```bash
# Check for tmux session
tmux attach -t chess-training

# If session doesn't exist, resume from checkpoint
python scripts/resume_training.py
```

### Checkpoint validation failed (NaN detected)
```bash
# Training diverged - check learning rate and try earlier checkpoint
python scripts/resume_training.py --checkpoint /workspace/output/chess-lora/checkpoint-1000
```

### Pod was stopped/restarted
```bash
# Your /workspace is persistent! Just resume
python scripts/resume_training.py
```

### Out of memory during resume
```bash
# Reduce batch size
python scripts/resume_training.py --batch_size 2 --grad_accum 8
```

---

## Post-Training

After training completes:

```bash
# 1. Merge adapter
python -m src.division2.training.merge_adapters \
    --adapter_path /workspace/output/chess-lora/final_adapter \
    --output_path /workspace/output/chess-merged \
    --verify

# 2. Upload to HuggingFace
huggingface-cli upload YOUR_USERNAME/chess-llama /workspace/output/chess-merged

# 3. Submit to AIcrowd
aicrowd submit --challenge global-chess-challenge-2025 \
    --hf-repo YOUR_USERNAME/chess-llama
```

---

## Data Files

| File | Samples | Purpose |
|------|---------|---------|
| `train.jsonl` | 756,685 | Training (weight-free for packing) |
| `val.jsonl` | 10,999 | Validation |
| `train (1).jsonl` | 756,685 | Original with weights (backup) |
| `val (1).jsonl` | 10,999 | Original with weights (backup) |
