# Global Chess Challenge 2025

Chess AI for the AIcrowd Global Chess Challenge 2025.

## Quick Start (One Command!)

```bash
# On a fresh RunPod GPU instance (web terminal recommended):
git clone https://github.com/stanleyngugi/global-chess-challenge.git
cd global-chess-challenge
HF_TOKEN=hf_your_token ./scripts/run.sh
```

That's it! The script automatically:
1. Installs all dependencies (PyTorch, Flash Attention, etc.)
2. Downloads training data from HuggingFace (617MB)
3. Authenticates with HuggingFace
4. Validates the environment
5. Starts training (~12 hours)

## Prerequisites

1. **GPU Instance**: RunPod A40/A100 (40-80GB VRAM)
2. **HuggingFace Token**: Get from https://huggingface.co/settings/tokens
3. **Llama Access**: Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## After Training (~12 hours)

```bash
# 1. Merge adapter into base model
python -m src.division2.training.merge_adapters \
    --adapter_path /workspace/output/chess-lora/final_adapter \
    --output_path /workspace/output/chess-merged \
    --verify

# 2. Upload to HuggingFace
huggingface-cli upload YOUR_USERNAME/chess-llama /workspace/output/chess-merged
```

## Alternative: tmux (Survives SSH Disconnect)

```bash
HF_TOKEN=hf_your_token ./scripts/start_training_tmux.sh
```

Reconnect after disconnect: `tmux attach -t chess-training`

## Project Structure

```
├── scripts/
│   ├── run.sh                 # ONE-COMMAND training launcher
│   ├── train_a100.sh          # Detailed training script
│   └── start_training_tmux.sh # tmux wrapper for SSH resilience
├── src/
│   ├── division2/
│   │   ├── data/
│   │   │   ├── train.jsonl    # 756K samples (auto-downloaded)
│   │   │   └── val.jsonl      # 11K validation samples
│   │   └── training/
│   │       ├── train_lora.py      # LoRA fine-tuning
│   │       └── merge_adapters.py  # Merge LoRA into base
│   └── templates/
│       └── chess_agent_prompt.jinja  # Competition prompt
└── requirements.txt
```

## Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | Llama-3.1-8B-Instruct | Competition baseline |
| LoRA r | 64 | High capacity for chess |
| Epochs | 1 | 756K samples sufficient |
| Learning rate | 1e-4 | Safe for r=64 |
| Packing | Enabled | 10x speedup |

**Estimated time:** ~12 hours on A40/A100

## Training Data

Training data is automatically downloaded from HuggingFace: `stan4u/global_chess_trainining`

If you need to use local data instead:
```bash
# Copy from existing RunPod workspace
cp /workspace/data/train.jsonl src/division2/data/

# Then run training (will skip download)
HF_TOKEN=hf_your_token ./scripts/run.sh
```

## Competition Details

- **Deadline**: January 31, 2026
- **Hardware**: AWS Trainium trn1.2xlarge
- **Baseline ACPL**: 71.921 (must beat to qualify)
- **Output Format**: `<rationale>...</rationale><uci_move>e2e4</uci_move>`
