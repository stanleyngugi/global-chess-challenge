# Division 2: Model Intelligence

This directory contains the training pipeline for fine-tuning Llama-3.1-8B to play grandmaster-level chess.

## Strategy Overview

### The Core Insight: "Distillation, Not Discovery"

We are not asking Llama to discover chess principles (which takes billions of games). We are **distilling** the explicit knowledge of:
- **Stockfish 16**: Via Q-gap weighted training on Lichess evaluations
- **Syzygy Tablebases**: Via perfect endgame positions

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Output Format | `<uci_move>e2e4</uci_move>` only | Competition only scores the move |
| Training Method | Weighted SFT | High Q-gap positions get higher loss weight |
| Data Sources | Lichess + Syzygy | Engine evals + perfect endgames |
| LoRA Rank | 64 | Balance of capacity and efficiency |
| Hardware | NVIDIA GPU (train) → Trainium (deploy) | Weights are portable |

## Directory Structure

```
src/division2/
├── data/
│   ├── config.py              # Configuration for all data processing
│   ├── lichess_processor.py   # Q-gap filtering of Lichess evaluations
│   ├── syzygy_generator.py    # Perfect endgame position generation
│   └── dataset_builder.py     # Combines sources into training format
├── training/
│   ├── weighted_trainer.py    # Custom trainer with weighted loss
│   ├── train_lora.py          # Main LoRA training script
│   └── merge_adapters.py      # Merge LoRA into base model
└── validation/
    ├── validate_model.py      # Test against Stockfish
    └── format_test.py         # Ensure format compliance
```

## Quick Start

### 1. Download Data

**Lichess Evaluations** (40GB, Parquet):
```bash
# Option A: HuggingFace CLI
huggingface-cli download Lichess/chess-position-evaluations --local-dir ./data/lichess

# Option B: Direct download
# Go to https://huggingface.co/datasets/Lichess/chess-position-evaluations
```

**Syzygy Tablebases** (1GB for 3-4-5 piece):
```bash
# Download from https://syzygy-tables.info/
# Recommended: 3-4-5 piece tables (~1GB)
# Optional: 6-piece tables (~150GB) for better coverage
```

### 2. Prepare Training Data

```bash
./scripts/prepare_data.sh \
    --lichess-dir ./data/lichess \
    --syzygy-dir ./data/syzygy \
    --output-dir ./data/processed
```

This will:
1. Filter Lichess positions by Q-gap (keep only critical decisions)
2. Generate perfect endgame positions from Syzygy
3. Combine and format for training

**Output:** `train.jsonl` (~1M samples) and `val.jsonl`

### 3. Train LoRA Adapter

```bash
./scripts/train.sh \
    --train-file ./data/processed/train.jsonl \
    --output-dir ./output/chess-lora \
    --epochs 3
```

**Requirements:**
- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
- For 4-bit training (8GB VRAM): add `--use-4bit`

**Training time:** ~4-8 hours on A100, ~12-24 hours on RTX 4090

### 4. Merge Adapter

```bash
./scripts/merge.sh \
    --adapter-path ./output/chess-lora/final_adapter \
    --output-path ./output/chess-merged \
    --verify
```

This produces a standalone model that can be deployed on Neuron.

### 5. Validate

```bash
./scripts/validate.sh \
    --model-path ./output/chess-merged \
    --num-games 100
```

**Target metrics:**
- Format compliance: 100%
- ACPL: < 50 (competitive), < 35 (winning)

### 6. Deploy to Neuron

On an AWS Trainium instance:
```bash
optimum-cli export neuron \
    --model ./output/chess-merged \
    --batch_size 1 \
    --sequence_length 4096 \
    --tensor_parallel_size 2 \
    ./output/neuron-model
```

## Data Pipeline Details

### Q-Gap Filtering

The key insight: not all positions are equally important.

```
Q-gap = Score(Best Move) - Score(Second Best Move)

High Q-gap (>100cp): Critical decision, one move is clearly best
Medium Q-gap (50-100cp): Important but less critical
Low Q-gap (<50cp): Many moves are equally good (skip)
```

We weight samples by Q-gap so the model focuses on critical positions.

### Weighted Loss

```python
weight = sigmoid((q_gap - 50) / 50)
loss = CrossEntropy(prediction, target) * weight
```

| Q-Gap | Weight | Meaning |
|-------|--------|---------|
| 0 cp | 0.27 | Low importance |
| 50 cp | 0.50 | Medium |
| 100 cp | 0.73 | High |
| 200 cp | 0.95 | Critical |

### Dataset Composition

| Source | Target | Weight | Purpose |
|--------|--------|--------|---------|
| Lichess high Q-gap | 500K | 1.0 | Critical decisions |
| Lichess medium Q-gap | 300K | 0.5 | Important decisions |
| Syzygy endgames | 100K | 1.0 | Perfect endgame play |
| **Total** | **~1M** | | |

## Training Configuration

### LoRA Settings

```python
LoraConfig(
    r=64,                    # Rank
    lora_alpha=128,          # Alpha (2x rank)
    target_modules=[
        "q_proj", "k_proj",  # Attention
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
)
```

### Training Hyperparameters

```python
TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    warmup_ratio=0.03,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)
```

## Troubleshooting

### Out of Memory

If you get OOM errors:
1. Reduce batch size: `--batch-size 2`
2. Use 4-bit quantization: `--use-4bit`
3. Reduce max sequence length in config

### Format Errors

If the model outputs invalid format:
1. Check training data format is correct
2. Increase training epochs
3. Lower learning rate

### Poor Chess Play

If ACPL is too high:
1. Increase training data
2. Add more endgame positions
3. Check Q-gap filtering is working

## References

- [DeepMind Searchless Chess](https://arxiv.org/abs/2402.04494) - Action-value prediction
- [ChessLLM](https://arxiv.org/abs/2501.17186) - Long-round data (+350 Elo)
- [Chess-R1](https://github.com/krafton-ai/Chess-R1) - GRPO for chess
