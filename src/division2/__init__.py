"""
Division 2: Model Intelligence
==============================

This module contains the training pipeline for fine-tuning Llama-3.1-8B
to play grandmaster-level chess through action-value distillation.

Training Pipeline:
------------------
1. train_lora.py     - LoRA fine-tuning with SFTTrainer + packing
2. merge_adapters.py - Merge LoRA into base model for deployment

Usage:
------
    # Train
    python -m src.division2.training.train_lora --train_file data/train.jsonl
    
    # Merge
    python -m src.division2.training.merge_adapters --adapter_path ... --output_path ...
"""

# Note: We don't import submodules here to avoid circular dependencies
# and missing file issues. Import directly from submodules:
#
#   from src.division2.training.train_lora import train_chess_model
#   from src.division2.training.merge_adapters import merge_lora_adapters

__all__ = []
