"""
Training module for Division 2 chess model fine-tuning.

Pipeline:
---------
1. train_lora.py     - LoRA fine-tuning with SFTTrainer + packing
2. merge_adapters.py - Merge LoRA into base model for deployment

Note: WeightedSFTTrainer is deprecated in favor of SFTTrainer with packing=True.
      Per-sample weighting is incompatible with packing; we use sample duplication instead.
"""

from .train_lora import train_chess_model
from .merge_adapters import merge_lora_adapters

__all__ = [
    "train_chess_model",
    "merge_lora_adapters",
]
