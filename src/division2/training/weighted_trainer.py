"""
Weighted Trainer Module - DEPRECATED

NOTE: This module is DEPRECATED in favor of TRL's SFTTrainer with packing=True.

RESEARCH FINDING:
- Packing provides 10x training speedup (12h vs 120h)
- Packing is INCOMPATIBLE with per-sample weights
- We achieved opening emphasis via 6x sample duplication instead

The new training pipeline uses:
- SFTTrainer from TRL library
- packing=True for efficiency
- No per-sample weights

This file is kept for reference only.
For the production training script, see: train_lora.py
"""

# This module is deprecated - see train_lora.py for production training
raise DeprecationWarning(
    "WeightedSFTTrainer is deprecated. "
    "Use SFTTrainer with packing=True instead (see train_lora.py)"
)
