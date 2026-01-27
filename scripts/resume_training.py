#!/usr/bin/env python3
"""
Resume Training from Checkpoint

This script resumes training from a saved checkpoint after:
- SSH disconnection
- Pod restart
- Training crash
- Manual interruption (Ctrl+C)

It handles the PEFT-specific quirks of checkpoint loading and validates
the checkpoint before resuming.

Usage:
    # Auto-find latest checkpoint
    python scripts/resume_training.py
    
    # Resume from specific checkpoint
    python scripts/resume_training.py --checkpoint /workspace/output/chess-lora/checkpoint-2500
    
    # Resume with different output dir
    python scripts/resume_training.py --output_dir /workspace/output/chess-lora-v2

IMPORTANT: This script is designed to work with the train_lora.py configuration.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest valid checkpoint in the output directory."""
    checkpoints = []
    
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                # Validate it has required files
                if (item / "adapter_model.safetensors").exists():
                    checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Return checkpoint with highest step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def validate_checkpoint(checkpoint_dir: Path) -> bool:
    """Validate checkpoint has all required files."""
    required_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
    ]
    
    for fname in required_files:
        if not (checkpoint_dir / fname).exists():
            logger.error(f"Missing required file: {fname}")
            return False
    
    # Check for NaN in adapter weights
    try:
        from safetensors import safe_open
        adapter_path = checkpoint_dir / "adapter_model.safetensors"
        with safe_open(str(adapter_path), framework="pt") as f:
            for key in list(f.keys())[:5]:
                tensor = f.get_tensor(key)
                if torch.isnan(tensor).any():
                    logger.error(f"NaN detected in checkpoint weights: {key}")
                    return False
    except Exception as e:
        logger.warning(f"Could not validate weights: {e}")
    
    return True


def get_checkpoint_info(checkpoint_dir: Path) -> dict:
    """Get training state from checkpoint."""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    
    if not trainer_state_path.exists():
        return {}
    
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    return {
        "global_step": state.get("global_step", 0),
        "epoch": state.get("epoch", 0),
        "total_flos": state.get("total_flos", 0),
    }


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer with proper pad_token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Fix pad_token collision (critical for Llama 3.1)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        reserved_token = "<|reserved_special_token_0|>"
        if reserved_token in tokenizer.get_vocab():
            tokenizer.pad_token = reserved_token
        else:
            tokenizer.add_special_tokens({'pad_token': reserved_token})
    
    tokenizer.padding_side = "right"
    return tokenizer


def resume_training(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "/workspace/output/chess-lora",
    train_file: str = "src/division2/data/train.jsonl",
    val_file: str = "src/division2/data/val.jsonl",
):
    """Resume training from checkpoint."""
    
    output_dir = Path(output_dir)
    
    # Find checkpoint
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
    else:
        logger.info(f"Looking for checkpoints in {output_dir}...")
        checkpoint_dir = find_latest_checkpoint(output_dir)
    
    if checkpoint_dir is None:
        logger.error("No checkpoint found!")
        logger.error("Cannot resume - you need to start training from scratch.")
        logger.error("Run: bash scripts/train_a100.sh")
        sys.exit(1)
    
    logger.info(f"Found checkpoint: {checkpoint_dir}")
    
    # Validate checkpoint
    if not validate_checkpoint(checkpoint_dir):
        logger.error("Checkpoint validation failed!")
        sys.exit(1)
    
    # Get checkpoint info
    info = get_checkpoint_info(checkpoint_dir)
    logger.info(f"Checkpoint info:")
    logger.info(f"  Global step: {info.get('global_step', 'unknown')}")
    logger.info(f"  Epoch: {info.get('epoch', 'unknown')}")
    
    # Configuration (must match original training)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    lora_r = 64
    lora_alpha = 128
    lora_dropout = 0.05
    max_seq_length = 2048
    
    # Training config
    num_epochs = 1
    learning_rate = 1e-4
    per_device_batch_size = 4
    gradient_accumulation_steps = 4
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = setup_tokenizer(model_name)
    
    # Determine attention implementation
    attn_impl = "eager"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Using SDPA")
    
    # Load base model
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    
    # Update model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Load the LoRA adapter from checkpoint
    logger.info(f"Loading LoRA adapter from checkpoint...")
    model = PeftModel.from_pretrained(
        model,
        checkpoint_dir,
        is_trainable=True,  # CRITICAL: Must be True to resume training
    )
    
    # Verify adapter is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    if trainable_params == 0:
        logger.error("No trainable parameters! Adapter loading failed.")
        sys.exit(1)
    
    # Load datasets
    logger.info(f"Loading training data from {train_file}")
    train_dataset = load_dataset(
        "json",
        data_files={"train": train_file},
        split="train"
    )
    logger.info(f"Training samples: {len(train_dataset):,}")
    
    eval_dataset = None
    if Path(val_file).exists():
        logger.info(f"Loading validation data from {val_file}")
        eval_dataset = load_dataset(
            "json",
            data_files={"validation": val_file},
            split="validation"
        )
        logger.info(f"Validation samples: {len(eval_dataset):,}")
    
    # Training arguments (must match original)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        
        # Epochs and batching
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Optimizer settings
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,
        tf32=True,
        fp16=False,
        
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Packing
        packing=True,
        max_seq_length=max_seq_length,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        eval_accumulation_steps=1,
        
        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Seed
        seed=42,
        data_seed=42,
        
        # Dataset field
        dataset_text_field="text",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Resume training
    logger.info("=" * 60)
    logger.info(f"RESUMING TRAINING FROM CHECKPOINT")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Step: {info.get('global_step', 'unknown')}")
    logger.info("=" * 60)
    
    # The key parameter: resume_from_checkpoint
    trainer.train(resume_from_checkpoint=str(checkpoint_dir))
    
    # Save final adapter
    final_adapter_path = output_dir / "final_adapter"
    logger.info(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return str(final_adapter_path)


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (auto-detects latest if not specified)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output/chess-lora",
        help="Training output directory"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="src/division2/data/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="src/division2/data/val.jsonl",
        help="Path to validation data"
    )
    
    args = parser.parse_args()
    
    adapter_path = resume_training(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        train_file=args.train_file,
        val_file=args.val_file,
    )
    
    print(f"\nTraining complete! Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
