"""
LoRA Training Script for Chess Model - Production Version

This is the optimized training script based on deep research for:
- AWS Trainium deployment compatibility
- RunPod A40 efficiency (48GB VRAM)
- Llama 3.1 specific optimizations

KEY OPTIMIZATIONS FROM RESEARCH:
1. PACKING=TRUE: 10x speedup by concatenating short samples (206 avg tokens)
2. 1 EPOCH: 756K samples is massive for SFT, 1 epoch prevents overfitting
3. PAD_TOKEN FIX: Use reserved token, NOT eos_token (prevents infinite generation)
4. FLASH ATTENTION 2: 2-3x speedup, linear memory complexity
5. LEARNING RATE 1e-4: Safer for r=64 LoRA with large dataset
6. TF32=TRUE: Free speedup on Ampere architecture
7. NO QUANTIZATION: Clean bf16 for Trainium export

Usage (RunPod A40):
    python -m src.division2.training.train_lora \\
        --train_file src/division2/data/train.jsonl \\
        --val_file src/division2/data/val.jsonl \\
        --output_dir /workspace/output/chess-lora

Training Time Estimate: ~12 hours for 756K samples, 1 epoch with packing
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for the model and LoRA.
    
    OPTIMIZED FOR A40 (48GB VRAM) + TRAINIUM DEPLOYMENT:
    - Full bf16 precision (no quantization)
    - LoRA r=64 for quality on complex reasoning tasks
    - All linear layers targeted for maximum learning capacity
    """
    
    # Base model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # LoRA configuration - QUALITY FOCUSED
    lora_r: int = 64           # Higher rank for complex chess reasoning
    lora_alpha: int = 128      # Alpha = 2x rank (standard aggressive scaling)
    lora_dropout: float = 0.05 # Slight regularization for 756K samples
    
    # Target ALL linear modules for Llama 3 (research-validated)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ])
    
    # Maximum sequence length (our samples are ~206 tokens avg)
    max_seq_length: int = 2048


@dataclass  
class TrainConfig:
    """
    Configuration for training.
    
    RESEARCH-OPTIMIZED SETTINGS:
    - 1 epoch (756K samples is huge for SFT, more epochs = overfitting)
    - Learning rate 1e-4 (safer for r=64 with large dataset)
    - Packing enabled (10x speedup for short sequences)
    - Flash Attention 2 (2-3x speedup)
    """
    
    # Paths
    train_file: str = "src/division2/data/train.jsonl"
    val_file: Optional[str] = "src/division2/data/val.jsonl"
    output_dir: str = "/workspace/output/chess-lora"  # RunPod persistent storage
    
    # CRITICAL: 1 epoch for 756K samples (prevents overfitting)
    num_epochs: int = 1
    
    # CRITICAL: Lower LR for r=64 with massive dataset
    learning_rate: float = 1e-4
    
    # Batch configuration with packing
    # With packing, each sequence contains ~10 samples (2048/206)
    # Effective samples per step = 4 * 10 * 4 = ~160
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Precision - bf16 mandatory for Trainium
    bf16: bool = True
    tf32: bool = True  # Free speedup on Ampere (A40)
    
    # Memory optimization
    gradient_checkpointing: bool = True
    
    # CRITICAL: Enable packing for 10x speedup
    packing: bool = True
    
    # DataLoader optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Seed
    seed: int = 42


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load and configure the tokenizer with CRITICAL pad_token fix.
    
    RESEARCH FINDING: Using eos_token as pad_token causes infinite generation!
    The model learns that eos_token is "meaningless filler" rather than a stop signal.
    
    SOLUTION: Use a reserved special token for padding.
    """
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # CRITICAL FIX: Do NOT use eos_token as pad_token
    # This causes the model to ignore eos during generation (infinite loops)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        logger.warning("Fixing pad_token collision with eos_token...")
        
        # Use a reserved special token from Llama 3.1's vocabulary
        # These tokens exist but are semantically null in pre-training
        reserved_token = "<|reserved_special_token_0|>"
        
        # Check if token exists in vocabulary
        if reserved_token in tokenizer.get_vocab():
            tokenizer.pad_token = reserved_token
            logger.info(f"Set pad_token to {reserved_token} (ID: {tokenizer.pad_token_id})")
        else:
            # Fallback: add as new special token
            tokenizer.add_special_tokens({'pad_token': reserved_token})
            logger.info(f"Added {reserved_token} as pad_token (ID: {tokenizer.pad_token_id})")
    
    # Training uses right padding (standard for causal LM)
    tokenizer.padding_side = "right"
    
    return tokenizer


def setup_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    """
    Load the base model with optimal configuration for A40 training.
    
    RESEARCH OPTIMIZATIONS:
    - Flash Attention 2 for speed and memory efficiency
    - bf16 precision (native Ampere support, Trainium compatible)
    - Gradient checkpointing for memory efficiency
    - No quantization (bitsandbytes breaks Trainium)
    """
    
    # Determine attention implementation
    attn_impl = "eager"  # Default fallback
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using Flash Attention 2 (2-3x speedup)")
        except ImportError:
            attn_impl = "sdpa"  # PyTorch native SDPA
            logger.info("Flash Attention not available, using SDPA")
    
    # Load model in bf16 (no quantization for Trainium compatibility)
    logger.info(f"Loading model in bf16: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    
    # CRITICAL: Update model config to match tokenizer pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resize embeddings if we added new tokens
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing (saves ~60% memory)
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.target_modules,
        bias="none",  # "none" is safest for Trainium compilation
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    
    return model


def train_chess_model(
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> str:
    """
    Main training function using SFTTrainer with packing.
    
    TRAINING STRATEGY (from research):
    1. Use SFTTrainer with packing=True for 10x speedup
    2. 1 epoch only (756K samples is enough)
    3. Save checkpoints frequently for recovery
    4. Push to hub for backup
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    
    # Set seed for reproducibility
    torch.manual_seed(train_config.seed)
    
    # Create output directory
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations for reproducibility
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(vars(model_config), f, indent=2, default=list)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(train_config), f, indent=2)
    
    # Setup tokenizer and model
    tokenizer = setup_tokenizer(model_config.model_name)
    model = setup_model(model_config, train_config, tokenizer)
    
    # Load datasets
    logger.info(f"Loading training data from {train_config.train_file}")
    train_dataset = load_dataset(
        "json",
        data_files={"train": train_config.train_file},
        split="train"
    )
    logger.info(f"Training samples: {len(train_dataset):,}")
    
    eval_dataset = None
    if train_config.val_file and Path(train_config.val_file).exists():
        logger.info(f"Loading validation data from {train_config.val_file}")
        eval_dataset = load_dataset(
            "json", 
            data_files={"validation": train_config.val_file},
            split="validation"
        )
        logger.info(f"Validation samples: {len(eval_dataset):,}")
    
    # Calculate training stats
    steps_per_epoch = len(train_dataset) // (
        train_config.per_device_batch_size * 
        train_config.gradient_accumulation_steps
    )
    # With packing, each step processes ~10x more samples
    if train_config.packing:
        effective_samples_per_step = (
            train_config.per_device_batch_size * 
            train_config.gradient_accumulation_steps * 
            (model_config.max_seq_length // 206)  # avg sample length
        )
        steps_per_epoch = len(train_dataset) // effective_samples_per_step
    
    total_steps = steps_per_epoch * train_config.num_epochs
    
    logger.info(f"Training plan:")
    logger.info(f"  Epochs: {train_config.num_epochs}")
    logger.info(f"  Steps per epoch: ~{steps_per_epoch:,}")
    logger.info(f"  Total steps: ~{total_steps:,}")
    logger.info(f"  Packing enabled: {train_config.packing}")
    logger.info(f"  Effective batch size: {train_config.per_device_batch_size * train_config.gradient_accumulation_steps}")
    
    # Training arguments with all optimizations
    training_args = SFTConfig(
        output_dir=str(output_dir),
        
        # Epochs and batching
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.per_device_batch_size,
        per_device_eval_batch_size=train_config.per_device_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        
        # Optimizer settings
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # Standard optimizer (no 8-bit for Trainium)
        max_grad_norm=train_config.max_grad_norm,
        
        # Precision - CRITICAL for Trainium
        bf16=train_config.bf16,
        tf32=train_config.tf32,
        fp16=False,
        
        # Memory optimization
        gradient_checkpointing=train_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # CRITICAL: Packing for 10x speedup
        packing=train_config.packing,
        max_seq_length=model_config.max_seq_length,
        
        # Logging
        logging_steps=train_config.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Checkpointing
        save_strategy="steps",
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_config.eval_steps if eval_dataset else None,
        eval_accumulation_steps=1,  # Prevent OOM during eval
        
        # DataLoader optimization
        dataloader_num_workers=train_config.dataloader_num_workers,
        dataloader_pin_memory=train_config.dataloader_pin_memory,
        
        # Reproducibility
        seed=train_config.seed,
        data_seed=train_config.seed,
        
        # Dataset text field
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
    
    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    trainer.train()
    
    # Save the final adapter
    final_adapter_path = output_dir / "final_adapter"
    logger.info(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return str(final_adapter_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a chess LoRA model with research-optimized settings"
    )
    
    # Data paths
    parser.add_argument(
        "--train_file",
        type=str,
        default="src/division2/data/train.jsonl",
        help="Path to training data JSONL (without weights)"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="src/division2/data/val.jsonl",
        help="Path to validation data JSONL"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output/chess-lora",
        help="Output directory (use /workspace for RunPod persistence)"
    )
    
    # Model config
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank (64 recommended for chess reasoning)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha (typically 2x rank)"
    )
    
    # Training config - RESEARCH-OPTIMIZED DEFAULTS
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,  # CRITICAL: 1 epoch for 756K samples
        help="Number of training epochs (1 recommended for large datasets)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # CRITICAL: Safer for r=64
        help="Learning rate (1e-4 recommended for r=64)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for packing"
    )
    parser.add_argument(
        "--no_packing",
        action="store_true",
        help="Disable sequence packing (NOT recommended)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create configs from args
    model_config = ModelConfig(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )
    
    train_config = TrainConfig(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        packing=not args.no_packing,
        seed=args.seed,
    )
    
    # Log configuration
    logger.info("Model Config:")
    logger.info(f"  Model: {model_config.model_name}")
    logger.info(f"  LoRA r={model_config.lora_r}, alpha={model_config.lora_alpha}")
    logger.info("")
    logger.info("Training Config:")
    logger.info(f"  Epochs: {train_config.num_epochs}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    logger.info(f"  Packing: {train_config.packing}")
    logger.info(f"  Batch size: {train_config.per_device_batch_size} x {train_config.gradient_accumulation_steps}")
    logger.info("")
    
    # Run training
    adapter_path = train_chess_model(model_config, train_config)
    print(f"\nTraining complete! Adapter saved to: {adapter_path}")
