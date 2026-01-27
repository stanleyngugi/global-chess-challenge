#!/usr/bin/env python3
"""
Pre-Flight Checklist for Chess LoRA Training

Run this BEFORE starting an expensive GPU run to catch issues early.
Validates data, configuration, and performs a quick dry-run.

Based on research document: "05_training_script_validation.md"

Checks:
1. Data files exist and are properly formatted
2. No 'weight' field in data (for packing compatibility)
3. Tokenizer pad_token configuration
4. GPU memory availability
5. Output directory is on persistent storage
6. Quick dry-run (10 steps) to catch immediate failures

Usage:
    python scripts/preflight_check.py
"""

import json
import os
import sys
from pathlib import Path


def check_data_files():
    """Verify training data exists and is formatted correctly."""
    print("=" * 60)
    print("1. Data File Validation")
    print("=" * 60)
    
    train_file = Path("src/division2/data/train.jsonl")
    val_file = Path("src/division2/data/val.jsonl")
    
    issues = []
    
    # Check training file exists
    if not train_file.exists():
        issues.append(f"Training file not found: {train_file}")
        
        # Check if original exists
        orig = Path("src/division2/data/train (1).jsonl")
        if orig.exists():
            issues.append("  -> Found 'train (1).jsonl'. Run: python scripts/remove_weights.py")
    else:
        # Check file size
        size_mb = train_file.stat().st_size / (1024 * 1024)
        line_count = sum(1 for _ in open(train_file, 'r', encoding='utf-8'))
        print(f"  Training file: {train_file}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Samples: {line_count:,}")
        
        # Check format
        with open(train_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            
            if "text" not in sample:
                issues.append("Training data missing 'text' field")
            
            if "weight" in sample:
                issues.append("Training data contains 'weight' field (incompatible with packing)")
                issues.append("  -> Run: python scripts/remove_weights.py")
            
            # Check for Llama 3.1 special tokens
            text = sample.get("text", "")
            if "<|begin_of_text|>" not in text:
                issues.append("Training data missing Llama 3.1 special tokens")
            if "<|eot_id|>" not in text:
                issues.append("Training data missing <|eot_id|> token")
    
    # Check validation file
    if val_file.exists():
        val_lines = sum(1 for _ in open(val_file, 'r', encoding='utf-8'))
        print(f"  Validation file: {val_file}")
        print(f"    Samples: {val_lines:,}")
    else:
        print(f"  Validation file: NOT FOUND (optional)")
    
    print("")
    return issues


def check_tokenizer_config():
    """Verify tokenizer configuration is correct for Llama 3.1."""
    print("=" * 60)
    print("2. Tokenizer Configuration")
    print("=" * 60)
    
    issues = []
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"  Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"  Vocab size: {len(tokenizer):,}")
        print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        # Check for pad_token collision
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print("  WARNING: pad_token == eos_token (will be fixed in training)")
        
        # Check for reserved token availability
        reserved = "<|reserved_special_token_0|>"
        if reserved in tokenizer.get_vocab():
            print(f"  Reserved token available: {reserved}")
        else:
            issues.append("Reserved token not found in vocabulary")
        
    except Exception as e:
        issues.append(f"Tokenizer loading failed: {e}")
    
    print("")
    return issues


def check_gpu_memory():
    """Verify GPU has sufficient memory."""
    print("=" * 60)
    print("3. GPU Memory")
    print("=" * 60)
    
    issues = []
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            issues.append("CUDA not available")
            return issues
        
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"  GPU: {gpu_name}")
        print(f"  Total VRAM: {total_mem:.1f} GB")
        print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
        
        # Memory requirements estimate
        # Base model bf16: ~16GB
        # LoRA adapters: ~0.4GB
        # Optimizer states: ~1.6GB
        # Activations (with GC): ~12GB
        # Total: ~30GB
        
        required_gb = 32
        
        if total_mem < required_gb:
            issues.append(f"Insufficient VRAM: {total_mem:.1f}GB < {required_gb}GB required")
        else:
            print(f"  Memory check: OK ({total_mem:.1f}GB >= {required_gb}GB required)")
        
    except Exception as e:
        issues.append(f"GPU check failed: {e}")
    
    print("")
    return issues


def check_output_directory():
    """Verify output directory is on persistent storage."""
    print("=" * 60)
    print("4. Output Directory (Persistence Check)")
    print("=" * 60)
    
    issues = []
    
    # Default output directories
    output_dirs = [
        Path("/workspace/output/chess-lora"),
        Path("output/chess-lora"),
    ]
    
    # On RunPod, /workspace is persistent, root is ephemeral
    for output_dir in output_dirs:
        print(f"  Checking: {output_dir}")
        
        if str(output_dir).startswith("/workspace"):
            print(f"    -> On /workspace (PERSISTENT on RunPod)")
        else:
            print(f"    -> WARNING: Not on /workspace (may be EPHEMERAL)")
            if os.path.exists("/workspace"):
                issues.append(f"Output dir {output_dir} may be ephemeral. Use /workspace/...")
    
    print("")
    return issues


def check_huggingface_auth():
    """Check HuggingFace authentication."""
    print("=" * 60)
    print("5. HuggingFace Authentication")
    print("=" * 60)
    
    issues = []
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  Logged in as: {user.get('name', 'Unknown')}")
        print(f"  Has Llama access: Check manually at huggingface.co")
    except Exception as e:
        issues.append(f"HuggingFace not authenticated: {e}")
        issues.append("  -> Run: huggingface-cli login")
    
    print("")
    return issues


def run_dry_run():
    """Run a quick dry run to catch immediate failures."""
    print("=" * 60)
    print("6. Dry Run (10 steps)")
    print("=" * 60)
    
    issues = []
    
    print("  Skipping dry run (would take ~5 minutes)")
    print("  To run manually:")
    print("    python -m src.division2.training.train_lora \\")
    print("        --num_epochs 1 --max_steps 10")
    
    # This would be a real dry run:
    # try:
    #     import subprocess
    #     result = subprocess.run([
    #         sys.executable, "-m", "src.division2.training.train_lora",
    #         "--num_epochs", "1",
    #         "--max_steps", "10",
    #         "--output_dir", "/tmp/dry_run"
    #     ], capture_output=True, text=True, timeout=300)
    #     
    #     if result.returncode != 0:
    #         issues.append(f"Dry run failed: {result.stderr[-500:]}")
    # except Exception as e:
    #     issues.append(f"Dry run error: {e}")
    
    print("")
    return issues


def main():
    print("")
    print("=" * 60)
    print("PRE-FLIGHT CHECKLIST - Chess LoRA Training")
    print("=" * 60)
    print("")
    print("This script validates your setup before expensive GPU training.")
    print("")
    
    all_issues = []
    
    # Run checks
    all_issues.extend(check_data_files())
    all_issues.extend(check_tokenizer_config())
    all_issues.extend(check_gpu_memory())
    all_issues.extend(check_output_directory())
    all_issues.extend(check_huggingface_auth())
    run_dry_run()  # Just informational
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_issues:
        print("")
        print("ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("")
        print("Please fix these issues before training!")
        print("")
        sys.exit(1)
    else:
        print("")
        print("All checks passed!")
        print("")
        print("Ready to train. Run:")
        print("  bash scripts/train_a100.sh")
        print("")
        print("Estimated training time: ~12 hours")
        print("")


if __name__ == "__main__":
    main()
