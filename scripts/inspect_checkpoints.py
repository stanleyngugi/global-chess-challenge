#!/usr/bin/env python3
"""
Checkpoint Inspector - View and Validate Training Checkpoints

This script inspects saved checkpoints to help you:
1. Find the latest valid checkpoint
2. Verify checkpoint integrity (adapter weights, optimizer state)
3. See training progress (step, loss, learning rate)
4. Decide whether to resume or start fresh

Usage:
    python scripts/inspect_checkpoints.py --output_dir /workspace/output/chess-lora
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
import sys


def find_checkpoints(output_dir: Path) -> List[Path]:
    """Find all checkpoint directories sorted by step number."""
    checkpoints = []
    
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return [cp[1] for cp in checkpoints]


def validate_checkpoint(checkpoint_dir: Path) -> Dict:
    """Validate a checkpoint directory and return its status."""
    result = {
        "path": str(checkpoint_dir),
        "step": int(checkpoint_dir.name.split("-")[1]),
        "valid": True,
        "issues": [],
        "files": {},
        "training_state": None,
    }
    
    # Required files for PEFT checkpoint
    required_files = [
        "adapter_model.safetensors",  # LoRA weights
        "adapter_config.json",         # LoRA config
        "trainer_state.json",          # Training progress
    ]
    
    # Optional but important files
    optional_files = [
        "optimizer.pt",               # Optimizer state
        "scheduler.pt",               # LR scheduler state
        "rng_state.pth",             # Random state for reproducibility
    ]
    
    # Check required files
    for fname in required_files:
        fpath = checkpoint_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            result["files"][fname] = f"{size_kb:.1f} KB"
        else:
            result["valid"] = False
            result["issues"].append(f"Missing required file: {fname}")
    
    # Check optional files
    for fname in optional_files:
        fpath = checkpoint_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            result["files"][fname] = f"{size_mb:.1f} MB"
        else:
            result["issues"].append(f"Missing optional file: {fname} (resume may not be exact)")
    
    # Parse trainer state
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path) as f:
                state = json.load(f)
            
            result["training_state"] = {
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                "total_flos": state.get("total_flos"),
                "best_metric": state.get("best_metric"),
                "log_history": state.get("log_history", [])[-3:],  # Last 3 log entries
            }
        except Exception as e:
            result["issues"].append(f"Could not parse trainer_state.json: {e}")
    
    # Check adapter weights for NaN (quick check)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        try:
            from safetensors import safe_open
            with safe_open(adapter_path, framework="pt") as f:
                # Check first tensor for NaN
                keys = list(f.keys())[:5]  # Check first 5 tensors
                import torch
                for key in keys:
                    tensor = f.get_tensor(key)
                    if torch.isnan(tensor).any():
                        result["valid"] = False
                        result["issues"].append(f"NaN detected in adapter weights: {key}")
                        break
        except Exception as e:
            result["issues"].append(f"Could not validate adapter weights: {e}")
    
    return result


def get_last_loss_from_state(training_state: Dict) -> Optional[float]:
    """Extract the last training loss from log history."""
    if not training_state or "log_history" not in training_state:
        return None
    
    log_history = training_state.get("log_history", [])
    for entry in reversed(log_history):
        if "loss" in entry:
            return entry["loss"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect training checkpoints")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output/chess-lora",
        help="Training output directory"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("CHECKPOINT INSPECTOR")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    
    if not output_dir.exists():
        print(f"\nERROR: Directory does not exist: {output_dir}")
        print("\nNo training has been started yet, or output is in a different location.")
        sys.exit(1)
    
    # Find checkpoints
    checkpoints = find_checkpoints(output_dir)
    
    if not checkpoints:
        print("\nNo checkpoints found!")
        print("\nThis means either:")
        print("  1. Training hasn't started yet")
        print("  2. Training crashed before first checkpoint (step 500)")
        print("  3. Checkpoints were saved to a different directory")
        
        # Check for final adapter
        final_adapter = output_dir / "final_adapter"
        if final_adapter.exists():
            print(f"\nBUT: Found final_adapter at {final_adapter}")
            print("     Training appears to have COMPLETED successfully!")
        
        sys.exit(0)
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    print("-" * 70)
    
    valid_checkpoints = []
    
    for checkpoint_dir in checkpoints:
        result = validate_checkpoint(checkpoint_dir)
        
        step = result["step"]
        status = "VALID" if result["valid"] else "INVALID"
        
        print(f"\n[{status}] checkpoint-{step}")
        print(f"  Path: {result['path']}")
        
        # Show files
        print(f"  Files:")
        for fname, size in result["files"].items():
            print(f"    - {fname}: {size}")
        
        # Show training state
        if result["training_state"]:
            state = result["training_state"]
            loss = get_last_loss_from_state(state)
            print(f"  Training State:")
            print(f"    - Global step: {state['global_step']}")
            print(f"    - Epoch: {state['epoch']:.4f}")
            if loss:
                print(f"    - Last loss: {loss:.4f}")
        
        # Show issues
        if result["issues"]:
            print(f"  Issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")
        
        if result["valid"]:
            valid_checkpoints.append(result)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if valid_checkpoints:
        latest = valid_checkpoints[-1]
        print(f"\nLatest valid checkpoint: checkpoint-{latest['step']}")
        
        if latest["training_state"]:
            step = latest["training_state"]["global_step"]
            epoch = latest["training_state"]["epoch"]
            
            # Estimate progress (assuming ~4700 total steps)
            estimated_total = 4700
            progress = min(100, step / estimated_total * 100)
            
            print(f"  Step: {step} / ~{estimated_total} ({progress:.1f}% complete)")
            print(f"  Epoch: {epoch:.4f}")
            
            loss = get_last_loss_from_state(latest["training_state"])
            if loss:
                print(f"  Loss: {loss:.4f}")
        
        print(f"\nTo resume training from this checkpoint:")
        print(f"  python scripts/resume_training.py --checkpoint {latest['path']}")
    else:
        print("\nNo valid checkpoints found!")
        print("You will need to restart training from scratch.")
    
    # Check for final adapter
    final_adapter = output_dir / "final_adapter"
    if final_adapter.exists():
        print(f"\nNOTE: final_adapter exists at {final_adapter}")
        print("      Training may have already completed!")


if __name__ == "__main__":
    main()
