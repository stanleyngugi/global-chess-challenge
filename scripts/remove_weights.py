#!/usr/bin/env python3
"""
Remove Sample Weights from Training Data

This script removes the "weight" field from all training samples to enable
sequence packing, which provides a 10x training speedup.

RATIONALE (from deep research):
- Packing concatenates multiple samples into single sequences for efficiency
- With packing, we can't apply per-sample scalar weights (they'd span multiple samples)
- We already duplicated opening samples 6x, so emphasis is already built in
- The 10x speedup (12h vs 120h) is worth removing explicit weights

Usage:
    python scripts/remove_weights.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def remove_weights_from_jsonl(input_path: str, output_path: str) -> dict:
    """
    Remove 'weight' field from all samples in a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        Statistics about the conversion
    """
    stats = {
        "total_samples": 0,
        "weights_removed": 0,
        "weight_distribution": {}
    }
    
    # Count lines first for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc=f"Processing {Path(input_path).name}"):
            line = line.strip()
            if not line:
                continue
                
            sample = json.loads(line)
            stats["total_samples"] += 1
            
            # Track weight distribution before removal
            if "weight" in sample:
                weight = sample["weight"]
                weight_str = str(weight)
                stats["weight_distribution"][weight_str] = \
                    stats["weight_distribution"].get(weight_str, 0) + 1
                stats["weights_removed"] += 1
                
                # Remove the weight field
                del sample["weight"]
            
            # Write the cleaned sample (text only)
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    return stats


def main():
    # Paths
    data_dir = Path("src/division2/data")
    
    # Input files (with spaces in names)
    train_input = data_dir / "train (1).jsonl"
    val_input = data_dir / "val (1).jsonl"
    
    # Output files (clean names, no weights)
    train_output = data_dir / "train.jsonl"
    val_output = data_dir / "val.jsonl"
    
    print("=" * 60)
    print("Removing Sample Weights for Packing Compatibility")
    print("=" * 60)
    print()
    print("REASON: Packing provides 10x speedup but requires no per-sample weights")
    print("        Opening emphasis is already achieved via 6x duplication")
    print()
    
    # Process training file
    if train_input.exists():
        print(f"Processing: {train_input}")
        train_stats = remove_weights_from_jsonl(str(train_input), str(train_output))
        
        print(f"\nTraining data statistics:")
        print(f"  Total samples: {train_stats['total_samples']:,}")
        print(f"  Weights removed: {train_stats['weights_removed']:,}")
        print(f"  Weight distribution:")
        for weight, count in sorted(train_stats['weight_distribution'].items()):
            pct = count / train_stats['total_samples'] * 100
            print(f"    {weight}: {count:,} ({pct:.1f}%)")
        print(f"\nOutput: {train_output}")
    else:
        print(f"WARNING: Training file not found: {train_input}")
    
    print()
    
    # Process validation file
    if val_input.exists():
        print(f"Processing: {val_input}")
        val_stats = remove_weights_from_jsonl(str(val_input), str(val_output))
        
        print(f"\nValidation data statistics:")
        print(f"  Total samples: {val_stats['total_samples']:,}")
        print(f"  Weights removed: {val_stats['weights_removed']:,}")
        print(f"\nOutput: {val_output}")
    else:
        print(f"WARNING: Validation file not found: {val_input}")
    
    print()
    print("=" * 60)
    print("Weight removal complete!")
    print("=" * 60)
    print()
    print("New files created (use these for training with packing=True):")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print()
    print("The original files with weights are preserved:")
    print(f"  - {train_input}")
    print(f"  - {val_input}")
    print()


if __name__ == "__main__":
    main()
