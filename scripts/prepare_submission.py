#!/usr/bin/env python3
"""
=============================================================================
Global Chess Challenge 2025 - Submission Preparation (Simplified)
=============================================================================

This script handles the final stages of the workflow:
1. Model training (LoRA fine-tuning)
2. Adapter merging (LoRA -> full model)  
3. Deployment packaging (ready for AIcrowd)

NOTE: Data generation is done via notebooks/colab_data_factory.ipynb
      and scripts/generate_opening_data.py (separately).

Usage:
    python scripts/prepare_submission.py --stage train --epochs 3
    python scripts/prepare_submission.py --stage merge
    python scripts/prepare_submission.py --stage package --hf-repo your-username/model
"""

import argparse
import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output"
SUBMISSION_DIR = PROJECT_ROOT / "submission"


class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_step(step: int, total: int, text: str):
    print(f"{Colors.GREEN}[{step}/{total}]{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}ERROR: {text}{Colors.RESET}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}WARNING: {text}{Colors.RESET}")


def print_success(text: str):
    print(f"{Colors.GREEN}SUCCESS: {text}{Colors.RESET}")


def run_command(cmd: list, cwd: Path = None, check: bool = True):
    cwd = cwd or PROJECT_ROOT
    print(f"{Colors.BLUE}Running: {' '.join(cmd)}{Colors.RESET}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=False, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        raise


def stage_train(args):
    """Stage: Train LoRA adapter."""
    print_header("Training LoRA Adapter")
    
    # Check for training data
    train_file = SRC_DIR / "division2" / "data" / "train (1).jsonl"
    if not train_file.exists():
        # Try alternative path
        train_file = PROJECT_ROOT / "data" / "train.jsonl"
        if not train_file.exists():
            print_error(f"Training data not found at {train_file}")
            print("Please run notebooks/colab_data_factory.ipynb first.")
            return False
    
    val_file = SRC_DIR / "division2" / "data" / "val (1).jsonl"
    
    # Build training command
    cmd = [
        sys.executable,
        "-m", "src.division2.training.train_lora",
        "--train_file", str(train_file),
        "--output_dir", str(OUTPUT_DIR / "chess-lora"),
    ]
    
    if val_file.exists():
        cmd.extend(["--val_file", str(val_file)])
    
    if args.epochs:
        cmd.extend(["--num_epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.base_model:
        cmd.extend(["--model_name", args.base_model])
    
    print_step(1, 1, f"Training for {args.epochs or 3} epochs...")
    print(f"  Training data: {train_file}")
    print(f"  Output: {OUTPUT_DIR / 'chess-lora'}")
    
    try:
        run_command(cmd)
        print_success("Training complete!")
        return True
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False


def stage_merge(args):
    """Stage: Merge LoRA adapter into base model."""
    print_header("Merging LoRA Adapter")
    
    adapter_path = OUTPUT_DIR / "chess-lora" / "final_adapter"
    if not adapter_path.exists():
        adapter_path = OUTPUT_DIR / "chess-lora"
    
    merged_path = OUTPUT_DIR / "chess-merged"
    
    if not adapter_path.exists():
        print_error(f"LoRA adapter not found at {adapter_path}")
        print("Run train stage first.")
        return False
    
    cmd = [
        sys.executable,
        "-m", "src.division2.training.merge_adapters",
        "--adapter_path", str(adapter_path),
        "--output_path", str(merged_path),
    ]
    
    if args.base_model:
        cmd.extend(["--base_model", args.base_model])
    if args.verify:
        cmd.append("--verify")
    
    print_step(1, 1, "Merging LoRA adapter into base model...")
    
    try:
        run_command(cmd)
        print_success(f"Model merged to {merged_path}")
        return True
    except Exception as e:
        print_error(f"Merge failed: {e}")
        return False


def stage_package(args):
    """Stage: Package for submission."""
    print_header("Creating Submission Package")
    
    merged_path = OUTPUT_DIR / "chess-merged"
    
    if not merged_path.exists():
        print_error(f"Merged model not found at {merged_path}")
        print("Run merge stage first.")
        return False
    
    # Create submission directory
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    print_step(1, 4, "Copying model files...")
    model_dest = SUBMISSION_DIR / "model"
    if model_dest.exists():
        shutil.rmtree(model_dest)
    shutil.copytree(merged_path, model_dest)
    
    print_step(2, 4, "Copying prompt template...")
    template_src = SRC_DIR / "templates" / "chess_agent_prompt.jinja"
    template_dest = SUBMISSION_DIR / "chess_agent_prompt.jinja"
    if template_src.exists():
        shutil.copy2(template_src, template_dest)
    else:
        print_warning(f"Template not found at {template_src}")
    
    print_step(3, 4, "Creating manifest...")
    manifest = {
        "created_at": datetime.now().isoformat(),
        "model_path": "./model",
        "prompt_template": "./chess_agent_prompt.jinja",
        "base_model": args.base_model or "meta-llama/Llama-3.1-8B-Instruct",
        "training_config": {
            "method": "LoRA",
            "rank": 64,
            "alpha": 128,
        },
        "competition": {
            "name": "Global Chess Challenge 2025",
            "hardware": "AWS Trainium trn1.2xlarge",
            "tensor_parallel_size": 2,
        },
        "important_notes": [
            "Only MODEL + TEMPLATE run in competition",
            "No custom Python code (opening books, etc.) runs",
            "Model must produce good moves from single inference"
        ]
    }
    
    with open(SUBMISSION_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print_step(4, 4, "Creating README...")
    readme = """# Submission Package - Global Chess Challenge 2025

## Files
- `model/` - Fine-tuned model weights (push to HuggingFace Hub)
- `chess_agent_prompt.jinja` - Prompt template (required for submission)
- `manifest.json` - Metadata

## Submission Commands

1. Push to HuggingFace:
   ```bash
   huggingface-cli upload YOUR_USERNAME/chess-model ./model
   ```

2. Submit to AIcrowd:
   ```bash
   aicrowd submit-model \\
       --challenge global-chess-challenge-2025 \\
       --hf-repo YOUR_USERNAME/chess-model \\
       --prompt_template_path ./chess_agent_prompt.jinja
   ```

## Important

- Only MODEL + TEMPLATE run in competition
- No custom Python code runs (opening books, best-of-N, etc.)
- Your model must produce valid UCI moves in `<uci_move>` tags
"""
    
    with open(SUBMISSION_DIR / "README.md", "w") as f:
        f.write(readme)
    
    print_success("Submission package created!")
    print(f"\nLocation: {SUBMISSION_DIR}")
    print("\nNext steps:")
    print(f"  1. huggingface-cli upload YOUR_USER/chess-model {model_dest}")
    print("  2. aicrowd submit-model --challenge global-chess-challenge-2025 ...")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Submission Preparation")
    
    parser.add_argument(
        "--stage",
        choices=["train", "merge", "package", "all"],
        default="all",
        help="Which stage to run"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--hf-repo", type=str)
    
    args = parser.parse_args()
    
    print_header("Global Chess Challenge 2025 - Submission Preparation")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stages = {
        "train": stage_train,
        "merge": stage_merge,
        "package": stage_package,
    }
    
    if args.stage == "all":
        stage_order = ["train", "merge", "package"]
    else:
        stage_order = [args.stage]
    
    for stage_name in stage_order:
        success = stages[stage_name](args)
        if not success:
            print_error(f"Stage '{stage_name}' failed")
            return 1
    
    print_header("Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
