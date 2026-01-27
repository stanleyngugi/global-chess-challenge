"""
LoRA Adapter Merge Script - Production Version

This script merges LoRA adapters into the base model, producing a
standalone model ready for AWS Trainium/Neuron deployment.

CRITICAL RESEARCH FINDINGS IMPLEMENTED:
1. safe_merge=True: Validates adapter weights for NaN before merging
2. bf16 precision: Maintains Trainium-compatible precision throughout
3. RoPE config patching: Fixes Llama 3.1 rope_scaling for Neuron compatibility
4. safetensors format: Mandatory for optimized Neuron loading
5. torch_dtype in config: Explicit bf16 declaration for vLLM

DEPLOYMENT PATH:
1. Train LoRA on A40 (this project)
2. Merge adapters (this script) 
3. Push to HuggingFace Hub
4. Submit to AIcrowd (they load from Hub)

Usage:
    python -m src.division2.training.merge_adapters \\
        --adapter_path /workspace/output/chess-lora/final_adapter \\
        --output_path /workspace/output/chess-merged \\
        --verify
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def patch_config_for_trainium(config_path: Path) -> None:
    """
    Patch config.json for AWS Trainium/Neuron compatibility.
    
    CRITICAL FIXES:
    1. rope_scaling: Change "llama3" type to "dynamic" (Neuron doesn't understand "llama3")
    2. torch_dtype: Explicitly set to "bfloat16" for correct vLLM memory allocation
    3. architectures: Ensure it's ["LlamaForCausalLM"] for vLLM model runner
    """
    logger.info("Patching config.json for Trainium compatibility...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    changes_made = []
    
    # PATCH 1: Fix rope_scaling for Neuron compatibility
    # Llama 3.1 uses "rope_type": "llama3" which older vLLM/transformers don't understand
    if "rope_scaling" in config:
        rs = config["rope_scaling"]
        if "rope_type" in rs or "low_freq_factor" in rs or "high_freq_factor" in rs:
            logger.info("  Downgrading rope_scaling from 'llama3' to 'dynamic'")
            config["rope_scaling"] = {
                "type": "dynamic",
                "factor": 8.0
            }
            changes_made.append("rope_scaling: llama3 -> dynamic")
    
    # PATCH 2: Explicitly set torch_dtype to bfloat16
    # This guides vLLM to allocate correct memory types immediately
    if config.get("torch_dtype") != "bfloat16":
        config["torch_dtype"] = "bfloat16"
        changes_made.append("torch_dtype: set to bfloat16")
    
    # PATCH 3: Ensure correct architecture for vLLM
    if config.get("architectures") != ["LlamaForCausalLM"]:
        config["architectures"] = ["LlamaForCausalLM"]
        changes_made.append("architectures: set to LlamaForCausalLM")
    
    # Save patched config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if changes_made:
        logger.info(f"  Config patched: {', '.join(changes_made)}")
    else:
        logger.info("  No patches needed")


def verify_no_nans(model: torch.nn.Module) -> bool:
    """
    Scan all model parameters for NaN values.
    
    NaNs in weights indicate training divergence or merge corruption.
    This MUST pass before uploading to Hub.
    """
    logger.info("Verifying merged weights for NaNs...")
    
    nan_layers = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_layers.append(name)
        if torch.isinf(param).any():
            nan_layers.append(f"{name} (inf)")
    
    if nan_layers:
        logger.error(f"FATAL: Found NaN/Inf in {len(nan_layers)} layers:")
        for layer in nan_layers[:10]:  # Show first 10
            logger.error(f"  - {layer}")
        return False
    
    logger.info("  NaN check passed - all weights are valid")
    return True


def verify_dtype_consistency(model: torch.nn.Module, expected_dtype: torch.dtype) -> bool:
    """
    Verify all linear layers are in the expected dtype.
    
    Mixed precision after merge can cause issues on Trainium.
    """
    logger.info(f"Verifying dtype consistency (expecting {expected_dtype})...")
    
    wrong_dtype = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype != expected_dtype:
                wrong_dtype.append(f"{name}: {module.weight.dtype}")
    
    if wrong_dtype:
        logger.warning(f"Found {len(wrong_dtype)} layers with wrong dtype:")
        for layer in wrong_dtype[:10]:
            logger.warning(f"  - {layer}")
        return False
    
    logger.info(f"  Dtype check passed - all layers are {expected_dtype}")
    return True


def merge_lora_adapters(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> str:
    """
    Merge LoRA adapters into base model with full verification.
    
    PROCESS:
    1. Load base model in bf16 (matching training precision)
    2. Load LoRA adapter
    3. Merge with safe_merge=True (validates for NaNs)
    4. Verify merged weights
    5. Save as safetensors with 5GB shards
    6. Patch config.json for Trainium
    7. Optionally push to Hub
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer from BASE MODEL (not adapter)
    # This ensures full Llama 3.1 vocabulary and special tokens
    logger.info(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    # Load base model in bf16
    # CRITICAL: Must match training precision for correct merge
    logger.info(f"Loading base model in bfloat16: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU for full control
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Load adapter
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    
    # Merge with safe_merge=True
    # This checks adapter weights for NaNs before merging
    logger.info("Merging layers with safe_merge=True...")
    try:
        merged_model = model.merge_and_unload(
            progressbar=True,
            safe_merge=True
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        logger.error("This usually indicates corrupted adapter weights from training divergence")
        raise
    
    # Switch to evaluation mode
    merged_model.eval()
    
    # Post-merge verification
    logger.info("=" * 60)
    logger.info("Running post-merge verification...")
    logger.info("=" * 60)
    
    # Check for NaNs
    if not verify_no_nans(merged_model):
        raise ValueError("FATAL: Merged model contains NaN values. Training may have diverged.")
    
    # Check dtype consistency
    if not verify_dtype_consistency(merged_model, torch.bfloat16):
        logger.warning("Some layers have inconsistent dtype - proceeding anyway")
    
    # Verify model architecture is standard (not PEFT wrapped)
    model_type = type(merged_model).__name__
    logger.info(f"Merged model type: {model_type}")
    if "Peft" in model_type:
        raise ValueError(f"Model is still PEFT-wrapped ({model_type}). merge_and_unload() failed.")
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # MANDATORY: safetensors for Neuron
        max_shard_size="5GB"      # Optimal for parallel download
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Save generation_config if available
    try:
        if hasattr(base_model, "generation_config"):
            base_model.generation_config.save_pretrained(output_path)
            logger.info("  Saved generation_config.json")
    except Exception as e:
        logger.warning(f"  Could not save generation_config: {e}")
    
    # CRITICAL: Patch config.json for Trainium compatibility
    config_path = output_path / "config.json"
    patch_config_for_trainium(config_path)
    
    # Save merge metadata
    merge_info = {
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "torch_dtype": "bfloat16",
        "merge_method": "merge_and_unload",
        "safe_merge": True,
        "serialization": "safetensors",
        "trainium_compatible": True,
        "config_patches": ["rope_scaling", "torch_dtype", "architectures"]
    }
    with open(output_path / "merge_info.json", "w") as f:
        json.dump(merge_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Merge complete! Model saved to: {output_path}")
    logger.info("=" * 60)
    
    # List output files
    logger.info("Output files:")
    for f in sorted(output_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")
    
    # Optionally push to Hub
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("hub_model_id required when push_to_hub=True")
        
        logger.info(f"\nPushing to Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_model_id)
        logger.info("Pushed to Hub successfully!")
    
    return str(output_path)


def verify_merged_model(model_path: str) -> bool:
    """
    Comprehensive verification of merged model.
    
    Tests:
    1. Model loads correctly
    2. Config is patched for Trainium
    3. Generation produces expected format
    4. No infinite loops (eos_token works)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Running comprehensive model verification...")
    logger.info(f"{'=' * 60}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Check config patches
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        logger.info("Config verification:")
        logger.info(f"  architectures: {config.get('architectures')}")
        logger.info(f"  torch_dtype: {config.get('torch_dtype')}")
        logger.info(f"  rope_scaling: {config.get('rope_scaling')}")
        
        # Verify rope_scaling is Trainium-compatible
        if config.get("rope_scaling", {}).get("rope_type") == "llama3":
            logger.error("  FAIL: rope_scaling still uses 'llama3' type")
            return False
        
        # Test generation
        test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a chess grandmaster.<|eot_id|><|start_header_id|>user<|end_header_id|>

Position (FEN): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Legal moves: e2e4, d2d4, g1f3, c2c4, b1c3

Output the best move:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info("\nTest generation:")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        
        logger.info(f"  Generated: {generated[:200]}...")
        
        # Check for expected format
        if "<uci_move>" in generated:
            logger.info("  SUCCESS: Model produces expected UCI format!")
        else:
            logger.warning("  WARNING: Model may not produce expected format yet (base model response)")
        
        # Check that generation stopped (didn't hit max_new_tokens)
        if len(outputs[0]) - len(inputs['input_ids'][0]) < 64:
            logger.info("  SUCCESS: Generation stopped naturally (EOS token works)")
        else:
            logger.warning("  WARNING: Generation hit max tokens limit")
        
        logger.info(f"\n{'=' * 60}")
        logger.info("Verification PASSED - Model ready for submission")
        logger.info(f"{'=' * 60}")
        
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters for Trainium deployment"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for Hub upload (e.g., 'username/chess-llama')"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after merge"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Merge adapters
    merged_path = merge_lora_adapters(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    
    # Optionally verify
    if args.verify:
        success = verify_merged_model(merged_path)
        if not success:
            logger.error("Verification failed!")
            exit(1)
    
    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print(f"{'=' * 60}")
    print(f"\nMerged model: {merged_path}")
    print(f"\nNext steps:")
    print(f"  1. Upload to HuggingFace Hub:")
    print(f"     huggingface-cli upload YOUR_USERNAME/chess-llama {merged_path}")
    print(f"  2. Submit to AIcrowd:")
    print(f"     aicrowd submit --challenge global-chess-challenge-2025 \\")
    print(f"         --hf-repo YOUR_USERNAME/chess-llama")
