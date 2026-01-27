#!/usr/bin/env python3
"""
Environment Verification Script

Run this BEFORE training to verify all dependencies are correctly installed
and compatible with AWS Trainium deployment.

Checks:
1. Python version (3.10 recommended)
2. NumPy < 2.0 (CRITICAL for Neuron)
3. PyTorch with CUDA and BF16 support
4. Transformers >= 4.43 (Llama 3.1 RoPE support)
5. PEFT version
6. Flash Attention availability
7. Llama 3.1 config validation

Usage:
    python scripts/verify_environment.py
"""

import sys
from packaging import version


def check_python():
    """Check Python version."""
    py_ver = sys.version_info
    print(f"Python: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    
    if py_ver.major != 3:
        return False, "Python 3 required"
    
    if py_ver.minor < 10:
        return False, "Python 3.10+ recommended for Trainium compatibility"
    
    if py_ver.minor > 11:
        return True, "WARNING: Python 3.12+ may have compatibility issues"
    
    return True, "OK"


def check_numpy():
    """Check NumPy version - CRITICAL."""
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        
        if version.parse(np.__version__) >= version.parse("2.0.0"):
            return False, "CRITICAL: NumPy 2.0+ breaks AWS Neuron. Downgrade to 1.26.4"
        
        return True, "OK"
    except ImportError:
        return False, "NumPy not installed"


def check_pytorch():
    """Check PyTorch and CUDA."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            return False, "CUDA not available - GPU training not possible"
        
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  BF16 support: {torch.cuda.is_bf16_supported()}")
        
        if not torch.cuda.is_bf16_supported():
            return False, "BF16 not supported - Ampere+ GPU required"
        
        return True, "OK"
    except ImportError:
        return False, "PyTorch not installed"


def check_transformers():
    """Check transformers version for Llama 3.1 support."""
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        
        ver = version.parse(transformers.__version__)
        
        if ver < version.parse("4.43.0"):
            return False, "Transformers < 4.43 doesn't support Llama 3.1 RoPE scaling"
        
        if ver >= version.parse("4.48.0"):
            return True, "WARNING: Transformers 4.48+ may have Neuron compatibility issues"
        
        return True, "OK"
    except ImportError:
        return False, "Transformers not installed"


def check_peft():
    """Check PEFT version."""
    try:
        import peft
        print(f"PEFT: {peft.__version__}")
        
        ver = version.parse(peft.__version__)
        
        if ver < version.parse("0.10.0"):
            return False, "PEFT < 0.10 may have LoRA merge issues"
        
        if ver >= version.parse("0.14.0"):
            return True, "WARNING: PEFT 0.14+ has reported regressions"
        
        return True, "OK"
    except ImportError:
        return False, "PEFT not installed"


def check_accelerate():
    """Check accelerate."""
    try:
        import accelerate
        print(f"Accelerate: {accelerate.__version__}")
        return True, "OK"
    except ImportError:
        return False, "Accelerate not installed"


def check_safetensors():
    """Check safetensors."""
    try:
        import safetensors
        print(f"Safetensors: {safetensors.__version__}")
        return True, "OK"
    except ImportError:
        return False, "Safetensors not installed (required for Trainium)"


def check_trl():
    """Check TRL for SFTTrainer."""
    try:
        import trl
        print(f"TRL: {trl.__version__}")
        return True, "OK"
    except ImportError:
        return False, "TRL not installed (required for SFTTrainer with packing)"


def check_flash_attention():
    """Check Flash Attention 2."""
    try:
        import flash_attn
        print(f"Flash Attention: {flash_attn.__version__}")
        return True, "OK (2-3x speedup enabled)"
    except ImportError:
        return True, "Not installed (will use SDPA fallback - slower)"


def check_llama_config():
    """Verify Llama 3.1 config support."""
    try:
        from transformers import LlamaConfig
        
        # Test that Llama 3.1 RoPE config is accepted
        config = LlamaConfig(
            rope_scaling={
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        )
        print("Llama 3.1 RoPE config: Supported")
        return True, "OK"
    except Exception as e:
        return False, f"Llama 3.1 RoPE config not supported: {e}"


def check_datasets():
    """Check datasets library."""
    try:
        import datasets
        print(f"Datasets: {datasets.__version__}")
        return True, "OK"
    except ImportError:
        return False, "Datasets not installed"


def main():
    print("=" * 60)
    print("Environment Verification for Chess LoRA Training")
    print("=" * 60)
    print("")
    
    checks = [
        ("Python", check_python),
        ("NumPy", check_numpy),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("PEFT", check_peft),
        ("Accelerate", check_accelerate),
        ("Safetensors", check_safetensors),
        ("TRL", check_trl),
        ("Datasets", check_datasets),
        ("Flash Attention", check_flash_attention),
        ("Llama Config", check_llama_config),
    ]
    
    results = []
    
    print("Component Versions:")
    print("-" * 40)
    
    for name, check_fn in checks:
        try:
            passed, message = check_fn()
            results.append((name, passed, message))
        except Exception as e:
            results.append((name, False, f"Error: {e}"))
        print("")
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    failures = []
    warnings = []
    
    for name, passed, message in results:
        if not passed:
            status = "FAIL"
            failures.append((name, message))
        elif "WARNING" in message:
            status = "WARN"
            warnings.append((name, message))
        else:
            status = "OK"
        
        print(f"  [{status:4}] {name}: {message}")
    
    print("")
    
    if failures:
        print("CRITICAL FAILURES:")
        for name, message in failures:
            print(f"  - {name}: {message}")
        print("")
        print("Fix these issues before training!")
        sys.exit(1)
    
    if warnings:
        print("WARNINGS (may cause issues):")
        for name, message in warnings:
            print(f"  - {name}: {message}")
        print("")
    
    print("=" * 60)
    print("Environment verification PASSED!")
    print("=" * 60)
    print("")
    print("Ready to train. Run:")
    print("  bash scripts/train_a100.sh")
    

if __name__ == "__main__":
    main()
