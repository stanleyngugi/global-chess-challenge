"""
vLLM Configuration for AWS Neuron
=================================
Configuration and launch scripts for running vLLM on AWS Trainium.

Key Constraints:
- TP=2 is MANDATORY for trn1.2xlarge
- Static shapes required (bucketing)
- BF16 dtype for native performance
- Enforce eager mode for stability
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .model_config import (
    InferenceConfig,
    NeuronConfig,
    ModelConfig,
    MODEL_REGISTRY,
    ModelType,
    NEURON_CONFIG,
    validate_model_for_neuron,
)


@dataclass
class VLLMServerConfig:
    """
    Configuration for vLLM server deployment on Neuron.
    
    CRITICAL SETTINGS FOR STABILITY:
    - max_model_len: 4096 (NOT 8192, to leave room for KV cache with concurrency=4)
    - max_num_seqs: 4 (matches competition evaluator concurrency)
    - block_size: 8 or 16 (Neuron uses block-based KV cache)
    - tensor_parallel_size: 2 (MANDATORY for trn1.2xlarge)
    
    Memory Budget (trn1.2xlarge, 32GB HBM):
    - Model weights (Llama-8B BF16): ~16GB
    - KV cache (4 seqs x 4096 tokens): ~8GB
    - Activations + overhead: ~8GB
    """
    
    # Model settings
    model_id: str
    
    # Neuron-specific settings (DO NOT CHANGE)
    tensor_parallel_size: int = 2  # MANDATORY for trn1.2xlarge
    device: str = "neuron"
    dtype: str = "bfloat16"
    
    # Memory management - CONSERVATIVE FOR STABILITY
    max_model_len: int = 4096  # Context length - DO NOT EXCEED for 8B model
    max_num_seqs: int = 4  # Match competition evaluator concurrency=4
    block_size: int = 8  # Smaller blocks for Neuron stability
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Stability flags - CRITICAL
    enforce_eager: bool = True  # Bypass CUDA graph issues on Neuron
    disable_log_stats: bool = True  # Reduce logging overhead
    disable_log_requests: bool = False  # Keep request logs for debugging
    
    # Optional: API key requirement
    api_key: Optional[str] = None
    
    # Neuron-specific environment variables
    neuron_env: dict = None
    
    def __post_init__(self):
        """Initialize default Neuron environment variables."""
        if self.neuron_env is None:
            self.neuron_env = {
                "NEURON_RT_NUM_CORES": "2",  # Match tensor_parallel_size
                "NEURON_CC_FLAGS": "--model-type=transformer",
                "VLLM_GUIDED_DECODING_BACKEND": "outlines",
                "TOKENIZERS_PARALLELISM": "false",
            }
    
    def to_args(self) -> list[str]:
        """Convert configuration to vLLM CLI arguments."""
        args = [
            "--model", self.model_id,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--device", self.device,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--block-size", str(self.block_size),
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        if self.enforce_eager:
            args.append("--enforce-eager")
        
        if self.disable_log_stats:
            args.append("--disable-log-stats")
        
        if self.disable_log_requests:
            args.append("--disable-log-requests")
        
        if self.api_key:
            args.extend(["--api-key", self.api_key])
        
        return args
    
    def to_command(self) -> str:
        """Get the full command to launch vLLM server."""
        args = self.to_args()
        return f"python -m vllm.entrypoints.openai.api_server {' '.join(args)}"
    
    def to_env_dict(self) -> dict[str, str]:
        """Get environment variables for the server."""
        env = {
            "VLLM_GUIDED_DECODING_BACKEND": "outlines",
            "TOKENIZERS_PARALLELISM": "false",
            "NEURON_RT_NUM_CORES": str(self.tensor_parallel_size),
            "NEURON_CC_FLAGS": "--model-type=transformer --enable-fast-loading-neuron-binaries",
        }
        if self.neuron_env:
            env.update(self.neuron_env)
        return env


def create_server_config(
    model_type: ModelType = ModelType.LLAMA_3_8B,
    max_model_len: int = 4096,
    port: int = 8000,
) -> VLLMServerConfig:
    """
    Create a vLLM server configuration for the specified model.
    
    Args:
        model_type: Which model to use
        max_model_len: Maximum context length
        port: Server port
    
    Returns:
        VLLMServerConfig instance
    """
    model_config = MODEL_REGISTRY[model_type]
    
    # Validate compatibility
    is_valid, issues = validate_model_for_neuron(model_config)
    if not is_valid:
        raise ValueError(f"Model {model_type} is not compatible with Neuron: {issues}")
    
    return VLLMServerConfig(
        model_id=model_config.model_id,
        max_model_len=max_model_len,
        port=port,
    )


def generate_launch_script(
    config: VLLMServerConfig,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a bash script to launch the vLLM server.
    
    Args:
        config: Server configuration
        output_path: Optional path to write the script
    
    Returns:
        The script content as a string
    """
    env_vars = config.to_env_dict()
    env_exports = "\n".join(f"export {k}={v}" for k, v in env_vars.items())
    command = config.to_command()
    
    script = f'''#!/bin/bash
# vLLM Launch Script for AWS Trainium (trn1.2xlarge)
# Generated by Configuration System
#
# Hardware: AWS Trainium trn1.2xlarge (2 NeuronCores, 32GB HBM)

set -e  # Exit on error

echo "=================================================="
echo "vLLM Server for Chess Agent"
echo "=================================================="
echo "Model: {config.model_id}"
echo "TP Size: {config.tensor_parallel_size}"
echo "Max Context: {config.max_model_len}"
echo "Port: {config.port}"
echo "=================================================="

# Environment variables for stability
{env_exports}

# Check if running on Neuron
if command -v neuron-ls &> /dev/null; then
    echo "Neuron devices detected:"
    neuron-ls
else
    echo "WARNING: neuron-ls not found. Are you on a Trainium instance?"
fi

# Launch vLLM server
echo ""
echo "Starting vLLM server..."
echo "Command: {command}"
echo ""

{command}
'''
    
    if output_path:
        path = Path(output_path)
        path.write_text(script)
        if os.name != 'nt':
            os.chmod(path, 0o755)
        print(f"Launch script written to: {path}")
    
    return script


# Pre-configured server configurations for quick access
LLAMA_SERVER_CONFIG = create_server_config(ModelType.LLAMA_3_8B)
QWEN_SERVER_CONFIG = create_server_config(ModelType.QWEN_2_5_7B)
