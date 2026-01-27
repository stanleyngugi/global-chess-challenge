"""
Configuration module for Global Chess Challenge 2025.

This module contains:
- Model configurations (Llama, Qwen)
- Hardware configurations (Neuron/Trainium)
- vLLM server configurations
- Competition constants
"""

from .model_config import (
    ModelType,
    ModelConfig,
    MODEL_REGISTRY,
    NeuronConfig,
    InferenceConfig,
    CompetitionConfig,
    NEURON_CONFIG,
    COMPETITION_CONFIG,
    get_default_inference_config,
    validate_model_for_neuron,
)

from .vllm_config import (
    VLLMServerConfig,
    create_server_config,
    generate_launch_script,
    LLAMA_SERVER_CONFIG,
    QWEN_SERVER_CONFIG,
)

__all__ = [
    # Model configs
    "ModelType",
    "ModelConfig", 
    "MODEL_REGISTRY",
    "NeuronConfig",
    "InferenceConfig",
    "CompetitionConfig",
    "NEURON_CONFIG",
    "COMPETITION_CONFIG",
    "get_default_inference_config",
    "validate_model_for_neuron",
    # vLLM configs
    "VLLMServerConfig",
    "create_server_config",
    "generate_launch_script",
    "LLAMA_SERVER_CONFIG",
    "QWEN_SERVER_CONFIG",
]
