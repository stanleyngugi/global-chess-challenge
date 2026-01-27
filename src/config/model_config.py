"""
Model Configuration Module
==========================
Centralized configuration for model specifications, hardware constraints,
and deployment parameters for the Global Chess Challenge 2025.

Key Constraints:
- AWS Trainium trn1.2xlarge: 2 NeuronCores, 32GB HBM total
- Tensor Parallelism: TP=2 is MANDATORY
- KV Heads must be divisible by TP degree
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ModelType(Enum):
    """Supported model architectures for Neuron deployment."""
    LLAMA_3_8B = "llama-3.1-8b"
    QWEN_2_5_7B = "qwen2.5-7b"


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a model architecture."""
    model_id: str
    architecture: str
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_context_length: int
    rope_theta: float
    license: str
    requires_auth: bool
    
    def is_tp_compatible(self, tp_degree: int) -> bool:
        """Check if model is compatible with given tensor parallelism degree."""
        return (
            self.num_attention_heads % tp_degree == 0 and
            self.num_key_value_heads % tp_degree == 0
        )
    
    def get_heads_per_core(self, tp_degree: int) -> tuple[int, int]:
        """Get Q heads and KV heads per NeuronCore."""
        return (
            self.num_attention_heads // tp_degree,
            self.num_key_value_heads // tp_degree
        )


# Model Registry: Verified configurations for Neuron deployment
MODEL_REGISTRY: dict[ModelType, ModelConfig] = {
    ModelType.LLAMA_3_8B: ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        architecture="LlamaForCausalLM",
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA with 4x ratio
        hidden_size=4096,
        num_layers=32,
        vocab_size=128256,
        max_context_length=131072,  # 128K native
        rope_theta=500000.0,
        license="llama3.1",
        requires_auth=True,
    ),
    ModelType.QWEN_2_5_7B: ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        architecture="Qwen2ForCausalLM",
        num_attention_heads=28,
        num_key_value_heads=4,  # GQA with 7x ratio
        hidden_size=3584,
        num_layers=28,
        vocab_size=152064,
        max_context_length=32768,  # 32K native, 128K with YaRN
        rope_theta=1000000.0,
        license="apache-2.0",
        requires_auth=False,
    ),
}


@dataclass(frozen=True)
class NeuronConfig:
    """Hardware configuration for AWS Trainium deployment."""
    instance_type: str = "trn1.2xlarge"
    neuron_cores: int = 2
    hbm_per_core_gb: int = 16
    total_hbm_gb: int = 32
    tensor_parallel_size: int = 2  # MANDATORY for this instance


@dataclass
class InferenceConfig:
    """
    Runtime inference configuration.
    
    CRITICAL: These defaults are tuned for stability on trn1.2xlarge
    with concurrency=4 (competition evaluator setting).
    """
    # Model settings
    model_type: ModelType = ModelType.LLAMA_3_8B
    dtype: str = "bfloat16"
    
    # Context and generation
    max_model_len: int = 4096  # Safe for memory with concurrency=4
    max_new_tokens: int = 256  # Enough for move + rationale
    
    # Sampling parameters
    temperature: float = 0.0  # Greedy for single-shot consistency
    top_p: float = 1.0
    top_k: int = -1  # Disabled when temperature=0
    
    # Retry configuration
    max_retries: int = 3
    retry_temperatures: tuple[float, ...] = (0.0, 0.3, 0.7)
    
    # vLLM settings
    max_num_seqs: int = 4  # Match competition concurrency=4
    block_size: int = 8  # Smaller blocks for Neuron
    enforce_eager: bool = True  # Stability on Neuron
    
    @property
    def model_config(self) -> ModelConfig:
        """Get the model configuration for the selected model type."""
        return MODEL_REGISTRY[self.model_type]
    
    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID."""
        return self.model_config.model_id


@dataclass(frozen=True)
class CompetitionConfig:
    """Competition-specific constants."""
    # Output format
    UCI_MOVE_PATTERN: str = r"<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>"
    RATIONALE_PATTERN: str = r"<rationale>(.*?)</rationale>"
    
    # Valid UCI move regex (standalone)
    UCI_MOVE_REGEX: str = r"^[a-h][1-8][a-h][1-8][qrbn]?$"
    
    # Retry limits
    MAX_INVALID_OUTPUTS: int = 3
    
    # Evaluation
    BASELINE_ACPL: float = 71.921  # Must beat this to qualify for finals
    
    # Submission
    CHALLENGE_NAME: str = "global-chess-challenge-2025"


# Singleton instances for easy access
NEURON_CONFIG = NeuronConfig()
COMPETITION_CONFIG = CompetitionConfig()


def get_default_inference_config(
    model_type: ModelType = ModelType.LLAMA_3_8B
) -> InferenceConfig:
    """Get default inference configuration for a model type."""
    return InferenceConfig(model_type=model_type)


def validate_model_for_neuron(
    model_config: ModelConfig,
    neuron_config: NeuronConfig = NEURON_CONFIG
) -> tuple[bool, list[str]]:
    """
    Validate if a model configuration is compatible with Neuron deployment.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check TP compatibility
    tp = neuron_config.tensor_parallel_size
    if not model_config.is_tp_compatible(tp):
        issues.append(
            f"KV heads ({model_config.num_key_value_heads}) not divisible by TP={tp}. "
            f"This will trigger GQA.CONVERT_TO_MHA fallback and likely OOM."
        )
    
    # Estimate memory usage (rough)
    params_billions = (
        model_config.hidden_size * model_config.vocab_size +
        model_config.num_layers * (
            4 * model_config.hidden_size ** 2 +  # QKV + O projections
            3 * model_config.hidden_size * 4 * model_config.hidden_size  # FFN (approx)
        )
    ) / 1e9
    
    weight_memory_gb = params_billions * 2  # BF16 = 2 bytes per param
    
    if weight_memory_gb > neuron_config.total_hbm_gb * 0.7:  # Leave 30% for KV cache
        issues.append(
            f"Estimated weight memory ({weight_memory_gb:.1f}GB) may exceed safe threshold. "
            f"Consider reducing max_model_len or using smaller model."
        )
    
    return len(issues) == 0, issues
