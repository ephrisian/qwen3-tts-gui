"""Model loading utilities for Qwen3-TTS."""

import torch

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_id, device, dtype, attn):
    """Load a Qwen3TTSModel from pretrained weights.

    Args:
        model_id: Hugging Face model ID or local path.
        device: "cuda" or "cpu".
        dtype: "bfloat16", "float16", or "float32".
        attn: Attention implementation ("flash_attention_2", "sdpa", "eager").

    Returns:
        Loaded Qwen3TTSModel instance.

    Raises:
        ImportError: If qwen-tts package is not installed.
        Exception: If model loading fails.
    """
    if Qwen3TTSModel is None:
        raise ImportError(
            "'qwen-tts' package is not installed. "
            "Install it with: pip install -U qwen-tts"
        )

    torch_dtype = DTYPE_MAP[dtype]

    print(f"Loading model: {model_id}")
    print(f"Device: {device} | Dtype: {dtype} | Attention: {attn}")

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    return model
