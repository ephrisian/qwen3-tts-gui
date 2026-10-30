"""Model loading utilities for Qwen3-TTS."""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None


def get_dtype_map():
    if not TORCH_AVAILABLE:
        return {}
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }


def is_cuda_available():
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def load_model(model_id, device, dtype, attn):
    """Load a Qwen3TTSModel from pretrained weights.

    Args:
        model_id: Hugging Face model ID or local path.
        device: "cuda" or "cpu".
        dtype: "bfloat16", "float16", or "float32".
        attn: Attention implementation ("flash_attention_2", "sdpa", "eager").

    Returns:
        Loaded Qwen3TTSModel instance.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install PyTorch with: pip install torch"
        )

    if Qwen3TTSModel is None:
        raise ImportError(
            "'qwen-tts' package is not installed. "
            "Install it with: pip install -U qwen-tts"
        )

    dtype_map = get_dtype_map()
    torch_dtype = dtype_map.get(dtype, torch.float32)

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
