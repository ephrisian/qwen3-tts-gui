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


class ModelManager:
    """Lazily loads models by kind and keeps at most `max_loaded` in memory.

    kind is a key of config.MODEL_MAP: 'voice-design' or 'voice-clone'.
    `settings` is a callable returning (device, dtype, attn); changing them reloads models.
    """

    def __init__(self, settings, max_loaded=2):
        import threading
        from collections import OrderedDict
        self._settings = settings
        self._max = max_loaded
        self._models = OrderedDict()
        self._loaded_with = None
        self._lock = threading.Lock()

    def get(self, kind, notify=None):
        from config import MODEL_MAP
        with self._lock:
            current = tuple(self._settings())
            if current != self._loaded_with:
                self._models.clear()
                self._loaded_with = current
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind]
            while len(self._models) >= self._max:
                self._models.popitem(last=False)
                self._free_memory()
            if notify:
                notify(f"Loading {kind} model…")
            self._models[kind] = load_model(MODEL_MAP[kind], *current)
            return self._models[kind]

    @staticmethod
    def _free_memory():
        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
