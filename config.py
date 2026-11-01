"""Configuration constants for Qwen3-TTS GUI."""

import os

# ------------------------------------------------------------------ directories
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "output")

# ------------------------------------------------------------------ tab / mode mappings
TAB_MODE_MAP = {
    "VoiceDesign": "voice-design",
    "VoiceClone": "voice-clone",
}

MODE_TAB_MAP = {
    "voice-design": "VoiceDesign",
    "voice-clone": "VoiceClone",
}

# Hugging Face model IDs; weights are downloaded on first use into HF_HOME
# (the project-local .hf_cache by default).
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".hf_cache"))
MODEL_MAP = {
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice-clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Available models for the dropdown (display name -> path)
MODELS = {
    "Qwen3-TTS 1.7B VoiceDesign": MODEL_MAP["voice-design"],
    "Qwen3-TTS 1.7B Base (VoiceClone)": MODEL_MAP["voice-clone"],
}

# ------------------------------------------------------------------ languages
# Languages supported by Qwen3-TTS (lower-case names as the model expects).
LANGUAGES = [
    "auto", "english", "chinese", "japanese", "korean",
    "german", "french", "russian", "portuguese", "spanish", "italian",
]
DEFAULT_LANGUAGE = "english"

# ------------------------------------------------------------------ dtypes / attention / devices
DTYPES = ["bfloat16", "float16", "float32"]
ATTNS = ["flash_attention_2", "sdpa", "eager"]
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN = "flash_attention_2"

# ------------------------------------------------------------------ chunking defaults
DEFAULT_MIN_WORDS = "200"
DEFAULT_MAX_WORDS = "250"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
