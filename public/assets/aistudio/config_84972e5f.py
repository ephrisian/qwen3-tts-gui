"""Configuration constants for Qwen3-TTS GUI."""

import os

# Contain all Hugging Face model downloads within this project folder
# so the entire workspace can be deleted at once for portability.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Output directory (auto-managed, no user input needed)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_MAP = {
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice-clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

SUPPORTED_SPEAKERS = {
    "Vivian": "Bright, slightly edgy young female voice. | Chinese",
    "Serena": "Warm, gentle young female voice. | Chinese",
    "Uncle_Fu": "Seasoned male voice with a low, mellow timbre. | Chinese",
    "Dylan": "Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect)",
    "Eric": "Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect)",
    "Ryan": "Dynamic male voice with strong rhythmic drive. | English",
    "Aiden": "Sunny American male voice with a clear midrange. | English",
    "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre. | Japanese",
    "Sohee": "Warm Korean female voice with rich emotion. | Korean",
}

LANGUAGES = [
    "auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

DTYPES = ["bfloat16", "float16", "float32"]
ATTNS = ["flash_attention_2", "sdpa", "eager"]

# Tab display name -> internal mode name mapping
TAB_MODE_MAP = {
    "CustomVoice": "custom-voice",
    "VoiceDesign": "voice-design",
    "VoiceClone": "voice-clone",
}

# Internal mode name -> tab display name mapping
MODE_TAB_MAP = {v: k for k, v in TAB_MODE_MAP.items()}

# Default UI values
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN = "flash_attention_2"
DEFAULT_MIN_WORDS = "200"
DEFAULT_MAX_WORDS = "250"
DEFAULT_SPEAKER = "Vivian"
DEFAULT_LANGUAGE = "auto"

# Window settings
WINDOW_TITLE = "Qwen3-TTS"
WINDOW_GEOMETRY = "1280x720"
WINDOW_MINSIZE = (960, 540)

# Audio playback defaults
DEFAULT_VOLUME = 0.8
VOLUME_SCALE_LENGTH = 150

# File dialog filters
WAV_FILETYPES = [("WAV files", "*.wav"), ("All files", "*.*")]
AUDIO_FILETYPES = [("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg")]
NPZ_FILETYPES = [("NPZ files", "*.npz")]
