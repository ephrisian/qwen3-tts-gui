"""On-disk library of designed voices.

Each voice lives in voices/<slug>/ :
    voice.json        metadata (description, seed, sample text, file names, usage notes)
    reference.wav     the approved VoiceDesign sample
    clone_prompt.npz  prebuilt Base-model clone prompt (speaker embedding + ref codes + transcript)
    emotions/         cached per-emotion clone prompts (created on demand)
"""

import json
import os
import re
import shutil
from datetime import datetime

import numpy as np

from audio_utils import load_voice_clone_prompt, save_voice_clone_prompt, write_wav

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

SAMPLE_TEXT = (
    "Are you certain the quick, brown fox jumped over the lazy, yawning dog? "
    "Honestly, I couldn't believe my own eyes! The majestic creature moved with "
    "such vivid, explosive power. Breathing deeply, we just watched the bizarre "
    "scene unfold, marveling at the quiet, azure sky above."
)

USAGE_NOTES = {
    "voice_clone": (
        "Pass clone_prompt.npz as voice_clone_prompt to Qwen3TTSModel.generate_voice_clone() on the "
        "Base model, or use reference.wav + ref_text with create_voice_clone_prompt()."
    ),
    "voice_design": (
        "description is the VoiceDesign instruct. Seed does not guarantee the same voice for new text; "
        "use the clone prompt for a stable identity."
    ),
}


def slugify(name):
    slug = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return slug[:60]


def voice_dir(name, root=None):
    return os.path.join(root or VOICES_DIR, slugify(name))


def list_voices(root=None):
    """Return sorted voice names that have a valid voice.json."""
    root = root or VOICES_DIR
    names = []
    if os.path.isdir(root):
        for entry in os.listdir(root):
            cfg = os.path.join(root, entry, "voice.json")
            if os.path.isfile(cfg):
                try:
                    with open(cfg, encoding="utf-8") as f:
                        names.append(json.load(f)["name"])
                except (OSError, ValueError, KeyError):
                    continue
    return sorted(names, key=str.lower)


def voice_exists(name, root=None):
    return os.path.isfile(os.path.join(voice_dir(name, root), "voice.json"))


def save_voice(name, wav, sr, prompt, description, language, seed,
               model_ids=None, root=None, overwrite=False):
    """Write a voice folder and return its config dict."""
    if not slugify(name):
        raise ValueError("Voice name must contain letters or digits.")
    path = voice_dir(name, root)
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"A voice named '{name}' already exists.")
        shutil.rmtree(path)
    os.makedirs(path)

    write_wav(os.path.join(path, "reference.wav"), np.asarray(wav), sr)
    save_voice_clone_prompt(os.path.join(path, "clone_prompt.npz"), prompt)

    config = {
        "schema": 1,
        "name": name.strip(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "description": description,
        "seed": int(seed),
        "sample_text": SAMPLE_TEXT,
        "ref_text": SAMPLE_TEXT,
        "ref_audio": "reference.wav",
        "clone_prompt": "clone_prompt.npz",
        "sample_rate": int(sr),
        "models": model_ids or {},
        "usage": USAGE_NOTES,
    }
    with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config


def load_voice(name, root=None):
    """Return (config, clone_prompt, voice_directory) for a saved voice."""
    path = voice_dir(name, root)
    with open(os.path.join(path, "voice.json"), encoding="utf-8") as f:
        config = json.load(f)
    prompt = load_voice_clone_prompt(os.path.join(path, config["clone_prompt"]))
    return config, prompt, path


def delete_voice(name, root=None):
    path = voice_dir(name, root)
    if os.path.isfile(os.path.join(path, "voice.json")):
        shutil.rmtree(path)
