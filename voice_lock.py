"""Helpers for reproducible voices and emotion-controlled long-form cloning."""

import dataclasses
import random
import re

import numpy as np

# Short emotion names expand to richer VoiceDesign style descriptions.
# Any other tag text is used verbatim as the description.
EMOTION_PRESETS = {
    "neutral": "calm, even, neutral delivery",
    "happy": "cheerful, bright and smiling, upbeat energy",
    "sad": "sorrowful, subdued and heavy, slow and quiet with a trembling softness",
    "angry": "furious, harsh and forceful, loud with sharp clipped delivery",
    "fearful": "terrified, breathy and shaky, quick and anxious",
    "whisper": "whispering softly and intimately, hushed breathy voice",
    "excited": "extremely excited, fast, energetic and animated",
    "tender": "tender, warm and gentle, soft intimate delivery",
    "sarcastic": "dry and sarcastic, flat with exaggerated emphasis",
    "serious": "grave, serious and measured, low and deliberate",
}

DEFAULT_EMOTION_REF_TEXT = (
    "I never thought I would be standing here, telling you all of this today."
)

_TAG_RE = re.compile(r"\[([^\[\]\n]{1,100})\]")


def seed_everything(seed):
    """Seed every RNG that influences sampling so a call is reproducible."""
    if seed is None:
        return
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seed(mode, value, default=42):
    """Return an int seed for 'Fixed' mode, otherwise None (random)."""
    if mode != "Fixed":
        return None
    value = (value or "").strip()
    return int(value) if value else default


def parse_emotion_segments(text, default_emotion=None):
    """Split text on [emotion] tags into [(emotion_or_None, segment_text)].

    A tag applies to all following text until the next tag.
    """
    segments = []
    emotion = default_emotion
    pos = 0
    for m in _TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if not re.search(r"[^\W\d_]", tag):  # ignore things like [1]
            continue
        chunk = text[pos:m.start()].strip()
        if chunk:
            segments.append((emotion, chunk))
        emotion = tag
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        segments.append((emotion, tail))
    return segments


def emotion_description(tag):
    return EMOTION_PRESETS.get(tag.lower(), tag)


def with_anchor_embedding(item, anchor_item):
    """Return `item` with the speaker embedding of `anchor_item`.

    The ICL reference (codes + transcript) carries the emotion while the
    shared speaker embedding pins the voice identity, limiting drift.
    """
    return dataclasses.replace(item, ref_spk_embedding=anchor_item.ref_spk_embedding)


class EmotionPromptBank:
    """Builds and caches one clone prompt per emotion, anchored to one voice."""

    def __init__(self, vd_model, base_model, anchor_item, voice_description,
                 language, seed, ref_text=None, cache_dir=None):
        self.vd_model = vd_model
        self.base_model = base_model
        self.anchor = anchor_item
        self.voice_description = voice_description
        self.language = language
        self.seed = seed
        self.ref_text = ref_text or DEFAULT_EMOTION_REF_TEXT
        self.cache_dir = cache_dir
        self._cache = {}

    def _cache_path(self, emotion):
        import hashlib
        import os
        raw = repr((self.voice_description, emotion_description(emotion), self.ref_text, self.language, self.seed))
        return os.path.join(self.cache_dir, hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16] + ".npz")

    def get(self, emotion):
        """Prompt item list for an emotion (None/neutral -> the anchor itself)."""
        if emotion is None:
            return [self.anchor]
        key = emotion.lower()
        if key not in self._cache and self.cache_dir:
            import os
            from audio_utils import load_voice_clone_prompt
            path = self._cache_path(emotion)
            if os.path.exists(path):
                try:
                    self._cache[key] = load_voice_clone_prompt(path)
                except Exception:
                    pass
        if key not in self._cache:
            instruct = f"{self.voice_description}. {emotion_description(emotion)}"
            seed_everything(self.seed)
            wavs, sr = self.vd_model.generate_voice_design(
                text=self.ref_text, language=self.language, instruct=instruct
            )
            wav = np.asarray(wavs[0] if hasattr(wavs, "__getitem__") else wavs)
            item = self.base_model.create_voice_clone_prompt(
                ref_audio=(wav, sr), ref_text=self.ref_text, x_vector_only_mode=False
            )[0]
            self._cache[key] = [with_anchor_embedding(item, self.anchor)]
            if self.cache_dir:
                import os
                from audio_utils import save_voice_clone_prompt
                os.makedirs(self.cache_dir, exist_ok=True)
                save_voice_clone_prompt(self._cache_path(emotion), self._cache[key])
        return self._cache[key]

