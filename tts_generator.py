"""UI-independent TTS generation logic for Qwen3-TTS.

Two ways to get one consistent voice:
  * VoiceClone   - saved/designed voice; emotion comes from per-emotion reference
                   clips generated with VoiceDesign and anchored to the voice's
                   speaker embedding (the Base model has no instruct input).
  * VoiceDesign  - used to create voices (and emotion references), not for long-form.
"""

import os

import numpy as np

from audio_utils import load_voice_clone_prompt, save_voice_clone_prompt
from text_utils import chunk_text
import voice_library
from voice_lock import (
    EmotionPromptBank, parse_emotion_segments, seed_everything,
)


def _first(wavs):
    return np.asarray(wavs[0] if hasattr(wavs, "__getitem__") else wavs)


class StopRequested(Exception):
    pass


class TTSGenerator:
    def __init__(self, models, notify=print, should_stop=lambda: False):
        self.models = models
        self.notify = notify
        self.should_stop = should_stop

    # ---- helpers
    @staticmethod
    def chunk(text, min_words, max_words):
        if min_words and max_words and max_words > 0:
            return chunk_text(text, min_words=min_words, max_words=max_words)
        return [text]

    def _jobs(self, text, min_words, max_words):
        """Split on [emotion] tags first, then chunk each segment: [(emotion, chunk)]."""
        jobs = []
        for emotion, seg in parse_emotion_segments(text):
            for c in self.chunk(seg, min_words, max_words):
                jobs.append((emotion, c))
        return jobs

    # ---- VoiceDesign: voice creation
    def design_sample(self, description, language, seed):
        model = self.models.get("voice-design", self.notify)
        seed_everything(seed)
        wavs, sr = model.generate_voice_design(
            text=voice_library.SAMPLE_TEXT, language=language, instruct=description
        )
        return _first(wavs), sr

    def build_voice_prompt(self, wav, sr):
        base = self.models.get("voice-clone", self.notify)
        return base.create_voice_clone_prompt(
            ref_audio=(wav, sr), ref_text=voice_library.SAMPLE_TEXT, x_vector_only_mode=False
        )

    def build_clone_prompt(self, ref_audio, ref_text, x_vector_only=False):
        """Build a voice-clone prompt from reference audio without generating audio."""
        model = self.models.get("voice-clone", self.notify)
        return model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=x_vector_only
        )

    # ---- VoiceClone
    def clone(self, text, language, min_words, max_words, seed=None, temperature=0.7,
              voice_name=None, ref_audio="", ref_text=None, x_vector_only=False,
              reuse_prompt=None, save_prompt=None, description=""):
        voice_cfg = voice_dir = None
        if voice_name:
            voice_cfg, anchor_prompt, voice_dir = voice_library.load_voice(voice_name)
        elif not ref_audio and not reuse_prompt:
            raise ValueError("Pick a saved voice, or provide reference audio / a reusable prompt.")

        model = self.models.get("voice-clone", self.notify)
        if x_vector_only:
            ref_text = None
        if not voice_cfg:
            if reuse_prompt:
                anchor_prompt = load_voice_clone_prompt(reuse_prompt)
            else:
                anchor_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=x_vector_only
                )

        if save_prompt:
            save_voice_clone_prompt(save_prompt, anchor_prompt)
            self.notify(f"Voice-clone prompt saved to: {save_prompt}")

        jobs = self._jobs(text, min_words, max_words)
        bank = None
        if any(e for e, _ in jobs):
            if not isinstance(anchor_prompt, list):
                raise ValueError("Emotion tags need a prompt saved as a list of items; rebuild the prompt.")
            description = voice_cfg["description"] if voice_cfg else (description or "").strip()
            if not description:
                raise ValueError("Describe the voice (description box) to use [emotion] tags.")
            bank = EmotionPromptBank(
                vd_model=self.models.get("voice-design", self.notify),
                base_model=model,
                anchor_item=anchor_prompt[0],
                voice_description=description,
                language=language,
                # A saved voice reuses its own seed so emotion references reproduce exactly.
                seed=voice_cfg["seed"] if voice_cfg else (seed or 0),
                cache_dir=os.path.join(voice_dir, "emotions") if voice_dir else None,
            )

        # Lower sampling temperature keeps timbre steadier across long text.
        gen_kwargs = dict(temperature=temperature, subtalker_temperature=temperature)
        out, sr = [], None
        for i, (emotion, t) in enumerate(jobs):
            if self.should_stop():
                break
            self.notify(f"Chunk {i+1}/{len(jobs)}" + (f" [{emotion}]" if emotion else ""))
            prompt = bank.get(emotion) if bank else anchor_prompt
            seed_everything(seed)
            wavs, sr = model.generate_voice_clone(
                text=t, language=language, voice_clone_prompt=prompt, **gen_kwargs
            )
            out.append(_first(wavs))
        return out, sr
