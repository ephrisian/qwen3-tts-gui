"""TTS generation logic for Qwen3-TTS Gradio Application with Seed Control."""

import os
import random
import sys
import uuid
from datetime import datetime
import numpy as np

from text_utils import chunk_text
from audio_utils import save_outputs, write_wav
from config import OUTPUT_DIR, MODE_TAB_MAP

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def generate_synthetic_audio(text, seed=42, duration_per_word=0.28, sr=24000):
    """Generate a clean synthetic harmonic waveform for testing/demo when model weights are not loaded.

    Produces pleasant speech-like harmonic formants varying predictably with seed and text.
    """
    rng = np.random.RandomState(seed % (2**31))
    words = len(text.split())
    duration = max(1.2, min(12.0, words * duration_per_word))
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Base pitch determined by seed
    f0 = 120.0 + (seed % 140)  # 120Hz to 260Hz pitch range
    wobble = 1.0 + 0.05 * np.sin(2 * np.pi * 3.5 * t)

    # Harmonic series with envelope
    y = np.zeros(n_samples, dtype=np.float32)
    harmonics = [1.0, 0.7, 0.45, 0.25, 0.15, 0.08]
    for h_idx, amp in enumerate(harmonics, start=1):
        phase = rng.uniform(0, 2 * np.pi)
        y += amp * np.sin(2 * np.pi * (f0 * h_idx) * wobble * t + phase)

    # Apply speech rhythm envelope (syllable pulses)
    syllable_rate = 4.2
    syllable_env = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t) ** 2
    # Attack / decay window
    fade_len = int(0.04 * sr)
    window = np.ones(n_samples, dtype=np.float32)
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)

    audio = y * syllable_env * window
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.75 * (audio / peak)
    return audio


class TTSGenerator:
    """Handles text-to-speech generation for CustomVoice, VoiceDesign, and VoiceClone."""

    def __init__(self, output_dir=OUTPUT_DIR):
        self.model = None
        self.output_dir = output_dir
        self.stop_requested = False
        os.makedirs(self.output_dir, exist_ok=True)

    def set_model(self, model):
        self.model = model

    def get_auto_output_path(self, mode):
        """Generate an auto output path based on the current mode and timestamp."""
        tab_name = MODE_TAB_MAP.get(mode, "CustomVoice")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:4]
        filename = f"{tab_name}_{timestamp}_{unique}.wav"
        return os.path.join(self.output_dir, filename)

    def apply_chunking(self, text, min_words=200, max_words=250):
        """Chunk long text into manageable segments."""
        if not text or not text.strip():
            return []
        if min_words and max_words and max_words > 0:
            return chunk_text(text.strip(), min_words=min_words, max_words=max_words)
        return [text.strip()]

    def generate_custom_voice(self, text, language="auto", speaker="Vivian", instruct=None,
                              min_words=200, max_words=250, output_path=None, progress_callback=None):
        """Synthesize audio using CustomVoice mode with pre-trained speaker personas."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        self.stop_requested = False
        texts = self.apply_chunking(text, min_words, max_words)
        output_file = output_path or self.get_auto_output_path("custom-voice")

        all_wavs = []
        sr = 24000

        for i, t in enumerate(texts):
            if self.stop_requested:
                break
            if progress_callback:
                progress_callback(i / len(texts), f"Generating chunk {i+1}/{len(texts)}...")
            print(f"CustomVoice chunk {i+1}/{len(texts)}: {t[:60]}...")

            if self.model is not None and hasattr(self.model, "generate_custom_voice"):
                wavs, sr = self.model.generate_custom_voice(
                    text=t, language=language, speaker=speaker, instruct=instruct or None
                )
                w = wavs[0] if hasattr(wavs, "__getitem__") and not isinstance(wavs, np.ndarray) else wavs
                all_wavs.append(np.asarray(w))
            else:
                # Synthetic fallback for demonstration / testing
                synth_audio = generate_synthetic_audio(t, seed=hash(speaker) % 100000)
                all_wavs.append(synth_audio)

        save_outputs(all_wavs, sr, output_file, stop_requested=self.stop_requested)
        if progress_callback:
            progress_callback(1.0, "Generation complete.")

        return {
            "output_path": output_file,
            "filename": os.path.basename(output_file),
            "chunks": len(texts),
            "sample_rate": sr,
            "mode": "CustomVoice",
        }

    def generate_voice_design(self, text, instruct, language="auto",
                              seed_mode="Random", seed_value=42,
                              min_words=200, max_words=250,
                              output_path=None, progress_callback=None):
        """Synthesize audio using VoiceDesign mode with natural language description.

        Supports both 'Random' seed and 'Fixed' seed controls for reproducible speaker generation.

        Args:
            text: Text to synthesize.
            instruct: Natural language voice description (required).
            language: Target language.
            seed_mode: 'Random' or 'Fixed'.
            seed_value: Integer seed used when seed_mode == 'Fixed'.
            min_words: Minimum words per chunk.
            max_words: Maximum words per chunk.
            output_path: Optional destination path.
            progress_callback: Callback for progress updates (fraction, message).

        Returns:
            Dict containing output_path, actual_seed, seed_mode, filename, etc.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        if not instruct or not instruct.strip():
            raise ValueError("Instruct (voice description) is required for VoiceDesign mode.")

        self.stop_requested = False

        # --- SEED RESOLUTION ---
        is_random = str(seed_mode).strip().lower() == "random"
        if is_random:
            actual_seed = random.randint(0, 2**31 - 1)
        else:
            try:
                actual_seed = int(seed_value)
            except (ValueError, TypeError):
                actual_seed = 42

        # Apply seeds across Python, NumPy, and PyTorch (CPU and CUDA)
        random.seed(actual_seed)
        np.random.seed(actual_seed % (2**32))
        if TORCH_AVAILABLE and torch is not None:
            try:
                torch.manual_seed(actual_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actual_seed)
            except Exception as e:
                print(f"Warning: Failed setting torch seed: {e}", file=sys.stderr)

        texts = self.apply_chunking(text, min_words, max_words)
        output_file = output_path or self.get_auto_output_path("voice-design")

        all_wavs = []
        sr = 24000

        for i, t in enumerate(texts):
            if self.stop_requested:
                break
            if progress_callback:
                progress_callback(i / len(texts), f"Generating chunk {i+1}/{len(texts)} (Seed: {actual_seed})...")
            print(f"VoiceDesign chunk {i+1}/{len(texts)} [Seed: {actual_seed}]: {t[:60]}...")

            if self.model is not None and hasattr(self.model, "generate_voice_design"):
                # Attempt passing seed kwarg if accepted by model implementation
                try:
                    wavs, sr = self.model.generate_voice_design(
                        text=t, language=language, instruct=instruct.strip(), seed=actual_seed
                    )
                except TypeError:
                    # Model function doesn't take explicit seed kwarg; uses torch.manual_seed
                    wavs, sr = self.model.generate_voice_design(
                        text=t, language=language, instruct=instruct.strip()
                    )
                w = wavs[0] if hasattr(wavs, "__getitem__") and not isinstance(wavs, np.ndarray) else wavs
                all_wavs.append(np.asarray(w))
            else:
                synth_audio = generate_synthetic_audio(t, seed=actual_seed)
                all_wavs.append(synth_audio)

        save_outputs(all_wavs, sr, output_file, stop_requested=self.stop_requested)
        if progress_callback:
            progress_callback(1.0, f"Generation complete! Seed: {actual_seed}")

        return {
            "output_path": output_file,
            "filename": os.path.basename(output_file),
            "actual_seed": actual_seed,
            "seed_mode": "Random" if is_random else "Fixed",
            "chunks": len(texts),
            "sample_rate": sr,
            "mode": "VoiceDesign",
        }

    def generate_voice_clone(self, text, ref_audio, ref_text=None,
                             x_vector_only=False, reuse_prompt=None, save_prompt=None,
                             language="auto", min_words=200, max_words=250,
                             output_path=None, progress_callback=None):
        """Synthesize audio using VoiceClone mode with reference audio prompt."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        if not ref_audio and not reuse_prompt:
            raise ValueError("Reference audio file or reuse prompt is required for VoiceClone mode.")

        self.stop_requested = False
        texts = self.apply_chunking(text, min_words, max_words)
        output_file = output_path or self.get_auto_output_path("voice-clone")

        voice_clone_prompt = None
        if reuse_prompt and os.path.exists(reuse_prompt):
            data = np.load(reuse_prompt)
            voice_clone_prompt = {k: data[k] for k in data.files}

        all_wavs = []
        sr = 24000

        for i, t in enumerate(texts):
            if self.stop_requested:
                break
            if progress_callback:
                progress_callback(i / len(texts), f"Cloning voice chunk {i+1}/{len(texts)}...")
            print(f"VoiceClone chunk {i+1}/{len(texts)}: {t[:60]}...")

            if self.model is not None and hasattr(self.model, "generate_voice_clone"):
                wavs, sr = self.model.generate_voice_clone(
                    text=t,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=None if x_vector_only else (ref_text or None),
                    x_vector_only_mode=x_vector_only,
                    voice_clone_prompt=voice_clone_prompt,
                )
                w = wavs[0] if hasattr(wavs, "__getitem__") and not isinstance(wavs, np.ndarray) else wavs
                all_wavs.append(np.asarray(w))
            else:
                synth_audio = generate_synthetic_audio(t, seed=999)
                all_wavs.append(synth_audio)

        if save_prompt and self.model is not None and hasattr(self.model, "create_voice_clone_prompt"):
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=None if x_vector_only else (ref_text or None),
                x_vector_only_mode=x_vector_only,
            )
            np.savez(save_prompt, **prompt)

        save_outputs(all_wavs, sr, output_file, stop_requested=self.stop_requested)
        if progress_callback:
            progress_callback(1.0, "Voice clone generation complete.")

        return {
            "output_path": output_file,
            "filename": os.path.basename(output_file),
            "chunks": len(texts),
            "sample_rate": sr,
            "mode": "VoiceClone",
        }

    def request_stop(self):
        """Signal ongoing generation to halt after the current chunk."""
        self.stop_requested = True
