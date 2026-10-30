"""TTS generation logic for Qwen3-TTS GUI."""

import sys

import numpy as np

from text_utils import chunk_text, split_text
from audio_utils import save_outputs


class TTSGenerator:
    """Handles text-to-speech generation for all three modes."""

    def __init__(self, app):
        self.app = app

    def get_language(self, mode):
        """Get the language setting for the given mode."""
        if mode == "custom-voice":
            return self.app.var_cv_lang.get()
        if mode == "voice-design":
            return self.app.var_vd_lang.get()
        if mode == "voice-clone":
            return self.app.var_vc_lang.get()
        return "auto"

    def get_texts(self, mode):
        """Get the text to synthesize for the given mode.

        The entire text content is treated as a single item. Commas are not
        used as separators.
        """
        if mode == "custom-voice":
            raw = self.app.txt_cv_text.get("1.0", "end").strip()
        elif mode == "voice-design":
            raw = self.app.txt_vd_text.get("1.0", "end").strip()
        else:
            raw = self.app.txt_vc_text.get("1.0", "end").strip()
        return [raw] if raw else []

    def apply_chunking(self, texts):
        """Apply text chunking based on min/max words settings."""
        min_words_str = self.app.var_min_words.get().strip()
        max_words_str = self.app.var_max_words.get().strip()
        min_words = int(min_words_str) if min_words_str else 200
        max_words = int(max_words_str) if max_words_str else 250

        if min_words and max_words and max_words > 0:
            chunked_texts = []
            for t in texts:
                chunks = chunk_text(t, min_words=min_words, max_words=max_words)
                chunked_texts.extend(chunks)
            texts = chunked_texts
            print(f"Text split into {len(texts)} chunk(s) ({min_words}-{max_words} words each).")
        return texts

    def generate(self, mode):
        """Run generation for the specified mode.

        This is called from a background thread.
        """
        try:
            language = self.get_language(mode)
            texts = self.get_texts(mode)
            if not texts:
                self.app.root.after(0, lambda: self.app._on_generation_done(False, "No text provided."))
                return

            texts = self.apply_chunking(texts)

            if mode == "custom-voice":
                self._gen_custom_voice(texts, language)
            elif mode == "voice-design":
                self._gen_voice_design(texts, language)
            elif mode == "voice-clone":
                self._gen_voice_clone(texts, language)

            if not self.app.stop_requested:
                self.app.root.after(0, lambda: self.app._on_generation_done(True, "Generation complete."))
            else:
                self.app.root.after(0, lambda: self.app._on_generation_done(False, "Stopped by user."))

        except Exception as e:
            error_msg = str(e)
            print(f"Generation Error: {error_msg}", file=sys.stderr)
            self.app.root.after(0, lambda: self.app._on_generation_done(False, error_msg))

    # ---- mode-specific generation

    def _gen_custom_voice(self, texts, language):
        speaker = self.app.var_cv_speaker.get()
        instruct = self.app.txt_cv_instruct.get("1.0", "end").strip() or None

        all_wavs = []
        sr = None
        for i, t in enumerate(texts):
            if self.app.stop_requested:
                break
            print(f"  Generating chunk {i+1}/{len(texts)}: {t[:70]}...")
            wavs, sr = self.app.model.generate_custom_voice(
                text=t, language=language, speaker=speaker, instruct=instruct
            )
            w = wavs[0] if hasattr(wavs, "__getitem__") else wavs
            all_wavs.append(np.asarray(w))
        self._save_outputs(all_wavs, sr)

    def _gen_voice_design(self, texts, language):
        instruct = self.app.txt_vd_instruct.get("1.0", "end").strip()
        if not instruct:
            raise ValueError("Instruct (voice description) is required for VoiceDesign mode.")

        all_wavs = []
        sr = None
        for i, t in enumerate(texts):
            if self.app.stop_requested:
                break
            print(f"  Generating chunk {i+1}/{len(texts)}: {t[:70]}...")
            wavs, sr = self.app.model.generate_voice_design(
                text=t, language=language, instruct=instruct
            )
            w = wavs[0] if hasattr(wavs, "__getitem__") else wavs
            all_wavs.append(np.asarray(w))
        self._save_outputs(all_wavs, sr)

    def _gen_voice_clone(self, texts, language):
        ref_audio = self.app.var_vc_ref_audio.get().strip()
        if not ref_audio:
            raise ValueError("Reference audio file is required for VoiceClone mode.")

        ref_text = self.app.txt_vc_ref_text.get("1.0", "end").strip() or None
        x_vector_only = self.app.var_vc_xvector.get()
        if x_vector_only:
            ref_text = None

        reuse_prompt = self.app.var_vc_reuse_prompt.get().strip() or None
        save_prompt = self.app.var_vc_save_prompt.get().strip() or None

        voice_clone_prompt = None
        if reuse_prompt:
            data = np.load(reuse_prompt)
            voice_clone_prompt = {k: data[k] for k in data.files}

        all_wavs = []
        sr = None
        for i, t in enumerate(texts):
            if self.app.stop_requested:
                break
            print(f"  Generating chunk {i+1}/{len(texts)}: {t[:70]}...")
            wavs, sr = self.app.model.generate_voice_clone(
                text=t,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                voice_clone_prompt=voice_clone_prompt,
            )
            w = wavs[0] if hasattr(wavs, "__getitem__") else wavs
            all_wavs.append(np.asarray(w))

        if save_prompt and voice_clone_prompt is None:
            voice_clone_prompt = self.app.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
            )
            np.savez(save_prompt, **voice_clone_prompt)
            self.app.root.after(0, lambda: self.app.notification_msg.set(
                f"Voice-clone prompt saved to: {save_prompt}"
            ))

        self._save_outputs(all_wavs, sr)

    def _save_outputs(self, wavs, sr):
        """Save generated outputs using the audio_utils module."""
        output = self.app.var_output.get().strip()

        def notification_callback(msg):
            self.app.root.after(0, lambda: self.app.notification_msg.set(msg))

        def generation_saved_callback(output_path):
            self.app.current_audio_file = output_path
            self.app.root.after(0, self.app._on_generation_saved, output_path)

        save_outputs(
            wavs, sr, output,
            stop_requested=self.app.stop_requested,
            notification_callback=notification_callback,
            generation_saved_callback=generation_saved_callback,
        )
