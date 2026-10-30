"""Main Qwen3-TTS GUI application class."""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame
import uuid

from config import OUTPUT_DIR
from gui_components import (
    build_notebook, build_common_controls, build_custom_voice_tab,
    build_voice_design_tab, build_voice_clone_tab, build_status_bar,
)
from model_loader import load_model
from tts_generator import TTSGenerator


class QwenTTSGUI:
    """Qwen3-TTS GUI: Simple graphical interface for CustomVoice, VoiceDesign, and VoiceClone modes."""

    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-TTS")
        self.root.geometry("1280x720")
        self.root.minsize(960, 540)

        self.model = None
        self.current_mode = tk.StringVar(value="custom-voice")
        self.generation_thread = None
        self.stop_requested = False
        self.current_audio_file = None
        self.recent_files = []
        self.notification_msg = tk.StringVar(value="")

        # Output directory (auto-managed, no user input needed)
        self.output_dir = OUTPUT_DIR

        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Warning: Could not initialize pygame mixer: {e}", file=sys.stderr)

        # Build UI
        self._build_ui()

        # Create the TTS generator (depends on UI being built)
        self.generator = TTSGenerator(self)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        build_notebook(self, self.root)
        build_common_controls(self, self.root)
        build_custom_voice_tab(self)
        build_voice_design_tab(self)
        build_voice_clone_tab(self)
        build_status_bar(self, self.root)

    # ------------------------------------------------------------------ callbacks
    def _on_tab_change(self, event=None):
        """Auto-update Model ID and output path based on the active tab."""
        tab = self.notebook.select()
        tab_text = self.notebook.tab(tab, "text")
        from config import TAB_MODE_MAP, MODEL_MAP
        mode = TAB_MODE_MAP.get(tab_text, "custom-voice")
        self.current_mode.set(mode)
        self.var_model.set(MODEL_MAP[mode])
        self.var_output.set(self._get_auto_output_path())

    def _get_auto_output_path(self):
        """Generate an auto output path based on the current tab and timestamp."""
        from datetime import datetime
        from config import MODE_TAB_MAP
        mode = self.current_mode.get()
        tab_name = MODE_TAB_MAP.get(mode, "CustomVoice")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:3]
        filename = f"{tab_name}_{timestamp}_{unique}.wav"
        return os.path.join(self.output_dir, filename)

    def _on_speaker_change(self, event=None):
        from config import SUPPORTED_SPEAKERS
        speaker = self.var_cv_speaker.get()
        desc = SUPPORTED_SPEAKERS.get(speaker, "")
        self.lbl_cv_speaker_desc.config(text=desc)

    def _list_speakers(self):
        from config import SUPPORTED_SPEAKERS
        lines = ["Supported speakers for CustomVoice models:\n"]
        for spk, desc in SUPPORTED_SPEAKERS.items():
            lines.append(f"  {spk}: {desc}")
        self.notification_msg.set("\n".join(lines))

    def _browse_playback_file(self):
        """Open a file dialog to select a WAV file for playback."""
        path = filedialog.askopenfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=self.output_dir,
        )
        if path:
            self.current_audio_file = path
            self.var_output.set(path)
            self.btn_play.config(state=tk.NORMAL)
            self.btn_play_stop.config(state=tk.NORMAL)
            self.lbl_playback_status.config(text=f"Loaded: {os.path.basename(path)}", foreground="green")
            self.lbl_status.config(text=f"Ready - {os.path.basename(path)}", foreground="green")
            print(f"Loaded playback file: {path}", file=sys.stderr)

    def _browse_ref_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg")])
        if path:
            self.var_vc_ref_audio.set(path)

    def _browse_reuse_prompt(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if path:
            self.var_vc_reuse_prompt.set(path)

    def _browse_save_prompt(self):
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ files", "*.npz")])
        if path:
            self.var_vc_save_prompt.set(path)

    def _load_model(self):
        if self.generation_thread and self.generation_thread.is_alive():
            self.notification_msg.set("Busy: A generation is already in progress.")
            return

        model_id = self.var_model.get().strip()
        if not model_id:
            print("Error: Please enter a model ID.", file=sys.stderr)
            self.notification_msg.set("Error: Please enter a model ID.")
            return

        self.lbl_model_status.config(text="Loading model…", foreground="orange")
        self.progress.start()
        self.root.update()

        def _load():
            try:
                import torch
                torch_dtype = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }[self.var_dtype.get()]

                device = self.var_device.get()
                attn = self.var_attn.get()

                m = load_model(model_id, device, self.var_dtype.get(), attn)
                self.model = m
                self.root.after(0, lambda: self._on_model_loaded(True, model_id))
            except Exception as e:
                self.root.after(0, lambda: self._on_model_loaded(False, str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, success, info):
        self.progress.stop()
        if success:
            self.lbl_model_status.config(text=f"Model loaded: {info}", foreground="green")
        else:
            self.lbl_model_status.config(text=f"Load failed: {info}", foreground="red")
            print(f"Model Load Error: {info}", file=sys.stderr)
            self.notification_msg.set(f"Model Load Error: {info}")

    # ------------------------------------------------------------------ generation
    def _start_generation(self, mode):
        if self.generation_thread and self.generation_thread.is_alive():
            self.notification_msg.set("Busy: A generation is already in progress.")
            return

        if self.model is None:
            print("Error: Please load a model first.", file=sys.stderr)
            self.notification_msg.set("Error: Please load a model first.")
            return

        # Ensure output path is set for current tab
        self.var_output.set(self._get_auto_output_path())

        self.stop_requested = False
        self.btn_stop.config(state=tk.NORMAL)
        self.progress.start()
        self.lbl_status.config(text="Generating…", foreground="orange")

        self.generation_thread = threading.Thread(target=self.generator.generate, args=(mode,), daemon=True)
        self.generation_thread.start()

    def _request_stop(self):
        self.stop_requested = True
        self.lbl_status.config(text="Stop requested…", foreground="orange")

    # ------------------------------------------------------------------ playback
    def _play_audio(self):
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            print("Error: No audio file to play.", file=sys.stderr)
            self.notification_msg.set("Error: No audio file to play.")
            return
        try:
            pygame.mixer.music.load(self.current_audio_file)
            pygame.mixer.music.set_volume(self.var_volume.get())
            pygame.mixer.music.play()
            self.lbl_playback_status.config(text=f"Playing: {os.path.basename(self.current_audio_file)}", foreground="blue")
        except Exception as e:
            print(f"Playback Error: {e}", file=sys.stderr)
            self.notification_msg.set(f"Playback Error: {e}")

    def _stop_audio(self):
        pygame.mixer.music.stop()
        self.lbl_playback_status.config(text="Playback stopped", foreground="green")

    def _set_volume(self, val):
        pygame.mixer.music.set_volume(float(val))

    # ------------------------------------------------------------------ completion callbacks
    def _on_generation_done(self, success, message):
        self.progress.stop()
        self.btn_stop.config(state=tk.DISABLED)
        if success:
            self.lbl_status.config(text=message, foreground="green")
        else:
            self.lbl_status.config(text=f"Error: {message}", foreground="red")
            print(f"Generation Error: {message}", file=sys.stderr)
            self.notification_msg.set(f"Generation Error: {message}")

    def _on_generation_saved(self, output):
        self.notification_msg.set(f"Audio saved to: {output}")
        self.btn_play.config(state=tk.NORMAL)
        self.btn_play_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Ready - {os.path.basename(output)}", foreground="green")
