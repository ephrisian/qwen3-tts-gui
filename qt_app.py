"""Qt (PySide6) front end for Qwen3-TTS.

All model work lives in tts_generator / voice_library / model_loader; this file
only builds widgets and runs the backend on worker threads.
"""

import os
import random
import sys
import threading
import uuid
from datetime import datetime

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import audio_player
import voice_library
from audio_utils import save_outputs, write_wav
from config import (
    DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_LANGUAGE, DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS, ATTNS, DTYPES, LANGUAGES, MODEL_MAP, OUTPUT_DIR,
)
from model_loader import ModelManager
from qt_audiobook import AudiobookTab
from tts_generator import TTSGenerator
from voice_lock import EMOTION_PRESETS

DIRECTION_HELP = (
    "Direct the performance like an actor: put a [tag] before a passage and it applies until the next "
    "tag. Use a preset ([angry], [whisper], [sad]…) or any free-text style, e.g. "
    "[shaky voice, holding back tears]. Emphasis is shaped by wording and punctuation (!, ?, …, "
    "short sentences); Qwen3-TTS has no markup for stressing single words."
)


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class Task(QtCore.QObject):
    """Run a function on a plain thread and report back on the GUI thread."""

    notify = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()

    def start(self, fn):
        def run():
            try:
                self.done.emit(fn(self))
            except Exception as e:  # surfaced in the status bar
                print(f"Error: {e}", file=sys.stderr)
                self.failed.emit(str(e))
        threading.Thread(target=run, daemon=True).start()


def text_edit(height_lines, placeholder="", text="", read_only=False):
    """QPlainTextEdit gives native undo/redo (Ctrl+Z, Ctrl+Y/Ctrl+Shift+Z) and Ctrl+A."""
    w = QtWidgets.QPlainTextEdit()
    w.setUndoRedoEnabled(True)
    w.setPlaceholderText(placeholder)
    w.setReadOnly(read_only)
    if text:
        w.setPlainText(text)
        w.document().clearUndoRedoStacks()
    fm = w.fontMetrics()
    w.setMinimumHeight(fm.lineSpacing() * height_lines + 16)
    return w


def combo(items, current=None):
    c = QtWidgets.QComboBox()
    c.addItems(items)
    if current is not None:
        c.setCurrentText(current)
    return c


def seed_row(default_mode, default_value):
    mode = combo(["Fixed", "Random"], default_mode)
    value = QtWidgets.QSpinBox()
    value.setRange(0, 2**31 - 1)
    value.setValue(default_value)
    value.setEnabled(default_mode == "Fixed")
    mode.currentTextChanged.connect(lambda m: value.setEnabled(m == "Fixed"))
    return mode, value


def tab_page(body_layout_builder):
    """Return (page, body_layout, bar_layout): scrollable body + fixed action bar."""
    page = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(page)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    body = QtWidgets.QWidget()
    body_layout = QtWidgets.QVBoxLayout(body)
    scroll.setWidget(body)
    outer.addWidget(scroll, 1)
    bar = QtWidgets.QHBoxLayout()
    outer.addLayout(bar)
    return page, body_layout, bar


class DirectionBar(QtWidgets.QWidget):
    """Insert an [emotion] tag at the cursor of a text editor."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Direction:"))
        self.preset = combo(list(EMOTION_PRESETS))
        lay.addWidget(self.preset)
        btn = QtWidgets.QPushButton("Insert [tag] at cursor")
        btn.clicked.connect(self.insert)
        lay.addWidget(btn)
        lay.addStretch(1)

    def insert(self):
        self.editor.insertPlainText(f"[{self.preset.currentText()}] ")
        self.editor.setFocus()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, models=None):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS")
        self.models = models or ModelManager(self.model_settings)
        self.task = None
        self.busy = False
        self.pending_voice = None
        self.last_output = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_settings())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_design_tab(), "Voice Creator (VoiceDesign)")
        self.tabs.addTab(self._build_clone_tab(), "VoiceClone")
        self.audiobook_tab = AudiobookTab(self, text_edit, combo, seed_row)
        self.tabs.addTab(self.audiobook_tab, "Audiobook")
        root.addWidget(self.tabs, 1)
        root.addLayout(self._build_status_bar())
        self.refresh_voice_lists()

        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1100, screen.width() - 40), min(860, screen.height() - 60))

    # ------------------------------------------------------------------ settings / status
    def model_settings(self):
        return (self.cmb_device.currentText(), self.cmb_dtype.currentText(), self.cmb_attn.currentText())

    def _build_settings(self):
        box = QtWidgets.QGroupBox("Global Settings")
        lay = QtWidgets.QHBoxLayout(box)
        self.cmb_device = combo(["cuda", "cpu"], DEFAULT_DEVICE if cuda_available() else "cpu")
        self.cmb_dtype = combo(DTYPES, DEFAULT_DTYPE)
        self.cmb_attn = combo(ATTNS, DEFAULT_ATTN)
        self.spn_min = QtWidgets.QSpinBox()
        self.spn_min.setRange(0, 5000)
        self.spn_min.setValue(int(DEFAULT_MIN_WORDS))
        self.spn_max = QtWidgets.QSpinBox()
        self.spn_max.setRange(0, 5000)
        self.spn_max.setValue(int(DEFAULT_MAX_WORDS))
        for label, w in (("Device", self.cmb_device), ("Dtype", self.cmb_dtype), ("Attention", self.cmb_attn),
                         ("Min words/chunk", self.spn_min), ("Max words/chunk", self.spn_max)):
            lay.addWidget(QtWidgets.QLabel(label + ":"))
            lay.addWidget(w)
        lay.addStretch(1)
        box.setToolTip("Models load automatically the first time a tab needs them.")
        return box

    def _build_status_bar(self):
        lay = QtWidgets.QVBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        lay.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        row.addWidget(self.lbl_status, 1)
        self.btn_play_last = QtWidgets.QPushButton("▶ Play last output")
        self.btn_play_last.setEnabled(False)
        self.btn_play_last.clicked.connect(lambda: self.play(self.last_output))
        self.btn_stop_audio = QtWidgets.QPushButton("■ Stop audio")
        self.btn_stop_audio.clicked.connect(audio_player.stop)
        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(OUTPUT_DIR)))
        self.btn_stop = QtWidgets.QPushButton("Stop generation")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        for b in (self.btn_play_last, self.btn_stop_audio, self.btn_open_out, self.btn_stop):
            row.addWidget(b)
        lay.addLayout(row)
        return lay

    def status(self, msg):
        self.lbl_status.setText(msg)

    def play(self, path):
        try:
            audio_player.play(path)
        except Exception as e:
            self.status(f"Playback error: {e}")

    def request_stop(self):
        if self.task:
            self.task.stop_event.set()
            self.status("Stop requested…")

    def set_busy(self, busy):
        self.busy = busy
        self.progress.setRange(0, 0 if busy else 1)
        self.btn_stop.setEnabled(busy)
        for b in self.action_buttons:
            b.setEnabled(not busy)

    def run_task(self, fn, on_done):
        if self.busy:
            self.status("Busy: a task is already running.")
            return
        task = Task()
        self.task = task
        task.notify.connect(self.status)
        task.done.connect(lambda r: (self.set_busy(False), on_done(r)))
        task.failed.connect(lambda m: (self.set_busy(False), self.status(f"Error: {m}")))
        self.set_busy(True)
        task.start(fn)

    def generator(self, task):
        return TTSGenerator(self.models, notify=task.notify.emit, should_stop=task.stop_event.is_set)

    def output_path(self, prefix):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(OUTPUT_DIR, f"{prefix}_{stamp}_{uuid.uuid4().hex[:3]}.wav")

    def finish_long_form(self, prefix):
        def on_done(result):
            wavs, sr = result
            if not wavs:
                self.status("Stopped before any audio was produced.")
                return
            path = self.output_path(prefix)
            save_outputs(wavs, sr, path)
            self.last_output = path
            self.btn_play_last.setEnabled(True)
            self.status(("Stopped early; partial audio saved to: " if self.task.stop_event.is_set()
                         else "Done. Audio saved to: ") + path)
        return on_done

    # ------------------------------------------------------------------ Voice Creator
    def _build_design_tab(self):
        page, lay, bar = tab_page(None)
        lay.addWidget(QtWidgets.QLabel("Sample text (fixed — every voice is auditioned on the same passage):"))
        lay.addWidget(text_edit(6, text=voice_library.SAMPLE_TEXT, read_only=True))

        lay.addWidget(QtWidgets.QLabel("Voice description:"))
        self.txt_vd_instruct = text_edit(
            5, "Describe timbre, age, gender, pace, accent, mood…",
            "Warm, gentle female voice, calm and soothing tone.")
        lay.addWidget(self.txt_vd_instruct)

        form = QtWidgets.QFormLayout()
        self.cmb_vd_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        form.addRow("Language:", self.cmb_vd_lang)
        self.cmb_vd_seed_mode, self.spn_vd_seed = seed_row("Random", 42)
        self.lbl_vd_seed_used = QtWidgets.QLabel("")
        seed_lay = QtWidgets.QHBoxLayout()
        for w in (self.cmb_vd_seed_mode, self.spn_vd_seed, self.lbl_vd_seed_used):
            seed_lay.addWidget(w)
        seed_lay.addStretch(1)
        form.addRow("Seed:", seed_lay)
        lay.addLayout(form)
        note = QtWidgets.QLabel(
            "Random tries a new voice each time (the seed used is recorded with the saved voice). "
            "Fixed repeats the same sampling. A saved voice is locked via its reference clip, not the seed.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        lay.addWidget(note)

        saved = QtWidgets.QGroupBox("Saved voices")
        sl = QtWidgets.QHBoxLayout(saved)
        self.cmb_vd_saved = QtWidgets.QComboBox()
        self.cmb_vd_saved.setMinimumWidth(220)
        btn_load = QtWidgets.QPushButton("Load into creator")
        btn_load.clicked.connect(self.load_saved_into_creator)
        btn_play = QtWidgets.QPushButton("▶ Play reference")
        btn_play.clicked.connect(self.play_saved_reference)
        btn_del = QtWidgets.QPushButton("Delete")
        btn_del.clicked.connect(self.delete_saved)
        for w in (self.cmb_vd_saved, btn_load, btn_play, btn_del):
            sl.addWidget(w)
        sl.addStretch(1)
        lay.addWidget(saved)
        lay.addStretch(1)

        self.btn_vd_generate = QtWidgets.QPushButton("Generate sample")
        self.btn_vd_generate.clicked.connect(self.generate_sample)
        self.btn_vd_play = QtWidgets.QPushButton("▶ Play sample")
        self.btn_vd_play.clicked.connect(self.play_sample)
        self.edt_vd_name = QtWidgets.QLineEdit()
        self.edt_vd_name.setPlaceholderText("Voice name")
        self.btn_vd_save = QtWidgets.QPushButton("Save voice")
        self.btn_vd_save.setEnabled(False)
        self.btn_vd_save.clicked.connect(self.save_voice)
        for w in (self.btn_vd_generate, self.btn_vd_play):
            bar.addWidget(w)
        bar.addSpacing(20)
        bar.addWidget(self.edt_vd_name, 1)
        bar.addWidget(self.btn_vd_save)
        return page

    def generate_sample(self):
        description = self.txt_vd_instruct.toPlainText().strip()
        if not description:
            self.status("Error: enter a voice description first.")
            return
        language = self.cmb_vd_lang.currentText()
        seed = (self.spn_vd_seed.value() if self.cmb_vd_seed_mode.currentText() == "Fixed"
                else random.randint(0, 2**31 - 1))
        audio_player.stop()

        def work(task):
            wav, sr = self.generator(task).design_sample(description, language, seed)
            preview = os.path.join(OUTPUT_DIR, "_voice_preview.wav")
            write_wav(preview, wav, sr)
            return dict(wav=wav, sr=sr, seed=seed, description=description,
                        language=language, preview=preview)

        def done(result):
            self.pending_voice = result
            self.lbl_vd_seed_used.setText(f"Seed used: {result['seed']}")
            self.btn_vd_save.setEnabled(True)
            self.status("Sample ready. Listen, then name and save the voice if you like it.")
            self.play(result["preview"])

        self.run_task(work, done)

    def play_sample(self):
        if self.pending_voice:
            self.play(self.pending_voice["preview"])
        else:
            self.status("Generate a sample first.")

    def save_voice(self):
        pending = self.pending_voice
        name = self.edt_vd_name.text().strip()
        if not pending:
            self.status("Error: generate and audition a sample first.")
            return
        if not voice_library.slugify(name):
            self.status("Error: enter a voice name.")
            return
        overwrite = False
        if voice_library.voice_exists(name):
            answer = QtWidgets.QMessageBox.question(
                self, "Overwrite voice?", f"A voice named '{name}' already exists. Replace it?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
            overwrite = True

        def work(task):
            prompt = self.generator(task).build_voice_prompt(pending["wav"], pending["sr"])
            voice_library.save_voice(
                name, pending["wav"], pending["sr"], prompt, pending["description"],
                pending["language"], pending["seed"],
                model_ids=dict(MODEL_MAP),
                overwrite=overwrite,
            )
            return name

        def done(saved_name):
            self.refresh_voice_lists()
            self.cmb_vd_saved.setCurrentText(saved_name)
            self.status(f"Voice '{saved_name}' saved to voices/{voice_library.slugify(saved_name)}/")

        self.run_task(work, done)

    def load_saved_into_creator(self):
        name = self.cmb_vd_saved.currentText()
        if not name:
            return
        cfg, _, _ = voice_library.load_voice(name)
        self.txt_vd_instruct.setPlainText(cfg["description"])
        self.cmb_vd_lang.setCurrentText(cfg["language"])
        self.cmb_vd_seed_mode.setCurrentText("Fixed")
        self.spn_vd_seed.setValue(cfg["seed"])
        self.edt_vd_name.setText(cfg["name"])
        self.status(f"Voice '{name}' loaded into the creator.")

    def play_saved_reference(self):
        name = self.cmb_vd_saved.currentText()
        if name:
            self.play(os.path.join(voice_library.voice_dir(name), "reference.wav"))

    def delete_saved(self):
        name = self.cmb_vd_saved.currentText()
        if not name:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete voice?", f"Delete '{name}' and all its files?")
        if answer == QtWidgets.QMessageBox.Yes:
            voice_library.delete_voice(name)
            self.refresh_voice_lists()
            self.status(f"Voice '{name}' deleted.")

    def refresh_voice_lists(self):
        names = voice_library.list_voices()
        for cmb, items in ((self.cmb_vd_saved, names), (self.cmb_vc_voice, [""] + names)):
            current = cmb.currentText()
            cmb.clear()
            cmb.addItems(items)
            cmb.setCurrentText(current if current in items else "")
        if hasattr(self, "audiobook_tab"):
            self.audiobook_tab.refresh_voices()

    # ------------------------------------------------------------------ VoiceClone
    def _build_clone_tab(self):
        page, lay, bar = tab_page(None)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Saved voice:"))
        self.cmb_vc_voice = QtWidgets.QComboBox()
        self.cmb_vc_voice.setMinimumWidth(240)
        self.cmb_vc_voice.currentTextChanged.connect(self.on_clone_voice_changed)
        row.addWidget(self.cmb_vc_voice)
        btn = QtWidgets.QPushButton("Refresh")
        btn.clicked.connect(self.refresh_voice_lists)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addWidget(QtWidgets.QLabel("Text to synthesize:"))
        self.txt_vc_text = text_edit(10, text="Hello, I am cloning this voice.")
        lay.addWidget(self.txt_vc_text, 1)
        lay.addWidget(DirectionBar(self.txt_vc_text))
        help_lbl = QtWidgets.QLabel(
            "The Base (clone) model takes no instructions, so each [tag] is performed by generating a short "
            "emotional reference with VoiceDesign (voice description + tag), anchored to your voice's speaker "
            "embedding, and cloning from it. Identity stays close but not identical to the neutral voice; "
            "audition emotional passages. " + DIRECTION_HELP)
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: gray")
        lay.addWidget(help_lbl)

        form = QtWidgets.QFormLayout()
        self.cmb_vc_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        form.addRow("Language:", self.cmb_vc_lang)
        self.cmb_vc_seed_mode, self.spn_vc_seed = seed_row("Fixed", 42)
        self.spn_vc_temp = QtWidgets.QDoubleSpinBox()
        self.spn_vc_temp.setRange(0.1, 1.5)
        self.spn_vc_temp.setSingleStep(0.05)
        self.spn_vc_temp.setValue(0.7)
        self.spn_vc_temp.setToolTip("Lower = steadier voice, flatter delivery. Library default is 0.9.")
        r = QtWidgets.QHBoxLayout()
        for w in (self.cmb_vc_seed_mode, self.spn_vc_seed, QtWidgets.QLabel("Temperature:"), self.spn_vc_temp):
            r.addWidget(w)
        r.addStretch(1)
        form.addRow("Seed:", r)
        lay.addLayout(form)

        lay.addWidget(QtWidgets.QLabel("Voice description (used for [tag] direction; filled from a saved voice):"))
        self.txt_vc_instruct = text_edit(3, "e.g. Warm, gentle female voice, calm and soothing tone.")
        lay.addWidget(self.txt_vc_instruct)

        self.grp_ref = QtWidgets.QGroupBox("Or clone from reference audio (ignored when a saved voice is selected)")
        g = QtWidgets.QFormLayout(self.grp_ref)
        self.edt_vc_ref = QtWidgets.QLineEdit()
        b1 = QtWidgets.QPushButton("Browse…")
        b1.clicked.connect(lambda: self.browse(self.edt_vc_ref, "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"))
        g.addRow("Reference audio:", self.row_with(self.edt_vc_ref, b1))
        self.txt_vc_ref_text = text_edit(3, "Exact transcript of the reference audio")
        g.addRow("Transcript:", self.txt_vc_ref_text)
        self.chk_vc_xvec = QtWidgets.QCheckBox("X-vector only (speaker embedding only, no transcript needed)")
        g.addRow("", self.chk_vc_xvec)
        self.edt_vc_reuse = QtWidgets.QLineEdit()
        b2 = QtWidgets.QPushButton("Browse…")
        b2.clicked.connect(lambda: self.browse(self.edt_vc_reuse, "NPZ (*.npz)"))
        g.addRow("Reuse prompt (NPZ):", self.row_with(self.edt_vc_reuse, b2))
        self.edt_vc_save = QtWidgets.QLineEdit()
        b3 = QtWidgets.QPushButton("Browse…")
        b3.clicked.connect(lambda: self.browse(self.edt_vc_save, "NPZ (*.npz)", save=True))
        g.addRow("Save prompt to (NPZ):", self.row_with(self.edt_vc_save, b3))
        lay.addWidget(self.grp_ref)

        self.btn_vc_generate = QtWidgets.QPushButton("Generate")
        self.btn_vc_generate.clicked.connect(self.generate_clone)
        bar.addWidget(self.btn_vc_generate)
        bar.addStretch(1)
        return page

    @staticmethod
    def row_with(edit, button):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        lay.addWidget(button)
        return w

    def browse(self, edit, flt, save=False):
        fn = QtWidgets.QFileDialog.getSaveFileName if save else QtWidgets.QFileDialog.getOpenFileName
        path, _ = fn(self, "Select file", "", flt)
        if path:
            edit.setText(path)

    def on_clone_voice_changed(self, name):
        self.grp_ref.setEnabled(not name)
        if name:
            try:
                cfg, _, _ = voice_library.load_voice(name)
            except Exception as e:
                self.status(f"Could not load voice: {e}")
                return
            self.txt_vc_instruct.setPlainText(cfg["description"])
            self.cmb_vc_lang.setCurrentText(cfg["language"])
            self.status(f"Using saved voice '{name}'.")

    def generate_clone(self):
        text = self.txt_vc_text.toPlainText().strip()
        if not text:
            self.status("Error: no text provided.")
            return
        args = dict(
            text=text, language=self.cmb_vc_lang.currentText(),
            min_words=self.spn_min.value(), max_words=self.spn_max.value(),
            seed=self.spn_vc_seed.value() if self.cmb_vc_seed_mode.currentText() == "Fixed" else None,
            temperature=self.spn_vc_temp.value(),
            voice_name=self.cmb_vc_voice.currentText() or None,
            ref_audio=self.edt_vc_ref.text().strip(),
            ref_text=self.txt_vc_ref_text.toPlainText().strip() or None,
            x_vector_only=self.chk_vc_xvec.isChecked(),
            reuse_prompt=self.edt_vc_reuse.text().strip() or None,
            save_prompt=self.edt_vc_save.text().strip() or None,
            description=self.txt_vc_instruct.toPlainText(),
        )
        self.run_task(lambda t: self.generator(t).clone(**args), self.finish_long_form("VoiceClone"))

    @property
    def action_buttons(self):
        return (self.btn_vd_generate, self.btn_vd_save, self.btn_vc_generate,
                self.audiobook_tab.btn_render, self.audiobook_tab.btn_preview)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
