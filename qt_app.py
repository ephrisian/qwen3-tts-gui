"""Qt (PySide6) front end for Qwen3-TTS.

All model work lives in tts_generator / voice_library / model_loader; this file
only builds widgets and runs the backend on worker threads.
"""

import json
import os
import random
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import audio_player
import icons
import task_errors
import voice_library
import voice_list
from audio_utils import (save_outputs, write_wav, load_wav, load_voice_clone_prompt, audio_problem,
                         audio_file_filter)
from config import (
    DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_LANGUAGE, DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS, ATTNS, DTYPES, LANGUAGES, MODELS, MODEL_MAP, OUTPUT_DIR,
)
from model_loader import ModelManager
from drama_studio import (PROVIDERS as DRAMA_PROVIDERS, environment_key as drama_env_key,
                          load_settings as load_drama_settings, save_settings as save_drama_settings)
from qt_audiobook import AudiobookTab
from qt_drama import DramaTab
from theme import COLORS, SIDEBAR_WIDTH, apply_theme, set_role, style_numeric
from tts_generator import TTSGenerator
from voice_lock import EMOTION_PRESETS, parse_emotion_segments


# Explanations behind the (i) buttons in Clone Voice > Advanced settings. Kept to what the backend does:
# Seed is applied with seed_everything() before each chunk; Temperature is the sampling temperature
# (talker and sub-talker); X-vector only and the prompt file follow qwen_tts' own descriptions.
SETTING_LABEL_WIDTH = 120
SEED_TIP = ("Fixed starts every chunk from the same random state, which makes results more repeatable "
            "(it does not guarantee identical audio). Random picks a new one each time.")
TEMPERATURE_TIP = ("How much the delivery varies. Lower is steadier and more conservative, higher is more "
                   "varied. Range 0.1 to 1.5; the default here is 0.7.")
DIRECTION_TAGS_TIP = ("Put a [tag] before a passage to change how it is delivered; it applies until the next tag. "
                      "Use a preset such as [sad] or [whisper], or any short style like [shaky voice]. "
                      "Tags need a voice description (a saved voice's own, or the Description field in Reference audio): the "
                      "clone model takes no instructions, so each tag is performed from a short reference made with it.")
DESCRIPTION_TIP = ("A short description of the voice, saved with it. It is used only for [tag] directions: each tag is "
                   "performed from a short reference made with this description, so a saved voice without one can't "
                   "use tags. Generating here from a recording or prompt file with tags uses the same description.")
XVECTOR_TIP = ("Uses only the speaker embedding of the recording, so no transcript is needed. "
               "Without it, the transcript must match the recording.")
PROMPT_FILE_TIP = ("Reuses a clone prompt (.npz) saved earlier instead of the recording. "
                   "Only open prompt files you trust.")

# Optional starting points for the voice description. Selecting one appends its phrase;
# deselecting (or editing it away) removes only that phrase, so typed text is never lost.
SUGGESTIONS = (
    ("Narration", "Clear, steady narrator voice, ideal for narration."),
    ("Relaxing", "Soft, slow, soothing and relaxed delivery."),
    ("Epic", "Deep, powerful, dramatic trailer-style voice."),
    ("Professional", "Polished, confident, neutral professional tone."),
)

# Create Voice content stops growing past this width so wide windows leave whitespace, not stretched fields.
CONTENT_MAX_WIDTH = 1120
PAGE_CREATE = 0

# Starts a Create Voice generation. Qt maps Ctrl to Command on macOS, so show the native spelling.
# Lucide icon for each sidebar entry; the text label always stays next to it.
NAV_ICONS = {"Create Voice": "wand-sparkles", "Clone Voice": "copy", "My Voices": "mic-vocal",
             "Audiobooks": "book-open", "Drama Studio": "play", "Settings": "settings"}
GENERATE_SHORTCUT = "Ctrl+Return"
READY_STATUS = "Sample ready. Listen, then name and save the voice if you like it."
STOP_AUDIO_TIP = "Stop the audio that is playing"
STALE_STATUS = "Showing the previous sample. Generate voice to hear your changes."
# Create Voice generation stages. Only stages the backend really reports are shown: the model
# manager announces "Loading <kind> model…" (a first load may download it, and it does not say which),
# and the app itself knows when the model call has returned and generation starts.
STAGE_PREPARING = "Preparing…"
STAGE_LOADING = "Loading the voice model…"
STAGE_GENERATING = "Generating the sample…"
LOADING_HINT = "The first generation may download the voice model. This is required once."
GENERATING_HINT = "This may take several minutes, depending on your device and sample length."
FOOTER_CREATING = "Creating the sample…"
SEEK_STEP = 5.0   # seconds for the back / forward buttons


def shortcut_text(sequence=GENERATE_SHORTCUT):
    return QtGui.QKeySequence(sequence).toString(QtGui.QKeySequence.NativeText)


# Preview panel pages (Create Voice).
PV_EMPTY, PV_BUSY, PV_READY, PV_ERROR = range(4)
CV_EMPTY, CV_BUSY, CV_READY, CV_ERROR, CV_PARTIAL, CV_STOPPED, CV_SAVED = range(7)   # Clone Voice result surface states


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
        self.failure = None   # task_errors.Failure of the last exception: type, message, traceback

    def start(self, fn):
        def run():
            try:
                self.done.emit(fn(self))
            except BaseException as e:  # a worker thread cannot propagate; never leave the UI busy
                self.failure = task_errors.Failure.from_exception(e)
                task_errors.log_failure(self.failure)   # full traceback to the console
                self.failed.emit(self.failure.message)
        threading.Thread(target=run, daemon=True).start()


def text_edit(height_lines, placeholder="", text="", read_only=False):
    """QPlainTextEdit gives native undo/redo (Ctrl+Z, Ctrl+Y/Ctrl+Shift+Z) and Ctrl+A."""
    w = QtWidgets.QPlainTextEdit()
    w.setUndoRedoEnabled(True)
    w.setTabChangesFocus(True)  # Tab moves on instead of inserting a tab character (keyboard trap)
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
    style_numeric(value, name="Seed value")
    mode.setAccessibleName("Seed mode")
    mode.currentTextChanged.connect(lambda m: value.setEnabled(m == "Fixed"))
    return mode, value


def page_header(title, subtitle=None):
    """Screen heading: one title and at most one short description."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 8)
    lay.setSpacing(4)
    lbl = QtWidgets.QLabel(title)
    set_role(lbl, "title")
    lay.addWidget(lbl)
    if subtitle:
        sub = QtWidgets.QLabel(subtitle)
        sub.setWordWrap(True)
        set_role(sub, "subtitle")
        lay.addWidget(sub)
    return box


def format_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class InfoPopup(QtWidgets.QLabel):
    """The explanation shown while an InfoButton has keyboard focus. A native tooltip can't be used
    for this: Qt dismisses it on the very next key event, which is the Tab key's own release."""

    _shared = None
    owner = None

    def __init__(self):
        super().__init__(None, QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint)
        self.setObjectName("infoPopup")
        self.setWordWrap(True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(300)
        self.setTextFormat(QtCore.Qt.PlainText)

    @classmethod
    def show_for(cls, button, text):
        if cls._shared is None:
            cls._shared = cls()
        pop = cls._shared
        cls.owner = button
        pop.setText(text)
        pop.adjustSize()
        pos = button.mapToGlobal(QtCore.QPoint(0, button.height() + 4))
        screen = button.screen().availableGeometry() if button.screen() else None
        if screen:   # keep it on screen
            pos.setX(max(screen.left(), min(pos.x(), screen.right() - pop.width())))
            if pos.y() + pop.height() > screen.bottom():
                pos.setY(button.mapToGlobal(QtCore.QPoint(0, 0)).y() - pop.height() - 4)
        pop.move(pos)
        pop.show()

    @classmethod
    def hide_popup(cls):
        cls.owner = None
        if cls._shared is not None:
            cls._shared.hide()

    @classmethod
    def is_open(cls):
        return cls._shared is not None and cls._shared.isVisible()


class InfoButton(QtWidgets.QPushButton):
    """A small keyboard-focusable (i) beside a label. The explanation shows on hover (native tooltip) and
    while it has keyboard focus (InfoPopup), is read by screen readers as the description, and Escape
    dismisses it."""

    def __init__(self, topic, text, parent=None):
        super().__init__(parent)
        self.topic, self.text_plain = topic, text
        set_role(self, "info")
        icons.apply(self, "info", 14)
        self.setIconSize(QtCore.QSize(14, 14))
        self.setFixedSize(22, 22)
        self.setCursor(QtCore.Qt.WhatsThisCursor)
        self.setAccessibleName(f"About {topic}")
        self.setAccessibleDescription(text)
        # rich text is what makes Qt wrap a tooltip instead of stretching it across the screen
        self.setToolTip(f"<p style='white-space:normal; width:260px'>{text}</p>")

    def event(self, event):
        if event.type() == QtCore.QEvent.ToolTip:   # hover: one bubble style for mouse and keyboard
            InfoPopup.show_for(self, self.text_plain)
            return True
        return super().event(event)

    def mousePressEvent(self, event):
        if InfoPopup.is_open() and InfoPopup.owner is self and not self.hasFocus():
            InfoPopup.hide_popup()
        else:
            InfoPopup.show_for(self, self.text_plain)
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if not self.hasFocus():
            InfoPopup.hide_popup()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if event.reason() in (QtCore.Qt.TabFocusReason, QtCore.Qt.BacktabFocusReason, QtCore.Qt.ShortcutFocusReason):
            InfoPopup.show_for(self, self.text_plain)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        InfoPopup.hide_popup()

    def hideEvent(self, event):
        super().hideEvent(event)
        InfoPopup.hide_popup()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape and InfoPopup.is_open():
            InfoPopup.hide_popup()
            return
        if event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            InfoPopup.show_for(self, self.text_plain)   # activating it shows the explanation again
            return
        super().keyPressEvent(event)


class WrapLabel(QtWidgets.QLabel):
    """Word-wrapped label that reserves the height its text needs at its current width.

    Box layouts don't always hand a width to height-for-width children, which clips wrapped text.
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def setText(self, text):
        super().setText(text)
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        height = self.heightForWidth(self.width()) if self.width() > 0 else 0
        if height > 0 and height != self.minimumHeight():
            self.setMinimumHeight(height)


class PageSwitcher(QtWidgets.QWidget):
    """Shows one page at a time. Unlike QStackedWidget, hidden pages take no part in layout, so the
    container is exactly as tall as the visible page (a stack sizes itself to its tallest page)."""

    def __init__(self):
        super().__init__()
        self._pages = []
        self._index = -1
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def insertWidget(self, index, widget):
        self._pages.insert(index, widget)
        self.layout().addWidget(widget)
        widget.setVisible(False)

    def setCurrentIndex(self, index):
        self._index = index
        for i, page in enumerate(self._pages):
            page.setVisible(i == index)

    def currentIndex(self):
        return self._index

    def currentWidget(self):
        return self._pages[self._index]

    def widget(self, index):
        return self._pages[index]

    def count(self):
        return len(self._pages)


class WaveformView(QtWidgets.QWidget):
    """Peak bars drawn from the real generated samples. The played part takes the accent colour and a
    playhead marks the real playback position; a click seeks (the keyboard uses the seek slider)."""

    BARS = 48
    seek_requested = QtCore.Signal(float)   # 0..1 of the sample's length

    def __init__(self):
        super().__init__()
        self._peaks = []
        self._progress = None   # None: nothing is playing or paused, so no playhead
        self.setFixedHeight(56)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAccessibleName("Waveform of the generated sample")

    def set_samples(self, wav):
        a = np.abs(np.asarray(wav, dtype=np.float32)).reshape(-1)
        peaks = [float(c.max()) for c in np.array_split(a, self.BARS) if c.size] if a.size else []
        top = max(peaks, default=0) or 1.0
        self._peaks = [p / top for p in peaks]
        self.update()

    def set_progress(self, fraction):
        fraction = None if fraction is None else min(1.0, max(0.0, fraction))
        if fraction != self._progress:
            self._progress = fraction
            self.update()

    def mousePressEvent(self, event):
        if self._peaks and event.button() == QtCore.Qt.LeftButton and self.width() > 0:
            self.seek_requested.emit(min(1.0, max(0.0, event.position().x() / self.width())))

    def paintEvent(self, _event):
        if not self._peaks:
            return
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.NoPen)
        played, rest = QtGui.QColor(COLORS["accent"]), QtGui.QColor(COLORS["text_muted"])
        step = self.width() / len(self._peaks)
        bar = max(2.0, step - 3)
        mid = self.height() / 2
        head = None if self._progress is None else self._progress * self.width()
        for i, p in enumerate(self._peaks):
            h = max(3.0, p * self.height())
            painter.setBrush(played if head is not None and i * step + bar / 2 <= head else rest)
            painter.drawRoundedRect(QtCore.QRectF(i * step, mid - h / 2, bar, h), 1.5, 1.5)
        if head is not None:
            painter.setBrush(QtGui.QColor(COLORS["text"]))
            painter.drawRect(QtCore.QRectF(min(head, self.width() - 2), 0, 2, self.height()))


class PreviewPlayer(QtWidgets.QWidget):
    """Waveform, seek slider, position / length and Back - Play/Pause/Resume - Forward for ONE audio file, driven by
    audio_player's single session. It knows nothing about voices, generation or which screen hosts it: the host
    decides when audio exists (`set_audio` / `clear`) and listens to `changed` (refresh the footer) and `failed`.
    Behaviour moved unchanged from the Create Voice Preview."""

    changed = QtCore.Signal()    # after any play / pause / stop / seek the player itself caused
    failed = QtCore.Signal(str)  # audio_player.play raised: the text for the host's own notice

    def __init__(self, noun="sample"):
        super().__init__()
        self._noun = noun
        self._path = None
        self._duration = 0.0   # seconds, from the samples themselves
        self._start = 0.0      # where Play starts when nothing is playing (set by seeking while stopped)
        self._last = "stopped"
        self._syncing = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.wave = WaveformView()
        self.wave.setAccessibleName(f"Waveform of the generated {noun}")
        self.wave.seek_requested.connect(lambda fraction: self.seek(fraction * self._duration))
        lay.addWidget(self.wave)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setAccessibleName("Seek")
        self.slider.setTracking(False)   # dragging only moves the handle; the seek happens on release
        self.slider.setSingleStep(5000)
        self.slider.setPageStep(10000)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderMoved.connect(lambda ms: self._show_position(ms / 1000.0))
        lay.addWidget(self.slider)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.lbl_pos = QtWidgets.QLabel("0:00")
        self.lbl_pos.setAccessibleName("Position")
        set_role(self.lbl_pos, "muted")
        row.addWidget(self.lbl_pos)
        row.addStretch(1)
        self.btn_back = QtWidgets.QPushButton()
        self.btn_back.setAccessibleName("Back 5 seconds")
        self.btn_back.setToolTip("Back 5 seconds")
        self.btn_back.clicked.connect(lambda: self.seek(self.position() - SEEK_STEP))
        self.btn_play = QtWidgets.QPushButton("Play")
        icons.apply(self.btn_play, "play")
        self.btn_play.setToolTip(f"Play the generated {noun}")
        self.btn_play.clicked.connect(self.toggle)
        self.btn_forward = QtWidgets.QPushButton()
        self.btn_forward.setAccessibleName("Forward 5 seconds")
        self.btn_forward.setToolTip("Forward 5 seconds")
        self.btn_forward.clicked.connect(lambda: self.seek(self.position() + SEEK_STEP))
        for b, glyph in ((self.btn_back, "rotate-ccw"), (self.btn_forward, "rotate-cw")):
            set_role(b, "quiet")
            icons.apply(b, glyph)
        row.addWidget(self.btn_back)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_forward)
        row.addStretch(1)
        self.lbl_len = QtWidgets.QLabel("0:00")
        self.lbl_len.setAccessibleName("Length")
        set_role(self.lbl_len, "muted")
        row.addWidget(self.lbl_len)
        lay.addLayout(row)

    # ---- what the host controls
    def set_audio(self, path, samples, sr):
        """Show `path` (its `samples` at `sr` give the waveform and length). Playback starts only when asked."""
        samples = np.asarray(samples)
        self.wave.set_samples(samples)
        self._duration = samples.reshape(-1).size / sr
        self._path = path
        self._start = 0.0
        self._last = "stopped"
        self.lbl_len.setText(format_duration(self._duration))
        self.slider.setRange(0, int(self._duration * 1000))
        self.refresh()

    def clear(self):
        """No audio. Stops this player's own playback; audio of another screen is left alone."""
        if self.state() != "stopped":
            audio_player.stop()
        self._path, self._duration, self._start, self._last = None, 0.0, 0.0, "stopped"
        self.wave.set_samples([])
        self.wave.set_progress(None)
        self.lbl_len.setText("0:00")
        self.lbl_pos.setText("0:00")
        self.slider.setRange(0, 0)
        for w in (self.slider, self.btn_back, self.btn_forward):
            w.setEnabled(False)
        self._set_play_label("Play", "stopped", True)

    def stop(self):
        """Stop this player's own playback (another screen's audio is left alone); the audio stays loaded."""
        if self.state() != "stopped":
            audio_player.stop()
            self.changed.emit()
        self._start = 0.0
        self.refresh()

    def reset_start(self):
        """Stopped from outside (footer Stop, new generation): the next Play begins at the start."""
        self._start = 0.0

    def has_audio(self):
        return self._path is not None

    def path(self):
        return self._path

    def duration(self):
        return self._duration

    def focus_chain(self):
        return [self.slider, self.btn_back, self.btn_play, self.btn_forward]

    # ---- state
    def state(self):
        """'playing' / 'paused' / 'stopped' for this player's own audio. Audio that belongs to another screen
        (a voice reference, a render, another preview) counts as stopped here."""
        if not self._path or audio_player.current_path() != self._path:
            return "stopped"
        return audio_player.state()

    def position(self):
        if self.state() == "stopped":
            return self._start
        return audio_player.position() or 0.0

    # ---- actions
    def toggle(self):
        if not self._path:
            return
        state = self.state()
        if state == "playing":
            audio_player.pause() if audio_player.can_control() else audio_player.stop()
        elif state == "paused":
            audio_player.resume()
        else:
            start = self._start if 0 < self._start < self._duration - 0.1 else 0.0
            try:
                audio_player.play(self._path, start=start) if start > 0 else audio_player.play(self._path)
            except Exception as e:
                self.failed.emit(str(e))
        self.changed.emit()
        self.refresh()

    def seek(self, seconds):
        if not self._path or self._duration <= 0 or not audio_player.can_control():
            return
        seconds = min(max(0.0, seconds), max(0.0, self._duration - 0.05))
        if self.state() == "stopped":
            self._start = seconds
        else:
            audio_player.seek(seconds)
        self.refresh()

    def _on_slider(self, ms):
        if not self._syncing:
            self.seek(ms / 1000.0)

    def _show_position(self, seconds):
        self.lbl_pos.setText(format_duration(seconds))
        self.slider.setAccessibleDescription(f"{format_duration(seconds)} of {format_duration(self._duration)}")

    def _set_play_label(self, label, state, control):
        if self.btn_play.text() != label:
            self.btn_play.setText(label)
            icons.apply(self.btn_play, ("pause" if control else "square") if label != "Play" else "play")
            self.btn_play.setToolTip({"Pause": "Pause playback", "Stop": "Stop playback"}.get(
                label, f"Resume the {self._noun}" if state == "paused" else f"Play the generated {self._noun}"))
        if state == "paused":
            self.btn_play.setToolTip("Resume playback")

    def refresh(self):
        """Mirror audio_player's real state into the controls. The host's timer calls it, and every action does."""
        if not self._path:
            return
        state = self.state()
        if state == "stopped" and self._last in ("playing", "paused"):
            self._start = 0.0   # finished or stopped: the next Play starts from the beginning
        self._last = state
        control = audio_player.can_control()
        playing = state == "playing"
        self._set_play_label(("Pause" if control else "Stop") if playing else "Play", state, control)
        for w in (self.slider, self.btn_back, self.btn_forward):
            w.setEnabled(control)
        position = min(self.position(), self._duration)
        if not self.slider.isSliderDown():
            self._syncing = True
            self.slider.setValue(int(position * 1000))
            self._syncing = False
            self._show_position(position)
        if not control:
            self.lbl_pos.setText("–")
        active = state != "stopped" or self._start > 0
        self.wave.set_progress(position / self._duration if active and self._duration else None)


class ActivityBar(QtWidgets.QWidget):
    """Restrained indeterminate indicator: one short segment gliding along a quiet track.

    It says only "work is happening". It has no value, so it can never suggest a percentage or an
    estimate (a stock indeterminate QProgressBar wraps around and reads as two segments). The animation
    runs only while the widget is visible.
    """

    indeterminate = True

    def __init__(self):
        super().__init__()
        self.setFixedHeight(4)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAccessibleName("Activity in progress")
        self._t = 0.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1800)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._anim.valueChanged.connect(self._step)

    def _step(self, value):
        self._t = float(value)
        self.update()

    def showEvent(self, event):
        self._anim.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._anim.stop()
        super().hideEvent(event)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        radius = self.height() / 2
        painter.setBrush(QtGui.QColor(COLORS["border"]))
        painter.drawRoundedRect(self.rect(), radius, radius)
        width = self.width() * 0.28
        phase = 1 - abs(2 * self._t - 1)          # glide out and back, never wrapping around
        painter.setBrush(QtGui.QColor(COLORS["accent"]))
        painter.drawRoundedRect(QtCore.QRectF((self.width() - width) * phase, 0, width, self.height()), radius, radius)


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
        self.btn_insert = btn
        lay.addWidget(btn)
        lay.addStretch(1)

    def insert(self):
        self.editor.insertPlainText(f"[{self.preset.currentText()}] ")
        self.editor.setFocus()


class MainWindow(QtWidgets.QMainWindow):
    _failure_details = ""   # sanitized technical text of the last Create Voice failure
    _clone_failure_details = ""   # the same for Clone Voice, so the two screens never show each other's details
    _gen_t0 = None          # time.monotonic() when the running generation started
    _gen_stage = STAGE_PREPARING

    def __init__(self, models=None):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS")
        self.models = models or ModelManager(self.model_settings)
        self.task = None
        self.busy = False
        self.pending_voice = None
        self.last_output = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ---- menu bar
        menu = self.menuBar()
        tools = menu.addMenu("Tools")
        act_voice_mgr = tools.addAction("Voice Manager…")
        act_voice_mgr.triggered.connect(self.open_voice_manager)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        shell = QtWidgets.QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        # Widgets other pages depend on (status label, settings spin boxes) are built first.
        settings_page = self._build_settings_page()
        footer = self._build_footer()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_design_tab())
        clone_page = self._build_clone_tab()
        voices_page = self._build_voices_tab()
        self.audiobook_tab = AudiobookTab(self, text_edit, combo, seed_row, header=page_header)
        for dense in (self.audiobook_tab,):  # not redesigned yet: keep them compact
            dense.setProperty("density", "compact")
        self.stack.addWidget(clone_page)
        self.stack.addWidget(voices_page)
        self.stack.addWidget(self.audiobook_tab)   # it scrolls its own body and keeps its header and Render in view
        self.drama_tab = DramaTab(self)
        self.drama_tab.setProperty("density", "compact")
        drama_lay = self.drama_tab.layout()
        drama_lay.setContentsMargins(24, 24, 24, 16)
        drama_lay.insertWidget(0, page_header("Drama Studio", "Cast management, LLM script planning, and ComfyUI video rendering."))
        drama_scroll = QtWidgets.QScrollArea()
        drama_scroll.setWidgetResizable(True)
        drama_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        drama_scroll.setWidget(self.drama_tab)
        self.stack.addWidget(drama_scroll)
        self.stack.addWidget(settings_page)

        shell.addWidget(self._build_sidebar(("Create Voice", "Clone Voice", "My Voices", "Audiobooks", "Drama Studio"), "Settings"))
        main_col = QtWidgets.QVBoxLayout()
        main_col.setContentsMargins(0, 0, 0, 0)
        main_col.setSpacing(0)
        main_col.addWidget(self.stack, 1)
        main_col.addWidget(footer)
        shell.addLayout(main_col, 1)
        self.show_page(0)
        try:
            voice_library.recover_interrupted_saves()   # finish a replace that was cut short by a crash
        except OSError:
            pass
        self.refresh_voice_lists()

        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(1024, screen.width()), min(640, screen.height()))
        self.resize(min(1240, screen.width() - 40), min(860, screen.height() - 60))

        # Buttons take keyboard focus only, so a mouse click never leaves a focus ring on
        # the selected navigation item; the description field gets the initial focus.
        for b in self.findChildren(QtWidgets.QPushButton):
            b.setFocusPolicy(QtCore.Qt.TabFocus)
        self._sync_nav_focus()  # the loop above must not give every nav item a tab stop
        for area in self.findChildren(QtWidgets.QScrollArea):  # a scroll area is not a control: no invisible stop
            area.setFocusPolicy(QtCore.Qt.NoFocus)
        self._apply_tab_order()
        self.txt_vd_instruct.setFocus()

        # Ctrl+Return starts a Create Voice generation (a disabled button ignores the click).
        QtGui.QShortcut(QtGui.QKeySequence(GENERATE_SHORTCUT), self, activated=self._shortcut_generate)

    # ------------------------------------------------------------------ settings / status
    def model_settings(self):
        return (self.cmb_device.currentText(), self.cmb_dtype.currentText(), self.cmb_attn.currentText())

    def _build_sidebar(self, entries, settings_label):
        """Left navigation. Selection is a neutral surface; the accent is reserved for actions."""
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(SIDEBAR_WIDTH)
        lay = QtWidgets.QVBoxLayout(side)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(4)
        brand = QtWidgets.QLabel("Qwen3-TTS Studio")
        set_role(brand, "brand")
        lay.addWidget(brand)
        lay.addSpacing(16)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []

        def add_nav(label):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            icons.apply(btn, NAV_ICONS.get(label, ""), icons.SIZE_NAV, "nav")
            btn.setToolTip(f"{label} (Ctrl+{len(self.nav_buttons) + 1})")
            set_role(btn, "nav")
            self.nav_group.addButton(btn, len(self.nav_buttons))
            self.nav_buttons.append(btn)
            lay.addWidget(btn)

        for label in entries:
            add_nav(label)
        lay.addStretch(1)
        add_nav(settings_label)
        self.nav_group.idClicked.connect(self.show_page)
        for i in range(len(self.nav_buttons)):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda i=i: self.show_page(i))
        return side

    def show_page(self, index):
        # Hiding the focused control would hand focus to the first control of the next page, which then
        # shows a focus ring nobody asked for (looks like a highlighted button after a mouse click).
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None and self.stack.isAncestorOf(focused) and index != self.stack.currentIndex():
            focused.clearFocus()
        self.stack.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self._sync_nav_focus()
        # Keyboard focus on a navigation item that is no longer selected (it is no longer a tab stop)
        # would keep drawing its focus ring, e.g. Settings while Create Voice is open. Follow the selection.
        if focused in self.nav_buttons and focused is not self.nav_buttons[index]:
            self.nav_buttons[index].setFocus(QtCore.Qt.OtherFocusReason)
        self._update_footer()

    def _sync_nav_focus(self):
        # The navigation is one tab stop (the selected item); arrow keys move between items.
        for button in self.nav_buttons:
            button.setFocusPolicy(QtCore.Qt.TabFocus if button.isChecked() else QtCore.Qt.NoFocus)

    def _apply_tab_order(self):
        """Tab follows the visual task order: navigation, creation form, primary action, preview
        actions, secondary settings, footer. Qt skips hidden and disabled widgets, and only the
        checked navigation item is a stop (arrow keys move between items)."""
        chain = [
            *self.nav_buttons,
            self.txt_vd_instruct, *(chip for chip, _name, _phrase in self.vd_chips),
            self.txt_vd_sample, self.btn_vd_reset_text, self.btn_vd_generate,
            self.btn_vd_notice_details, self.txt_vd_spoken, self.btn_vd_edit_text,
            self.sld_vd_seek, self.btn_vd_back, self.btn_vd_play, self.btn_vd_forward, self.edt_vd_name, self.btn_vd_save, self.btn_vd_download,
            self.btn_vd_regenerate,
            self.btn_vd_retry, self.btn_vd_error_settings, self.btn_vd_details, self.btn_vd_copy_details,
            self.txt_vd_details,
            self.cmb_vd_lang, self.btn_vd_advanced,
            self.cmb_vd_seed_mode, self.spn_vd_seed, self.chk_vd_autosave, self.cmb_vd_name_gender,
            self.btn_vd_rand_name, self.btn_vd_open_settings,
            # Clone Voice, in reading order (hidden pages are skipped by Qt)
            self.btn_vc_src_saved, self.btn_vc_src_ref, self.cmb_vc_voice, self.btn_vc_refresh,
            self.edt_vc_ref, self.btn_vc_ref_browse, self.txt_vc_ref_text, self.chk_vc_xvec,
            self.txt_vc_text, self.cmb_vc_lang, self.btn_vc_generate, self.btn_vc_details, self.btn_vc_error_settings, *self.vc_player.focus_chain(),
            self.info_buttons["Description"], self.txt_vc_instruct,
            self.edt_vc_new_name, self.btn_vc_save_voice, self.btn_vc_use_saved, self.btn_vc_advanced,
            self.info_buttons["Direction tags"], self.dir_vc.preset, self.dir_vc.btn_insert,
            self.info_buttons["Seed"], self.cmb_vc_seed_mode, self.spn_vc_seed,
            self.info_buttons["Temperature"], self.spn_vc_temp,
            self.chk_vc_npz, self.edt_vc_reuse, self.btn_vc_reuse_browse,
            self.edt_vc_save, self.btn_vc_save_browse,
            self.btn_play_last, self.btn_stop_audio, self.btn_open_out, self.btn_stop,
        ]
        for first, second in zip(chain, chain[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def _shortcut_generate(self):
        if self.stack.currentIndex() == 0:
            self.btn_vd_generate.click()

    def model_settings(self):
        return (self.cmb_device.currentText(), self.cmb_dtype.currentText(), self.cmb_attn.currentText())

    def _build_settings_page(self):
        """Application-wide options (same widgets and defaults as before), grouped by scope.

        Task options (language, seed, auto-save, naming) stay on the screen that uses them.
        """
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        page = QtWidgets.QWidget()
        scroll.setWidget(page)
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(32, 32, 32, 24)
        outer.setSpacing(16)
        outer.addWidget(page_header(
            "Settings", "Application-wide options. Language, seed and naming live on each screen."))
        self.cmb_device = combo(["cuda", "cpu"], DEFAULT_DEVICE if cuda_available() else "cpu")
        self.cmb_dtype = combo(DTYPES, DEFAULT_DTYPE)
        self.cmb_attn = combo(ATTNS, DEFAULT_ATTN)
        self.spn_min = QtWidgets.QSpinBox()
        self.spn_min.setRange(0, 5000)
        self.spn_min.setValue(int(DEFAULT_MIN_WORDS))
        self.spn_max = QtWidgets.QSpinBox()
        self.spn_max.setRange(0, 5000)
        self.spn_max.setValue(int(DEFAULT_MAX_WORDS))
        style_numeric(self.spn_min, name="Min words per chunk")
        style_numeric(self.spn_max, name="Max words per chunk")

        def section(title, description, rows):
            head = QtWidgets.QLabel(title)
            set_role(head, "heading")
            outer.addWidget(head)
            desc = QtWidgets.QLabel(description)
            desc.setWordWrap(True)
            desc.setMaximumWidth(560)
            set_role(desc, "muted")
            outer.addWidget(desc)
            form = QtWidgets.QFormLayout()
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(12)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            for label, w in rows:
                lbl = QtWidgets.QLabel(label)
                lbl.setBuddy(w)
                form.addRow(lbl, w)
            outer.addLayout(form)

        section("Compute", "Used by every screen. Models reload when these change.",
                (("Device", self.cmb_device), ("Dtype", self.cmb_dtype), ("Attention", self.cmb_attn)))
        outer.addSpacing(8)
        section("Long-form text", "Used by Clone Voice and Audiobooks. Voice creation is not split into chunks.",
                (("Min words per chunk", self.spn_min), ("Max words per chunk", self.spn_max)))
        outer.addSpacing(8)
        self._drama_settings = load_drama_settings()
        self.cmb_drama_provider = combo(list(DRAMA_PROVIDERS), self._drama_settings["provider"])
        self.edt_drama_llm_url = QtWidgets.QLineEdit(self._drama_settings["llm_base_url"])
        self.edt_drama_llm_url.setMinimumWidth(340)
        self.edt_drama_llm_model = QtWidgets.QLineEdit(self._drama_settings["llm_model"])
        self.edt_drama_llm_model.setMinimumWidth(340)
        self.edt_drama_api_key = QtWidgets.QLineEdit(drama_env_key(self.cmb_drama_provider.currentText()))
        self.edt_drama_api_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edt_drama_api_key.setMinimumWidth(340)
        self.edt_drama_comfy = QtWidgets.QLineEdit(self._drama_settings["comfy_url"])
        self.edt_drama_comfy.setMinimumWidth(340)
        self.cmb_drama_provider.currentTextChanged.connect(self._drama_provider_changed)
        self.btn_drama_save_conn = QtWidgets.QPushButton("Save connection settings")
        self.btn_drama_save_conn.clicked.connect(self._save_drama_connection)
        section("Drama Studio (LLM & ComfyUI)",
                "API keys are kept in memory for this session only and are never saved to disk. "
                "OPENROUTER_API_KEY or OPENAI_API_KEY load automatically when set. ComfyUI may be local or remote.",
                (("LLM provider", self.cmb_drama_provider), ("OpenAI-compatible URL", self.edt_drama_llm_url),
                 ("LLM model", self.edt_drama_llm_model), ("API key", self.edt_drama_api_key),
                 ("ComfyUI URL", self.edt_drama_comfy)))
        outer.addWidget(self.btn_drama_save_conn, 0, QtCore.Qt.AlignLeft)
        outer.addSpacing(8)
        note = QtWidgets.QLabel(
            "Defaults are unchanged from earlier versions. flash_attention_2 targets NVIDIA GPUs with the "
            "flash-attn package installed; behavior on other hardware has not been verified here. "
            "Models are downloaded on first use.")
        note.setWordWrap(True)
        note.setMaximumWidth(560)
        set_role(note, "muted")
        outer.addWidget(note)
        outer.addStretch(1)
        return scroll

    def _drama_provider_changed(self, provider):
        url, model, _ = DRAMA_PROVIDERS[provider]
        self.edt_drama_llm_url.setText(url)
        self.edt_drama_llm_model.setText(model)
        self.edt_drama_api_key.setText(drama_env_key(provider))

    def drama_connection_settings(self):
        return {
            "provider": self.cmb_drama_provider.currentText(),
            "llm_base_url": self.edt_drama_llm_url.text().strip(),
            "llm_model": self.edt_drama_llm_model.text().strip(),
            "comfy_url": self.edt_drama_comfy.text().strip(),
        }

    def drama_api_key(self):
        return self.edt_drama_api_key.text().strip()

    def _save_drama_connection(self):
        self._drama_settings = save_drama_settings(self.drama_connection_settings())
        self.status("Drama connection settings saved; the API key was not written to disk.")

    def _build_footer(self):
        """Global status strip: progress, last message, and audio/output shortcuts."""
        footer = QtWidgets.QFrame()
        footer.setObjectName("footer")
        lay = QtWidgets.QVBoxLayout(footer)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(3)
        lay.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(24, 8, 24, 8)
        row.setSpacing(8)
        # Decorative glyph for the two states that benefit from one; the status text always says it too.
        self.lbl_status_icon = QtWidgets.QLabel()
        self.lbl_status_icon.setFixedSize(icons.SIZE_COMPACT, icons.SIZE_COMPACT)
        self.lbl_status_icon.setVisible(False)
        row.addWidget(self.lbl_status_icon)
        row.setSpacing(8)
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAccessibleName("Status")
        row.addWidget(self.lbl_status, 1)
        self.btn_play_last = QtWidgets.QPushButton("Play last output")
        self.btn_play_last.setEnabled(False)
        self.btn_play_last.clicked.connect(lambda: self.play(self.last_output))
        self.btn_stop_audio = QtWidgets.QPushButton("Stop audio")
        self.btn_stop_audio.clicked.connect(self.stop_audio)
        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        self.btn_stop = QtWidgets.QPushButton("Stop generation")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_play_last.setToolTip("Play the most recent Clone Voice or Audiobook render")
        self.btn_stop_audio.setToolTip(STOP_AUDIO_TIP)
        self.btn_open_out.setToolTip("Open the folder where renders are saved")
        self.btn_stop.setToolTip("Stop the running render after its current chunk or segment")
        self.btn_stop.setVisible(False)
        for b, glyph in ((self.btn_play_last, "play"), (self.btn_stop_audio, "square"),
                         (self.btn_open_out, "folder-open"), (self.btn_stop, "square")):
            set_role(b, "quiet")
            icons.apply(b, glyph)
            row.addWidget(b)
        lay.addLayout(row)
        # Stop audio is only live while the player reports audio; the player is asked, never assumed.
        # Hidden (so neither visible nor focusable) unless the player has a session, but it keeps its
        # space so the footer doesn't shift when it appears.
        policy = self.btn_stop_audio.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.btn_stop_audio.setSizePolicy(policy)
        self.btn_stop_audio.setVisible(False)
        self.audio_timer = QtCore.QTimer(self)
        self.audio_timer.setInterval(250)
        self.audio_timer.timeout.connect(self._refresh_audio_state)
        self.audio_timer.start()
        # The Preview player follows audio_player's real state (position, pause, end) while it is on screen.
        self.player_timer = QtCore.QTimer(self)
        self.player_timer.setInterval(100)
        self.player_timer.timeout.connect(self._tick_player)
        self.player_timer.start()
        self.gen_timer = QtCore.QTimer(self)   # ticks only while a sample is being generated
        self.gen_timer.setInterval(1000)
        self.gen_timer.timeout.connect(self._tick_elapsed)
        return footer

    def status(self, msg):
        self.lbl_status.setText(msg)
        kind = "error" if msg.startswith(("Error", "Playback error", "Couldn't")) else (
            "ok" if msg == READY_STATUS else None)
        glyph = {"error": "circle-exclamation-point", "ok": "circle-check"}.get(kind)
        if glyph:
            self.lbl_status_icon.setPixmap(icons.icon(glyph, icons.SIZE_COMPACT, "error" if kind == "error" else "default", gap=0)
                                           .pixmap(icons.SIZE_COMPACT, icons.SIZE_COMPACT))
        self.lbl_status_icon.setVisible(bool(glyph))

    def _refresh_audio_state(self):
        session = audio_player.state() != "stopped"   # playing or paused, from any page
        if session == self.btn_stop_audio.isHidden():
            self.btn_stop_audio.setVisible(session)

    def stop_audio(self):
        audio_player.stop()
        self.vd_player.reset_start()
        self.vc_player.reset_start()
        self._refresh_audio_state()
        self._refresh_player()
        self.vc_player.refresh()

    def open_output_folder(self):
        """Open the renders folder; recreate it if it was removed, and say so if the system can't open it."""
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except OSError as exc:
            self.status(f"Couldn't open the output folder: {exc}")
            return
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(OUTPUT_DIR)):
            self.status(f"Couldn't open the output folder: {OUTPUT_DIR}")

    def play(self, path, start=0.0):
        """Start `path`; returns the error text if it can't play, else None."""
        try:
            audio_player.play(path, start=start) if start > 0 else audio_player.play(path)
            self._refresh_audio_state()
            self._refresh_player()
        except Exception as e:
            self.status(f"Playback error: {e}")
            self._refresh_player()
            return str(e)
        return None

    def open_voice_manager(self):
        dlg = VoiceManagerDialog(self)
        dlg.exec()

    def request_stop(self):
        if self.task:
            self.task.stop_event.set()
            self.status("Stop requested…")

    def set_busy(self, busy, stoppable=True, footer_progress=True):
        self.busy = busy
        self.progress.setRange(0, 0 if busy and footer_progress else 1)
        # Only the clone loop and audiobook render check the stop flag; other tasks can't be cancelled.
        self.btn_stop.setEnabled(busy and stoppable)
        self.btn_stop.setVisible(busy and stoppable)
        for b in self.action_buttons:
            b.setEnabled(not busy)
        self._refresh_save_button()  # re-enabling everything must not enable Save without a sample or name
        if hasattr(self, "audiobook_tab"):
            self.audiobook_tab.refresh_actions()   # Preview segment needs exactly one selected segment

    def _update_footer(self):
        # Create Voice has its own Play; "last output" only applies to Clone Voice and Audiobook renders.
        self.btn_play_last.setVisible(self.stack.currentIndex() != 0)

    def run_task(self, fn, on_done, on_fail=None, on_notify=None, stoppable=True,
                 footer_progress=True, footer_status=True):
        """footer_progress / footer_status False: the screen shows this task's progress itself
        (Create Voice), so the footer neither animates nor repeats the stage text."""
        if self.busy:
            self.status("Busy: a task is already running.")
            return
        task = Task()
        self.task = task
        if footer_status:
            task.notify.connect(self.status)
        if on_notify:
            task.notify.connect(on_notify)

        def failed(message):
            self.set_busy(False)
            if footer_status:
                self.status(f"Error: {message}")
            if on_fail:
                on_fail(message)

        task.done.connect(lambda r: (self.set_busy(False), on_done(r)))
        task.failed.connect(failed)
        self.set_busy(True, stoppable, footer_progress)
        task.start(fn)

    def generator(self, task):
        return TTSGenerator(self.models, notify=task.notify.emit, should_stop=task.stop_event.is_set)

    def output_path(self, prefix):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(OUTPUT_DIR, f"{prefix}_{stamp}_{uuid.uuid4().hex[:3]}.wav")

    def finish_long_form(self, prefix):
        def on_done(result, early=None):
            """early: True/False when the caller knows whether chunks were really skipped; None = the Stop flag."""
            wavs, sr = result
            if not wavs:
                self.status("Stopped before any audio was produced.")
                return None
            path = self.output_path(prefix)
            save_outputs(wavs, sr, path)
            self.last_output = path
            self.btn_play_last.setEnabled(True)
            if early is None:
                early = self.task.stop_event.is_set()
            self.status(("Stopped early; partial audio saved to: " if early else "Done. Audio saved to: ") + path)
            return path
        return on_done

    # ------------------------------------------------------------------ Voice Creator
    def _build_design_tab(self):
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        page.setWidget(body)
        frame = QtWidgets.QHBoxLayout(body)
        frame.setContentsMargins(32, 32, 32, 24)
        self.vd_content = QtWidgets.QWidget()
        self.vd_content.setMaximumWidth(CONTENT_MAX_WIDTH)
        frame.addWidget(self.vd_content, 100, QtCore.Qt.AlignTop)
        frame.addStretch(1)
        outer = QtWidgets.QVBoxLayout(self.vd_content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(24)

        outer.addWidget(page_header(
            "Create a voice with AI", "Describe how you want it to sound, then listen to a sample."))
        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(32)
        left = self._build_creation_column()
        # Model, Language and Advanced settings configure generation, so they sit under Generate voice
        # in the form column rather than spanning under the Preview.
        left.addSpacing(8)
        divider = QtWidgets.QFrame()
        set_role(divider, "divider")
        left.addWidget(divider)
        left.addLayout(self._build_options_strip())
        self.vd_advanced = self._build_advanced_panel()
        self.vd_advanced.setVisible(False)
        left.addWidget(self.vd_advanced)
        left.addStretch(1)
        cols.addLayout(left, 5)
        cols.addWidget(self._build_preview_panel(), 4, QtCore.Qt.AlignTop)
        outer.addLayout(cols)

        self.txt_vd_instruct.textChanged.connect(self._on_description_changed)
        self.cmb_vd_lang.currentTextChanged.connect(lambda _t: self._refresh_stale_note())
        for cmb in (self.cmb_device, self.cmb_dtype, self.cmb_attn):
            cmb.currentTextChanged.connect(lambda _t: self._update_compute_summary())
        self._update_compute_summary()
        self._set_preview(PV_EMPTY)
        return page

    # ---- left column: describe and generate
    def _build_creation_column(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        lbl = QtWidgets.QLabel("Voice description")
        self.lbl_vd_description = lbl
        set_role(lbl, "field")
        col.addWidget(lbl)
        self.txt_vd_instruct = text_edit(
            4, "e.g. A warm, calm voice with a friendly tone",
            "Warm, gentle female voice, calm and soothing tone.")
        self.txt_vd_instruct.setFixedHeight(112)
        self.txt_vd_instruct.setTabChangesFocus(True)  # Tab/Shift+Tab leave the field; Enter still adds a line
        self.txt_vd_instruct.setAccessibleName("Voice description")
        lbl.setBuddy(self.txt_vd_instruct)
        col.addWidget(self.txt_vd_instruct)
        self.lbl_vd_field_error = QtWidgets.QLabel("Describe how you want the voice to sound.")
        set_role(self.lbl_vd_field_error, "error-text")
        self.lbl_vd_field_error.setVisible(False)
        col.addWidget(self.lbl_vd_field_error)

        col.addSpacing(8)
        caption = QtWidgets.QLabel("Suggestions")
        set_role(caption, "caption")
        col.addWidget(caption)
        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(8)
        self.vd_chips = []
        for name, phrase in SUGGESTIONS:
            chip = QtWidgets.QPushButton(name)
            chip.setCheckable(True)
            chip.setToolTip(f"Add to the description: {phrase}")
            set_role(chip, "chip")
            chip.toggled.connect(lambda on, ph=phrase: self._toggle_suggestion(ph, on))
            self.vd_chips.append((chip, name, phrase))
            chips.addWidget(chip)
        chips.addStretch(1)
        col.addLayout(chips)

        col.addSpacing(8)
        sample_head = QtWidgets.QHBoxLayout()
        sample_lbl = QtWidgets.QLabel("Sample text")
        set_role(sample_lbl, "field")
        sample_head.addWidget(sample_lbl)
        sample_head.addStretch(1)
        self.btn_vd_reset_text = QtWidgets.QPushButton("Use standard text")
        set_role(self.btn_vd_reset_text, "quiet")
        self.btn_vd_reset_text.setToolTip("Restore the standard sample passage")
        self.btn_vd_reset_text.clicked.connect(lambda: self.txt_vd_sample.setPlainText(voice_library.SAMPLE_TEXT))
        sample_head.addWidget(self.btn_vd_reset_text)
        col.addLayout(sample_head)
        self.txt_vd_sample = text_edit(3, text=voice_library.SAMPLE_TEXT)
        self.txt_vd_sample.setFixedHeight(self.txt_vd_sample.fontMetrics().lineSpacing() * 3 + 20)
        self.txt_vd_sample.setAccessibleName("Sample text")
        self.txt_vd_sample.setAccessibleDescription("This is what the generated voice will say")
        self.txt_vd_sample.setToolTip("Longer text takes longer to generate.")
        sample_lbl.setBuddy(self.txt_vd_sample)
        self.txt_vd_sample.textChanged.connect(self._on_sample_text_changed)
        col.addWidget(self.txt_vd_sample)
        sample_hint = QtWidgets.QLabel("This is what the generated voice will say.")
        set_role(sample_hint, "muted")
        col.addWidget(sample_hint)

        col.addSpacing(16)
        gen = QtWidgets.QHBoxLayout()
        self.btn_vd_generate = QtWidgets.QPushButton("Generate voice")
        set_role(self.btn_vd_generate, "primary")
        icons.apply(self.btn_vd_generate, "audio-lines", icons.SIZE_COMPACT, "primary")
        self.btn_vd_generate.setMinimumWidth(180)
        self.btn_vd_generate.setToolTip(f"Generate voice ({shortcut_text()})")
        self.btn_vd_generate.setAccessibleDescription(f"Keyboard shortcut {shortcut_text()}")
        self.btn_vd_generate.clicked.connect(self.generate_sample)
        gen.addWidget(self.btn_vd_generate)
        gen.addSpacing(12)
        self.lbl_vd_shortcut = QtWidgets.QLabel(f"or press {shortcut_text()}")
        set_role(self.lbl_vd_shortcut, "muted")
        gen.addWidget(self.lbl_vd_shortcut)
        gen.addStretch(1)
        col.addLayout(gen)
        col.addStretch(1)
        return col

    # ---- right column: preview states
    def _build_preview_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("previewPanel")
        self.preview_panel = panel
        panel.setMinimumWidth(340)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(12)

        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Preview")
        self.lbl_vd_preview_title = title
        set_role(title, "heading")
        head.addWidget(title)
        head.addStretch(1)
        lay.addLayout(head)

        self.vd_preview = PageSwitcher()
        lay.addWidget(self.vd_preview)

        # empty
        empty = QtWidgets.QWidget()
        em = QtWidgets.QVBoxLayout(empty)
        em.setContentsMargins(0, 0, 0, 0)
        em.setSpacing(4)
        msg = WrapLabel("Your generated sample will appear here.")
        set_role(msg, "notice")
        hint = WrapLabel("Describe a voice, then choose Generate voice.")
        set_role(hint, "muted")
        em.addWidget(msg)
        em.addWidget(hint)
        em.addStretch(1)
        self.vd_preview.insertWidget(PV_EMPTY, empty)

        # busy
        busy = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(busy)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        self.lbl_vd_busy = WrapLabel(STAGE_PREPARING)
        self.lbl_vd_busy.setAccessibleName("Generation status")
        bl.addWidget(self.lbl_vd_busy)
        self.vd_busy_bar = ActivityBar()
        bl.addWidget(self.vd_busy_bar)
        self.lbl_vd_elapsed = QtWidgets.QLabel("")   # real time since Generate voice, never an estimate
        self.lbl_vd_elapsed.setAccessibleName("Time elapsed")
        bl.addWidget(self.lbl_vd_elapsed)
        self.lbl_vd_busy_hint = WrapLabel(LOADING_HINT)
        set_role(self.lbl_vd_busy_hint, "muted")
        self.lbl_vd_busy_hint.setVisible(False)   # a short explanation of the stage the backend is really in
        bl.addWidget(self.lbl_vd_busy_hint)
        bl.addStretch(1)
        self.vd_preview.insertWidget(PV_BUSY, busy)

        # ready
        ready = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(ready)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        self.lbl_vd_notice = WrapLabel("")
        self.lbl_vd_notice.setAccessibleName("Sample notice")
        set_role(self.lbl_vd_notice, "notice")
        self.lbl_vd_notice.setVisible(False)
        rl.addWidget(self.lbl_vd_notice)
        notice_row = QtWidgets.QHBoxLayout()
        self.btn_vd_notice_details = QtWidgets.QPushButton("Show details")
        set_role(self.btn_vd_notice_details, "quiet")
        self.btn_vd_notice_details.clicked.connect(self._open_failure_dialog)
        self.btn_vd_notice_details.setVisible(False)
        notice_row.addWidget(self.btn_vd_notice_details)
        notice_row.addStretch(1)
        rl.addLayout(notice_row)
        self.vd_regen_bar = ActivityBar()   # the one indicator while a new sample replaces an old one
        self.vd_regen_bar.setVisible(False)
        rl.addWidget(self.vd_regen_bar)
        text_lbl = QtWidgets.QLabel("Spoken text")
        set_role(text_lbl, "field")
        rl.addWidget(text_lbl)
        self.txt_vd_spoken = text_edit(3, read_only=True)   # the exact words spoken in the audio below
        self.txt_vd_spoken.setTextInteractionFlags(   # select with mouse or keyboard; arrows scroll and move the caret
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        self.txt_vd_spoken.setProperty("role", "reader")   # flat text, not an input: selectable and scrollable
        self.txt_vd_spoken.setFixedHeight(self.txt_vd_spoken.fontMetrics().lineSpacing() * 3 + 8)  # ~3 lines; scrolls
        self.txt_vd_spoken.setAccessibleName("Sample text used for this sample")
        text_lbl.setBuddy(self.txt_vd_spoken)
        rl.addWidget(self.txt_vd_spoken)
        text_row = QtWidgets.QHBoxLayout()
        self.lbl_vd_spoken_note = WrapLabel("Text used for this sample")
        set_role(self.lbl_vd_spoken_note, "muted")
        text_row.addWidget(self.lbl_vd_spoken_note, 1)
        self.btn_vd_edit_text = QtWidgets.QPushButton("Edit text")
        set_role(self.btn_vd_edit_text, "quiet")
        icons.apply(self.btn_vd_edit_text, "pencil")
        self.btn_vd_edit_text.setToolTip("Change the Sample text for the next sample. This audio is not edited.")
        self.btn_vd_edit_text.clicked.connect(self.edit_sample_text)
        text_row.addWidget(self.btn_vd_edit_text)
        rl.addLayout(text_row)

        self.vd_player = PreviewPlayer("sample")
        self.vd_player.changed.connect(self._refresh_audio_state)
        self.vd_player.failed.connect(self._on_preview_play_failed)
        rl.addWidget(self.vd_player)
        # the names the rest of the window (and its tests) already use for these controls
        self.vd_wave, self.sld_vd_seek = self.vd_player.wave, self.vd_player.slider
        self.btn_vd_back, self.btn_vd_play, self.btn_vd_forward = (
            self.vd_player.btn_back, self.vd_player.btn_play, self.vd_player.btn_forward)
        self.lbl_vd_time, self.lbl_vd_total = self.vd_player.lbl_pos, self.vd_player.lbl_len
        self.lbl_vd_origin = WrapLabel("")  # which inputs produced this audio
        self.lbl_vd_origin.setAccessibleName("Sample made from")
        set_role(self.lbl_vd_origin, "muted")
        rl.addWidget(self.lbl_vd_origin)
        self.frm_vd_stale = QtWidgets.QFrame()
        set_role(self.frm_vd_stale, "callout")
        callout = QtWidgets.QVBoxLayout(self.frm_vd_stale)
        callout.setContentsMargins(12, 10, 12, 10)
        self.lbl_vd_stale = WrapLabel(
            "Description, language or sample text changed. "
            "Save keeps the original; Generate voice makes a new sample.")
        set_role(self.lbl_vd_stale, "notice")
        callout.addWidget(self.lbl_vd_stale)
        self.frm_vd_stale.setVisible(False)
        self.lbl_vd_stale.setVisible(False)
        rl.addWidget(self.frm_vd_stale)
        rl.addSpacing(4)
        divider = QtWidgets.QFrame()
        set_role(divider, "divider")
        rl.addWidget(divider)
        rl.addSpacing(4)
        name_lbl = QtWidgets.QLabel("Voice name")
        set_role(name_lbl, "field")
        rl.addWidget(name_lbl)
        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(8)
        self.edt_vd_name = QtWidgets.QLineEdit()
        self.edt_vd_name.setPlaceholderText("Name this voice")
        self.edt_vd_name.setAccessibleName("Voice name")
        name_lbl.setBuddy(self.edt_vd_name)
        self.edt_vd_name.textChanged.connect(lambda _t: self._refresh_save_button())
        save_row.addWidget(self.edt_vd_name, 1)
        self.btn_vd_save = QtWidgets.QPushButton("Save")
        icons.apply(self.btn_vd_save, "save")
        self.btn_vd_save.setEnabled(False)
        self.btn_vd_save.clicked.connect(self.save_voice)
        save_row.addWidget(self.btn_vd_save)
        rl.addLayout(save_row)
        self.lbl_vd_name_hint = WrapLabel("")
        set_role(self.lbl_vd_name_hint, "muted")
        self.lbl_vd_name_hint.setVisible(False)
        rl.addWidget(self.lbl_vd_name_hint)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        self.btn_vd_download = QtWidgets.QPushButton("Download")
        icons.apply(self.btn_vd_download, "download")
        self.btn_vd_download.setToolTip("Export the sample as a WAV file")
        self.btn_vd_download.clicked.connect(self.download_sample)
        self.btn_vd_regenerate = QtWidgets.QPushButton("Regenerate")
        icons.apply(self.btn_vd_regenerate, "refresh-cw")
        self.btn_vd_regenerate.setToolTip("Generate another sample from the current description")
        self.btn_vd_regenerate.clicked.connect(self.generate_sample)
        actions.addWidget(self.btn_vd_download)
        actions.addWidget(self.btn_vd_regenerate)
        actions.addStretch(1)
        rl.addLayout(actions)
        self.vd_preview.insertWidget(PV_READY, ready)

        # error: what failed, what to do, then the technical details on request
        err = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(err)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(8)
        err_title = QtWidgets.QLabel("Couldn't generate the sample")
        set_role(err_title, "error-title")
        el.addWidget(err_title)
        self.lbl_vd_error = WrapLabel("")
        self.lbl_vd_error.setAccessibleName("Error")
        el.addWidget(self.lbl_vd_error)
        self.lbl_vd_error_action = WrapLabel("")
        self.lbl_vd_error_action.setAccessibleName("What you can do")
        set_role(self.lbl_vd_error_action, "muted")
        el.addWidget(self.lbl_vd_error_action)
        err_row = QtWidgets.QHBoxLayout()
        err_row.setSpacing(8)
        self.btn_vd_retry = QtWidgets.QPushButton("Retry")
        self.btn_vd_retry.clicked.connect(self.generate_sample)
        icons.apply(self.btn_vd_retry, "refresh-cw")
        self.btn_vd_error_settings = QtWidgets.QPushButton("Open Settings")
        self.btn_vd_error_settings.clicked.connect(lambda: self.show_page(len(self.nav_buttons) - 1))
        err_row.addWidget(self.btn_vd_retry)
        err_row.addWidget(self.btn_vd_error_settings)
        err_row.addStretch(1)
        el.addLayout(err_row)
        det_row = QtWidgets.QHBoxLayout()
        det_row.setSpacing(8)
        self.btn_vd_details = QtWidgets.QPushButton("Show details")
        self.btn_vd_details.setCheckable(True)
        self.btn_vd_details.setAccessibleName("Technical details")
        self.btn_vd_details.setToolTip("Show the technical details of this failure")
        set_role(self.btn_vd_details, "quiet")
        icons.apply(self.btn_vd_details, "chevron-right")
        self.btn_vd_copy_details = QtWidgets.QPushButton("Copy details")
        self.btn_vd_copy_details.setToolTip("Copy the technical details (private paths and credentials removed)")
        self.btn_vd_copy_details.clicked.connect(self.copy_failure_details)
        set_role(self.btn_vd_copy_details, "quiet")
        icons.apply(self.btn_vd_copy_details, "copy")
        det_row.addWidget(self.btn_vd_details)
        det_row.addWidget(self.btn_vd_copy_details)
        det_row.addStretch(1)
        el.addLayout(det_row)
        # A read-only text box, not a label: it takes keyboard focus, scrolls, and selects with the keyboard.
        self.txt_vd_details = text_edit(4, read_only=True)
        self.txt_vd_details.setAccessibleName("Technical details text")
        self.txt_vd_details.setMaximumHeight(150)
        self.txt_vd_details.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.txt_vd_details.setVisible(False)
        self.btn_vd_details.toggled.connect(self._toggle_error_details)
        el.addWidget(self.txt_vd_details)
        el.addStretch(1)
        self.vd_preview.insertWidget(PV_ERROR, err)
        return panel

    # ---- bottom strip: model / language / advanced
    def _build_options_strip(self):
        strip = QtWidgets.QVBoxLayout()   # Model + Language on one row, Advanced settings beneath: fits the form column
        strip.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(20)
        strip.addLayout(row)

        model_box = QtWidgets.QVBoxLayout()
        model_box.setSpacing(4)
        cap = QtWidgets.QLabel("Model")
        set_role(cap, "caption")
        self.lbl_vd_model = QtWidgets.QLabel(next(n for n, mid in MODELS.items() if mid == MODEL_MAP["voice-design"]))
        self.lbl_vd_model.setToolTip("Voice creation always uses this model; the clone model is loaded when a voice is saved.")
        self.lbl_vd_model.setAccessibleName("Model")
        self.lbl_vd_model.setMinimumHeight(36)  # same height as the language combo beside it
        model_box.addWidget(cap)
        model_box.addWidget(self.lbl_vd_model)
        row.addLayout(model_box)

        lang_box = QtWidgets.QVBoxLayout()
        lang_box.setSpacing(4)
        cap2 = QtWidgets.QLabel("Language")
        set_role(cap2, "caption")
        self.cmb_vd_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        self.cmb_vd_lang.setMinimumWidth(130)
        cap2.setBuddy(self.cmb_vd_lang)
        lang_box.addWidget(cap2)
        lang_box.addWidget(self.cmb_vd_lang)
        row.addLayout(lang_box)

        row.addStretch(1)
        adv_row = QtWidgets.QHBoxLayout()
        strip.addLayout(adv_row)
        self.btn_vd_advanced = QtWidgets.QPushButton("Advanced settings")
        self.btn_vd_advanced.setCheckable(True)
        self.btn_vd_advanced.setAccessibleName("Advanced settings")
        self._set_advanced_icon(False)
        set_role(self.btn_vd_advanced, "quiet")
        self.btn_vd_advanced.toggled.connect(self._toggle_advanced)
        adv_row.addWidget(self.btn_vd_advanced)
        adv_row.addStretch(1)
        return strip

    def _build_advanced_panel(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        seed_lay = QtWidgets.QHBoxLayout()
        seed_lay.setSpacing(8)
        seed_lbl = QtWidgets.QLabel("Seed")
        seed_lbl.setMinimumWidth(48)
        self.cmb_vd_seed_mode, self.spn_vd_seed = seed_row("Random", 42)
        self.lbl_vd_seed_used = QtWidgets.QLabel("")
        seed_lbl.setBuddy(self.cmb_vd_seed_mode)
        for w in (seed_lbl, self.cmb_vd_seed_mode, self.spn_vd_seed):
            seed_lay.addWidget(w)
        seed_lay.addStretch(1)
        lay.addLayout(seed_lay)
        lay.addWidget(self.lbl_vd_seed_used)
        note = QtWidgets.QLabel(
            "Random tries a new voice each time (the seed used is recorded with the saved voice). "
            "Fixed repeats the same sampling. A saved voice is locked via its reference clip, not the seed.")
        note.setWordWrap(True)
        set_role(note, "muted")
        lay.addWidget(note)

        auto_row = QtWidgets.QHBoxLayout()
        auto_row.setSpacing(8)
        self.chk_vd_autosave = QtWidgets.QCheckBox("Auto-save generated voices")
        self.chk_vd_autosave.setToolTip("Automatically save each generated sample as a voice.")
        gender_lbl = QtWidgets.QLabel("Name gender")
        self.cmb_vd_name_gender = QtWidgets.QComboBox()
        self.cmb_vd_name_gender.addItems(["Random", "Male", "Female"])
        self.cmb_vd_name_gender.setToolTip("Gender used for auto-generated voice names.")
        gender_lbl.setBuddy(self.cmb_vd_name_gender)
        self.btn_vd_rand_name = QtWidgets.QPushButton("Suggest name")
        self.btn_vd_rand_name.setToolTip("Fill the voice name with a random, unused name")
        self.btn_vd_rand_name.clicked.connect(self.fill_random_name)
        lay.addWidget(self.chk_vd_autosave)   # the form column is narrow: one control row each
        for w in (gender_lbl, self.cmb_vd_name_gender, self.btn_vd_rand_name):
            auto_row.addWidget(w)
        auto_row.addStretch(1)
        lay.addLayout(auto_row)

        compute = QtWidgets.QHBoxLayout()
        self.lbl_vd_compute = QtWidgets.QLabel("")
        set_role(self.lbl_vd_compute, "muted")
        self.lbl_vd_compute.setAccessibleName("Current compute settings")
        self.btn_vd_open_settings = QtWidgets.QPushButton("Open Settings")
        set_role(self.btn_vd_open_settings, "quiet")
        self.btn_vd_open_settings.clicked.connect(lambda: self.show_page(len(self.nav_buttons) - 1))
        self.lbl_vd_compute.setWordWrap(True)
        lay.addWidget(self.lbl_vd_compute)
        compute.addWidget(self.btn_vd_open_settings)
        compute.addStretch(1)
        lay.addLayout(compute)
        return panel

    def _toggle_advanced(self, on):
        self.vd_advanced.setVisible(on)
        self._set_advanced_icon(on)

    def _set_advanced_icon(self, expanded):
        """The chevron shows the state, and the accessible description says it in words."""
        icons.apply(self.btn_vd_advanced, "chevron-down" if expanded else "chevron-right")
        self.btn_vd_advanced.setAccessibleDescription("Expanded" if expanded else "Collapsed")

    def _update_compute_summary(self):
        device, dtype, attn = self.model_settings()
        self.lbl_vd_compute.setText(f"Device {device} · Dtype {dtype} · Attention {attn}")

    # ---- preview state helpers
    def _set_preview(self, state):
        self.vd_preview.setCurrentIndex(state)
        self.preview_panel.setProperty("empty", state == PV_EMPTY)  # quiet, unfilled panel until audio exists
        style = self.preview_panel.style()
        style.unpolish(self.preview_panel)
        style.polish(self.preview_panel)
        self._refresh_stale_note()

    def _relayout_preview(self):
        # Wrapped labels change height with their text; recompute so nothing is clipped or left oversized.
        self.vd_preview.updateGeometry()
        layout = self.preview_panel.layout()
        layout.invalidate()
        layout.activate()
        self.vd_content.layout().invalidate()

    def _vd_notice(self, text, details=False):
        self.lbl_vd_notice.setText(text)
        self.lbl_vd_notice.setVisible(bool(text))
        self.btn_vd_notice_details.setVisible(bool(text) and details)
        self._relayout_preview()

    def _on_generation_notice(self, message):
        """Backend stage messages: only the model-loading one is reported by the backend, the rest is ours."""
        stage = STAGE_LOADING if message.startswith("Loading ") and message.endswith(" model…") else message
        self._gen_stage = stage
        self.lbl_vd_busy.setText(stage)
        hint = {STAGE_LOADING: LOADING_HINT, STAGE_GENERATING: GENERATING_HINT}.get(stage)
        self.lbl_vd_busy_hint.setText(hint or "")
        self.lbl_vd_busy_hint.setVisible(bool(hint))
        self._tick_elapsed()

    def _generation_elapsed(self):
        return format_duration(max(0.0, time.monotonic() - self._gen_t0)) if self._gen_t0 else ""

    def _tick_elapsed(self):
        """Real elapsed time of the running generation (a monotonic clock), shown beside the stage."""
        if not self._gen_t0:
            return
        elapsed = self._generation_elapsed()
        self.lbl_vd_elapsed.setText(f"{elapsed} elapsed")
        if self.vd_preview.currentIndex() == PV_READY and self.busy:
            self._vd_notice(f"{self._gen_stage} {elapsed} elapsed. Your current sample stays available.")

    def _end_generation_clock(self):
        self._gen_t0 = None
        self.gen_timer.stop()

    def _toggle_error_details(self, on):
        self.txt_vd_details.setVisible(on)
        self.btn_vd_details.setText("Hide details" if on else "Show details")
        icons.apply(self.btn_vd_details, "chevron-down" if on else "chevron-right")
        self._relayout_preview()

    def copy_failure_details(self, text=None):
        QtWidgets.QApplication.clipboard().setText(self._failure_details if not isinstance(text, str) else text)
        self.status("Technical details copied.")

    def _open_failure_dialog(self, details=None):
        """Technical details of a failure whose previous good result stays on screen. Defaults to Create Voice's."""
        details = self._failure_details if details is None else details
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Technical details")
        dialog.setMinimumSize(560, 320)
        lay = QtWidgets.QVBoxLayout(dialog)
        box = text_edit(10, read_only=True)
        box.setAccessibleName("Technical details text")
        box.setPlainText(details)
        lay.addWidget(box, 1)
        buttons = QtWidgets.QDialogButtonBox()
        copy = buttons.addButton("Copy details", QtWidgets.QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: self.copy_failure_details(details))
        buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        dialog.exec()

    def _show_ready(self, result):
        wav, sr = np.asarray(result["wav"]), result["sr"]
        self.vd_player.set_audio(result["preview"], wav, sr)
        self.txt_vd_spoken.setPlainText(result.get("text") or voice_library.SAMPLE_TEXT)
        self.lbl_vd_origin.setText(self._origin_text(result))
        self.lbl_vd_origin.setToolTip(result["description"])
        # Always reachable (tooltip, screen-reader description); shown inline only when it matters (stale).
        self.lbl_vd_preview_title.setToolTip(self.lbl_vd_origin.text())
        self.txt_vd_spoken.setAccessibleDescription(self.lbl_vd_origin.text())
        self._vd_notice("")
        self._set_preview(PV_READY)

    def _show_error(self, message):
        """Explain a failed generation. The cause is named only when it is reliably identifiable;
        otherwise the message stays generic, and the technical details carry the real exception."""
        failure = (getattr(self.task, "failure", None) if self.task else None) or task_errors.Failure.from_message(message)
        explanation = task_errors.classify(failure)
        summary = explanation.summary if explanation else task_errors.UNKNOWN_SUMMARY
        action = explanation.action if explanation else task_errors.UNKNOWN_ACTION
        device, dtype, attn = self.model_settings()
        self._failure_details = task_errors.details_text(failure, context={
            "Model": MODEL_MAP["voice-design"],
            "Settings": f"device={device}, dtype={dtype}, attention={attn}",
            "Language": self.cmb_vd_lang.currentText()})
        self._end_generation_clock()
        self.vd_regen_bar.setVisible(False)
        self.btn_vd_error_settings.setVisible(bool(explanation and explanation.settings))
        if self.pending_voice:  # keep the last good sample available
            self._set_preview(PV_READY)
            self._vd_notice(f"Couldn't generate a new sample. {summary} Your previous sample is still available.",
                            details=True)
            self.status("Couldn't generate a new sample. Your previous sample is still available.")
            return
        self.lbl_vd_error.setText(summary)
        self.lbl_vd_error_action.setText(action)
        self.txt_vd_details.setPlainText(self._failure_details)
        self.btn_vd_details.setChecked(False)
        self._set_preview(PV_ERROR)
        self._announce_error()
        self.status(f"Couldn't generate the sample: {explanation.short}." if explanation
                    else "Couldn't generate the sample. See the Preview for details.")

    def _announce_error(self):
        """Tell assistive technology about the error, and put focus on Retry unless the person is typing."""
        try:
            event = QtGui.QAccessibleEvent(self.lbl_vd_error, QtGui.QAccessible.Event.Alert)
            QtGui.QAccessible.updateAccessibility(event)
        except Exception:  # accessibility must never break error handling
            pass
        focus = QtWidgets.QApplication.focusWidget()
        typing = isinstance(focus, (QtWidgets.QPlainTextEdit, QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox))
        if self.isActiveWindow() and not typing:
            self.btn_vd_retry.setFocus()

    def _on_description_changed(self):
        self.lbl_vd_field_error.setVisible(False)
        self._refresh_stale_note()
        text = self.txt_vd_instruct.toPlainText()
        for chip, name, phrase in self.vd_chips:  # keep chips in step with manual edits
            on = phrase in text
            if chip.isChecked() != on:
                chip.blockSignals(True)
                chip.setChecked(on)
                chip.blockSignals(False)

    @staticmethod
    def _origin_text(sample):
        desc = " ".join(sample["description"].split())
        if len(desc) > 90:
            desc = desc[:87].rstrip() + "…"
        return f"Made from “{desc}” · {sample['language']}"

    def _sample_is_stale(self):
        """True when the description, language or sample text differ from those that produced the sample."""
        pending = self.pending_voice
        return bool(pending) and (
            self.txt_vd_instruct.toPlainText().strip() != pending["description"]
            or self.cmb_vd_lang.currentText() != pending["language"]
            or self._form_sample_text() != self._sample_text_of(pending))

    def _refresh_stale_note(self):
        stale = self._sample_is_stale() and self.vd_preview.currentIndex() == PV_READY
        self.lbl_vd_stale.setVisible(stale)
        self.frm_vd_stale.setVisible(stale)
        self.lbl_vd_origin.setVisible(stale)   # which inputs made this audio: shown when the form has moved on
        self.lbl_vd_preview_title.setText("Previous sample" if stale else "Preview")
        text_changed = bool(self.pending_voice) and self._form_sample_text() != self._sample_text_of(self.pending_voice)
        self.lbl_vd_spoken_note.setText("Text used for this sample (the field has changed)"
                                        if text_changed else "Text used for this sample")
        # The footer must not claim the sample matches the form while it does not.
        if stale and self.lbl_status.text() == READY_STATUS:
            self.status(STALE_STATUS)
        elif not stale and self.lbl_status.text() == STALE_STATUS:
            self.status(READY_STATUS)
        self._refresh_save_button()
        self._relayout_preview()

    def _refresh_save_button(self):
        """Save needs a sample and a usable name; its label says which inputs it will store."""
        name = self.edt_vd_name.text().strip()
        has_sample = bool(self.pending_voice)
        valid = bool(voice_library.slugify(name))
        stale = self._sample_is_stale()
        self.btn_vd_save.setText("Save previous sample" if stale else "Save")
        if stale:
            tip = ("Saves this audio with the description, language and text it was made from "
                   "(shown above), not your current edits.")
        else:
            tip = "Save as a named voice in My Voices"
        if has_sample and not valid:
            tip = "Enter a voice name to enable Save."
        self.btn_vd_save.setToolTip(tip)
        self.btn_vd_save.setEnabled(has_sample and valid and not self.busy)
        if not has_sample or valid:
            hint = ""
        elif not name:
            hint = "Enter a voice name to save."
        else:
            hint = "Use at least one letter or number in the name."
        self.lbl_vd_name_hint.setText(hint)
        self.lbl_vd_name_hint.setVisible(bool(hint))
        self.edt_vd_name.setAccessibleDescription(hint)

    def _toggle_suggestion(self, phrase, on):
        text = self.txt_vd_instruct.toPlainText()
        if on and phrase not in text:
            new = f"{text.rstrip()} {phrase}" if text.strip() else phrase
        elif not on and phrase in text:
            new = " ".join(text.replace(phrase, " ").split())
        else:
            return
        cur = self.txt_vd_instruct.textCursor()  # one undoable edit, so Ctrl+Z restores the old text
        cur.select(QtGui.QTextCursor.Document)
        cur.insertText(new)
        self.txt_vd_instruct.setTextCursor(cur)

    # ---- handlers
    def generate_sample(self):
        description = self.txt_vd_instruct.toPlainText().strip()
        if not description:
            self.status("Error: enter a voice description first.")
            self.lbl_vd_field_error.setVisible(True)
            self.txt_vd_instruct.setFocus()
            return
        text = self._form_sample_text()
        if not text:
            self.status("Error: enter the sample text, or choose Use standard text.")
            self.txt_vd_sample.setFocus()
            return
        language = self.cmb_vd_lang.currentText()
        seed = (self.spn_vd_seed.value() if self.cmb_vd_seed_mode.currentText() == "Fixed"
                else random.randint(0, 2**31 - 1))
        audio_player.stop()
        self.vd_player.reset_start()

        def work(task):
            # The same call design_sample makes: it returns at once when the model is already loaded.
            # Making it here lets the app say "Generating" only after the model call has returned.
            self.models.get("voice-design", task.notify.emit)
            task.notify.emit(STAGE_GENERATING)
            wav, sr = self.generator(task).design_sample(description, language, seed, text)
            preview = os.path.join(OUTPUT_DIR, "_voice_preview.wav")
            write_wav(preview, wav, sr)
            return dict(wav=wav, sr=sr, seed=seed, description=description,
                        language=language, preview=preview, text=text)

        def done(result):
            self._end_generation_clock()
            self.vd_regen_bar.setVisible(False)
            self.pending_voice = result
            self.lbl_vd_seed_used.setText(f"Seed used: {result['seed']}")
            self._show_ready(result)
            self.status(READY_STATUS)
            self.play(result["preview"])
            if self.chk_vd_autosave.isChecked():
                gender = self.cmb_vd_name_gender.currentText().lower()
                if gender == "random":
                    gender = random.choice(["male", "female"])
                base = voice_library.random_voice_name(gender)
                name = voice_library.unique_voice_name(base)
                self.edt_vd_name.setText(name)
                self.save_voice(autosaved=True)

        if self.busy:
            self.status("Busy: a task is already running.")
            return
        self._gen_stage = STAGE_PREPARING
        self._gen_t0 = time.monotonic()
        self.gen_timer.start()
        self.lbl_vd_busy.setText(STAGE_PREPARING)
        self.lbl_vd_elapsed.setText("0:00 elapsed")
        self.lbl_vd_busy_hint.setVisible(False)
        if self.pending_voice:  # the current sample stays playable until its replacement is ready
            self._set_preview(PV_READY)
            self._vd_notice(f"{STAGE_PREPARING} 0:00 elapsed. Your current sample stays available.")
            self.vd_regen_bar.setVisible(True)
        else:
            self._set_preview(PV_BUSY)
        # Preview owns the progress and stage text; the footer states the activity once, without animating.
        self.status(FOOTER_CREATING)
        self.run_task(work, done, on_fail=self._show_error, on_notify=self._on_generation_notice,
                      stoppable=False, footer_progress=False, footer_status=False)

    # ---- Preview player: one Play/Pause control over audio_player's own state
    def _form_sample_text(self):
        return self.txt_vd_sample.toPlainText().strip()

    @staticmethod
    def _sample_text_of(sample):
        return sample.get("text") or voice_library.SAMPLE_TEXT

    def play_sample(self):
        if not self.pending_voice:
            self.status("Generate a sample first.")
            return
        self.vd_player.toggle()

    def _on_preview_play_failed(self, error):
        self.status(f"Playback error: {error}")
        self._vd_notice(f"Couldn't play the sample: {error}")

    def _seek_preview(self, seconds):
        if self.pending_voice:
            self.vd_player.seek(seconds)

    def _refresh_player(self):
        """Mirror the real playback state into the Preview. Called by the timer and after every action."""
        if self.pending_voice:
            self.vd_player.refresh()

    def _tick_player(self):
        if not self.isVisible():
            return
        live = False
        if self.pending_voice and self.vd_preview.currentIndex() == PV_READY:
            self._refresh_player()
            live = True
        if self.vc_player.has_audio() and self.vc_player.isVisible():
            self.vc_player.refresh()
            live = True
        if live:
            self._refresh_audio_state()   # the footer reads the same player state in the same tick

    def edit_sample_text(self):
        """Take the person to the Sample text field (the existing audio is never edited)."""
        self.txt_vd_sample.setFocus()
        QtCore.QTimer.singleShot(
            0, lambda: self.stack.widget(PAGE_CREATE).ensureWidgetVisible(self.txt_vd_sample))

    def _on_sample_text_changed(self):
        self._refresh_stale_note()

    def download_sample(self):
        pending = self.pending_voice
        if not pending:
            self.status("Generate a sample first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Download sample", "voice-sample.wav", "WAV audio (*.wav)")
        if not path:
            return
        try:
            write_wav(path, np.asarray(pending["wav"]), pending["sr"])
        except Exception as e:
            self._vd_notice(f"Couldn't save the file: {e}")
            return
        self._vd_notice(f"Saved a copy to {path}")
        self.status(f"Sample exported to: {path}")

    def save_voice(self, autosaved=False):
        pending = self.pending_voice
        name = self.edt_vd_name.text().strip()
        if not pending:
            self.status("Error: generate and audition a sample first.")
            return
        if not voice_library.slugify(name):
            self.status("Error: enter a voice name.")
            self._vd_notice("Enter a voice name to save.")
            self.edt_vd_name.setFocus()
            return
        was_stale = self._sample_is_stale()  # the snapshot below keeps the sample's own inputs either way
        overwrite = False
        if voice_library.voice_exists(name) and autosaved:
            name = voice_library.unique_voice_name(name)
            self.edt_vd_name.setText(name)
        else:
            overwrite = self._confirm_replace_voice(name, blocked=self._vd_notice)
            if overwrite is None:
                return

        def work(task):
            text = self._sample_text_of(pending)   # the words actually spoken in this audio
            prompt = self.generator(task).build_voice_prompt(pending["wav"], pending["sr"], text)
            voice_library.save_voice(
                name, pending["wav"], pending["sr"], prompt, pending["description"],
                pending["language"], pending["seed"],
                model_ids=dict(MODEL_MAP),
                overwrite=overwrite,
                sample_text=text,
            )
            return name

        def done(saved_name):
            self.refresh_voice_lists()
            self.status(f"Voice '{saved_name}' saved to voices/{voice_library.slugify(saved_name)}/")
            note = f"Saved as '{saved_name}'. Find it in My Voices."
            if was_stale:
                note = f"Saved as '{saved_name}' with the description it was made from. Find it in My Voices."
            self._vd_notice(note)

        self._vd_notice("")
        self.run_task(work, done, on_fail=lambda m: self._vd_notice(f"Couldn't save the voice: {m}"),
                      stoppable=False)

    def fill_random_name(self):
        gender = self.cmb_vd_name_gender.currentText().lower()
        if gender == "random":
            gender = random.choice(["male", "female"])
        name = voice_library.random_voice_name(gender)
        name = voice_library.unique_voice_name(name)
        self.edt_vd_name.setText(name)
        self.status(f"Suggested name: {name}")

    def refresh_voice_lists(self):
        names = voice_library.list_voices()
        keep = self._selected_clone_voice()
        self._vc_syncing = True   # clearing/refilling must not look like a user choice (it would reset Language)
        try:
            self.cmb_vc_voice.clear()
            self.cmb_vc_voice.addItems(names)
            self.cmb_vc_voice.setCurrentIndex(self.cmb_vc_voice.findText(keep) if keep in names else -1)
        finally:
            self._vc_syncing = False
        self.lbl_vc_empty.setVisible(not names)
        self._refresh_voice_desc()
        if hasattr(self, "audiobook_tab"):
            self.audiobook_tab.refresh_voices()
        if hasattr(self, "drama_tab"):
            self.drama_tab.refresh_voices()

    # ------------------------------------------------------------------ VoiceClone
    # Exactly one voice source is active at a time; only it is sent to generation.

    def _build_clone_tab(self):
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        page.setWidget(body)
        frame = QtWidgets.QHBoxLayout(body)
        frame.setContentsMargins(32, 32, 32, 24)
        content = QtWidgets.QWidget()
        content.setMaximumWidth(CONTENT_MAX_WIDTH)
        frame.addWidget(content, 100, QtCore.Qt.AlignTop)
        frame.addStretch(1)
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(24)
        outer.addWidget(page_header("Clone Voice", "Use a saved voice or a recording to generate speech."))
        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(32)
        cols.addLayout(self._build_clone_column(), 5)
        cols.addWidget(self._build_clone_preview(), 4, QtCore.Qt.AlignTop)
        outer.addLayout(cols)
        self._vc_syncing = False
        self.btn_vc_src_saved.setChecked(True)
        self._on_clone_source_changed()
        self.chk_vc_npz.toggled.connect(self._on_clone_source_changed)
        self.chk_vc_xvec.toggled.connect(self._on_clone_source_changed)
        for w in (self.txt_vc_text, self.txt_vc_instruct, self.txt_vc_ref_text):
            w.textChanged.connect(self._clear_clone_error)
        for w in (self.edt_vc_ref, self.edt_vc_reuse):
            w.textChanged.connect(self._clear_clone_error)
        self._set_clone_preview(CV_EMPTY)
        return page

    def _clone_field(self, text, buddy=None):
        lbl = QtWidgets.QLabel(text)
        set_role(lbl, "field")
        if buddy is not None:
            lbl.setBuddy(buddy)
        return lbl

    def _label_with_info(self, text, topic, tip, buddy=None, width=None):
        """A field label with a small (i) right after it; the explanation is also its accessible description.
        `width` gives rows of controls one shared label column so the fields line up."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QtWidgets.QLabel(text)
        set_role(lbl, "field")
        if buddy is not None:
            lbl.setBuddy(buddy)
        info = InfoButton(topic, tip)
        row.addWidget(lbl)
        row.addWidget(info)
        row.addStretch(1)
        if width:
            box.setFixedWidth(width)
        self.info_buttons = getattr(self, "info_buttons", {})
        self.info_buttons[topic] = info
        return box

    def _build_clone_column(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)

        # ---- voice source: two explicit, exclusive modes
        col.addWidget(self._clone_field("Voice source"))
        seg = QtWidgets.QHBoxLayout()
        seg.setSpacing(0)
        self.grp_vc_source = QtWidgets.QButtonGroup(self)
        self.grp_vc_source.setExclusive(True)
        self.btn_vc_src_saved = QtWidgets.QPushButton("Saved voice")
        self.btn_vc_src_ref = QtWidgets.QPushButton("Reference audio")
        for i, b in enumerate((self.btn_vc_src_saved, self.btn_vc_src_ref)):
            b.setCheckable(True)
            set_role(b, "segment")
            b.setProperty("position", "left" if i == 0 else "right")
            self.grp_vc_source.addButton(b, i)
            seg.addWidget(b, 1)
        self.btn_vc_src_saved.setAccessibleDescription("Generate with a voice from your library")
        self.btn_vc_src_ref.setAccessibleDescription("Generate by copying a recording you provide")
        self.grp_vc_source.idToggled.connect(lambda _i, on: on and self._on_clone_source_changed())
        col.addLayout(seg)

        # ---- saved-voice panel
        self.pnl_vc_saved = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(self.pnl_vc_saved)
        sv.setContentsMargins(0, 8, 0, 0)
        sv.setSpacing(8)
        self.cmb_vc_voice = QtWidgets.QComboBox()
        self.cmb_vc_voice.setPlaceholderText("Select a voice…")
        self.cmb_vc_voice.setAccessibleName("Saved voice")
        self.cmb_vc_voice.currentIndexChanged.connect(self.on_clone_voice_changed)
        sv.addWidget(self._clone_field("Choose a saved voice", self.cmb_vc_voice))
        pick = QtWidgets.QHBoxLayout()
        pick.setSpacing(8)
        pick.addWidget(self.cmb_vc_voice, 1)
        self.btn_vc_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_vc_refresh.setToolTip("Reload the list of saved voices")
        self.btn_vc_refresh.clicked.connect(self.refresh_voice_lists)
        pick.addWidget(self.btn_vc_refresh)
        sv.addLayout(pick)
        self.lbl_vc_empty = WrapLabel("No saved voices yet. Create one, or switch to Reference audio.")
        set_role(self.lbl_vc_empty, "muted")
        sv.addWidget(self.lbl_vc_empty)
        col.addWidget(self.pnl_vc_saved)

        # ---- reference-audio panel
        self.pnl_vc_ref = QtWidgets.QWidget()
        rf = QtWidgets.QVBoxLayout(self.pnl_vc_ref)
        rf.setContentsMargins(0, 8, 0, 0)
        rf.setSpacing(8)
        self.edt_vc_ref = QtWidgets.QLineEdit()
        self.edt_vc_ref.setPlaceholderText("Choose a recording")
        self.edt_vc_ref.setAccessibleName("Reference audio file")
        self.btn_vc_ref_browse = QtWidgets.QPushButton("Browse…")
        self.btn_vc_ref_browse.clicked.connect(
            lambda: self.browse(self.edt_vc_ref, audio_file_filter()))
        self.lbl_vc_ref = self._clone_field("Reference audio", self.edt_vc_ref)
        rf.addWidget(self.lbl_vc_ref)
        rf.addWidget(self.row_with(self.edt_vc_ref, self.btn_vc_ref_browse))
        self.lbl_vc_transcript = self._clone_field("Transcript")
        rf.addWidget(self.lbl_vc_transcript)
        self.txt_vc_ref_text = text_edit(3, "Exact words spoken in the recording")
        self.txt_vc_ref_text.setFixedHeight(self.txt_vc_ref_text.fontMetrics().lineSpacing() * 3 + 20)
        self.txt_vc_ref_text.setAccessibleName("Transcript")
        self.lbl_vc_transcript.setBuddy(self.txt_vc_ref_text)
        rf.addWidget(self.txt_vc_ref_text)
        self.chk_vc_xvec = QtWidgets.QCheckBox("X-vector only (no transcript needed)")
        self.chk_vc_xvec.setToolTip(XVECTOR_TIP)
        self.chk_vc_xvec.setAccessibleDescription(XVECTOR_TIP)
        rf.addWidget(self.chk_vc_xvec)
        self.lbl_vc_npz_active = WrapLabel("Using a prompt file instead (set in Advanced settings).")
        set_role(self.lbl_vc_npz_active, "muted")
        rf.addWidget(self.lbl_vc_npz_active)
        col.addWidget(self.pnl_vc_ref)
        self._vc_saved_name = None

        # ---- text
        col.addSpacing(8)
        self.txt_vc_text = text_edit(5, text="Hello, I am cloning this voice.")
        self.txt_vc_text.setFixedHeight(self.txt_vc_text.fontMetrics().lineSpacing() * 4 + 20)
        self.txt_vc_text.setAccessibleName("Text to generate")
        col.addWidget(self._clone_field("Text to generate", self.txt_vc_text))
        col.addWidget(self.txt_vc_text)

        # ---- language
        col.addSpacing(4)
        self.cmb_vc_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        self.cmb_vc_lang.setMinimumWidth(160)
        col.addWidget(self._clone_field("Language", self.cmb_vc_lang))
        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(self.cmb_vc_lang)
        lang_row.addStretch(1)
        col.addLayout(lang_row)

        # ---- generate (above Advanced, like Create Voice); the problem, if any, is shown right above it
        col.addSpacing(4)
        self.lbl_vc_error = WrapLabel("")
        set_role(self.lbl_vc_error, "error-text")
        self.lbl_vc_error.setAccessibleName("Problem")
        self.lbl_vc_error.setVisible(False)
        col.addWidget(self.lbl_vc_error)
        col.addSpacing(4)
        gen = QtWidgets.QHBoxLayout()
        self.btn_vc_generate = QtWidgets.QPushButton("Generate audio")
        set_role(self.btn_vc_generate, "primary")
        icons.apply(self.btn_vc_generate, "audio-lines", icons.SIZE_COMPACT, "primary")
        self.btn_vc_generate.setMinimumWidth(180)
        self.btn_vc_generate.clicked.connect(self.generate_clone)
        gen.addWidget(self.btn_vc_generate)
        gen.addStretch(1)
        col.addLayout(gen)

        self.pnl_vc_save = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(self.pnl_vc_save)
        sl.setContentsMargins(0, 8, 0, 0)
        sl.setSpacing(8)
        # One description for the whole reference workflow: saved with the voice, and used for [tags] when generating here
        self.txt_vc_instruct = text_edit(2, "e.g. Warm, gentle female voice, calm and soothing tone.")
        self.txt_vc_instruct.setFixedHeight(self.txt_vc_instruct.fontMetrics().lineSpacing() * 2 + 20)
        self.txt_vc_instruct.setAccessibleName("Voice description")
        sl.addWidget(self._label_with_info("Description (optional)", "Description", DESCRIPTION_TIP,
                                           buddy=self.txt_vc_instruct))
        sl.addWidget(self.txt_vc_instruct)
        # saving the recording as a voice acts on the reference, not on generated audio
        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(8)
        self.edt_vc_new_name = QtWidgets.QLineEdit()
        self.edt_vc_new_name.setPlaceholderText("New voice name")
        self.edt_vc_new_name.setAccessibleName("New voice name")
        self.btn_vc_save_voice = QtWidgets.QPushButton("Save as voice")
        self.btn_vc_save_voice.setToolTip("Save this recording as a reusable voice (no audio is generated)")
        self.btn_vc_save_voice.clicked.connect(self.save_clone_as_voice)
        save_row.addWidget(self.edt_vc_new_name, 1)
        save_row.addWidget(self.btn_vc_save_voice)
        sl.addWidget(self._clone_field("Save the recording as a voice", self.edt_vc_new_name))
        sl.addLayout(save_row)
        use_row = QtWidgets.QHBoxLayout()
        self.lbl_vc_saved_note = WrapLabel("")
        set_role(self.lbl_vc_saved_note, "muted")
        self.btn_vc_use_saved = QtWidgets.QPushButton("Use this voice")
        set_role(self.btn_vc_use_saved, "quiet")
        self.btn_vc_use_saved.clicked.connect(self.use_saved_clone_voice)
        use_row.addWidget(self.lbl_vc_saved_note, 1)
        use_row.addWidget(self.btn_vc_use_saved)
        self.row_vc_use = QtWidgets.QWidget()
        self.row_vc_use.setLayout(use_row)
        use_row.setContentsMargins(0, 0, 0, 0)
        self.row_vc_use.setVisible(False)
        sl.addWidget(self.row_vc_use)
        col.addWidget(self.pnl_vc_save)
        # ---- advanced (collapsed): technical controls, semantics unchanged
        col.addSpacing(8)
        divider = QtWidgets.QFrame()
        set_role(divider, "divider")
        col.addWidget(divider)
        adv_row = QtWidgets.QHBoxLayout()
        self.btn_vc_advanced = QtWidgets.QPushButton("Advanced settings")
        self.btn_vc_advanced.setCheckable(True)
        self.btn_vc_advanced.setAccessibleName("Advanced settings")
        set_role(self.btn_vc_advanced, "quiet")
        self.btn_vc_advanced.toggled.connect(self._toggle_clone_advanced)
        adv_row.addWidget(self.btn_vc_advanced)
        adv_row.addStretch(1)
        col.addLayout(adv_row)
        self.vc_advanced = self._build_clone_advanced()
        self.vc_advanced.setVisible(False)
        col.addWidget(self.vc_advanced)
        self._set_clone_advanced_icon(False)
        col.addStretch(1)
        return col

    def _build_clone_advanced(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        lay.addWidget(self._label_with_info("Direction tags", "Direction tags", DIRECTION_TAGS_TIP))
        self.dir_vc = DirectionBar(self.txt_vc_text)
        lay.addWidget(self.dir_vc)

        # Saved voice mode only: shows what the voice was saved with. The editable description lives beside Save as voice.
        self.lbl_vc_description = self._clone_field("Voice description (from the saved voice)")
        lay.addWidget(self.lbl_vc_description)
        self.lbl_vc_voice_desc = WrapLabel("")
        self.lbl_vc_voice_desc.setAccessibleName("Saved voice description")
        set_role(self.lbl_vc_voice_desc, "muted")
        lay.addWidget(self.lbl_vc_voice_desc)

        # numeric fields use the shared field style (role "numeric"): the stepper buttons take no space and draw nothing,
        # while typing, the arrow keys and the mouse wheel still step by the same amounts
        self.cmb_vc_seed_mode, self.spn_vc_seed = seed_row("Fixed", 42)
        self.spn_vc_temp = QtWidgets.QDoubleSpinBox()
        self.spn_vc_temp.setRange(0.1, 1.5)
        self.spn_vc_temp.setSingleStep(0.05)
        self.spn_vc_temp.setValue(0.7)
        style_numeric(self.spn_vc_temp, name="Temperature")
        seed_lay = QtWidgets.QHBoxLayout()
        seed_lay.setSpacing(8)
        seed_lay.addWidget(self._label_with_info("Seed", "Seed", SEED_TIP, buddy=self.cmb_vc_seed_mode, width=SETTING_LABEL_WIDTH))
        seed_lay.addWidget(self.cmb_vc_seed_mode)
        seed_lay.addWidget(self.spn_vc_seed)
        seed_lay.addStretch(1)
        lay.addLayout(seed_lay)
        temp_lay = QtWidgets.QHBoxLayout()
        temp_lay.setSpacing(8)
        temp_lay.addWidget(self._label_with_info("Temperature", "Temperature", TEMPERATURE_TIP, buddy=self.spn_vc_temp, width=SETTING_LABEL_WIDTH))
        temp_lay.addWidget(self.spn_vc_temp)
        temp_lay.addStretch(1)
        lay.addLayout(temp_lay)

        # a reusable prompt is an alternative to reference audio, never combined with it
        self.pnl_vc_npz = QtWidgets.QWidget()
        nz = QtWidgets.QVBoxLayout(self.pnl_vc_npz)
        nz.setContentsMargins(0, 0, 0, 0)
        nz.setSpacing(8)
        self.chk_vc_npz = QtWidgets.QCheckBox("Use a prompt file instead of the recording")
        tip = "When on, the prompt file is the only source and the recording fields are ignored. " + PROMPT_FILE_TIP
        self.chk_vc_npz.setToolTip(tip)
        self.chk_vc_npz.setAccessibleDescription(tip)
        nz.addWidget(self.chk_vc_npz)
        self.edt_vc_reuse = QtWidgets.QLineEdit()
        self.edt_vc_reuse.setPlaceholderText("Choose a prompt file (.npz)")
        self.edt_vc_reuse.setAccessibleName("Prompt file to reuse")
        self.btn_vc_reuse_browse = QtWidgets.QPushButton("Browse…")
        self.btn_vc_reuse_browse.clicked.connect(lambda: self.browse(self.edt_vc_reuse, "NPZ (*.npz)"))
        self.row_vc_reuse = self.row_with(self.edt_vc_reuse, self.btn_vc_reuse_browse)
        nz.addWidget(self.row_vc_reuse)
        lay.addWidget(self.pnl_vc_npz)

        self.edt_vc_save = QtWidgets.QLineEdit()
        self.edt_vc_save.setPlaceholderText("Optional: save the prompt used (.npz)")
        self.edt_vc_save.setAccessibleName("Save prompt to")
        self.btn_vc_save_browse = QtWidgets.QPushButton("Browse…")
        self.btn_vc_save_browse.clicked.connect(lambda: self.browse(self.edt_vc_save, "NPZ (*.npz)", save=True))
        lay.addWidget(self._clone_field("Save prompt to (NPZ)", self.edt_vc_save))
        lay.addWidget(self.row_with(self.edt_vc_save, self.btn_vc_save_browse))
        return panel

    def _toggle_clone_advanced(self, on):
        self.vc_advanced.setVisible(on)
        self._set_clone_advanced_icon(on)

    def _set_clone_advanced_icon(self, expanded):
        icons.apply(self.btn_vc_advanced, "chevron-down" if expanded else "chevron-right")
        self.btn_vc_advanced.setAccessibleDescription("Expanded" if expanded else "Collapsed")

    # ---- right column: a small result surface (state text only; the shared player is a later phase)
    def _build_clone_preview(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("previewPanel")
        self.clone_preview_panel = panel
        panel.setMinimumWidth(340)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(12)
        title = QtWidgets.QLabel("Preview")
        set_role(title, "heading")
        lay.addWidget(title)
        self.lbl_vc_pv_title = WrapLabel("")
        set_role(self.lbl_vc_pv_title, "notice")
        self.lbl_vc_pv_title.setAccessibleName("Preview status")
        lay.addWidget(self.lbl_vc_pv_title)
        self.lbl_vc_pv_detail = WrapLabel("")
        set_role(self.lbl_vc_pv_detail, "muted")
        lay.addWidget(self.lbl_vc_pv_detail)
        err_row = QtWidgets.QHBoxLayout()
        err_row.setSpacing(8)
        self.btn_vc_details = QtWidgets.QPushButton("Show details")
        self.btn_vc_details.setAccessibleName("Technical details")
        self.btn_vc_details.setToolTip("Show the technical details of this failure (private paths and credentials removed)")
        set_role(self.btn_vc_details, "quiet")
        icons.apply(self.btn_vc_details, "chevron-right")
        self.btn_vc_details.clicked.connect(lambda: self._open_failure_dialog(self._clone_failure_details))
        self.btn_vc_error_settings = QtWidgets.QPushButton("Open Settings")
        set_role(self.btn_vc_error_settings, "quiet")
        icons.apply(self.btn_vc_error_settings, "settings")
        self.btn_vc_error_settings.clicked.connect(lambda: self.show_page(len(self.nav_buttons) - 1))
        for b in (self.btn_vc_details, self.btn_vc_error_settings):
            b.setVisible(False)
            err_row.addWidget(b)
        err_row.addStretch(1)
        lay.addLayout(err_row)
        self.lbl_vc_pv_prev = QtWidgets.QLabel("Previous audio")
        set_role(self.lbl_vc_pv_prev, "field")
        self.lbl_vc_pv_prev.setVisible(False)
        lay.addWidget(self.lbl_vc_pv_prev)
        self.vc_player = PreviewPlayer("audio")
        self.vc_player.setVisible(False)
        self.vc_player.changed.connect(self._refresh_audio_state)
        self.vc_player.failed.connect(self._on_clone_play_failed)
        lay.addWidget(self.vc_player)
        lay.addStretch(1)
        return panel

    def _on_clone_play_failed(self, error):
        self.status(f"Playback error: {error}")
        self.lbl_vc_pv_detail.setText(f"Couldn't play the audio: {error}")

    def _set_clone_preview(self, state, detail="", audio=None):
        """One state at a time: EMPTY, BUSY, READY, PARTIAL (stopped early, audio kept), STOPPED (stopped, no audio),
        ERROR, SAVED (the audio was generated and saved, only the Preview couldn't load it). `audio` = (path,
        samples, sample_rate) for READY and PARTIAL, which replace the player's audio. BUSY, STOPPED, ERROR and
        SAVED keep the previous audio on screen (as Create Voice keeps its last sample)."""
        self._vc_pv_state = state
        replaces = state in (CV_READY, CV_PARTIAL) and audio is not None
        keeps = state in (CV_BUSY, CV_STOPPED, CV_ERROR, CV_SAVED) and self.vc_player.has_audio()
        had_previous = self.vc_player.has_audio()
        title = {CV_EMPTY: "Your generated audio will appear here.",
                 CV_BUSY: "Generating audio…",
                 CV_READY: "Audio ready",
                 CV_PARTIAL: "Stopped early",
                 CV_STOPPED: "Stopped before any audio was produced",
                 CV_SAVED: "Audio saved. Preview unavailable",
                 CV_ERROR: "Couldn't generate new audio" if had_previous else "Couldn't generate the audio"}[state]
        if state == CV_STOPPED:
            detail = detail or ("Nothing new was saved. Your previous audio is still available." if had_previous
                                else "Nothing was saved.")
        detail = detail or {CV_EMPTY: "Choose a voice, enter your text, then generate audio."}.get(state, "")
        focus = QtWidgets.QApplication.focusWidget()
        if replaces:
            self.vc_player.clear()        # stops the previous audio if it is the one playing, then loads the new file
            self.vc_player.set_audio(*audio)
        elif not keeps:
            self.vc_player.clear()
        self.vc_player.setVisible(replaces or keeps)
        self.lbl_vc_pv_prev.setVisible(keeps)
        for b in (self.btn_vc_details, self.btn_vc_error_settings):
            b.setVisible(False)           # _show_clone_error turns these on for a failure
        self.lbl_vc_pv_title.setText(title)
        self.lbl_vc_pv_detail.setMinimumHeight(0)   # WrapLabel only recomputes its reserved height upward from a stale value
        self.lbl_vc_pv_detail.setText(detail)
        self.lbl_vc_pv_detail.setVisible(bool(detail))
        if focus is not None and self.clone_preview_panel.isAncestorOf(focus) and not focus.isVisibleTo(self.clone_preview_panel):
            self.txt_vc_text.setFocus()   # the control that had focus just disappeared; keep focus on something visible
        if state in (CV_ERROR, CV_PARTIAL, CV_SAVED):
            QtGui.QAccessible.updateAccessibility(QtGui.QAccessibleEvent(self.lbl_vc_pv_title, QtGui.QAccessible.Event.Alert))

    def _store_clone_details(self, failure, task):
        device, dtype, attn = self.model_settings()
        self._clone_failure_details = task_errors.details_text(failure, task=task, context={
            "Model": MODEL_MAP["voice-clone"],
            "Settings": f"device={device}, dtype={dtype}, attention={attn}",
            "Language": self.cmb_vc_lang.currentText(),
            "Source": self.clone_source_kind()})

    def _show_clone_saved_without_preview(self, path, early, exc):
        """The audio was generated and written; only reading it back for the Preview failed. That is not a failed
        generation: say it was saved, where, and that the footer's Play last output / Open output folder still work."""
        self._store_clone_details(task_errors.Failure.from_exception(exc), "Clone Voice Preview")
        lead = "Stopped early; only the part generated so far was saved. " if early else ""
        self._set_clone_preview(CV_SAVED, f"{lead}Saved to {path}. The Preview couldn't load this file. "
                                          "Use Play last output or Open output folder below to reach it.")
        self.btn_vc_details.setVisible(True)

    def _show_clone_error(self, message, failure=None):
        """Explain a failed generation the way Create Voice does: a cause only when it is reliably identifiable, an
        action, and sanitized technical details on request. The raw exception text is never put in the Preview."""
        failure = failure or (getattr(self.task, "failure", None) if self.task else None) \
            or task_errors.Failure.from_message(message)
        explanation = task_errors.classify(failure, noun="audio")
        summary = explanation.summary if explanation else task_errors.UNKNOWN_SUMMARY_AUDIO
        action = explanation.action if explanation else task_errors.UNKNOWN_ACTION_AUDIO
        self._store_clone_details(failure, "Clone Voice audio")
        had_previous = self.vc_player.has_audio()
        self._set_clone_preview(CV_ERROR, f"{summary} {action}" + (" Your previous audio is still available." if had_previous else ""))
        self.btn_vc_details.setVisible(True)
        self.btn_vc_error_settings.setVisible(bool(explanation and explanation.settings))
        self.status(f"Couldn't generate the audio: {explanation.short}." if explanation
                    else "Couldn't generate the audio. See the Preview for details.")

    # ---- source state
    def clone_source_kind(self):
        """'saved', 'reference' or 'prompt': the one source generation will use."""
        if self.btn_vc_src_saved.isChecked():
            return "saved"
        return "prompt" if self.chk_vc_npz.isChecked() else "reference"

    def _on_clone_source_changed(self, *_):
        saved = self.btn_vc_src_saved.isChecked()
        # one tab stop for the pair (the selected one); the arrow keys move between them
        self.btn_vc_src_saved.setFocusPolicy(QtCore.Qt.StrongFocus if saved else QtCore.Qt.ClickFocus)
        self.btn_vc_src_ref.setFocusPolicy(QtCore.Qt.ClickFocus if saved else QtCore.Qt.StrongFocus)
        npz = self.chk_vc_npz.isChecked() and not saved
        self.pnl_vc_saved.setVisible(saved)
        self.pnl_vc_ref.setVisible(not saved)
        self.pnl_vc_npz.setVisible(not saved)
        self.pnl_vc_save.setVisible(not saved)
        # the other source's entries are kept, but greyed out and never sent
        for w in (self.edt_vc_ref, self.btn_vc_ref_browse, self.chk_vc_xvec):
            w.setEnabled(not npz)
        self.txt_vc_ref_text.setEnabled(not npz and not self.chk_vc_xvec.isChecked())
        self.lbl_vc_npz_active.setVisible(npz)
        self.btn_vc_save_voice.setEnabled(not npz)
        self.edt_vc_new_name.setEnabled(not npz)
        self.row_vc_reuse.setEnabled(npz)
        self.lbl_vc_description.setVisible(saved)
        self.lbl_vc_voice_desc.setVisible(saved)
        self._refresh_voice_desc()
        self._clear_clone_error()

    def _refresh_voice_desc(self):
        name = self._selected_clone_voice()
        desc = self._saved_voice_description(name) if name else ""
        self.lbl_vc_voice_desc.setText(desc or ("This voice has no description, so [tags] can't be used."
                                                if name else "Choose a voice to see its description."))

    @staticmethod
    def _saved_voice_description(name):
        try:
            with open(os.path.join(voice_library.voice_dir(name), "voice.json"), encoding="utf-8") as f:
                return (json.load(f).get("description") or "").strip()
        except (OSError, ValueError):
            return ""

    def _selected_clone_voice(self):
        """The chosen saved voice, or None. The combo has no blank entry, so there is no hidden 'none' mode."""
        return self.cmb_vc_voice.currentText() if self.cmb_vc_voice.currentIndex() >= 0 else None

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

    def on_clone_voice_changed(self, _index=None):
        """A user's choice prefills Language; list refreshes and clearing never touch the form."""
        if self._vc_syncing:
            return
        name = self._selected_clone_voice()
        self._refresh_voice_desc()
        self._clear_clone_error()
        if not name:
            return
        try:
            cfg, _, _ = voice_library.load_voice(name)
        except Exception as e:
            self.status(f"Could not load voice: {e}")
            return
        if cfg.get("language") in LANGUAGES:
            self.cmb_vc_lang.setCurrentText(cfg["language"])
        self.status(f"Using saved voice '{name}'.")

    def _clear_clone_error(self, *_):
        self.lbl_vc_error.setVisible(False)
        self.lbl_vc_error.setText("")

    def _clone_error(self, message, widget=None, advanced=False):
        """Show a problem next to the form and move focus to the field that needs fixing."""
        self.lbl_vc_error.setText(message)
        self.lbl_vc_error.setVisible(True)
        self.status(f"Error: {message}")
        if advanced and not self.btn_vc_advanced.isChecked():
            self.btn_vc_advanced.setChecked(True)
        if widget is not None:
            widget.setFocus()
        return False

    def _validate_clone(self, for_save=False):
        """Check everything the model would need, before it is loaded. True when generation can start."""
        kind = "reference" if for_save else self.clone_source_kind()
        text = self.txt_vc_text.toPlainText().strip()
        if not for_save and not text:
            return self._clone_error("Enter the text to generate.", self.txt_vc_text)
        if kind == "saved":
            name = self._selected_clone_voice()
            if not name:
                return self._clone_error("Choose a saved voice.", self.cmb_vc_voice)
            if not voice_library.voice_exists(name):
                return self._clone_error(f"The voice '{name}' is no longer available. Refresh the list.",
                                         self.btn_vc_refresh)
        elif kind == "prompt":
            path = self.edt_vc_reuse.text().strip()
            if not path:
                return self._clone_error("Choose a prompt file, or turn off 'Use a saved prompt file'.",
                                         self.edt_vc_reuse, advanced=True)
            if not os.path.isfile(path):
                return self._clone_error("The prompt file was not found.", self.edt_vc_reuse, advanced=True)
        else:
            path = self.edt_vc_ref.text().strip()
            if not path:
                return self._clone_error("Choose a reference recording.", self.edt_vc_ref)
            if not os.path.isfile(path):
                return self._clone_error("The reference recording was not found.", self.edt_vc_ref)
            if for_save and os.path.splitext(path)[1].lower() != ".wav":
                return self._clone_error("Saving a voice needs a WAV recording.", self.edt_vc_ref)
            problem = audio_problem(path)   # header only: an unsupported or damaged file is explained before any model loads
            if problem:
                return self._clone_error(problem, self.edt_vc_ref)
            if not self.chk_vc_xvec.isChecked() and not self.txt_vc_ref_text.toPlainText().strip():
                return self._clone_error(
                    "Enter the transcript of the recording, or choose X-vector only.", self.txt_vc_ref_text)
        if not for_save:
            has_tags = any(emotion for emotion, _ in parse_emotion_segments(text))
            if has_tags:
                if kind == "saved":
                    if not self._saved_voice_description(self._selected_clone_voice()):
                        return self._clone_error(
                            "This voice has no saved description, so [tags] can't be used. Remove the tags.",
                            self.txt_vc_text)
                elif not self.txt_vc_instruct.toPlainText().strip():
                    return self._clone_error("Describe the voice (the Description field above Save as voice) to use [emotion] tags.",
                                             self.txt_vc_instruct)
                if kind == "prompt":
                    try:
                        prompt = load_voice_clone_prompt(self.edt_vc_reuse.text().strip())
                    except Exception as e:
                        return self._clone_error(f"The prompt file couldn't be read: {e}",
                                                 self.edt_vc_reuse, advanced=True)
                    if not isinstance(prompt, list):
                        return self._clone_error(
                            "[Tags] need a prompt saved as a list; rebuild this prompt file.",
                            self.edt_vc_reuse, advanced=True)
        self._clear_clone_error()
        return True

    def clone_args(self):
        """Arguments for generation: only the active source is filled; every other source stays empty."""
        kind = self.clone_source_kind()
        src = dict(voice_name=None, ref_audio="", ref_text=None, x_vector_only=False, reuse_prompt=None)
        if kind == "saved":
            src["voice_name"] = self._selected_clone_voice()
        elif kind == "prompt":
            src["reuse_prompt"] = self.edt_vc_reuse.text().strip()
        else:
            xvec = self.chk_vc_xvec.isChecked()
            src.update(ref_audio=self.edt_vc_ref.text().strip(), x_vector_only=xvec,
                       ref_text=None if xvec else (self.txt_vc_ref_text.toPlainText().strip() or None))
        return dict(
            text=self.txt_vc_text.toPlainText().strip(), language=self.cmb_vc_lang.currentText(),
            min_words=self.spn_min.value(), max_words=self.spn_max.value(),
            seed=self.spn_vc_seed.value() if self.cmb_vc_seed_mode.currentText() == "Fixed" else None,
            temperature=self.spn_vc_temp.value(),
            save_prompt=self.edt_vc_save.text().strip() or None,
            description="" if kind == "saved" else self.txt_vc_instruct.toPlainText(),
            **src,
        )

    def generate_clone(self):
        if self.busy:
            self.status("Busy: a task is already running.")
            return   # the current Preview must survive a refused click
        if not self._validate_clone():
            return
        args = self.clone_args()
        on_done = self.finish_long_form("VoiceClone")
        made = {}
        self.vc_player.stop()   # like Create Voice: playback ends when a generation starts, the previous audio stays loaded
        self._set_clone_preview(CV_BUSY, "Getting ready…")   # a previous result stays on screen until the new one is ready

        def work(task):
            made["generator"] = self.generator(task)
            return made["generator"].clone(**args)

        def done(result):
            wavs = result[0]
            stop_pressed = bool(self.task and self.task.stop_event.is_set())
            # Stop only means "early" when chunks were really skipped; pressed after the last chunk the audio is complete.
            expected = getattr(made.get("generator"), "clone_chunks", None)
            early = stop_pressed and (len(wavs) < expected if expected is not None else True)
            try:
                path = on_done(result, early=early)   # writes the file, sets last_output and the footer status
            except Exception as exc:                  # nothing usable was written: a real failure
                self._show_clone_error(str(exc), task_errors.Failure.from_exception(exc))
                return
            audio = None
            if path:
                try:
                    samples, rate = load_wav(path)
                    audio = (path, samples, rate)
                except Exception as exc:              # the file exists; only the Preview can't show it
                    self._show_clone_saved_without_preview(path, early, exc)
                    return
            if audio is None:
                self._set_clone_preview(CV_STOPPED)
            elif early:
                self._set_clone_preview(CV_PARTIAL, f"Only the part generated so far was saved to {path}", audio)
            else:
                self._set_clone_preview(CV_READY, f"Saved to {path}", audio)

        self.run_task(work, done, on_fail=self._show_clone_error,
                      on_notify=lambda m: self._vc_pv_state == CV_BUSY and self.lbl_vc_pv_detail.setText(m))

    def save_clone_as_voice(self):
        """Build a clone prompt from reference audio and save it as a named voice."""
        if self.busy:
            self.status("Busy: a task is already running.")
            return
        name = self.edt_vc_new_name.text().strip()
        if not voice_library.slugify(name):
            self._clone_error("Enter a name for the voice.", self.edt_vc_new_name)
            return
        if not self._validate_clone(for_save=True):
            return
        overwrite = self._confirm_replace_voice(name)
        if overwrite is None:
            return
        ref_audio = self.edt_vc_ref.text().strip()
        ref_text = None if self.chk_vc_xvec.isChecked() else (self.txt_vc_ref_text.toPlainText().strip() or None)
        x_vector_only = self.chk_vc_xvec.isChecked()
        language = self.cmb_vc_lang.currentText()
        description = self.txt_vc_instruct.toPlainText().strip()
        # Exactly what clone() would pass for [tag] emotion samples from this form: the Fixed value, or 0 when Random
        # (tts_generator uses `seed or 0` for them). Recording anything else would describe a voice nobody heard.
        seed = self.spn_vc_seed.value() if self.cmb_vc_seed_mode.currentText() == "Fixed" else 0
        sample_text = ref_text or ""   # the exact transcript the user supplied; empty when there is none (X-vector only)

        def work(task):
            wav, sr = load_wav(ref_audio)
            prompt = self.generator(task).build_clone_prompt(
                ref_audio=(wav, sr), ref_text=ref_text, x_vector_only=x_vector_only)
            voice_library.save_voice(
                name, wav, sr, prompt, description, language, seed,
                model_ids=dict(MODEL_MAP), overwrite=overwrite,   # decided on the GUI thread, never here
                sample_text=sample_text)
            return name

        def done(saved_name):
            self.refresh_voice_lists()
            self._vc_saved_name = saved_name
            self.lbl_vc_saved_note.setText(f"Saved '{saved_name}'.")
            self.btn_vc_use_saved.setText(f"Use '{saved_name}'")
            self.row_vc_use.setVisible(True)
            self.status(f"Voice '{saved_name}' saved to voices/{voice_library.slugify(saved_name)}/")

        self.run_task(work, done, on_fail=lambda m: self._clone_error(m, self.edt_vc_new_name), stoppable=False)

    def _confirm_replace_voice(self, name, blocked=None):
        """None = cancelled/blocked, False = the name is free, True = the user agreed to replace it.
        Decided here, on the GUI thread; save_voice only replaces when told to. `blocked(message)` reports a
        folder that is not a voice (default: the Clone error line)."""
        path = voice_library.voice_dir(name)
        if not os.path.lexists(path):
            return False
        if not voice_library.voice_exists(name):
            message = (f"A folder named '{os.path.basename(path)}' already exists in the voices folder "
                       "but isn't a voice. Choose another name.")
            if blocked:
                blocked(message)
            else:
                self._clone_error(message, self.edt_vc_new_name)
            return None
        existing = name
        try:
            with open(os.path.join(path, "voice.json"), encoding="utf-8") as f:
                existing = json.load(f).get("name") or name
        except (OSError, ValueError):
            pass
        lost = voice_library.describe_voice_assets(name)
        detail = ("Replacing it deletes " + (", ".join(lost[:-1]) + " and " + lost[-1] if len(lost) > 1 else
                  (lost[0] if lost else "its files")) + ". They belong to the old recording and can't be kept "
                  "for the new one. The replacement is written first, so if saving fails the existing voice stays as it is.")
        if existing.strip().lower() != name.strip().lower():
            detail = (f"'{name}' and '{existing}' share the folder '{os.path.basename(path)}'. ") + detail
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Replace voice?")
        box.setText(f"A voice named '{existing}' already exists.")
        box.setInformativeText(detail)
        replace = box.addButton("Replace", QtWidgets.QMessageBox.DestructiveRole)
        keep = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        return True if box.clickedButton() is replace else None

    def use_saved_clone_voice(self):
        """Explicit switch to the voice that was just saved; saving never changes the source by itself."""
        name = self._vc_saved_name
        if not name:
            return
        self.refresh_voice_lists()
        idx = self.cmb_vc_voice.findText(name)
        self.btn_vc_src_saved.setChecked(True)
        if idx >= 0:
            self.cmb_vc_voice.setCurrentIndex(idx)
        self.row_vc_use.setVisible(False)
        self.cmb_vc_voice.setFocus()   # the button that had focus was just hidden

    # ------------------------------------------------------------------ Voices tab
    def _build_voices_tab(self):
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        page.setWidget(body)
        frame = QtWidgets.QHBoxLayout(body)
        frame.setContentsMargins(32, 32, 32, 24)
        content = QtWidgets.QWidget()
        content.setMaximumWidth(CONTENT_MAX_WIDTH)
        frame.addWidget(content, 100)
        frame.addStretch(1)
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)

        # Header: title on the left, the two ways to add a voice on the right (neutral, not primary).
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(page_header(
            "My Voices", "Saved voices are shared by Clone Voice and Audiobooks."), 1, QtCore.Qt.AlignTop)
        self.btn_voice_import = QtWidgets.QPushButton("Import voice…")
        self.btn_voice_import.setToolTip("Import a voice folder (must contain reference.wav + voice.json).")
        self.btn_voice_import.clicked.connect(self.import_voice)
        icons.apply(self.btn_voice_import, "download")
        self.btn_voice_create = QtWidgets.QPushButton("Create from audio…")
        self.btn_voice_create.setToolTip("Build a new voice from a WAV file + transcript.")
        self.btn_voice_create.clicked.connect(self.create_voice_from_audio)
        icons.apply(self.btn_voice_create, "wand-sparkles")
        for b in (self.btn_voice_import, self.btn_voice_create):
            head.addWidget(b, 0, QtCore.Qt.AlignTop)
        outer.addLayout(head)

        # List toolbar: the count is context for the list itself (the footer stays for action feedback).
        bar = QtWidgets.QHBoxLayout()
        self.lbl_voice_count = QtWidgets.QLabel("")
        set_role(self.lbl_voice_count, "muted")
        bar.addWidget(self.lbl_voice_count)
        bar.addStretch(1)
        self.btn_voice_refresh = QtWidgets.QPushButton("Refresh")
        set_role(self.btn_voice_refresh, "quiet")
        self.btn_voice_refresh.setToolTip("Re-scan the voices folder.")
        self.btn_voice_refresh.clicked.connect(self.refresh_voices_clicked)
        icons.apply(self.btn_voice_refresh, "refresh-cw")
        bar.addWidget(self.btn_voice_refresh)
        outer.addLayout(bar)

        # The list (native QListWidget, rows painted by a delegate) or, when empty, an explanation.
        self.lst_voices = voice_list.VoiceList()
        set_role(self.lst_voices, "voices")
        self.lst_voices.setItemDelegate(voice_list.VoiceRowDelegate(self.lst_voices))
        self.lst_voices.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.lst_voices.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lst_voices.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.lst_voices.setMouseTracking(True)
        self.lst_voices.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lst_voices.setUniformItemSizes(True)
        self.lst_voices.setMinimumHeight(180)
        self.lst_voices.setAccessibleName("Saved voices")
        self.lst_voices.setAccessibleDescription(
            "Use the arrow keys to move, Shift or Command with the arrow keys to select several. "
            "Return plays the reference audio.")
        outer.addWidget(self.lst_voices, 1)

        self.frm_voices_empty = QtWidgets.QFrame()
        set_role(self.frm_voices_empty, "callout")
        empty = QtWidgets.QVBoxLayout(self.frm_voices_empty)
        empty.setContentsMargins(24, 24, 24, 24)
        empty.setSpacing(8)
        empty_title = QtWidgets.QLabel("No saved voices yet")
        set_role(empty_title, "heading")
        self.lbl_voices_empty = QtWidgets.QLabel(
            "Create a voice from a description or a recording. Saved voices appear here.")
        self.lbl_voices_empty.setWordWrap(True)
        set_role(self.lbl_voices_empty, "muted")
        self.btn_voice_go_create = QtWidgets.QPushButton("Go to Create Voice")
        self.btn_voice_go_create.clicked.connect(self._go_to_create_voice)
        icons.apply(self.btn_voice_go_create, "wand-sparkles")
        empty.addWidget(empty_title)
        empty.addWidget(self.lbl_voices_empty)
        empty.addSpacing(8)
        empty.addWidget(self.btn_voice_go_create, 0, QtCore.Qt.AlignLeft)
        outer.addWidget(self.frm_voices_empty, 0, QtCore.Qt.AlignTop)

        # Selected-voice bar: names the selection and offers what applies to it.
        self.frm_voice_actions = QtWidgets.QFrame()
        set_role(self.frm_voice_actions, "callout")
        sel = QtWidgets.QHBoxLayout(self.frm_voice_actions)
        sel.setContentsMargins(16, 10, 16, 10)
        sel.setSpacing(8)
        self.lbl_voice_selection = QtWidgets.QLabel("")
        set_role(self.lbl_voice_selection, "heading")
        self.lbl_voice_selection.setMinimumWidth(0)
        self.lbl_voice_selection.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sel.addWidget(self.lbl_voice_selection, 1)
        self.btn_voice_play = QtWidgets.QPushButton("Play reference")
        self.btn_voice_play.clicked.connect(self.play_voice_tab)
        icons.apply(self.btn_voice_play, "play")
        self.btn_voice_rename = QtWidgets.QPushButton("Rename")
        self.btn_voice_rename.clicked.connect(self.rename_voice_tab)
        self.btn_voice_delete = QtWidgets.QPushButton("Delete")
        self.btn_voice_delete.clicked.connect(self.delete_voice_tab)
        act_transcript = QtGui.QAction("Reference transcript…", self)
        act_transcript.triggered.connect(self.edit_csm_transcript)
        act_f5 = QtGui.QAction("F5 reference…", self)
        act_f5.triggered.connect(self.edit_f5_reference)
        # Engine-specific references are rare and need exactly one voice, so they live in a menu.
        self.btn_voice_engine_refs = QtWidgets.QPushButton("Engine references")
        set_role(self.btn_voice_engine_refs, "menu")
        engine_menu = QtWidgets.QMenu(self.btn_voice_engine_refs)
        engine_menu.addAction(act_transcript)
        engine_menu.addAction(act_f5)
        self.btn_voice_engine_refs.setMenu(engine_menu)
        self.act_voice_transcript, self.act_voice_f5 = act_transcript, act_f5
        for b in (self.btn_voice_play, self.btn_voice_rename, self.btn_voice_delete, self.btn_voice_engine_refs):
            sel.addWidget(b)
        outer.addWidget(self.frm_voice_actions)
        outer.addStretch(0)   # takes the spare height only while the library is empty
        self._voices_outer = outer

        self.lst_voices.itemSelectionChanged.connect(self._refresh_voice_actions)
        self.lst_voices.itemActivated.connect(lambda item: self.play_voice_by_slug(item.data(voice_list.SLUG_ROLE)))
        self._fill_voices_list()  # silent: a library count is not the status of the Create Voice task
        return page

    def _go_to_create_voice(self):
        self.show_page(PAGE_CREATE)
        self.txt_vd_instruct.setFocus()   # the next thing the person does is describe a voice

    def refresh_voices_tab(self):
        """Re-read the library. Silent: callers report their own result in the footer."""
        self._fill_voices_list()

    def refresh_voices_clicked(self):
        self._fill_voices_list()
        self.status("Voice list refreshed.")

    def _fill_voices_list(self, select=None):
        """Rebuild the list; keep the current selection (or select `select`, a list of slugs)."""
        keep = set(select if select is not None else self.selected_voice_slugs())
        details = voice_library.list_voice_details()
        self.lst_voices.blockSignals(True)
        self.lst_voices.clear()
        for detail in details:
            item = voice_list.make_item(detail)
            self.lst_voices.addItem(item)
            if detail["slug"] in keep:
                item.setSelected(True)
        self.lst_voices.blockSignals(False)
        if keep and self.lst_voices.selectedItems():
            self.lst_voices.setCurrentItem(self.lst_voices.selectedItems()[0], QtCore.QItemSelectionModel.NoUpdate)
        self.lst_voices.setVisible(bool(details))
        self._voices_outer.setStretchFactor(self.lst_voices, 1 if details else 0)
        self._voices_outer.setStretch(self._voices_outer.count() - 1, 0 if details else 1)
        self.frm_voices_empty.setVisible(not details)
        self.frm_voice_actions.setVisible(bool(details))
        self.lbl_voice_count.setText(
            "" if not details else "1 voice" if len(details) == 1 else f"{len(details)} voices")
        self._refresh_voice_actions()
        return len(details)

    def _refresh_voice_actions(self):
        """Name the selection and enable only what applies: one voice for play, rename and
        references; one or more for delete. Handlers keep their own error fallback."""
        slugs = self.selected_voice_slugs()
        names = self.selected_voice_names()
        count = len(slugs)
        one = count == 1
        if count == 0:
            text, tip = "No voice selected", ""
            self.lbl_voice_selection.setProperty("role", "muted")
        else:
            text = names[0] if one else f"{count} voices selected"
            tip = names[0] if one else ", ".join(names)
            self.lbl_voice_selection.setProperty("role", "heading")
        self.lbl_voice_selection.style().unpolish(self.lbl_voice_selection)
        self.lbl_voice_selection.style().polish(self.lbl_voice_selection)
        metrics = self.lbl_voice_selection.fontMetrics()
        self.lbl_voice_selection.setText(metrics.elidedText(text, QtCore.Qt.ElideRight, 260))
        self.lbl_voice_selection.setToolTip(tip)
        self.lbl_voice_selection.setAccessibleName(text)

        need_one = "Select exactly one voice."
        for button, enabled_tip, accessible in (
                (self.btn_voice_play, "Play the selected voice's reference audio. "
                                      "Return on a row also plays it.", f"Play reference for {names[0] if one else ''}"),
                (self.btn_voice_rename, "Rename the selected voice.", f"Rename {names[0] if one else ''}"),
                (self.btn_voice_engine_refs,
                 "Add a reference transcript or F5 reference for the selected voice.", "")):
            button.setEnabled(one)
            button.setToolTip(enabled_tip if one else need_one)
            button.setAccessibleName(accessible.strip() if one else "")
        self.btn_voice_delete.setEnabled(count >= 1)
        self.btn_voice_delete.setToolTip(
            "Permanently delete the selected voice(s). You will be asked to confirm." if count
            else "Select one or more voices to delete.")
        self.btn_voice_delete.setAccessibleName(
            "" if not count else f"Delete {names[0]}" if one else f"Delete {count} voices")

    def selected_voice_slugs(self):
        return sorted({item.data(voice_list.SLUG_ROLE) for item in self.lst_voices.selectedItems()})

    def selected_voice_names(self):
        by_slug = {item.data(voice_list.SLUG_ROLE): item.data(voice_list.NAME_ROLE)
                   for item in self.lst_voices.selectedItems()}
        return [by_slug[s] for s in sorted(by_slug)]

    def selected_voice_slug(self):
        slugs = self.selected_voice_slugs()
        return slugs[0] if len(slugs) == 1 else None

    def rename_voice_tab(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice to rename.")
            return
        current = self.selected_voice_names()[0]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename voice", f"New name for '{current}':", text=current)
        if not ok or not name.strip():
            return
        try:
            voice_library.rename_voice(slug, name.strip())
            self._fill_voices_list(select=[voice_library.slugify(name.strip())])
            self.refresh_voice_lists()
            self.status(f"Renamed '{current}' to '{name.strip()}'.")
        except Exception as e:
            self.status(f"Rename failed: {e}")

    def delete_voice_tab(self):
        slugs = self.selected_voice_slugs()
        if not slugs:
            return
        names = self.selected_voice_names()
        shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        answer = QtWidgets.QMessageBox.question(
            self, "Delete voices?",
            f"Delete {len(slugs)} voice(s): {shown}? This cannot be undone.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for slug in slugs:
            voice_library.delete_voice(slug)
        self._fill_voices_list(select=[])
        self.refresh_voice_lists()
        self.status(f"Deleted {len(slugs)} voice(s).")

    def play_voice_tab(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice to play.")
            return
        self.play_voice_by_slug(slug)

    def edit_csm_transcript(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice.")
            return
        try:
            with open(os.path.join(voice_library.voice_dir(slug), "voice.json"), encoding="utf-8") as voice_file:
                config = json.load(voice_file)
            text, accepted = QtWidgets.QInputDialog.getMultiLineText(
                self, "Reference transcript",
                "Enter the exact words spoken in the entire reference recording:",
                config.get("csm_ref_text", config.get("ref_text", "")))
            if accepted:
                voice_library.set_csm_reference_text(slug, text)
                self.status(f"Reference transcript saved for '{slug}'.")
        except (OSError, ValueError) as exc:
            self.status(f"Error saving reference transcript: {exc}")

    def edit_f5_reference(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice.")
            return
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select F5 reference (12 seconds or less)", "", "WAV audio (*.wav)")
        if not source:
            return
        text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self, "F5 reference transcript", "Enter the exact words spoken in this recording:")
        if not accepted:
            return
        try:
            voice_library.set_f5_reference(slug, source, text)
            self.status(f"F5 reference saved for '{slug}'.")
        except (OSError, ValueError) as exc:
            self.status(f"Error saving F5 reference: {exc}")

    def play_voice_by_slug(self, slug):
        path = os.path.join(voice_library.voice_dir(slug), "reference.wav")
        if not os.path.isfile(path):
            self.status(f"Error: reference.wav not found for '{slug}'.")
            return
        self.play(path)

    def import_voice(self):
        src = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select voice folder to import", voice_library.VOICES_DIR)
        if not src:
            return
        cfg_path = os.path.join(src, "voice.json")
        wav_path = os.path.join(src, "reference.wav")
        if not os.path.isfile(cfg_path):
            self.status("Error: selected folder has no voice.json.")
            return
        if not os.path.isfile(wav_path):
            self.status("Error: selected folder has no reference.wav.")
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            self.status(f"Error reading voice.json: {e}")
            return
        name = cfg.get("name", os.path.basename(src))
        if not voice_library.slugify(name):
            self.status("Error: voice name is invalid.")
            return
        name = voice_library.unique_voice_name(name)
        dst = voice_library.voice_dir(name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        # Ensure voice.json name matches the new folder name
        cfg["name"] = name
        with open(os.path.join(dst, "voice.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        self.refresh_voices_tab()
        self.refresh_voice_lists()
        self.status(f"Imported voice '{name}'.")

    def create_voice_from_audio(self):
        """Build a new voice from a WAV file + transcript (no model generation needed for the prompt)."""
        wav_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select reference audio", "", audio_file_filter())
        if not wav_path:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New voice name", "Voice name:", text=os.path.splitext(os.path.basename(wav_path))[0])
        if not ok or not voice_library.slugify(name.strip()):
            self.status("Error: enter a valid voice name.")
            return
        name = voice_library.unique_voice_name(name.strip())
        language, ok2 = QtWidgets.QInputDialog.getItem(
            self, "Language", "Language:", LANGUAGES, 0, False)
        if not ok2:
            return
        description, ok3 = QtWidgets.QInputDialog.getMultiLineText(
            self, "Voice description", "Describe the voice (used for [emotion] tags):", "")
        if not ok3:
            return
        seed, ok4 = QtWidgets.QInputDialog.getInt(
            self, "Seed", "Seed (0 = random):", 0, 0, 2**31 - 1)
        if not ok4:
            return
        if seed == 0:
            seed = random.randint(0, 2**31 - 1)

        def work(task):
            wav, sr = load_wav(wav_path)
            prompt = self.generator(task).build_clone_prompt(
                ref_audio=(wav, sr), ref_text=None, x_vector_only=True)
            voice_library.save_voice(
                name, wav, sr, prompt, description, language, seed,
                model_ids=dict(MODEL_MAP), overwrite=False, sample_text="")   # X-vector only: no transcript
            return name

        def done(saved_name):
            self.refresh_voices_tab()
            self.refresh_voice_lists()
            self.status(f"Voice '{saved_name}' created from audio.")

        self.run_task(work, done, stoppable=False)

    @property
    def action_buttons(self):
        return (self.btn_vd_generate, self.btn_vd_save, self.btn_vd_regenerate, self.btn_vd_retry,
                self.btn_vc_generate, self.audiobook_tab.btn_render, self.audiobook_tab.btn_preview,
                *self.drama_tab.action_buttons)


# ------------------------------------------------------------------ Voice Manager dialog
class VoiceManagerDialog(QtWidgets.QDialog):
    """View, rename, and delete all saved voices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(400)

        lay = QtWidgets.QVBoxLayout(self)

        lay.addWidget(QtWidgets.QLabel("All saved voices (stored in voices/<name>/):"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Description", "Language", "Seed", "Created"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in (0, 2, 3, 4):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        lay.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_rename = QtWidgets.QPushButton("Rename selected")
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        for b in (btn_refresh, self.btn_rename, self.btn_delete, btn_close):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.refresh()

    def refresh(self):
        details = voice_library.list_voice_details()
        self.table.setRowCount(len(details))
        for r, d in enumerate(details):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(d["name"]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(d["description"]))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(d["language"]))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(d["seed"])))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(d["created"]))
            # store slug for operations
            self.table.item(r, 0).setData(QtCore.Qt.UserRole, d["slug"])

    def selected_slugs(self):
        slugs = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                slugs.append(item.data(QtCore.Qt.UserRole))
        return sorted(set(slugs))

    def rename_selected(self):
        slugs = self.selected_slugs()
        if len(slugs) != 1:
            QtWidgets.QMessageBox.information(
                self, "Rename voice", "Select exactly one voice to rename.")
            return
        slug = slugs[0]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename voice", f"New name for '{slug}':", text=slug)
        if not ok or not name.strip():
            return
        try:
            voice_library.rename_voice(slug, name.strip())
            self.refresh()
            if self.parent():
                self.parent().refresh_voice_lists()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename failed", str(e))

    def delete_selected(self):
        slugs = self.selected_slugs()
        if not slugs:
            return
        names = ", ".join(slugs)
        answer = QtWidgets.QMessageBox.question(
            self, "Delete voices?",
            f"Delete {len(slugs)} voice(s): {names}? This cannot be undone.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for slug in slugs:
            voice_library.delete_voice(slug)
        self.refresh()
        if self.parent():
            self.parent().refresh_voice_lists()


def configure_keyboard_navigation(app):
    """Let Tab reach every control on every platform.

    macOS only Tab-focuses text and list controls unless the system's Full Keyboard Access is on.
    Then the description box is the only Tab stop, Qt finds nowhere to move and inserts a tab
    character instead (a keyboard trap). The app promises full keyboard operation, so it opts in.
    """
    app.styleHints().setTabFocusBehavior(QtCore.Qt.TabFocusAllControls)


def main(api_enabled=True, api_host="127.0.0.1", api_port=8765):
    app = QtWidgets.QApplication(sys.argv)
    configure_keyboard_navigation(app)
    apply_theme(app)
    win = MainWindow()
    if api_enabled:
        from api_server import APIService, QwenAPIServer
        try:
            service = APIService(win.models, gui_busy=lambda: win.busy)
            api_server = QwenAPIServer(api_host, api_port, service)
            api_server.start()
        except OSError as exc:
            print(f"Could not start the API on {api_host}:{api_port}: {exc}. GUI will continue.")
        else:
            print(f"Qwen3-TTS API listening on {api_server.address}", flush=True)
            app.aboutToQuit.connect(api_server.stop)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
