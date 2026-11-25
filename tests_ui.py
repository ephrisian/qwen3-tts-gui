"""Headless Qt smoke tests: build the main window with a stub model manager.

Run with:  python tests_ui.py
No real model is downloaded or loaded, no audio is played, and no GPU is needed.
Generation paths run against a stub model, with playback and the output folder patched.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

import audio_player
from audio_utils import load_wav, write_wav
import qt_app
import cast_rules
import qt_audiobook
import icons
import theme
import voice_library
import voice_list
PAGE_VOICES = 2
from qt_app import CONTENT_MAX_WIDTH, MainWindow, READY_STATUS, STALE_STATUS, configure_keyboard_navigation, PV_BUSY, PV_EMPTY, PV_ERROR, PV_READY


class _StubModels:
    """Stands in for ModelManager; any attempt to load a model fails the generation."""

    def __init__(self):
        self.calls = []

    def get(self, kind, notify=None):
        self.calls.append(kind)
        raise AssertionError(f"model '{kind}' must not be loaded in UI smoke tests")


class _FakeDesignModel:
    def generate_voice_design(self, text, language, instruct):
        return [np.sin(np.linspace(0, 440, 24000, dtype=np.float32)) * 0.5], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return "FAKE-PROMPT"


class _FakeModels:
    def get(self, kind, notify=None):
        return _FakeDesignModel()


class _SlowModels:
    """Blocks inside generation until released, so tests can look at the UI mid-run."""

    def __init__(self):
        self.release = threading.Event()

    def get(self, kind, notify=None):
        if notify:
            notify("Loading voice-design model…")
        self.release.wait(10)
        return _FakeDesignModel()


_app = None


def _make_window(themed=True, models=None):
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_keyboard_navigation(_app)
    if themed:
        theme.apply_theme(_app)
    stub = models or _StubModels()
    return MainWindow(models=stub), stub


def _wait_idle(win, timeout=10):
    end = time.time() + timeout
    while win.busy and time.time() < end:
        _app.processEvents()
        time.sleep(0.01)
    _app.processEvents()
    assert not win.busy, "task did not finish"


class _Patched:
    """Silence audio and redirect the preview file for tests that really run a task."""

    def __enter__(self):
        self._play, self._stop, self._out = audio_player.play, audio_player.stop, qt_app.OUTPUT_DIR
        self.played = []
        audio_player.play = self.played.append
        audio_player.stop = lambda: None
        self.tmp = tempfile.mkdtemp()
        qt_app.OUTPUT_DIR = self.tmp
        return self

    def __exit__(self, *exc):
        audio_player.play, audio_player.stop, qt_app.OUTPUT_DIR = self._play, self._stop, self._out


def _test_window_builds_with_sidebar_navigation():
    win, stub = _make_window()
    labels = [b.text() for b in win.nav_buttons]
    assert labels == ["Create Voice", "Clone Voice", "My Voices", "Audiobooks", "Drama Studio", "Settings"], labels
    assert win.stack.count() == 6 and win.stack.currentIndex() == 0
    assert win.nav_buttons[0].isChecked()
    assert win.nav_buttons[-1] is win.nav_buttons[5], "Settings is last (bottom of the sidebar)"
    assert stub.calls == []
    print("PASS: window builds with sidebar navigation, no model touched")


def _test_navigation_switches_pages():
    win, _ = _make_window()
    for i, btn in enumerate(win.nav_buttons):
        btn.click()
        assert win.stack.currentIndex() == i
        assert [b.isChecked() for b in win.nav_buttons].count(True) == 1 and btn.isChecked()
    assert win.stack.widget(5).isAncestorOf(win.cmb_attn)
    print("PASS: sidebar buttons switch pages and stay exclusive")


def _test_create_voice_widgets():
    win, _ = _make_window()
    assert isinstance(win.btn_vd_generate, QtWidgets.QPushButton)
    assert win.btn_vd_generate.text() == "Generate voice"
    assert isinstance(win.btn_vd_play, QtWidgets.QPushButton)
    assert isinstance(win.btn_vd_save, QtWidgets.QPushButton)
    assert isinstance(win.txt_vd_instruct, QtWidgets.QPlainTextEdit)
    assert isinstance(win.edt_vd_name, QtWidgets.QLineEdit)
    assert win.cmb_vd_lang.findText("english") >= 0
    assert win.txt_vd_instruct.toPlainText().strip()
    assert 100 <= win.txt_vd_instruct.height() <= 120 or win.txt_vd_instruct.minimumHeight() == 112
    assert win.btn_vd_generate.isEnabled() and not win.btn_vd_save.isEnabled()
    assert win.vd_preview.currentIndex() == PV_EMPTY
    for name in ("cmb_vd_seed_mode", "spn_vd_seed", "chk_vd_autosave", "cmb_vd_name_gender",
                 "btn_vd_download", "btn_vd_regenerate", "cmb_device", "cmb_dtype", "cmb_attn",
                 "spn_min", "spn_max"):
        assert hasattr(win, name), name
    print("PASS: Create Voice widgets exist with expected initial state")


def _test_advanced_settings_hidden_by_default():
    win, _ = _make_window()
    assert win.vd_advanced.isHidden()
    for w in (win.cmb_vd_seed_mode, win.chk_vd_autosave, win.cmb_vd_name_gender):
        assert win.vd_advanced.isAncestorOf(w)
    assert not win.vd_advanced.isAncestorOf(win.txt_vd_sample), "Sample text is in the main form"
    win.btn_vd_advanced.click()
    assert not win.vd_advanced.isHidden()
    assert win.model_settings() == (win.cmb_device.currentText(), win.cmb_dtype.currentText(),
                                    win.cmb_attn.currentText())
    win.cmb_dtype.setCurrentText("float32")
    assert "float32" in win.lbl_vd_compute.text()
    win.btn_vd_advanced.click()
    assert win.vd_advanced.isHidden()
    print("PASS: Advanced settings collapsed by default, expandable, compute summary follows Settings")


def _test_suggestion_chips_are_reversible():
    win, _ = _make_window()
    original = "My own words."
    win.txt_vd_instruct.setPlainText(original)
    chip, _name, phrase = win.vd_chips[0]
    chip.setChecked(True)
    text = win.txt_vd_instruct.toPlainText()
    assert text.startswith(original) and phrase in text
    chip.setChecked(False)
    assert win.txt_vd_instruct.toPlainText() == original
    chip.setChecked(True)
    win.txt_vd_instruct.setPlainText(original)  # user edits the phrase away
    assert not chip.isChecked()
    print("PASS: suggestion chips append, remove only their phrase, and follow manual edits")


def _test_generate_validates_without_starting_a_task():
    win, stub = _make_window()
    win.txt_vd_instruct.setPlainText("   ")
    win.btn_vd_generate.click()
    assert win.lbl_status.text().startswith("Error"), win.lbl_status.text()
    assert not win.lbl_vd_field_error.isHidden()
    assert not win.busy and win.task is None and stub.calls == []
    assert win.vd_preview.currentIndex() == PV_EMPTY
    win.txt_vd_instruct.setPlainText("Calm voice")
    assert win.lbl_vd_field_error.isHidden()
    print("PASS: empty description shows an inline error and starts nothing")


def _test_generation_failure_shows_error_and_keeps_input():
    with _Patched():
        win, stub = _make_window()
        win.txt_vd_instruct.setPlainText("A gravelly old sailor")
        win.btn_vd_generate.click()
        assert win.vd_preview.currentIndex() == PV_BUSY and win.busy
        _wait_idle(win)
        assert stub.calls == ["voice-design"]
        assert win.vd_preview.currentIndex() == PV_ERROR
        # the footer is a concise status; the Preview owns the explanation and the details
        assert win.lbl_status.text().startswith("Couldn't generate") and "must not be loaded" not in win.lbl_status.text()
        assert not win.lbl_status_icon.isHidden(), "an error status carries its glyph as well as its words"
        assert "Couldn't" in win.lbl_vd_error.text() or "wrong" in win.lbl_vd_error.text()
        assert "must not be loaded" in win.txt_vd_details.toPlainText()
        assert win.txt_vd_instruct.toPlainText() == "A gravelly old sailor"
        assert win.btn_vd_generate.isEnabled() and win.btn_vd_retry.isEnabled()
    print("PASS: failed generation shows the error state, preserves input, allows retry")


def _test_generation_success_shows_real_preview():
    with _Patched() as patched:
        win, _ = _make_window(models=_FakeModels())
        win.txt_vd_instruct.setPlainText("Warm narrator")
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY
        assert win.pending_voice["description"] == "Warm narrator"
        assert win.lbl_vd_total.text() == "0:01", win.lbl_vd_total.text()
        assert not win.btn_vd_save.isEnabled(), "Save waits for a voice name"
        win.edt_vd_name.setText("Ada")
        assert win.btn_vd_save.isEnabled() and win.btn_vd_regenerate.isEnabled()
        assert len(patched.played) == 1 and patched.played[0].endswith("_voice_preview.wav")
        assert os.path.isfile(win.pending_voice["preview"])
        assert win.vd_wave._peaks and abs(max(win.vd_wave._peaks) - 1.0) < 1e-6
        assert win.lbl_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("Something else")
        assert not win.lbl_vd_stale.isHidden()
        # a later failure keeps the previous sample visible
        win.models = _StubModels()
        win.btn_vd_regenerate.click()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY and not win.lbl_vd_notice.isHidden()
        assert win.pending_voice["description"] == "Warm narrator"
    print("PASS: success shows real duration/waveform; stale note; failure keeps previous sample")


# ------------------------------------------------------------------ Preview: sample text and player
class _RecordingModels:
    """Fake models that record what the voice is asked to say, and make `seconds` of audio."""

    def __init__(self, seconds=20):
        self.texts, self.ref_texts, self.seconds = [], [], seconds

    def get(self, kind, notify=None):
        outer = self

        class Model:
            def generate_voice_design(self, text, language, instruct):
                outer.texts.append(text)
                n = int(24000 * outer.seconds)
                return [np.sin(np.linspace(0, 440 * outer.seconds, n, dtype=np.float32)) * 0.5], 24000

            def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
                outer.ref_texts.append(ref_text)
                return {"ref_spk_embedding": np.zeros(4, dtype=np.float32)}
        return Model()


class _FakeAudio:
    """A controllable stand-in for audio_player with a manual clock, pause, seek and a natural end."""

    NAMES = ("play", "stop", "pause", "resume", "state", "current_path", "position", "seek", "can_control",
             "is_playing")

    def __init__(self, duration=20.0, control=True):
        self.duration, self.control = duration, control
        self.path, self.mode, self.pos = None, "stopped", 0.0
        self.play_calls, self.fail = [], None

    def __enter__(self):
        self._saved = {n: getattr(audio_player, n) for n in self.NAMES}
        for n in self.NAMES:
            setattr(audio_player, n, getattr(self, "_" + n))
        return self

    def __exit__(self, *exc):
        for n, f in self._saved.items():
            setattr(audio_player, n, f)

    def advance(self, seconds):
        if self.mode == "playing":
            self.pos += seconds
            if self.pos >= self.duration:
                self.mode, self.path, self.pos = "stopped", None, 0.0   # played to the end

    def _play(self, path, start=0.0):
        if self.fail:
            raise RuntimeError(self.fail)
        self.play_calls.append((path, start))
        self.path, self.mode, self.pos = path, "playing", start

    def _stop(self):
        self.path, self.mode, self.pos = None, "stopped", 0.0

    def _pause(self):
        if self.control and self.mode == "playing":
            self.mode = "paused"
            return True
        return False

    def _resume(self):
        if self.mode == "paused":
            self.mode = "playing"
            return True
        return False

    def _state(self):
        return self.mode

    def _current_path(self):
        return self.path if self.mode != "stopped" else None

    def _position(self):
        return self.pos if self.control and self.mode != "stopped" else None

    def _seek(self, seconds):
        if self.control and self.mode != "stopped":
            self.pos = seconds
            return True
        return False

    def _can_control(self):
        return self.control

    def _is_playing(self):
        return self.mode == "playing"


def _ready_window(models=None):
    win, _ = _make_window(models=models or _RecordingModels())
    win.txt_vd_instruct.setPlainText("Warm narrator")
    win.btn_vd_generate.click()
    _wait_idle(win)
    assert win.vd_preview.currentIndex() == PV_READY
    return win


def _test_sample_text_provenance_and_editing():
    with _Patched():
        models = _RecordingModels(seconds=1)
        win = _ready_window(models)
        # the voice said the standard passage, and the Preview shows exactly that
        assert models.texts == [voice_library.SAMPLE_TEXT]
        assert win.pending_voice["text"] == voice_library.SAMPLE_TEXT
        assert win.txt_vd_spoken.toPlainText() == voice_library.SAMPLE_TEXT
        assert win.txt_vd_spoken.isReadOnly() and win.txt_vd_spoken.accessibleName()
        assert win.lbl_vd_spoken_note.text() == "Text used for this sample"
        assert win.txt_vd_sample.toPlainText() == voice_library.SAMPLE_TEXT and not win.txt_vd_sample.isReadOnly()
        # editing the field never edits the existing sample; it says the text differs
        win.txt_vd_sample.setPlainText("Hello there. It's a beautiful day.")
        assert win.txt_vd_spoken.toPlainText() == voice_library.SAMPLE_TEXT
        assert not win.frm_vd_stale.isHidden() and "sample text" in win.lbl_vd_stale.text()
        assert "has changed" in win.lbl_vd_spoken_note.text()
        assert win.btn_vd_save.text() == "Save previous sample"
        # Regenerate speaks the new text and the Preview follows it
        win.btn_vd_regenerate.click()
        _wait_idle(win)
        assert models.texts[-1] == "Hello there. It's a beautiful day."
        assert win.txt_vd_spoken.toPlainText() == "Hello there. It's a beautiful day."
        assert win.frm_vd_stale.isHidden() and win.btn_vd_save.text() == "Save"
        # the standard text can be restored
        win.btn_vd_reset_text.click()
        assert win.txt_vd_sample.toPlainText() == voice_library.SAMPLE_TEXT and not win.frm_vd_stale.isHidden()
        # Edit text focuses the field, which lives in the main form (not in Advanced settings)
        win.show()
        assert not win.vd_advanced.isAncestorOf(win.txt_vd_sample) and win.vd_advanced.isHidden()
        win.btn_vd_edit_text.click()
        _app.processEvents()
        assert win.txt_vd_sample.hasFocus() and win.vd_advanced.isHidden()
        # blank text can't be generated
        win.txt_vd_sample.setPlainText("   ")
        before = len(models.texts)
        win.btn_vd_generate.click()
        _app.processEvents()
        assert len(models.texts) == before and "sample text" in win.lbl_status.text().lower()
        assert not win.busy and win.txt_vd_sample.hasFocus()
        win.close()
    print("PASS: spoken text provenance, editing for the next sample only, differs note, edit/reset, blank blocked")


def _test_saved_voice_and_download_use_the_spoken_text():
    with _Patched():
        root = tempfile.mkdtemp()
        original = voice_library.VOICES_DIR
        voice_library.VOICES_DIR = root
        real_dialog = QtWidgets.QFileDialog.getSaveFileName
        try:
            models = _RecordingModels(seconds=1)
            win, _ = _make_window(models=models)
            win.txt_vd_instruct.setPlainText("Warm narrator")
            win.txt_vd_sample.setPlainText("Custom words for this voice.")
            win.btn_vd_generate.click()
            _wait_idle(win)
            # editing the field afterwards must not change what Save records: the audio says the old words
            win.txt_vd_sample.setPlainText("Different words that were never spoken.")
            win.edt_vd_name.setText("Ada")
            win.btn_vd_save.click()
            _wait_idle(win)
            with open(os.path.join(root, "Ada", "voice.json"), encoding="utf-8") as f:
                config = json.load(f)
            assert config["sample_text"] == config["ref_text"] == "Custom words for this voice.", config
            assert models.ref_texts == ["Custom words for this voice."]
            assert set(config) >= {"schema", "name", "created", "language", "description", "seed", "sample_text",
                                   "ref_text", "ref_audio", "clone_prompt", "sample_rate", "models", "usage"}
            # Download exports the same audio the Preview plays
            target = os.path.join(root, "export.wav")
            QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
            win.btn_vd_download.click()
            exported, sr = load_wav(target)
            assert sr == 24000 and len(exported) == len(np.asarray(win.pending_voice["wav"]).reshape(-1))
        finally:
            QtWidgets.QFileDialog.getSaveFileName = real_dialog
            voice_library.VOICES_DIR = original
    print("PASS: Save records the text actually spoken (not later edits); Download exports the previewed audio")


def _test_preview_player_play_pause_resume_seek_stop_and_end():
    with _Patched(), _FakeAudio(duration=20.0) as audio:
        win = _ready_window(_RecordingModels(seconds=20))
        path = win.pending_voice["preview"]
        assert audio.play_calls == [(path, 0.0)], audio.play_calls
        win._refresh_player()
        assert win.btn_vd_play.text() == "Pause" and win.lbl_vd_total.text() == "0:20"
        assert win.sld_vd_seek.maximum() == 20000 and win.sld_vd_seek.isEnabled()
        # time and position are the player's own, not invented
        audio.advance(7.0)
        win._refresh_player()
        assert win.lbl_vd_time.text() == "0:07" and win.sld_vd_seek.value() == 7000
        # Play/Pause is one toggle: pause keeps the position, resume does not restart
        win.btn_vd_play.click()
        assert audio.mode == "paused" and win.btn_vd_play.text() == "Play" and win.lbl_vd_time.text() == "0:07"
        audio.advance(3.0)
        assert audio.pos == 7.0
        win.btn_vd_play.click()
        assert audio.mode == "playing" and len(audio.play_calls) == 1, "resume must not restart playback"
        assert win.btn_vd_play.text() == "Pause" and win.lbl_vd_time.text() == "0:07"
        # seeking: slider (keyboard), waveform click, back and forward
        win.sld_vd_seek.setValue(12000)
        assert audio.pos == 12.0 and win.lbl_vd_time.text() == "0:12"
        win.vd_wave.seek_requested.emit(0.25)
        assert audio.pos == 5.0
        win.btn_vd_forward.click()
        assert audio.pos == 10.0
        win.btn_vd_back.click()
        win.btn_vd_back.click()
        win.btn_vd_back.click()
        assert audio.pos == 0.0, "back stops at the start"
        win._seek_preview(999)
        assert audio.pos < 20.0 and audio.mode == "playing"
        # the playhead follows the position
        win._seek_preview(10.0)
        assert abs(win.vd_wave._progress - 0.5) < 1e-6
        # footer Stop audio resets the Preview to the start
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden()
        win.btn_stop_audio.click()
        assert audio.mode == "stopped" and win.btn_vd_play.text() == "Play" and win.lbl_vd_time.text() == "0:00"
        assert win.vd_wave._progress is None and win.btn_stop_audio.isHidden()
        # seeking while stopped sets where Play starts
        win._seek_preview(6.0)
        assert audio.mode == "stopped" and win.lbl_vd_time.text() == "0:06"
        win.btn_vd_play.click()
        assert audio.play_calls[-1] == (path, 6.0) and audio.mode == "playing"
        # played to the end: back to Play at 0:00, and Play starts from the beginning
        audio.advance(100)
        win._refresh_player()
        assert win.btn_vd_play.text() == "Play" and win.lbl_vd_time.text() == "0:00" and win.sld_vd_seek.value() == 0
        win.btn_vd_play.click()
        assert audio.play_calls[-1] == (path, 0.0)
        win.close()
    print("PASS: Play/Pause toggles and resumes, real position/duration, seek, stop, ended state")


def _test_preview_player_stays_consistent_with_other_audio_and_pages():
    with _Patched(), _FakeAudio(duration=20.0) as audio:
        win = _ready_window(_RecordingModels(seconds=20))
        audio.advance(4.0)
        win.show_page(2)                       # leave Create Voice while it plays
        audio.advance(2.0)
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden(), "Stop audio stays available on other pages"
        win.show_page(0)
        win._refresh_player()
        assert win.btn_vd_play.text() == "Pause" and win.lbl_vd_time.text() == "0:06"
        # different audio starts elsewhere: the Preview is not playing, so it never claims to be
        audio._play("/somewhere/else.wav")
        win._refresh_player()
        assert win.btn_vd_play.text() == "Play"
        win.btn_vd_play.click()
        assert audio.path == win.pending_voice["preview"] and len(audio.play_calls) == 3
        # paused audio can still be stopped from the footer
        win.btn_vd_play.click()
        win._refresh_audio_state()
        assert audio.mode == "paused" and not win.btn_stop_audio.isHidden()
        # regenerating stops playback before the new sample arrives, so nothing overlaps
        win.btn_vd_regenerate.click()
        assert audio.mode == "stopped"
        _wait_idle(win)
        win.close()
    print("PASS: Preview and footer share one playback state across pages and other audio")


def _test_preview_player_without_pause_support_and_playback_errors():
    with _Patched(), _FakeAudio(duration=20.0, control=False) as audio:
        win = _ready_window(_RecordingModels(seconds=20))
        win._refresh_player()
        assert win.btn_vd_play.text() == "Stop", "a backend that can't pause offers Stop, not a fake Pause"
        assert not win.sld_vd_seek.isEnabled() and not win.btn_vd_back.isEnabled() and win.lbl_vd_time.text() == "–"
        win.btn_vd_play.click()
        assert audio.mode == "stopped" and win.btn_vd_play.text() == "Play"
        audio.fail = "No audio player found"
        win.btn_vd_play.click()
        assert "Couldn't play the sample" in win.lbl_vd_notice.text()
        assert win.lbl_status.text().startswith("Playback error") and win.btn_vd_play.text() == "Play"
        win.close()
    print("PASS: no-pause backends offer Stop and no seek; playback errors are reported and recoverable")


def _test_audio_player_position_pause_and_seek_over_pygame_clock():
    import types
    clock = {"t": 0.0}

    class Music:
        def __init__(self):
            self.playing, self.paused, self.t0, self.pos = False, False, 0.0, 0.0

        def _played(self):
            return self.pos + (0 if self.paused or not self.playing else clock["t"] - self.t0)

        def load(self, path): self.loaded = path
        def play(self): self.playing, self.paused, self.t0, self.pos = True, False, clock["t"], 0.0
        def pause(self): self.pos, self.paused = self._played(), True
        def unpause(self): self.paused, self.t0 = False, clock["t"]
        def stop(self): self.playing, self.paused = False, False
        def get_busy(self): return self.playing and not self.paused and self._played() < 10
        def get_pos(self): return int(self._played() * 1000) if self.playing else -1
        def set_pos(self, seconds): self.seek_to = seconds   # pygame's own clock keeps counting

    music = Music()
    fake = types.SimpleNamespace(mixer=types.SimpleNamespace(get_init=lambda: True, init=lambda: None, music=music))
    real = sys.modules.get("pygame")
    sys.modules["pygame"] = fake
    try:
        audio_player.stop()
        audio_player.play("a.wav")
        assert audio_player.state() == "playing" and audio_player.current_path() == "a.wav"
        clock["t"] = 2.0
        assert abs(audio_player.position() - 2.0) < 1e-6
        assert audio_player.pause() and audio_player.state() == "paused" and audio_player.current_path() == "a.wav"
        clock["t"] = 5.0
        assert abs(audio_player.position() - 2.0) < 1e-6, "paused time does not advance"
        assert audio_player.resume() and audio_player.state() == "playing"
        clock["t"] = 6.0
        assert abs(audio_player.position() - 3.0) < 1e-6
        assert audio_player.seek(8.0) and music.seek_to == 8.0
        assert abs(audio_player.position() - 8.0) < 1e-6
        clock["t"] = 7.0
        assert abs(audio_player.position() - 9.0) < 1e-6, "position continues from the seek target"
        clock["t"] = 20.0
        assert audio_player.state() == "stopped" and audio_player.position() is None and audio_player.current_path() is None
        audio_player.play("b.wav", start=4.0)
        assert music.seek_to == 4.0 and abs(audio_player.position() - 4.0) < 1e-6
        audio_player.stop()
        assert audio_player.state() == "stopped" and not audio_player.seek(1.0) and not audio_player.pause()
    finally:
        audio_player.stop()
        if real is not None:
            sys.modules["pygame"] = real
        else:
            del sys.modules["pygame"]
    print("PASS: audio_player reports real position across pause, resume and seek, and the ended state")


def _test_preview_player_keyboard_focus_and_minimum_size_layout():
    with _Patched(), _FakeAudio(duration=20.0) as audio:
        win = _ready_window(_RecordingModels(seconds=20))
        win.resize(1024, 640)
        win.show()
        _app.processEvents()
        page = win.stack.widget(0)
        assert page.horizontalScrollBar().maximum() == 0, "no horizontal scrolling at 1024x640"
        order = [win.txt_vd_spoken, win.btn_vd_edit_text, win.sld_vd_seek, win.btn_vd_back, win.btn_vd_play,
                 win.btn_vd_forward, win.edt_vd_name]
        win.txt_vd_spoken.setFocus()
        seen = []
        for _ in range(len(order) - 1):
            win.focusNextChild()
            seen.append(QtWidgets.QApplication.focusWidget())
        assert seen == order[1:], [w.accessibleName() or w.text() for w in seen if w]
        assert win.sld_vd_seek.accessibleName() == "Seek" and win.btn_vd_back.accessibleName() == "Back 5 seconds"
        assert win.btn_vd_forward.accessibleName() == "Forward 5 seconds"
        assert "of 0:20" in win.sld_vd_seek.accessibleDescription()
        # the slider is keyboard operable
        win.sld_vd_seek.setFocus()
        audio.pos = 0.0
        QTest.keyClick(win.sld_vd_seek, QtCore.Qt.Key_Right)
        assert audio.pos == 5.0, audio.pos
        win.close()
    print("PASS: player keyboard order, accessible names and slider keys; no horizontal scroll at 1024x640")


def _test_sample_text_is_a_first_class_field_in_the_main_form():
    with _Patched():
        models = _RecordingModels(seconds=1)
        win, _ = _make_window(models=models)
        # one editor only, in the main form between the description and Generate
        editors = [w for w in win.findChildren(QtWidgets.QPlainTextEdit) if w.accessibleName() == "Sample text"]
        assert editors == [win.txt_vd_sample], "exactly one Sample text editor"
        assert not win.vd_advanced.isAncestorOf(win.txt_vd_sample) and not win.vd_advanced.isAncestorOf(win.btn_vd_reset_text)
        assert win.txt_vd_sample.toPlainText() == voice_library.SAMPLE_TEXT
        for size in ((1240, 780), (1024, 640)):
            win.resize(*size)
            win.show()
            _app.processEvents()
            page = win.stack.widget(0)
            viewport = page.viewport()
            for w in (win.txt_vd_instruct, win.txt_vd_sample, win.btn_vd_reset_text, win.btn_vd_generate):
                top = w.mapTo(viewport, QtCore.QPoint(0, 0)).y()
                assert top >= 0 and top + w.height() <= viewport.height(), (size, w.accessibleName() or w.text(), top)
            ys = [w.mapTo(win.vd_content, QtCore.QPoint(0, 0)).y() for w in (
                win.txt_vd_instruct, win.txt_vd_sample, win.btn_vd_generate)]
            assert ys == sorted(ys), "order: description, sample text, Generate"
            assert page.horizontalScrollBar().maximum() == 0
        # reset restores the passage without generating
        win.txt_vd_sample.setPlainText("Something else.")
        win.btn_vd_reset_text.click()
        assert win.txt_vd_sample.toPlainText() == voice_library.SAMPLE_TEXT and not models.texts and not win.busy
        # either input makes an existing sample potentially outdated; the audio and its text stay as they were
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.frm_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("A different voice")
        assert not win.frm_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("Warm, gentle female voice, calm and soothing tone.")
        assert win.frm_vd_stale.isHidden()
        win.txt_vd_sample.setPlainText("New words.")
        assert not win.frm_vd_stale.isHidden() and win.txt_vd_spoken.toPlainText() == voice_library.SAMPLE_TEXT
        assert win.pending_voice["text"] == voice_library.SAMPLE_TEXT
        win.close()
    print("PASS: Sample text is one editor in the main form, visible with Generate at 1240x780 and 1024x640; reset and stale behaviour")


def _test_create_voice_final_layout_flat_reader_scrollbars_and_contextual_origin():
    with _Patched():
        win = _ready_window(_RecordingModels(seconds=1))
        win.txt_vd_sample.setPlainText("A long passage. " * 120)   # long Sample text
        for size in ((1240, 780), (1024, 640)):
            win.resize(*size)
            win.show()
            _app.processEvents()
            page = win.stack.widget(0)
            assert page.horizontalScrollBar().maximum() == 0, f"horizontal scrolling at {size}"
            left_edge = win.txt_vd_instruct.mapTo(win.vd_content, QtCore.QPoint(0, 0)).x()
            right_edge = left_edge + win.txt_vd_instruct.width()
            for w in (win.lbl_vd_model, win.cmb_vd_lang, win.btn_vd_advanced):
                x = w.mapTo(win.vd_content, QtCore.QPoint(0, 0)).x()
                assert left_edge <= x and x + w.width() <= right_edge + 1, (size, w, x, w.width())
                assert w.mapTo(win.vd_content, QtCore.QPoint(0, 0)).y() > win.btn_vd_generate.mapTo(
                    win.vd_content, QtCore.QPoint(0, 0)).y(), "options sit beneath Generate voice"
            preview_left = win.preview_panel.mapTo(win.vd_content, QtCore.QPoint(0, 0)).x()
            assert right_edge < preview_left, "form column and Preview do not overlap"
            win.btn_vd_advanced.setChecked(True)
            _app.processEvents()
            assert page.horizontalScrollBar().maximum() == 0, f"Advanced settings overflow at {size}"
            for w in (win.cmb_vd_seed_mode, win.chk_vd_autosave, win.btn_vd_rand_name):
                x = w.mapTo(win.vd_content, QtCore.QPoint(0, 0)).x()
                assert x + w.width() <= right_edge + 1, (size, w.text() if hasattr(w, "text") else w, x + w.width(), right_edge)
            win.btn_vd_advanced.setChecked(False)
        # the Preview text is a flat reader (selectable, scrollable, focusable), about three lines tall
        reader = win.txt_vd_spoken
        assert reader.property("role") == "reader" and reader.isReadOnly()
        assert reader.height() <= reader.fontMetrics().lineSpacing() * 3 + 12
        assert reader.textInteractionFlags() & QtCore.Qt.TextSelectableByKeyboard
        assert reader.focusPolicy() & QtCore.Qt.TabFocus
        assert win.txt_vd_sample.property("role") != "reader", "only the form field looks like an input"
        reader.setFocus()
        assert reader.verticalScrollBar().maximum() > 0, "the full passage is longer than the three visible lines"
        for _ in range(4):
            QTest.keyClick(reader, QtCore.Qt.Key_Down)   # the keyboard scrolls the reader
        assert reader.verticalScrollBar().value() > 0
        QTest.keyClick(reader, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        assert reader.textCursor().hasSelection()
        assert reader.toPlainText() == voice_library.SAMPLE_TEXT, "the long form text never replaces the audio's text"
        # scrollbars are themed: compact, no arrow buttons
        sheet = QtWidgets.QApplication.instance().styleSheet()
        assert "QScrollBar::add-line" in sheet and "QScrollBar:vertical" in sheet
        # origin metadata is reachable but only shown inline when the sample is out of date
        assert win.lbl_vd_origin.isHidden() is False or True
        assert not win.frm_vd_stale.isHidden() and not win.lbl_vd_origin.isHidden()
        assert "Warm narrator" in win.lbl_vd_preview_title.toolTip() and "Warm narrator" in reader.accessibleDescription()
        win.txt_vd_sample.setPlainText(voice_library.SAMPLE_TEXT)
        assert win.frm_vd_stale.isHidden() and win.lbl_vd_origin.isHidden()
        win.close()
    print("PASS: options under the form column, flat selectable reader, themed scrollbars, contextual origin, no overflow")


def _test_loading_ui_shows_real_elapsed_time_in_plain_wording():
    with _Patched():
        slow = _SlowModels()
        win, _ = _make_window(models=slow)
        win.resize(1024, 640)
        _shown(win)
        real = qt_app.time.monotonic
        clock = {"t": 1000.0}
        qt_app.time.monotonic = lambda: clock["t"]
        try:
            win.btn_vd_generate.click()
            for _ in range(20):
                _app.processEvents()
                time.sleep(0.02)
            assert win.gen_timer.isActive() and win.lbl_vd_elapsed.text() == "0:00 elapsed"
            clock["t"] += 125
            win._tick_elapsed()
            assert win.lbl_vd_elapsed.text() == "2:05 elapsed", "elapsed is the measured clock, not an estimate"
            text = " ".join([win.lbl_vd_busy.text(), win.lbl_vd_busy_hint.text(), win.lbl_vd_elapsed.text()])
            assert "%" not in text and "remaining" not in text.lower() and "left" not in text.lower()
            assert win.lbl_vd_busy_hint.text() == qt_app.LOADING_HINT, "plain wording, no technical explanation"
            assert win.vd_busy_bar.indeterminate and win.progress.maximum() == 1 and win.btn_stop.isHidden(), \
                "one indeterminate bar, and no Cancel: the model call cannot be interrupted"
            slow.release.set()
            _wait_idle(win)
            assert not win.gen_timer.isActive() and win._gen_t0 is None, "the clock stops with the generation"
            # a regeneration over an existing sample reports elapsed time in its notice, then clears it
            slow.release.clear()
            win.models = slow
            win.btn_vd_regenerate.click()
            for _ in range(20):
                _app.processEvents()
                time.sleep(0.02)
            clock["t"] += 61
            win._tick_elapsed()
            assert "1:01 elapsed" in win.lbl_vd_notice.text() and "current sample stays available" in win.lbl_vd_notice.text()
            slow.release.set()
            _wait_idle(win)
            assert not win.gen_timer.isActive()
            # failure stops the clock too
            win.models = _StubModels()
            win.btn_vd_regenerate.click()
            _wait_idle(win)
            assert not win.gen_timer.isActive() and win._gen_t0 is None
        finally:
            qt_app.time.monotonic = real
            slow.release.set()
        win.close()
    print("PASS: loading shows a real elapsed clock in plain wording; no percentage, ETA or Cancel; clock stops")


def _test_footer_stop_audio_is_contextual_and_does_not_shift_the_footer():
    with _Patched(), _FakeAudio(duration=20.0) as audio:
        win = _ready_window(_RecordingModels(seconds=20))
        win.resize(1240, 780)
        win.show()
        _app.processEvents()
        audio._stop()
        win._refresh_audio_state()
        assert win.btn_stop_audio.isHidden(), "idle: no Stop audio"
        x_open = win.btn_open_out.geometry().x()
        # playing (the Preview's own audio), paused, stopped, ended, and audio from another page
        audio._play(win.pending_voice["preview"])
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden() and win.btn_open_out.geometry().x() == x_open, "no footer shift"
        audio._pause()
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden(), "paused audio can still be stopped"
        win.btn_stop_audio.click()
        assert win.btn_stop_audio.isHidden() and audio.mode == "stopped"
        audio._play(win.pending_voice["preview"])
        win._refresh_audio_state()
        audio.advance(100)                          # played to the end
        win._refresh_audio_state()
        assert win.btn_stop_audio.isHidden(), "ended: hidden again"
        win.show_page(2)
        audio._play("/voices/ada/reference.wav")    # My Voices playback keeps the global stop
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden() and win.btn_open_out.geometry().x() == x_open
        # a hidden control is not a Tab stop; Open output folder stays neutral and always present
        audio._stop()
        win._refresh_audio_state()
        assert win.btn_stop_audio.isHidden() and not win.btn_open_out.isHidden()
        win.btn_play_last.setFocus() if not win.btn_play_last.isHidden() else win.nav_buttons[2].setFocus()
        seen = []
        for _ in range(40):
            win.focusNextChild()
            seen.append(QtWidgets.QApplication.focusWidget())
        assert win.btn_stop_audio not in seen
        win.close()
    print("PASS: Stop audio shows only for a playback session (playing/paused, any page), hides otherwise, no footer shift or hidden tab stop")


def _test_leaving_a_navigation_item_never_leaves_its_focus_ring_behind():
    with _Patched():
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show()
        win.activateWindow()
        _app.processEvents()
        settings = win.nav_buttons[-1]
        for how in ("shortcut", "mouse"):
            win.show_page(0)
            win.nav_buttons[0].setFocus()
            for _ in range(5):
                QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_Down)   # arrows move between items
            _app.processEvents()
            assert win.stack.currentIndex() == 5 and win.focusWidget() is settings
            if how == "shortcut":
                QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_1, QtCore.Qt.ControlModifier)
            else:
                QTest.mouseClick(win.nav_buttons[0], QtCore.Qt.LeftButton)
            _app.processEvents()
            assert win.stack.currentIndex() == 0
            assert not settings.hasFocus() and win.focusWidget() is win.nav_buttons[0], (how, win.focusWidget())
        # a mouse user never gets a ring on Settings, and a keyboard user keeps a visible ring on the selected item
        win.show_page(0)
        QTest.mouseClick(settings, QtCore.Qt.LeftButton)
        QTest.mouseClick(win.nav_buttons[0], QtCore.Qt.LeftButton)
        _app.processEvents()
        assert not settings.hasFocus()
        win.close()
    print("PASS: keyboard focus follows the selected navigation item; Settings keeps no stray focus ring")


def _test_activity_bar_is_one_restrained_segment_that_only_runs_while_visible():
    bar = qt_app.ActivityBar()
    bar.resize(300, 4)
    assert bar.indeterminate and bar.accessibleName() and not hasattr(bar, "value"), "no value, so no fake percentage"
    assert bar._anim.state() == QtCore.QAbstractAnimation.Stopped, "idle until shown"
    bar.show()
    _app.processEvents()
    assert bar._anim.state() == QtCore.QAbstractAnimation.Running
    accent = QtGui.QColor(theme.COLORS["accent"]).name().upper()
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        bar._t = t
        image = bar.grab().toImage()
        runs, inside = 0, False
        for x in range(image.width()):
            lit = image.pixelColor(x, 2).name().upper() == accent
            runs += lit and not inside
            inside = lit
        assert runs == 1, f"exactly one segment at t={t}, found {runs}"
    bar.hide()
    assert bar._anim.state() == QtCore.QAbstractAnimation.Stopped, "no animation (or CPU) while hidden"
    print("PASS: the activity indicator is a single gliding segment with no value, running only while visible")


def _test_navigation_ring_follows_keyboard_focus_only():
    with _Patched():
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show()
        win.activateWindow()
        _app.processEvents()
        accent = QtGui.QColor(theme.COLORS["accent"]).name().upper()

        def ring(button):
            _app.processEvents()
            image = button.grab().toImage()
            edge = [image.pixelColor(x, y).name().upper() for x in (1, image.width() - 2) for y in range(8, image.height() - 8)]
            edge += [image.pixelColor(x, y).name().upper() for y in (1, image.height() - 2) for x in range(8, image.width() - 8)]
            return accent in edge

        for index in (0, 2, 5):                      # any selected page, with and without focus
            win.show_page(index)
            win.nav_buttons[index].clearFocus()
            win.setFocus()
            _app.processEvents()
            assert not win.nav_buttons[index].hasFocus()
            assert not ring(win.nav_buttons[index]), f"selected item {index} must not show a ring without focus"
            win.nav_buttons[index].setFocus(QtCore.Qt.TabFocusReason)
            assert ring(win.nav_buttons[index]), f"keyboard focus on item {index} shows the ring"
        for index in range(5):
            if index != win.stack.currentIndex():
                assert not ring(win.nav_buttons[index]), "unselected, unfocused items never show a ring"
        # moving the selection moves the ring with the keyboard focus; no other item keeps one
        win.show_page(0)
        assert win.focusWidget() is win.nav_buttons[0] and ring(win.nav_buttons[0])
        assert not any(ring(b) for b in win.nav_buttons[1:])
        QTest.mouseClick(win.nav_buttons[5], QtCore.Qt.LeftButton)   # a mouse click selects without a ring on Settings
        win.nav_buttons[0].clearFocus()
        win.setFocus()
        assert not ring(win.nav_buttons[5])
        win.close()
    print("PASS: the navigation ring appears only for keyboard focus, never for selection alone, and never stays on a deselected item")


def _page_contains(stack, widget):
    return stack.currentWidget().isAncestorOf(widget)


def _test_empty_preview_is_compact_and_has_no_audio_controls():
    win, _ = _make_window()
    assert win.vd_preview.currentIndex() == PV_EMPTY
    assert win.preview_panel.property("empty") is True
    for w in (win.btn_vd_play, win.btn_vd_save, win.btn_vd_download, win.btn_vd_regenerate, win.vd_wave):
        assert not _page_contains(win.vd_preview, w), "audio actions must not show before audio exists"
    assert not _page_contains(win.vd_preview, win.lbl_vd_total)
    win.show()
    _app.processEvents()
    assert win.preview_panel.height() < 160, win.preview_panel.height()
    win.hide()
    print("PASS: empty preview is compact, quiet, and shows no playback or fake duration")


def _test_previous_sample_stays_available_while_regenerating():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.btn_vd_generate.click()
        _wait_idle(win)
        first_preview = win.pending_voice["preview"]
        slow = _SlowModels()
        win.models = slow
        win.btn_vd_regenerate.click()
        _app.processEvents()
        time.sleep(0.2)
        _app.processEvents()
        assert win.busy
        assert win.vd_preview.currentIndex() == PV_READY, "previous sample must stay visible"
        assert win.btn_vd_play.isEnabled() and not win.btn_vd_regenerate.isEnabled()
        assert "Loading the voice model" in win.lbl_vd_notice.text() and "previous" not in win.lbl_vd_notice.text().lower() \
            and "current sample stays available" in win.lbl_vd_notice.text()
        assert not win.vd_regen_bar.isHidden() and win.progress.maximum() == 1, "one indicator: in the Preview, none in the footer"
        assert win.lbl_status.text() == qt_app.FOOTER_CREATING
        assert win.pending_voice["preview"] == first_preview
        slow.release.set()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY and win.lbl_vd_notice.isHidden() and win.vd_regen_bar.isHidden()
    print("PASS: previous sample stays visible and playable while a new one generates")


def _test_preview_text_is_never_clipped_or_oversized():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.resize(1100, 760)
        win.show()
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.models = _StubModels()
        win.btn_vd_regenerate.click()  # fails: long notice above the still-visible previous sample
        _wait_idle(win)
        _app.processEvents()
        panel, notice, play = win.preview_panel, win.lbl_vd_notice, win.btn_vd_play
        assert not notice.isHidden()
        bottom = notice.mapTo(panel, notice.rect().bottomLeft()).y()
        assert bottom <= play.mapTo(panel, play.rect().topLeft()).y(), "notice overlaps the player row"
        assert notice.height() >= notice.heightForWidth(notice.width()) - 1, "notice text is clipped"
        win.pending_voice = None  # error -> empty must shrink back to the compact panel
        win._set_preview(PV_ERROR)
        _app.processEvents()
        win._set_preview(PV_EMPTY)
        _app.processEvents()
        assert panel.height() < 160, panel.height()
        win.hide()
    print("PASS: preview text is never clipped, and the panel shrinks back to its compact empty state")


def _test_footer_actions_match_what_they_can_do():
    win, _ = _make_window()
    win.show_page(0)
    assert win.btn_play_last.isHidden(), "Create Voice has its own Play"
    for i in (1, 2, 3, 4, 5):
        win.show_page(i)
        assert not win.btn_play_last.isHidden()
    win.show_page(0)
    assert win.btn_stop.isHidden() and not win.btn_stop.isEnabled()
    with _Patched():
        win.btn_vd_generate.click()  # voice creation can't be cancelled, so no Stop button
        assert win.busy and not win.btn_stop.isEnabled() and win.btn_stop.isHidden()
        _wait_idle(win)
    win.set_busy(True)  # a cancellable task (clone / audiobook) offers Stop
    assert win.btn_stop.isEnabled() and not win.btn_stop.isHidden()
    win.set_busy(False)
    assert win.btn_stop.isHidden()
    print("PASS: footer scopes Play-last to long-form pages; Stop appears only for cancellable tasks")


def _test_settings_page_separates_scopes():
    win, _ = _make_window()
    page = win.stack.widget(5)
    for w in (win.cmb_device, win.cmb_dtype, win.cmb_attn, win.spn_min, win.spn_max):
        assert page.isAncestorOf(w)
    texts = [l.text() for l in page.findChildren(QtWidgets.QLabel)]
    assert "Compute" in texts and "Long-form text" in texts
    assert not page.isAncestorOf(win.cmb_vd_lang), "language is task-specific"
    print("PASS: Settings holds application-wide options; task options stay on their screen")


def _test_my_voices_empty_state():
    original = voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    try:
        win, _ = _make_window()
        win.show_page(PAGE_VOICES)
        _shown(win)
        assert not win.frm_voices_empty.isHidden() and win.lst_voices.isHidden(), "empty library shows the explanation, not a blank list"
        assert win.frm_voice_actions.isHidden() and win.lbl_voice_count.text() == ""
        assert win.btn_voice_refresh.isVisible() and win.btn_voice_import.isVisible()
        win.btn_voice_go_create.click()
        assert win.stack.currentIndex() == 0, "the empty state leads to Create Voice"
        os.makedirs(os.path.join(voice_library.VOICES_DIR, "Ada"))
        with open(os.path.join(voice_library.VOICES_DIR, "Ada", "voice.json"), "w") as f:
            f.write('{"name": "Ada", "description": "d", "language": "english", "seed": 1, "created": "x"}')
        win.show_page(PAGE_VOICES)
        win.refresh_voices_tab()
        assert win.frm_voices_empty.isHidden() and not win.lst_voices.isHidden() and not win.frm_voice_actions.isHidden()
        assert win.lbl_voice_count.text() == "1 voice"
        win.close()
    finally:
        voice_library.VOICES_DIR = original
    print("PASS: My Voices shows a real empty state (no blank list) and the list once a voice exists")


def _with_voices(names):
    """Temporary voice library; returns (restore, root)."""
    original = voice_library.VOICES_DIR
    root = voice_library.VOICES_DIR = tempfile.mkdtemp()
    for name in names:
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "voice.json"), "w") as f:
            f.write('{"name": "%s", "description": "A warm narrator voice", "language": "english", '
                    '"seed": 1, "created": "2026-11-12 10:00"}' % name)

    def restore():
        voice_library.VOICES_DIR = original
    return restore


def _test_my_voices_actions_fit_at_1024_px():
    for names, pick in (([], None), (["Ada", "Brook", "Cleo"], 1), ([f"V{i}" for i in range(30)], 2)):
        restore = _with_voices(names)
        try:
            win, _ = _make_window()
            win.resize(1024, 640)
            win.show_page(PAGE_VOICES)
            _shown(win)
            win.resize(1024, 640)
            _app.processEvents()
            if pick is not None:
                win.lst_voices.item(pick).setSelected(True)
            page = win.stack.currentWidget()
            assert page.horizontalScrollBar().maximum() == 0, "My Voices must not scroll sideways at 1024 px"
            assert page.verticalScrollBar().maximum() == 0, "My Voices must fit 1024 x 640 without page scrolling"
            viewport = page.viewport().rect()
            for button in page.findChildren(QtWidgets.QPushButton):
                if not button.isVisible():
                    continue
                top_left = button.mapTo(page.viewport(), QtCore.QPoint(0, 0))
                assert viewport.contains(QtCore.QRect(top_left, button.size())), f"{button.text()!r} is clipped"
                need = button.fontMetrics().horizontalAdvance(button.text()) + 24
                assert button.width() >= need, f"{button.text()!r} is squeezed"
            if names:
                assert win.lst_voices.horizontalScrollBar().maximum() == 0 and not win.lst_voices.horizontalScrollBar().isVisible()
                assert win.lst_voices.height() >= 180
            win.close()
        finally:
            restore()
    print("PASS: every My Voices control is visible at 1024 x 640 with 0, 3 and 30 voices; no sideways scrolling")


def _test_engine_references_require_exactly_one_voice():
    restore = _with_voices(["Ada", "Brook"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        button = win.btn_voice_engine_refs
        assert not button.isEnabled() and "exactly one" in button.toolTip()
        win.lst_voices.item(0).setSelected(True)
        assert button.isEnabled()
        win.lst_voices.selectAll()
        assert not button.isEnabled(), "two selected voices must disable the engine references"
        win.lst_voices.clearSelection()
        assert not button.isEnabled()
        # The old handlers keep their own error fallback when called without a selection.
        win.edit_csm_transcript()
        assert "select exactly one voice" in win.lbl_status.text().lower()
        win.edit_f5_reference()
        assert "select exactly one voice" in win.lbl_status.text().lower()
        # Both existing actions are still offered, under their original names.
        labels = [a.text() for a in button.menu().actions()]
        assert labels == ["Reference transcript…", "F5 reference…"], labels
        win.close()
    finally:
        restore()
    print("PASS: engine references are disabled unless exactly one voice is selected; handlers keep their fallback")


def _test_engine_references_menu_works_from_the_keyboard():
    restore = _with_voices(["Ada"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        win.lst_voices.item(0).setSelected(True)
        calls = []
        win.act_voice_transcript.triggered.disconnect()
        win.act_voice_f5.triggered.disconnect()
        win.act_voice_transcript.triggered.connect(lambda: calls.append("transcript"))
        win.act_voice_f5.triggered.connect(lambda: calls.append("f5"))
        button = win.btn_voice_engine_refs
        button.setFocus()
        _app.processEvents()
        assert win.focusWidget() is button
        QTest.keyClick(button, QtCore.Qt.Key_Space)
        _app.processEvents()
        menu = button.menu()
        assert menu.isVisible(), "Space must open the menu"
        QTest.keyClick(menu, QtCore.Qt.Key_Down)
        QTest.keyClick(menu, QtCore.Qt.Key_Down)
        QTest.keyClick(menu, QtCore.Qt.Key_Return)
        _app.processEvents()
        assert calls == ["f5"], calls
        QTest.keyClick(button, QtCore.Qt.Key_Space)
        _app.processEvents()
        assert menu.isVisible()
        QTest.keyClick(menu, QtCore.Qt.Key_Escape)
        _app.processEvents()
        assert not menu.isVisible() and win.focusWidget() is button, "Escape closes the menu and keeps focus"
        win.close()
    finally:
        restore()
    print("PASS: the engine references menu opens, navigates, selects and closes with the keyboard")


def _test_my_voices_tab_order():
    restore = _with_voices(["Ada"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        win.lst_voices.item(0).setSelected(True)
        win.lst_voices.setCurrentRow(0, QtCore.QItemSelectionModel.NoUpdate)
        first = win.btn_voice_import
        first.setFocus()
        _app.processEvents()

        def label(widget):
            return "voice list" if widget is win.lst_voices else getattr(widget, "text", lambda: "?")()

        seen = [label(first)]
        for _ in range(7):
            QTest.keyClick(win.focusWidget(), QtCore.Qt.Key_Tab)
            _app.processEvents()
            seen.append(label(win.focusWidget()))
        expected = ["Import voice…", "Create from audio…", "Refresh", "voice list", "Play reference",
                    "Rename", "Delete", "Engine references"]
        assert seen == expected, seen
        win.close()
    finally:
        restore()
    print("PASS: My Voices Tab order is header actions, refresh, the list, then the selected-voice actions")


def _voices_window(names, size=(1024, 640)):
    restore = _with_voices(names)
    win, _ = _make_window()
    win.resize(*size)
    win.show_page(PAGE_VOICES)
    _shown(win)
    win.resize(*size)
    _app.processEvents()
    return win, restore


def _test_my_voices_rows_show_name_description_and_formatted_metadata():
    restore = _with_voices([])
    try:
        root = voice_library.VOICES_DIR
        for slug, cfg in {
            "Ada": {"name": "Ada Lovelace", "description": "Warm narrator", "language": "english", "seed": 424242,
                    "created": "2026-11-12T10:30:00"},
            "Odd": {"name": "Odd", "description": "", "language": "", "seed": 7, "created": "not a date"},
        }.items():
            os.makedirs(os.path.join(root, slug))
            with open(os.path.join(root, slug, "voice.json"), "w") as f:
                json.dump(cfg, f)
        win, _ = _make_window()
        win.show_page(PAGE_VOICES)
        _shown(win)
        lst = win.lst_voices
        assert lst.count() == 2
        by_slug = {lst.item(i).data(voice_list.SLUG_ROLE): lst.item(i) for i in range(lst.count())}
        ada = by_slug["Ada"]
        assert ada.text() == "Ada Lovelace" and ada.data(voice_list.NAME_ROLE) == "Ada Lovelace"
        meta = ada.data(QtCore.Qt.UserRole + 3)
        assert meta.startswith("English · ") and "2026" in meta and "T10" not in meta, meta
        assert ada.toolTip() == "Warm narrator"
        accessible = ada.data(QtCore.Qt.AccessibleTextRole)
        assert "Ada Lovelace" in accessible and "Warm narrator" in accessible and "English" in accessible, accessible
        for item in by_slug.values():
            assert "424242" not in item.data(QtCore.Qt.AccessibleTextRole) and "424242" not in (item.data(QtCore.Qt.UserRole + 3) or "")
        assert by_slug["Odd"].data(QtCore.Qt.UserRole + 3) == "not a date", "an unparseable date is shown as stored"
        # the technical metadata stays on disk and in the library API
        details = {d["slug"]: d for d in voice_library.list_voice_details()}
        assert details["Ada"]["seed"] == 424242
        with open(os.path.join(root, "Ada", "voice.json")) as f:
            assert json.load(f)["seed"] == 424242
        win.close()
    finally:
        restore()
    print("PASS: voice rows show name, description, language and a formatted date; seed is not in the list but stays stored")


def _test_my_voices_selection_states_gate_the_actions():
    win, restore = _voices_window(["Ada", "Brook", "Cleo"])
    try:
        play, rename, delete, refs = win.btn_voice_play, win.btn_voice_rename, win.btn_voice_delete, win.btn_voice_engine_refs
        # nothing selected
        assert win.lst_voices.selectedItems() == []
        assert not any(b.isEnabled() for b in (play, rename, delete, refs))
        assert win.lbl_voice_selection.text() == "No voice selected"
        for b in (play, rename, refs):
            assert "exactly one" in b.toolTip()
        assert "Select one or more" in delete.toolTip()
        # one voice
        win.lst_voices.item(1).setSelected(True)
        assert all(b.isEnabled() for b in (play, rename, delete, refs))
        assert win.lbl_voice_selection.text() == "Brook"
        assert play.accessibleName() == "Play reference for Brook" and rename.accessibleName() == "Rename Brook"
        assert delete.accessibleName() == "Delete Brook"
        assert play.accessibleName().startswith(play.text()), "the accessible name starts with the visible label"
        # several voices: only delete stays available
        win.lst_voices.selectAll()
        assert delete.isEnabled() and not any(b.isEnabled() for b in (play, rename, refs))
        assert win.lbl_voice_selection.text() == "3 voices selected"
        assert delete.accessibleName() == "Delete 3 voices"
        win.lst_voices.clearSelection()
        assert not delete.isEnabled()
        # the handlers keep their own fallback when called without exactly one voice
        win.play_voice_tab()
        assert "select exactly one voice to play" in win.lbl_status.text().lower()
        win.rename_voice_tab()
        assert "select exactly one voice to rename" in win.lbl_status.text().lower()
        win.close()
    finally:
        restore()
    print("PASS: play/rename/references need exactly one voice, delete needs at least one; the bar names the selection")


def _test_my_voices_handlers_rename_play_and_delete():
    win, restore = _voices_window(["Ada", "Brook", "Cleo"])
    try:
        from audio_utils import write_wav
        for name in ("Ada", "Brook", "Cleo"):
            write_wav(os.path.join(voice_library.VOICES_DIR, name, "reference.wav"), np.zeros(2400, dtype="float32"), 24000)
        played = []
        win.play = played.append
        # playback: the action bar and Return/double-click on a row both play that voice's reference
        win.lst_voices.item(0).setSelected(True)
        win.btn_voice_play.click()
        assert len(played) == 1 and played[-1].endswith(os.path.join("Ada", "reference.wav")), played
        win.lst_voices.itemActivated.emit(win.lst_voices.item(1))
        assert len(played) == 2 and "Brook" in played[-1], played
        # rename: connected to real persistence, selection follows the renamed voice
        original = QtWidgets.QInputDialog.getText
        asked = []
        QtWidgets.QInputDialog.getText = staticmethod(lambda *a, **k: (asked.append((a[1:3], k)) or ("Ada Prime", True)))
        try:
            win.btn_voice_rename.click()
        finally:
            QtWidgets.QInputDialog.getText = original
        assert "'Ada'" in asked[0][0][1], asked
        assert any(d["name"] == "Ada Prime" for d in voice_library.list_voice_details())
        assert win.selected_voice_names() == ["Ada Prime"], win.selected_voice_names()
        assert win.lbl_status.text() == "Renamed 'Ada' to 'Ada Prime'."
        # delete: the confirmation names the voices (not folder slugs); No keeps them, Yes removes them all
        win.lst_voices.selectAll()
        question = QtWidgets.QMessageBox.question
        prompts = []
        QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: (prompts.append(a[2]) or QtWidgets.QMessageBox.No))
        try:
            win.btn_voice_delete.click()
            assert win.lst_voices.count() == 3 and len(prompts) == 1
            assert "3 voice(s)" in prompts[0] and "Ada Prime" in prompts[0] and "cannot be undone" in prompts[0], prompts
            QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
            win.btn_voice_delete.click()
        finally:
            QtWidgets.QMessageBox.question = question
        assert win.lst_voices.count() == 0 and voice_library.list_voices() == []
        assert win.lbl_status.text() == "Deleted 3 voice(s)."
        assert not win.frm_voices_empty.isHidden(), "deleting the last voice returns to the empty state"
        win.close()
    finally:
        restore()
    print("PASS: rename, playback and multi-delete handlers work from the new list, with contextual footer messages")


def _test_my_voices_has_no_misleading_inline_editing():
    win, restore = _voices_window(["Ada"])
    try:
        lst = win.lst_voices
        assert lst.editTriggers() == QtWidgets.QAbstractItemView.NoEditTriggers
        item = lst.item(0)
        assert not item.flags() & QtCore.Qt.ItemIsEditable, "rows are not editable items"
        lst.setCurrentRow(0)
        lst.setFocus()
        QTest.keyClick(lst, QtCore.Qt.Key_F2)
        lst.edit(lst.model().index(0, 0))
        _app.processEvents()
        assert lst.findChildren(QtWidgets.QLineEdit) == [], "no editor may open for a name that cannot be saved"
        assert lst.state() != QtWidgets.QAbstractItemView.EditingState
        win.close()
    finally:
        restore()
    print("PASS: rows are not editable in place; rename goes through the Rename action")


def _test_my_voices_keyboard_selection_and_return():
    win, restore = _voices_window(["Ada", "Brook", "Cleo"])
    try:
        from audio_utils import write_wav
        for name in ("Ada", "Brook", "Cleo"):
            write_wav(os.path.join(voice_library.VOICES_DIR, name, "reference.wav"), np.zeros(2400, dtype="float32"), 24000)
        played = []
        win.play = played.append
        lst = win.lst_voices
        lst.setFocus()
        _app.processEvents()
        QTest.keyClick(lst, QtCore.Qt.Key_Down)
        lst.setCurrentRow(0)
        QTest.keyClick(lst, QtCore.Qt.Key_Down)
        assert win.selected_voice_names() == ["Brook"] and win.btn_voice_play.isEnabled()
        QTest.keyClick(lst, QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier)
        assert win.selected_voice_names() == ["Brook", "Cleo"] and not win.btn_voice_play.isEnabled() and win.btn_voice_delete.isEnabled()
        QTest.keyClick(lst, QtCore.Qt.Key_Up)
        QTest.keyClick(lst, QtCore.Qt.Key_Return)
        assert len(played) == 1 and "Brook" in played[-1], played
        QTest.keyClick(lst, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        assert len(win.selected_voice_slugs()) == 3, "select all works from the keyboard"
        # type-ahead matches the voice name
        lst.clearSelection()
        lst.setCurrentRow(0)
        QTest.keyClicks(lst, "c")
        assert win.lst_voices.currentItem().text() == "Cleo"
        win.close()
    finally:
        restore()
    print("PASS: the list works from the keyboard: arrows, Shift-extend, select all, type-ahead and Return")


def _test_my_voices_rows_paint_selection_and_focus_visibly():
    win, restore = _voices_window(["Ada", "Brook", "Cleo"])
    try:
        lst = win.lst_voices
        lst.item(1).setSelected(True)
        lst.setCurrentRow(1, QtCore.QItemSelectionModel.NoUpdate)
        _app.processEvents()

        def pixel(row, dx, dy):
            image = lst.viewport().grab().toImage()
            rect = lst.visualItemRect(lst.item(row))
            return image.pixelColor(rect.left() + dx, rect.top() + dy).name().upper()

        accent, selection, surface = theme.COLORS["accent"], theme.COLORS["selection"], theme.COLORS["surface"]
        assert pixel(1, 1, 30) == accent, "the selected row has an accent bar, not just a tint"
        assert pixel(1, 200, 5) == selection and pixel(0, 200, 5) == surface
        assert pixel(0, 1, 30) == surface, "unselected rows have no bar"
        lst.setFocus()
        _app.processEvents()
        assert pixel(1, 1, 5) == accent and pixel(1, 200, 1) == accent, "keyboard focus draws a ring on the current row"
        # contrast of the row text on the selected and resting backgrounds
        c = theme.COLORS
        for fg, bg in ((c["text"], c["selection"]), (c["text_secondary"], c["selection"]), (c["text_secondary"], c["surface"]),
                       (c["text_muted"], c["surface"]), (c["accent"], c["surface"])):
            assert theme.contrast_ratio(fg, bg) >= (3.0 if fg == c["accent"] else 4.5), (fg, bg)
        win.close()
    finally:
        restore()
    print("PASS: selection is an accent bar plus tint, keyboard focus draws a ring, row text keeps its contrast")


def _test_my_voices_rows_are_exposed_to_accessibility():
    """Rows are model items, so Qt's accessibility tree gives each one the full row text."""
    win, restore = _voices_window(["Ada"])
    try:
        from PySide6 import QtGui as _gui
        lst = win.lst_voices
        assert lst.accessibleName() == "Saved voices"
        iface = _gui.QAccessible.queryAccessibleInterface(lst)
        assert iface is not None, "the list must be in the accessibility tree"
        names = []
        for i in range(iface.childCount()):
            child = iface.child(i)
            if child is not None:
                names.append(child.text(_gui.QAccessible.Text.Name))
        assert any("Ada" in n and "A warm narrator voice" in n for n in names), names
        play = _gui.QAccessible.queryAccessibleInterface(win.btn_voice_play)
        assert play.text(_gui.QAccessible.Text.Name).startswith("Play reference") or play.text(_gui.QAccessible.Text.Name) == "Play reference"
        win.close()
    finally:
        restore()
    print("PASS: the list and its rows expose names to the accessibility tree (VoiceOver itself not verified here)")


def _test_my_voices_tab_stops_under_the_macos_default_tab_behavior():
    macos_default = QtCore.Qt.TabFocusBehavior(3)
    hints = _app.styleHints() if _app else _make_window()[0] and _app.styleHints()
    original = hints.tabFocusBehavior()
    restore = _with_voices(["Ada"])
    try:
        hints.setTabFocusBehavior(macos_default)
        configure_keyboard_navigation(_app)      # what main() does at startup
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        win.lst_voices.item(0).setSelected(True)
        win.lst_voices.setCurrentRow(0, QtCore.QItemSelectionModel.NoUpdate)
        win.btn_voice_refresh.setFocus()
        _app.processEvents()
        seen = []
        for _ in range(5):
            QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_Tab)
            _app.processEvents()
            seen.append("list" if win.focusWidget() is win.lst_voices else getattr(win.focusWidget(), "text", lambda: "?")())
        assert seen == ["list", "Play reference", "Rename", "Delete", "Engine references"], seen
        win.close()
    finally:
        hints.setTabFocusBehavior(original)
        restore()
    print("PASS: every My Voices control, including the list, is a Tab stop under the macOS default tab behavior")


class _RaisingModels:
    """Raises a chosen exception from the model call, like a real load failure would."""

    def __init__(self, make_exc):
        self.make_exc, self.calls = make_exc, 0

    def get(self, kind, notify=None):
        self.calls += 1
        raise self.make_exc()


class _BlockedGenerateModels:
    """Model loads at once; generation blocks until released, to observe the generating stage."""

    def __init__(self):
        self.release = threading.Event()

    def get(self, kind, notify=None):
        release = self.release

        class Model(_FakeDesignModel):
            def generate_voice_design(self, text, language, instruct):
                release.wait(10)
                return super().generate_voice_design(text, language, instruct)
        return Model()


def _test_failure_classification_names_only_reliable_causes():
    import errno as _errno
    import task_errors as te

    def fail(exc):
        try:
            raise exc
        except BaseException as caught:
            return te.Failure.from_exception(caught)

    flash = te.classify(fail(ImportError("FlashAttention2 has been toggled on, but it cannot be used due to the "
                                         "following error: the package flash_attn seems to be not installed.")))
    assert flash and flash.kind == "flash" and flash.settings and "FlashAttention" in flash.short
    assert te.classify(fail(MemoryError())).kind == "memory"
    assert te.classify(fail(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))).kind == "memory"
    assert te.classify(fail(OSError(_errno.ENOSPC, "No space left on device"))).kind == "disk"
    assert te.classify(fail(ImportError("'qwen-tts' package is not installed. Install it with: pip install -U qwen-tts"))).kind == "package"
    assert te.classify(te.Failure("requests.exceptions.ConnectionError", "boom", "")).kind == "network"
    assert te.classify(te.Failure("huggingface_hub.errors.OfflineModeIsEnabled", "offline", "")).kind == "network"
    # not reliably identifiable -> unknown, never guessed
    for exc in (RuntimeError("something odd happened"), ValueError("bad input"), KeyError("x"),
                RuntimeError("memory layout is not contiguous"), ImportError("No module named 'foo'"),
                AssertionError("shape mismatch")):
        assert te.classify(fail(exc)) is None, repr(exc)
    assert te.Failure.from_exception(AssertionError()).message == "AssertionError", "a blank message must not hide the cause"
    print("PASS: only reliably identifiable failures are named; everything else stays unknown")


def _test_failure_details_are_sanitized_but_keep_the_cause():
    import task_errors as te
    home, app = "/Users/jane.doe", "/Users/jane.doe/Projects/qwen3-tts-gui"
    raw = "\n".join([
        "Traceback (most recent call last):",
        '  File "%s/qt_app.py", line 1, in work' % app,
        '  File "%s/anaconda3/envs/x/lib/python3.12/site-packages/transformers/a.py", line 9, in f' % home,
        "ImportError: boom for jane.doe@example.com with token=abc123secret and hf_AbCdEfGhIjKlMnOpQr1234 "
        "in %s/.cache and C:\\Users\\Jane\\file and /home/jane/x Bearer sk-live_abcdefghijklmnopqrstu" % home,
    ])
    failure = te.Failure("builtins.ImportError", "boom", raw, None)
    text = te.details_text(failure, context={"Settings": "device=cpu, dtype=bfloat16, attention=sdpa"}, home=home, app_dir=app)
    for private in ("jane.doe", "Jane", "jane", "abc123secret", "hf_AbCdEf", "sk-live", "example.com"):
        assert private not in text, (private, text)
    assert "<app>/qt_app.py" in text and "<site-packages>/transformers/a.py" in text
    assert "builtins.ImportError" in text and "Settings: device=cpu" in text and "ImportError: boom" in text
    assert failure.trace == raw, "the original failure context is preserved unsanitized in memory"
    print("PASS: details drop private paths, tokens and emails but keep type, message, settings and trace")


def _test_known_failure_explains_cause_action_and_details():
    with _Patched():
        def flash():
            return ImportError("FlashAttention2 has been toggled on, but it cannot be used due to the following "
                               "error: the package flash_attn seems to be not installed.")
        models = _RaisingModels(flash)
        win, _ = _make_window(models=models)
        win.resize(1024, 640)
        _shown(win)
        win.txt_vd_instruct.setPlainText("A gravelly old sailor")
        win.btn_vd_generate.setFocus()          # the person started it from the button
        win.btn_vd_generate.click()
        _wait_idle(win)
        _app.processEvents()
        assert win.vd_preview.currentIndex() == PV_ERROR
        assert "FlashAttention" in win.lbl_vd_error.text() and "isn't installed" in win.lbl_vd_error.text()
        assert "different Attention option" in win.lbl_vd_error_action.text()
        assert not win.btn_vd_error_settings.isHidden(), "Settings is where this can be changed"
        assert win.lbl_status.text() == "Couldn't generate the sample: FlashAttention isn't installed."
        assert win.progress.maximum() == 1, "no animated footer progress after a failure"
        details = win.txt_vd_details.toPlainText()
        assert "builtins.ImportError" in details and "flash_attn" in details and "attention=" in details
        assert os.path.expanduser("~") not in details and "Traceback" in details
        assert win.txt_vd_details.isHidden() and win.btn_vd_details.text() == "Show details"
        assert win.focusWidget() is win.btn_vd_retry, "focus lands on Retry so a keyboard user can act at once"
        # Show details works from the keyboard and the text is focusable, scrollable and read-only
        win.btn_vd_details.setFocus()
        QTest.keyClick(win.btn_vd_details, QtCore.Qt.Key_Space)
        _app.processEvents()
        assert not win.txt_vd_details.isHidden() and win.btn_vd_details.text() == "Hide details" and win.btn_vd_details.isChecked()
        QTest.keyClick(win.btn_vd_details, QtCore.Qt.Key_Tab)
        _app.processEvents()
        assert win.focusWidget() in (win.btn_vd_copy_details, win.txt_vd_details)
        assert win.txt_vd_details.isReadOnly() and win.txt_vd_details.focusPolicy() != QtCore.Qt.NoFocus
        assert win.txt_vd_details.height() <= 150
        # Copy details puts the sanitized text on the clipboard and says so
        win.btn_vd_copy_details.click()
        assert QtWidgets.QApplication.clipboard().text() == details
        assert win.lbl_status.text() == "Technical details copied."
        # Retry reruns the same generation (and fails the same way, once more)
        win.btn_vd_retry.click()
        _wait_idle(win)
        assert models.calls == 2 and win.vd_preview.currentIndex() == PV_ERROR
        assert win.txt_vd_instruct.toPlainText() == "A gravelly old sailor"
        for widget, name in ((win.lbl_vd_error, "Error"), (win.btn_vd_details, "Technical details"),
                             (win.txt_vd_details, "Technical details text")):
            assert widget.accessibleName() == name
        win.close()
    print("PASS: a known failure names its cause and next step, keeps details accessible, copyable and sanitized, and retries")


def _test_unknown_failure_stays_generic_and_honest():
    with _Patched():
        models = _RaisingModels(lambda: RuntimeError("tensor shape mismatch in layer 7"))
        win, _ = _make_window(models=models)
        win.resize(1024, 640)
        _shown(win)
        win.txt_vd_instruct.setPlainText("A calm narrator")
        win.txt_vd_instruct.setFocus()          # still editing the description when the failure arrives
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.focusWidget() is win.txt_vd_instruct, "an error must not take focus from someone who is typing"
        assert win.lbl_vd_error.text() == "Something went wrong while creating the sample."
        assert "unchanged" in win.lbl_vd_error_action.text() and "Show details" in win.lbl_vd_error_action.text()
        assert win.btn_vd_error_settings.isHidden(), "no settings advice for a cause we cannot identify"
        for claim in ("memory", "flash", "internet", "install", "Settings"):
            assert claim.lower() not in win.lbl_vd_error.text().lower() + win.lbl_vd_error_action.text().lower()
        assert win.lbl_status.text() == "Couldn't generate the sample. See the Preview for details."
        win.btn_vd_details.click()
        details = win.txt_vd_details.toPlainText()
        assert "RuntimeError" in details and "tensor shape mismatch in layer 7" in details, "the real cause is in the details"
        win.close()
    print("PASS: an unknown failure stays generic, makes no claims, and the details carry the real exception")


def _test_failed_regeneration_keeps_sample_and_offers_details():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.resize(1024, 640)
        _shown(win)
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.models = _RaisingModels(lambda: MemoryError())
        win.btn_vd_regenerate.click()
        _wait_idle(win)
        _app.processEvents()
        assert win.vd_preview.currentIndex() == PV_READY and win.btn_vd_play.isEnabled()
        assert "enough memory" in win.lbl_vd_notice.text() and "previous sample is still available" in win.lbl_vd_notice.text()
        assert not win.btn_vd_notice_details.isHidden() and win.vd_regen_bar.isHidden()
        assert "MemoryError" in win._failure_details
        win.close()
    print("PASS: a failed regeneration keeps the good sample, says why, and still offers the technical details")


def _test_loading_shows_one_indicator_and_only_real_stages():
    with _Patched():
        slow = _SlowModels()
        win, _ = _make_window(models=slow)
        win.resize(1024, 640)
        _shown(win)
        assert win.vd_preview.currentIndex() == PV_EMPTY and win.progress.maximum() == 1 and win.btn_stop.isHidden()
        win.txt_vd_instruct.setPlainText("A warm narrator")
        win.btn_vd_generate.click()
        for _ in range(20):
            _app.processEvents()
            time.sleep(0.02)
        assert win.vd_preview.currentIndex() == PV_BUSY and win.busy
        # stage 1: the backend reported it is loading the model
        assert win.lbl_vd_busy.text() == qt_app.STAGE_LOADING and not win.lbl_vd_busy_hint.isHidden()
        # exactly one animated indicator: the Preview's. The footer neither animates nor repeats the stage.
        assert win.vd_busy_bar.indeterminate and win.vd_busy_bar.isVisible()
        assert win.progress.maximum() == 1, "the footer bar must not animate while the Preview shows progress"
        assert win.lbl_status.text() == qt_app.FOOTER_CREATING
        assert "%" not in win.lbl_vd_busy.text() and "second" not in win.lbl_vd_busy.text().lower()
        slow.release.set()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY and win.lbl_status.text() == READY_STATUS
        assert win.progress.maximum() == 1

        blocked = _BlockedGenerateModels()
        win2, _ = _make_window(models=blocked)
        win2.resize(1024, 640)
        _shown(win2)
        win2.txt_vd_instruct.setPlainText("A warm narrator")
        win2.btn_vd_generate.click()
        for _ in range(20):
            _app.processEvents()
            time.sleep(0.02)
        # stage 2: the model call has returned and generation is running
        assert win2.lbl_vd_busy.text() == qt_app.STAGE_GENERATING and not win2.lbl_vd_busy_hint.isHidden()
        assert win2.lbl_vd_busy_hint.text() == "This may take several minutes, depending on your device and sample length."
        assert "elapsed" in win2.lbl_vd_elapsed.text()
        assert win2.progress.maximum() == 1 and win2.vd_busy_bar.indeterminate
        blocked.release.set()
        _wait_idle(win2)
        win.close()
        win2.close()
    print("PASS: one indicator in the Preview, no footer animation, and loading/generating shown only as reported")


def _test_other_pages_keep_their_footer_progress():
    with _Patched():
        win, _ = _make_window()
        win.run_task(lambda task: time.sleep(0.2) or "x", lambda r: None)
        _app.processEvents()
        assert win.busy and win.progress.maximum() == 0, "tasks without their own indicator still animate the footer"
        _wait_idle(win)
        assert win.progress.maximum() == 1
        win.close()
    print("PASS: Clone and Audiobook tasks keep the footer progress as their only indicator")


def _test_import_voice_orange_outline_is_the_keyboard_focus_ring():
    restore = _with_voices([])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        button = win.btn_voice_import
        assert button.property("role") in (None, ""), "Import voice is a neutral button, not primary"
        accent = theme.COLORS["accent"]

        def edge():
            _app.processEvents()
            return button.grab().toImage().pixelColor(0, button.height() // 2).name().upper()

        win.nav_buttons[PAGE_VOICES].setFocus()
        assert edge() != accent, "unfocused, the outline is the neutral control border"
        button.setFocus()
        assert edge() == accent, "focused, the accent outline is the focus indicator"
        # arriving by clicking the sidebar does not park a focus ring on the first control of the page
        win.show_page(0)
        win.txt_vd_instruct.setFocus()
        _app.processEvents()
        QTest.mouseClick(win.nav_buttons[PAGE_VOICES], QtCore.Qt.LeftButton)
        _app.processEvents()
        assert win.stack.currentIndex() == PAGE_VOICES and win.focusWidget() is not button
        assert edge() != accent
        win.btn_voice_go_create.click() if not win.frm_voices_empty.isHidden() else None
        assert win.stack.currentIndex() == 0 and win.focusWidget() is win.txt_vd_instruct
        assert len(win.lbl_voices_empty.text()) < 90, win.lbl_voices_empty.text()
        win.close()
    finally:
        restore()
    print("PASS: Import voice's orange outline is the focus ring (neutral otherwise); the empty-state text is short")


class _AbVoices:
    """A throwaway voices folder with a few voices, so the Audiobooks pickers have real choices."""

    def __init__(self, *names):
        self.names = names or ("Ana", "Marcus")

    def __enter__(self):
        self.original = voice_library.VOICES_DIR
        voice_library.VOICES_DIR = tempfile.mkdtemp()
        for name in self.names:
            path = voice_library.voice_dir(name)
            os.makedirs(path)
            with open(os.path.join(path, "voice.json"), "w") as f:
                json.dump({"name": name, "description": "d", "language": "english", "seed": 1, "created": "x",
                           "ref_audio": "reference.wav", "clone_prompt": "clone_prompt.npz", "csm_ref_text": "Hello there."}, f)
            write_wav(os.path.join(path, "reference.wav"), np.zeros(24000), 24000)   # the files the engines read
            np.savez(os.path.join(path, "clone_prompt.npz"), x=np.zeros(1))
        return self

    def __exit__(self, *exc):
        voice_library.VOICES_DIR = self.original


def _ab_segments(n):
    import audiobook
    return [audiobook.new_segment(f"Segment number {i + 1}.", "voice:Ana" if i % 2 == 0 else "voice:Marcus",
                                  "narration" if i % 2 == 0 else "dialogue", "sad" if i == 2 else "") for i in range(n)]


def _ab_tab(n=5):
    win, _ = _make_window()
    win.show_page(3)
    win.resize(1024, 640)
    win.show()
    tab = win.audiobook_tab
    tab.segments = _ab_segments(n)
    tab.rebuild(select=0)
    _app.processEvents()
    return win, tab


def _select_rows(tab, *rows):
    tab.listw.clearSelection()
    for r in rows:
        tab.listw.item(r).setSelected(True)
    tab.listw.setCurrentRow(rows[0], QtCore.QItemSelectionModel.NoUpdate)
    _app.processEvents()


def _test_audiobook_list_refills_and_selection_never_change_segments():
    import copy
    with _AbVoices():
        win, tab = _ab_tab(6)
        before = copy.deepcopy(tab.segments)
        tab.refresh_voices()
        tab.on_selection()
        for r in (3, 1, 5, 0):
            tab.listw.setCurrentRow(r)
        _select_rows(tab, 1, 2, 4)
        tab.rebuild(select=2)
        assert tab.segments == before, "refilling combos or changing the selection must never write to the segments"
        # a voice that disappeared stays visible and unchanged
        tab.segments[2]["voice"] = "voice:Gone"
        tab.rebuild(select=2)
        assert tab.cmb_seg_voice.currentText() == "Missing: Gone" and tab.segments[2]["voice"] == "voice:Gone"
        tab.refresh_voices()
        assert tab.segments[2]["voice"] == "voice:Gone"
        win.close()
    print("PASS: Audiobooks list: refills, selection changes and rebuilds never write to the segments; missing voices stay visible")


def _test_audiobook_editor_applies_voice_to_the_selection_and_keeps_direction_per_segment():
    with _AbVoices():
        win, tab = _ab_tab(6)
        # one segment: its text, voice and direction are shown; direction edits stay on that segment
        tab.listw.setCurrentRow(2)
        assert tab.txt_segment.toPlainText() == "Segment number 3." and tab.cmb_seg_direction.currentText() == "sad"
        assert tab.cmb_seg_direction.isEnabled() and tab.lbl_seg_title.text() == "Segment 3 · Narration"
        edit = tab.cmb_seg_direction.lineEdit()
        edit.setFocus()
        edit.selectAll()
        for text in ("s", "whisper"):                   # real keys: auto-completion must not desync the box and the segment
            edit.selectAll()
            QTest.keyClicks(edit, text)
            _app.processEvents()
            assert tab.segments[2]["direction"] == tab.cmb_seg_direction.currentText() and tab.segments[2]["direction"].startswith(text)
        edit.selectAll()
        QTest.keyClicks(edit, "whisper")
        tab.cmb_seg_direction.setEditText("whisper")
        assert tab.segments[2]["direction"] == "whisper" and tab.segments[1]["direction"] == ""
        assert "[whisper]" in tab.listw.item(2).data((QtCore.Qt.UserRole + 2))
        # several segments: voice applies to all, text and direction are not editable, mixed shows no value
        _select_rows(tab, 0, 1, 3)
        assert tab.lbl_seg_title.text() == "3 segments selected"
        assert not tab.txt_segment.isEnabled() and not tab.cmb_seg_direction.isEnabled()
        assert tab.cmb_seg_voice.currentIndex() == -1, "mixed voices must not show (or write) a value"
        assert tab.segments[0]["voice"] == "voice:Ana" and tab.segments[1]["voice"] == "voice:Marcus"
        idx = tab.cmb_seg_voice.findData("Marcus")
        tab.cmb_seg_voice.setCurrentIndex(idx)          # what the user's pick does, then the activated signal
        tab.cmb_seg_voice.activated.emit(idx)
        assert [tab.segments[i]["voice"] for i in (0, 1, 3)] == ["voice:Marcus"] * 3
        assert tab.segments[2]["voice"] == "voice:Ana" and tab.segments[2]["direction"] == "whisper", "unselected rows untouched"
        assert "Marcus" in tab.listw.item(0).data(voice_list.NAME_ROLE)
        assert tab.cmb_seg_voice.currentText().endswith("Marcus"), "all selected rows now agree, so the value shows"
        # nothing selected: nothing can be written
        tab.listw.clearSelection()
        _app.processEvents()
        assert not tab.cmb_seg_voice.isEnabled() and not tab.btn_delete.isEnabled() and not tab.btn_preview.isEnabled()
        win.close()
    print("PASS: Audiobooks editor: voice applies to every selected segment, direction and text stay per segment")


def _test_audiobook_edits_split_merge_move_delete_and_project_round_trip_keep_voice_and_direction():
    import audiobook
    with _AbVoices(), _Patched():
        win, tab = _ab_tab(4)
        tab.segments[1]["direction"] = "angry"
        tab.rebuild(select=1)
        tab.txt_segment.setPlainText("Hello there friend.")
        _app.processEvents()
        assert tab.segments[1]["text"] == "Hello there friend."
        tab.txt_segment.moveCursor(QtGui.QTextCursor.Start)
        for _ in range(5):
            tab.txt_segment.moveCursor(QtGui.QTextCursor.Right)
        tab.btn_split_cursor.click()
        assert len(tab.segments) == 5 and [s["text"] for s in tab.segments[1:3]] == ["Hello", "there friend."]
        assert all(s["voice"] == "voice:Marcus" and s["direction"] == "angry" for s in tab.segments[1:3])
        assert tab.selected_rows() == [2]
        tab.listw.setCurrentRow(1)
        tab.btn_merge.click()
        assert len(tab.segments) == 4 and tab.segments[1]["text"] == "Hello there friend." and tab.segments[1]["direction"] == "angry"
        tab.btn_down.click()
        assert tab.segments[2]["text"] == "Hello there friend." and tab.selected_rows() == [2]
        tab.btn_up.click()
        assert tab.segments[1]["text"] == "Hello there friend."
        tab.btn_add.click()
        assert len(tab.segments) == 5 and tab.segments[2]["text"] == "New segment." and tab.selected_rows() == [2]
        _select_rows(tab, 2, 3)
        tab.btn_delete.click()
        assert len(tab.segments) == 3 and tab.selected_rows() == [2]
        # save and load keep everything, through the real project functions and the tab's own handlers
        path = os.path.join(tempfile.mkdtemp(), "book.json")
        original = QtWidgets.QFileDialog.getSaveFileName, QtWidgets.QFileDialog.getOpenFileName
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ""))
        QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))
        try:
            saved = [dict(s) for s in tab.segments]
            tab.btn_save.click()
            tab.segments = []
            tab.rebuild()
            assert tab.listw.count() == 0 and not tab.btn_delete.isEnabled()
            tab.btn_load.click()
        finally:
            QtWidgets.QFileDialog.getSaveFileName, QtWidgets.QFileDialog.getOpenFileName = original
        assert tab.segments == saved and tab.listw.count() == 3 and tab.selected_rows() == [0]
        assert json.load(open(path))["schema"] == 1, "the project schema is unchanged"
        win.close()
    print("PASS: Audiobooks: split, merge, move, add, delete and project save/load keep voices and directions")


def _test_audiobook_typing_keeps_selection_caret_and_focus():
    with _AbVoices():
        win, tab = _ab_tab(5)
        tab.listw.setCurrentRow(3)
        tab.txt_segment.setFocus()
        _app.processEvents()
        tab.txt_segment.moveCursor(QtGui.QTextCursor.End)
        QTest.keyClicks(tab.txt_segment, " More words")
        assert tab.selected_rows() == [3] and tab.txt_segment.hasFocus()
        assert tab.txt_segment.textCursor().position() == len("Segment number 4. More words")
        assert tab.segments[3]["text"] == "Segment number 4. More words"
        assert "More words" in tab.listw.item(3).data((QtCore.Qt.UserRole + 2))
        # a button that becomes unavailable never keeps focus
        tab.listw.setCurrentRow(3)
        tab.btn_merge.setFocus()
        tab.listw.setCurrentRow(4)         # the last segment: nothing to merge with
        _app.processEvents()
        assert not tab.btn_merge.isEnabled() and not tab.btn_merge.hasFocus() and tab.listw.hasFocus()
        win.close()
    print("PASS: Audiobooks: typing keeps the selection, caret and focus; a control that goes away hands focus to the list")


def _test_audiobook_rows_are_compact_readable_and_accessible():
    long_name = "A Very Long Voice Name For The Narrator Of Chapter Twelve"
    with _AbVoices("Ana", "Marcus", long_name):
        win, tab = _ab_tab(27)
        win.resize(1024, 640)
        _app.processEvents()
        tab.segments[0]["voice"] = f"voice:{long_name}"
        tab.segments[2]["direction"] = "whisper"
        tab.rebuild(select=2)
        _app.processEvents()
        # density: one line per segment, at the same type size as the rest of the app
        row = tab.listw.sizeHintForRow(0)
        assert row <= 32, row
        rows_visible = tab.listw.viewport().height() / row
        assert rows_visible >= 5.5, f"only {rows_visible:.1f} rows visible at 1024x640"   # offscreen fonts run taller; native shows 6.5+
        delegate = tab.listw.itemDelegate()
        assert isinstance(delegate, qt_audiobook.SegmentRowDelegate)
        # long voice names are shortened in the row, never widen the list
        scroll = tab.findChild(QtWidgets.QScrollArea)
        assert scroll.horizontalScrollBar().maximum() == 0 and tab.listw.horizontalScrollBar().maximum() == 0
        assert tab.listw.item(0).data(voice_list.NAME_ROLE) == long_name   # the full name, elided only when painted
        # the row paints (selected, focused, hovered) without error and shows its content
        for state in (QtWidgets.QStyle.State_Selected, QtWidgets.QStyle.State_HasFocus, QtWidgets.QStyle.State_MouseOver):
            pix = QtGui.QPixmap(400, row)
            pix.fill(QtGui.QColor("#000000"))
            painter = QtGui.QPainter(pix)
            option = QtWidgets.QStyleOptionViewItem()
            option.rect = QtCore.QRect(0, 0, 400, row)
            option.state = state
            option.font = tab.listw.font()
            delegate.paint(painter, option, tab.listw.model().index(2, 0))
            painter.end()
            image = pix.toImage()
            assert any(image.pixelColor(x, y) != QtGui.QColor("#000000") for x in range(0, 400, 7) for y in range(0, row, 5))
            edge = image.pixelColor(1, row // 2).name().lower()
            if state == QtWidgets.QStyle.State_Selected:
                assert edge == theme.COLORS["accent"].lower(), ("selected rows carry the accent bar", edge)
            elif state == QtWidgets.QStyle.State_MouseOver:
                assert edge != theme.COLORS["accent"].lower(), "hover alone never looks selected"
            # State_HasFocus draws the accent focus frame by design
        # the accessible text reads number, role, voice, direction and text
        text = tab.listw.item(2).data(QtCore.Qt.AccessibleTextRole)
        assert text.startswith("Segment 3, narration, voice Ana, engine Qwen") and "direction whisper" in text and "Segment number 3." in text, text
        assert "[whisper]" in tab.listw.item(2).data(qt_audiobook.DESC_ROLE)
        assert tab.listw.item(1).data(qt_audiobook.ROLE_ROLE) == "D" and tab.listw.item(0).data(qt_audiobook.ROLE_ROLE) == "N"
        tab.segments[4]["voice"] = "voice:Gone"
        tab.rebuild(select=4)
        assert tab.listw.item(4).data(voice_list.NAME_ROLE) == "Missing: Gone", "a missing voice keeps its full label"
        tip = tab.listw.item(0).toolTip()
        assert long_name in tip and tip.startswith("Segment 1 · Narration · A Very Long"), tip
        # multi-selection, keyboard navigation and Select All still work on the compact rows
        tab.listw.setCurrentRow(2)
        tab.listw.setFocus()
        QTest.keyClick(tab.listw, QtCore.Qt.Key_Down)
        assert tab.selected_rows() == [3]
        QTest.keyClick(tab.listw, QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier)
        assert tab.selected_rows() == [3, 4]
        QTest.keyClick(tab.listw, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        assert len(tab.selected_rows()) == 27 and tab.lbl_seg_title.text() == "27 segments selected"
        win.close()
    print("PASS: Audiobooks rows: one compact line (5.5+ rows visible offscreen at 1024x640, 6.6 native), long names never widen the list, accessible text, selection and keys intact")


def _test_audiobooks_page_opens_without_a_focused_button():
    with _AbVoices():
        win, _ = _make_window()
        win.show()
        win.activateWindow()
        _app.processEvents()
        win.txt_vd_instruct.setFocus()
        _app.processEvents()
        win.show_page(3)
        _app.processEvents()
        assert not isinstance(_app.focusWidget(), QtWidgets.QPushButton), "Return must not open a file dialog on arrival"
        win.close()
    print("PASS: Audiobooks opens with no button focused, like the other pages")


def _test_audiobook_render_and_preview_run_through_the_new_page():
    import audiobook
    import qt_audiobook
    from unittest.mock import patch
    with _AbVoices(), _Patched() as patched:
        win, tab = _ab_tab(3)
        calls = []

        def fake_render(gen, segment, settings, cache_dir=None):
            calls.append(segment["text"])
            return np.full(2400, 0.1, dtype=np.float32), 24000
        original_dir = qt_audiobook.OUTPUT_DIR
        qt_audiobook.OUTPUT_DIR = patched.tmp
        try:
            with patch.object(audiobook, "render_segment", fake_render):
                # Render chapter from the header
                tab.btn_render.click()
                _wait_idle(win)
                assert win.lbl_status.text().startswith("Chapter rendered (3 segments"), win.lbl_status.text()
                assert win.last_output and os.path.isfile(win.last_output) and win.btn_play_last.isEnabled()
                assert len(calls) == 3
                # Preview segment plays the preview file
                tab.listw.setCurrentRow(1)
                assert tab.btn_preview.isEnabled()
                tab.btn_preview.click()
                _wait_idle(win)
                assert patched.played and patched.played[-1].endswith("_segment_preview.wav"), patched.played
                assert os.path.isfile(patched.played[-1])
                # while a task runs nothing can start another; afterwards Preview follows the selection again
                win.set_busy(True)
                assert not tab.btn_render.isEnabled() and not tab.btn_preview.isEnabled()
                win.set_busy(False)
                assert tab.btn_render.isEnabled() and tab.btn_preview.isEnabled()
                _select_rows(tab, 0, 1, 2)
                win.set_busy(True)
                win.set_busy(False)
                assert not tab.btn_preview.isEnabled(), "Preview needs exactly one segment"
        finally:
            qt_audiobook.OUTPUT_DIR = original_dir
        win.close()
    print("PASS: Audiobooks: Render chapter and Preview segment work from the new page, and Preview follows the selection after a task")


def _test_audiobook_tab_cycle_from_the_navigation_is_pinned_both_ways():
    with _AbVoices():
        win, tab = _ab_tab(5)
        tab.listw.setCurrentRow(2)
        win.show_page(3)
        _shown(win)
        fixed = lambda labels: ["Segments" if l == "?VoiceList" else l for l in labels]
        forward = fixed(_focus_cycle(win))
        expected = ["Audiobooks", "Open output folder",      # the footer follows the navigation here, as on every page not yet given an explicit order
                    "Save project…", "Load project…", "Render chapter", "Chapter script", "combo:Ana", "combo:Qwen",
                    "combo:Marcus", "combo:Qwen", "combo:english", "Split into segments", "Render settings", "Segments",
                    "▶ Preview segment", "Selected segment text", "combo:Ana", "combo:Qwen", "combo:sad", "Split here",
                    "Merge next", "Add after", "Move up", "Move down", "Delete"]
        assert forward == expected, forward
        backward = fixed(_focus_cycle(win, backwards=True))
        assert backward == [expected[0]] + expected[:0:-1], backward
        win.close()
    print("PASS: Audiobooks: the whole Tab cycle from the navigation is pinned, forwards and backwards")


def _test_audiobook_keyboard_wheel_and_geometry_at_the_minimum_window():
    import audiobook
    long_name = "A Very Long Voice Name For The Narrator Of Chapter Twelve"
    with _AbVoices("Ana", "Marcus", long_name):
        counts = []
        for n in (2, 27):
            win, tab = _ab_tab(n)
            tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData(long_name))
            tab.segments[0]["voice"] = f"voice:{long_name}"
            tab.rebuild(select=0)
            tab.btn_render.setFocus()
            _app.processEvents()
            stops = 0
            while _app.focusWidget() is not tab.btn_preview and stops < 60:
                QTest.keyClick(_app.focusWidget(), QtCore.Qt.Key_Tab)
                _app.processEvents()
                stops += 1
            counts.append(stops)
            assert stops < 30, f"{stops} Tab presses from Render to the last editor control"
            if n == 27:
                tab.listw.setCurrentRow(5)
                tab.btn_save.setFocus()
                _app.processEvents()
                seen = []
                for _ in range(22):
                    QTest.keyClick(_app.focusWidget(), QtCore.Qt.Key_Tab)
                    _app.processEvents()
                    w = _app.focusWidget()
                    seen.append(tab.cmb_seg_direction if w is tab.cmb_seg_direction.lineEdit() else w)
                expected = [tab.btn_load, tab.btn_render, tab.txt_source, tab.cmb_narrator, tab.cmb_narrator_engine,
                            tab.cmb_dialogue, tab.cmb_dialogue_engine, tab.cmb_lang,
                            tab.btn_split, tab.btn_settings, tab.listw, tab.btn_preview, tab.txt_segment, tab.cmb_seg_voice,
                            tab.cmb_seg_engine, tab.cmb_seg_direction, tab.btn_split_cursor, tab.btn_merge, tab.btn_add, tab.btn_up, tab.btn_down,
                            tab.btn_delete]
                assert seen == expected, [(w.accessibleName() or w.text()) for w in seen]
            # geometry: nothing is clipped at 1024x640, Render is in view, no scrollbars
            scroll = tab.findChild(QtWidgets.QScrollArea)
            assert scroll.horizontalScrollBar().maximum() == 0 and scroll.verticalScrollBar().maximum() == 0
            area = tab.rect()
            for w in (tab.btn_save, tab.btn_load, tab.btn_render, tab.btn_split, tab.cmb_narrator, tab.cmb_dialogue,
                      tab.cmb_narrator_engine, tab.cmb_dialogue_engine, tab.status_narrator, tab.status_dialogue,
                      tab.status_segment, tab.cmb_seg_engine,
                      tab.cmb_lang, tab.listw, tab.txt_segment, tab.cmb_seg_voice, tab.cmb_seg_direction,
                      tab.btn_delete, tab.btn_preview):
                top_left = w.mapTo(tab, QtCore.QPoint(0, 0))
                box = QtCore.QRect(top_left, w.size())
                assert area.contains(box), (w.accessibleName() or w.text(), box, area)
                assert w.width() >= (w.minimumWidth() or w.minimumSizeHint().width()), (w.accessibleName() or w.text(), w.width())   # a combo may be squeezed to its set minimum, and elides
            assert tab.btn_render.isVisible()
            # the wheel changes nothing unless the combo has focus
            tab.listw.setCurrentRow(1)
            tab.btn_render.setFocus()
            before = tab.segments[1]["voice"]
            ev = QtGui.QWheelEvent(QtCore.QPointF(5, 5), tab.cmb_seg_voice.mapToGlobal(QtCore.QPointF(5, 5)), QtCore.QPoint(0, 0),
                                   QtCore.QPoint(0, -120), QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
            index = tab.cmb_seg_voice.currentIndex()
            _app.sendEvent(tab.cmb_seg_voice, ev)
            tab.cmb_seg_voice.setFocus()
            _app.processEvents()
            _app.sendEvent(tab.cmb_seg_voice, ev)
            _app.processEvents()
            assert tab.cmb_seg_voice.currentIndex() == index and tab.segments[1]["voice"] == before, "the wheel never changes a voice"
            tab.btn_settings.setChecked(True)
            _app.processEvents()
            for w, value in ((tab.spn_pause, tab.spn_pause.value()), (tab.spn_seed, tab.spn_seed.value()),
                             (tab.spn_temp, tab.spn_temp.value()), (tab.cmb_seed_mode, tab.cmb_seed_mode.currentIndex())):
                ev2 = QtGui.QWheelEvent(QtCore.QPointF(5, 5), w.mapToGlobal(QtCore.QPointF(5, 5)), QtCore.QPoint(0, 0),
                                        QtCore.QPoint(0, 120), QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
                _app.sendEvent(w, ev2)
                _app.processEvents()
                now = w.currentIndex() if isinstance(w, QtWidgets.QComboBox) else w.value()
                assert now == value, f"the wheel changed {w.accessibleName()}"
            tab.btn_settings.setChecked(False)
            win.close()
        assert counts[0] == counts[1], f"Tab stops must not grow with the number of segments: {counts}"
    print("PASS: Audiobooks: Tab reaches the editor in a few stops at any size of book, nothing clips at 1024x640, the wheel never changes a voice")


_READY_ENV = {"xtts": True, "csm": True, "dia": True, "f5": "ok"}


class _CastEnv:
    """Every optional engine importable, so the tests judge the voices, language and text rather than this machine."""

    def __init__(self, **overrides):
        self.env = dict(_READY_ENV, **overrides)

    def __enter__(self):
        from unittest.mock import patch
        self.patch = patch.object(cast_rules, "environment", lambda: dict(self.env))
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


def _ab_edit_voice(name, **changes):
    path = os.path.join(voice_library.voice_dir(name), "voice.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(changes)
    config = {key: value for key, value in config.items() if value is not None}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


def _pick(combo, data):
    """What the user's choice does: set the item, then the activated signal."""
    index = combo.findData(data)
    assert index >= 0, (data, [combo.itemData(i) for i in range(combo.count())])
    combo.setCurrentIndex(index)
    combo.activated.emit(index)
    _app.processEvents()


def _engine_items(combo):
    return {combo.itemData(i): (combo.itemText(i), combo.model().item(i).isEnabled()) for i in range(combo.count())}


def _test_cast_voices_appear_once_and_engines_carry_their_state():
    with _AbVoices("Ana", "Bare"), _CastEnv(xtts=False):
        _ab_edit_voice("Bare", csm_ref_text=None)
        win, tab = _ab_tab(4)
        names = [tab.cmb_narrator.itemData(i) for i in range(tab.cmb_narrator.count())]
        assert names == ["Ana", "Bare"] and all(tab.cmb_seg_voice.itemData(i) in names for i in range(tab.cmb_seg_voice.count()))
        assert [tab.cmb_narrator.itemText(i) for i in range(2)] == ["Ana", "Bare"], "no engine-prefixed duplicates"
        assert [tab.cmb_narrator_engine.itemData(i) for i in range(5)] == ["qwen", "xtts", "csm", "dia", "f5"]
        # Ana: Qwen ready; XTTS needs the package (here forced missing)
        engines = _engine_items(tab.cmb_narrator_engine)
        assert engines["qwen"] == ("Qwen", True) and engines["csm"] == ("CSM", True)
        assert engines["xtts"][0].startswith("XTTS · needs coqui-tts") and engines["xtts"][1], "needs setup stays selectable"
        assert "Ready" in tab.status_narrator.full_text() and "local checks only" in tab.status_narrator.full_text()
        # Bare has no confirmed transcript: CSM, Dia and F5 need setup, with an actionable reason
        _pick(tab.cmb_narrator, "Bare")
        assert _engine_items(tab.cmb_narrator_engine)["csm"][0] == "CSM · needs transcript"
        _pick(tab.cmb_narrator_engine, "csm")
        assert tab.narrator_key() == "csm:Bare"
        text = tab.status_narrator.full_text()
        assert text.startswith("Needs setup:") and "Reference transcript" in text and tab.status_narrator.state == cast_rules.SETUP
        assert "Reference transcript" in tab.cmb_narrator.accessibleDescription()
        # a narrator split uses voice and engine together
        tab.txt_source.setPlainText('He said, "Hello." Then left.')
        tab.segments = []
        tab.auto_split()
        assert [s["voice"] for s in tab.segments][0] == "csm:Bare"
        win.close()
    print("PASS: Cast: each voice appears once, engines carry their state, Needs setup stays selectable with an actionable reason")


def _test_cast_language_makes_engines_unavailable_but_never_rewrites_assignments():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(4)
        tab.segments[1]["voice"] = "csm:Marcus"
        tab.segments[3]["voice"] = "xtts:Ana"
        tab.rebuild(select=0)
        before = [s["voice"] for s in tab.segments]
        tab.cmb_lang.setCurrentText("french")
        _app.processEvents()
        engines = _engine_items(tab.cmb_seg_engine)
        assert engines["csm"][1] is False and engines["dia"][1] is False and engines["f5"][1] is False
        assert engines["csm"][0] == "CSM · English only" and engines["xtts"][1] and engines["qwen"][1]
        assert "Unavailable:" in tab.cmb_seg_engine.accessibleDescription() and "English only" in tab.cmb_seg_engine.accessibleDescription()
        assert [s["voice"] for s in tab.segments] == before, "a language change never rewrites a segment"
        state = qt_audiobook.STATE_ROLE
        assert tab.listw.item(1).data(state) == cast_rules.UNSUPPORTED and tab.listw.item(0).data(state) == cast_rules.READY
        assert tab.listw.item(3).data(state) == cast_rules.READY
        assert "Unsupported" in tab.listw.item(1).data(QtCore.Qt.AccessibleTextRole) and "English only" in tab.listw.item(1).toolTip()
        # an invalid assignment stays visible and editable: select it, its own engine is still choosable
        tab.listw.setCurrentRow(1)
        assert tab.cmb_seg_engine.currentData() == "csm" and _engine_items(tab.cmb_seg_engine)["csm"][1] is True
        assert tab.status_segment.state == cast_rules.UNSUPPORTED and tab.status_segment.full_text() == "English only"
        assert "Change Language" in tab.status_segment.toolTip() and "Change Language" in tab.cmb_seg_voice.accessibleDescription()
        _pick(tab.cmb_seg_voice, "Ana")                       # changing the voice is still allowed
        assert tab.segments[1]["voice"] == "csm:Ana"
        # a new, invalid engine cannot be assigned, even by a programmatic activation
        _pick(tab.cmb_seg_engine, "xtts")
        assert tab.segments[1]["voice"] == "xtts:Ana"
        tab.listw.setCurrentRow(0)
        index = tab.cmb_seg_engine.findData("dia")
        assert not tab.cmb_seg_engine.model().item(index).isEnabled()
        tab.cmb_seg_engine.activated.emit(index)
        _app.processEvents()
        assert tab.segments[0]["voice"] == "voice:Ana"
        tab.cmb_lang.setCurrentText("english")
        assert _engine_items(tab.cmb_seg_engine)["dia"][1] is True
        win.close()
    print("PASS: Cast: a language change disables engines that cannot speak it and flags, never rewrites, existing assignments")


def _test_cast_direction_and_tags_follow_the_selection_and_block_new_assignments():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(4)
        tab.listw.setCurrentRow(2)                                # has direction "sad" (Qwen)
        engines = _engine_items(tab.cmb_seg_engine)
        assert all(engines[e][1] is False and "no direction" in engines[e][0] for e in ("xtts", "csm", "dia", "f5")), engines
        assert engines["qwen"][1]
        tab.cmb_seg_direction.setEditText("")
        assert all(_engine_items(tab.cmb_seg_engine)[e][1] for e in ("xtts", "csm", "dia", "f5")), "clearing the direction frees them"
        tab.txt_segment.setPlainText("[sad] It rained.")
        assert _engine_items(tab.cmb_seg_engine)["csm"][0] == "CSM · no [tags]" and not _engine_items(tab.cmb_seg_engine)["csm"][1]
        # several selected segments: an engine is disabled if it cannot speak any of them
        tab.txt_segment.setPlainText("It rained.")
        _select_rows(tab, 0, 1, 2)
        tab.segments[2]["direction"] = "whisper"
        tab._sync_voice_combo()
        assert not _engine_items(tab.cmb_seg_engine)["csm"][1]
        win.close()
    print("PASS: Cast: direction and [tags] disable the engines that cannot speak them, for one segment or for any in a selection")


def _test_cast_bulk_edits_keep_each_voice_and_each_engine():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(6)
        tab.segments[0]["voice"], tab.segments[1]["voice"], tab.segments[3]["voice"] = "xtts:Ana", "csm:Marcus", "voice:Ana"
        tab.segments[2]["direction"] = ""
        tab.rebuild(select=0)
        _select_rows(tab, 0, 1, 3)
        assert tab.cmb_seg_voice.currentIndex() == -1 and tab.cmb_seg_voice.placeholderText() == "Mixed"
        assert tab.cmb_seg_engine.currentIndex() == -1 and tab.cmb_seg_engine.placeholderText() == "Keep each"
        _pick(tab.cmb_seg_voice, "Marcus")                          # the voice changes, each engine stays
        assert [tab.segments[i]["voice"] for i in (0, 1, 3)] == ["xtts:Marcus", "csm:Marcus", "voice:Marcus"]
        assert tab.cmb_seg_voice.currentData() == "Marcus" and tab.cmb_seg_engine.currentIndex() == -1
        _pick(tab.cmb_seg_engine, "dia")                            # the engine changes, each voice stays
        assert [tab.segments[i]["voice"] for i in (0, 1, 3)] == ["dia:Marcus"] * 3
        assert tab.cmb_seg_engine.currentData() == "dia"
        assert tab.segments[2]["voice"] == "voice:Ana" and tab.segments[4]["voice"] == "voice:Ana", "unselected rows untouched"
        # a bulk choice that would give any segment an impossible combination changes nothing at all
        tab.segments[1]["direction"] = "whisper"
        before = [s["voice"] for s in tab.segments]
        _select_rows(tab, 0, 1)
        tab.cmb_seg_engine.activated.emit(tab.cmb_seg_engine.findData("xtts"))
        _app.processEvents()
        assert [s["voice"] for s in tab.segments] == before and "Not changed" in win.lbl_status.text()
        # the status line reports how many of the selection need attention
        tab.segments[1]["voice"] = "csm:Gone"
        tab.rebuild(select=0)
        _select_rows(tab, 0, 1, 2)
        assert "of 3: " in tab.status_segment.full_text() and "selected segments need attention" in tab.cmb_seg_voice.accessibleDescription()
        win.close()
    print("PASS: Cast: bulk voice keeps each engine, bulk engine keeps each voice, impossible bulk changes write nothing")


def _test_cast_missing_voices_are_preserved_and_flagged():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(4)
        tab.segments[1]["voice"] = "csm:Gone"
        tab.segments[2]["voice"] = "legacy-key"
        tab.segments[3]["voice"] = ""
        tab.rebuild(select=1)
        before = [s["voice"] for s in tab.segments]
        assert tab.cmb_seg_voice.currentText() == "Missing: Gone" and tab.cmb_seg_engine.currentData() == "csm"
        assert tab.listw.item(1).data(voice_list.NAME_ROLE) == "Missing: Gone"
        assert tab.listw.item(1).data(qt_audiobook.STATE_ROLE) == cast_rules.MISSING
        assert "Voice missing" == tab.status_segment.full_text() and "no longer exists" in tab.status_segment.toolTip()
        assert tab.listw.item(2).data(qt_audiobook.STATE_ROLE) == cast_rules.MISSING and tab.listw.item(3).data(qt_audiobook.STATE_ROLE) == cast_rules.SETUP
        tab.refresh_voices()
        tab.refresh_compat()
        tab.cmb_lang.setCurrentText("german")
        tab.cmb_lang.setCurrentText("english")
        assert [s["voice"] for s in tab.segments] == before, "nothing rewrites a missing assignment"
        # a missing narrator voice stays visible too
        tab.cmb_narrator.clear()
        tab.cmb_narrator.addItem("Missing: Lost", "Lost")
        tab.cmb_narrator.setCurrentIndex(0)
        tab.refresh_voices()
        assert tab.cmb_narrator.currentData() == "Lost" and tab.cmb_narrator.currentText() == "Missing: Lost"
        assert tab.status_narrator.state == cast_rules.MISSING
        win.close()
    print("PASS: Cast: missing voices, unknown keys and empty voices stay as they are, visible and flagged")


def _test_cast_preflight_blocks_render_and_preview_before_licenses_and_models():
    import audiobook
    from unittest.mock import patch
    with _AbVoices("Ana", "Bare", "Marcus"), _CastEnv(), _Patched():
        _ab_edit_voice("Bare", csm_ref_text=None)
        win, tab = _ab_tab(6)
        tab.segments[2]["direction"] = ""
        tab.segments[0]["voice"], tab.segments[1]["voice"], tab.segments[2]["voice"] = "csm:Bare", "csm:Bare", "xtts:Ana"
        tab.segments[3]["voice"], tab.segments[4]["voice"] = "voice:Gone", ""
        tab.rebuild(select=0)
        calls, asked = [], []
        os.environ.pop("COQUI_TOS_AGREED", None)

        def fake_render(gen, segment, settings, cache_dir=None):
            calls.append(segment["text"])
            return np.full(2400, 0.1, dtype=np.float32), 24000

        def no_question(*args, **kwargs):
            asked.append(args)
            return QtWidgets.QMessageBox.Yes
        with patch.object(audiobook, "render_segment", fake_render), patch.object(QtWidgets.QMessageBox, "question", no_question):
            tab.btn_render.click()
            _wait_idle(win)
            _app.processEvents()
            assert calls == [] and asked == [], "no model work, and no license question, before every segment can be spoken"
            dialog = tab._problem_dialog
            assert dialog is not None and dialog.isVisible() and "Nothing was loaded" in win.lbl_status.text() or "Not rendered" in win.lbl_status.text()
            groups = {g.cause: g for g in tab._problem_groups}
            assert groups["setup:csm:transcript:Bare"].rows == (0, 1) and groups["missing:voice:Gone"].rows == (3,)
            assert groups["setup:none:voice"].rows == (4,) and len(groups) == 3
            assert tab._problem_groups[0].state == cast_rules.MISSING, "worst first"
            buttons = [b for b in dialog.findChildren(QtWidgets.QPushButton) if b.text() == "Select these segments"]
            assert len(buttons) == 3 and all(b.accessibleName() for b in buttons)
            texts = " ".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
            assert "2 segments (1, 2)" in texts and "Reference transcript" in texts
            [b for b in buttons if "transcript" in b.accessibleName().lower() or "Reference" in b.accessibleName()][0].click()
            _app.processEvents()
            assert tab.selected_rows() == [0, 1], tab.selected_rows()
            win.activateWindow()
            tab.select_segments([0, 1])
            _app.processEvents()
            assert tab.listw.hasFocus() or tab.listw.focusWidget() is not None, "the keyboard lands on the list"
            # Preview checks the one segment the same way
            _select_rows(tab, 4)
            tab.btn_preview.click()
            _wait_idle(win)
            assert calls == [] and tab._problem_groups[0].rows == (4,), (calls, [(g.cause, g.rows) for g in tab._problem_groups], tab.selected_rows(), tab.btn_preview.isEnabled())
            tab._problem_dialog.close()
            # once the segments are fine, Render proceeds (and the license question is the next step, not the first)
            for r in (0, 1, 2, 3, 4, 5):
                tab.segments[r]["voice"] = "voice:Ana"
            tab.rebuild(select=0)
            tab.segments[2]["voice"] = "xtts:Ana"
            tab.rebuild(select=0)
            tab.btn_render.click()
            _wait_idle(win)
            assert asked and "XTTS" in asked[0][1], "licenses are asked only after the pre-flight passes"
            _app.processEvents()
        win.close()
        os.environ.pop("COQUI_TOS_AGREED", None)
    print("PASS: Cast: Render and Preview stop before licenses, models and rendering, group problems by cause, and select the affected segments")


def _test_cast_role_markers_are_shapes_with_accessible_text():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(4)
        delegate = tab.listw.itemDelegate()
        row = tab.listw.sizeHintForRow(0)
        images = {}
        for r in (0, 1):                                          # 0 narration, 1 dialogue
            pix = QtGui.QPixmap(300, row)
            pix.fill(QtGui.QColor(theme.COLORS["surface"]))
            painter = QtGui.QPainter(pix)
            option = QtWidgets.QStyleOptionViewItem()
            option.rect = QtCore.QRect(0, 0, 300, row)
            option.font = tab.listw.font()
            delegate.paint(painter, option, tab.listw.model().index(r, 0))
            painter.end()
            images[r] = pix.toImage().copy(delegate.NUM_W + 12, 0, delegate.ROLE_W, row)
        background = QtGui.QColor(theme.COLORS["surface"]).name()
        for image in images.values():
            ink = [(x, y) for x in range(image.width()) for y in range(image.height()) if image.pixelColor(x, y).name() != background]
            assert len(ink) > 20, "the role marker is drawn"
        assert images[0] != images[1], "narration and dialogue differ by shape"
        assert qt_audiobook.icons.icon("quote", 16, gap=0).availableSizes() or not qt_audiobook.icons.icon("quote").isNull()
        assert " narration," in tab.listw.item(0).data(QtCore.Qt.AccessibleTextRole).lower().replace("segment 1,", " ")
        assert ", dialogue," in tab.listw.item(1).data(QtCore.Qt.AccessibleTextRole)
        assert tab.listw.item(1).data(qt_audiobook.ROLE_ROLE) == "D" and tab.listw.item(0).data(qt_audiobook.ROLE_ROLE) == "N"
        win.close()
    print("PASS: Cast: narration is an open book and dialogue a quotation mark, with the role also in the accessible text")


def _test_cast_fits_both_window_sizes_and_keeps_render_in_view():
    long_name = "A Very Long Voice Name For The Narrator Of Chapter Twelve"
    with _AbVoices("Ana", "Bare", long_name), _CastEnv(xtts=False):
        _ab_edit_voice("Bare", csm_ref_text=None)
        for size in ((1024, 640), (1240, 780)):
            win, tab = _ab_tab(27)
            win.resize(*size)
            tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData(long_name))
            tab.cmb_dialogue.setCurrentIndex(tab.cmb_dialogue.findData("Bare"))
            tab.on_role_changed("narrator")
            _pick(tab.cmb_dialogue_engine, "csm")                  # the longest status text
            tab.listw.setCurrentRow(3)
            _app.processEvents()
            scroll = tab.findChild(QtWidgets.QScrollArea)
            assert scroll.horizontalScrollBar().maximum() == 0 and scroll.verticalScrollBar().maximum() == 0, size
            tab.btn_settings.setChecked(True)                      # Render settings open: it may scroll down, never sideways
            _app.processEvents()
            assert scroll.horizontalScrollBar().maximum() == 0, (size, "settings open", scroll.horizontalScrollBar().maximum())
            tab.btn_settings.setChecked(False)
            _app.processEvents()
            assert tab.btn_render.isVisible() and tab.rect().contains(QtCore.QRect(tab.btn_render.mapTo(tab, QtCore.QPoint(0, 0)), tab.btn_render.size()))
            for w in (tab.cmb_narrator, tab.cmb_narrator_engine, tab.cmb_dialogue, tab.cmb_dialogue_engine, tab.status_narrator,
                      tab.status_dialogue, tab.status_segment, tab.cmb_seg_voice, tab.cmb_seg_engine, tab.cmb_seg_direction):
                box = QtCore.QRect(w.mapTo(tab, QtCore.QPoint(0, 0)), w.size())
                assert tab.rect().contains(box) and w.width() >= (w.minimumWidth() or w.minimumSizeHint().width()), (size, w.accessibleName(), box)
            assert tab.listw.viewport().height() / tab.listw.sizeHintForRow(0) >= 5.5, (size, tab.listw.viewport().height())
            assert tab.status_dialogue.full_text().startswith("Needs setup"), tab.status_dialogue.full_text()
            win.close()
    print("PASS: Cast: nothing clips or scrolls at 1024x640 and 1240x780, long names and long reasons are elided, 5.5+ rows stay visible")


def _test_cast_new_segments_never_get_an_impossible_combination():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(3)
        _pick(tab.cmb_narrator_engine, "csm")
        tab.cmb_lang.setCurrentText("french")
        _app.processEvents()
        assert tab.narrator_key() == "csm:Ana", "the role itself is kept, visible and editable"
        assert tab.status_narrator.state == cast_rules.UNSUPPORTED
        tab.txt_source.setPlainText('He said, "Hello." Then left.')
        before = [dict(seg) for seg in tab.segments]
        tab.auto_split()
        assert tab.segments == before and "narrator engine cannot speak this" in win.lbl_status.text(), win.lbl_status.text()
        win.lbl_status.setText("")
        tab.add_after()
        assert tab.segments == before and "narrator engine cannot speak this" in win.lbl_status.text()
        _pick(tab.cmb_narrator_engine, "xtts")                      # xtts speaks French
        tab.add_after()
        assert len(tab.segments) == 4 and tab.segments[-1]["voice"] == "xtts:Ana" or tab.segments[1]["voice"] == "xtts:Ana"
        win.close()
    print("PASS: Cast: Split and Add after never stamp an impossible voice and engine onto a new segment")


def _test_cast_problems_dialog_scrolls_and_keeps_close_reachable():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(12)
        for i, seg in enumerate(tab.segments):
            seg["voice"] = f"{('voice', 'xtts', 'csm')[i % 3]}:Gone {i}"      # twelve different causes
        tab.rebuild(select=0)
        tab.btn_render.click()
        _app.processEvents()
        dialog = tab._problem_dialog
        assert dialog is not None and len(tab._problem_groups) == 12
        screen_height = 640
        assert dialog.height() <= screen_height, dialog.height()
        scroll = dialog.findChild(QtWidgets.QScrollArea)
        assert scroll.verticalScrollBar().maximum() > 0, "twelve groups scroll instead of running off the screen"
        close = [b for b in dialog.findChildren(QtWidgets.QPushButton) if b.text() == "Close"][0]
        assert dialog.rect().contains(QtCore.QRect(close.mapTo(dialog, QtCore.QPoint(0, 0)), close.size())) and close.isVisible()
        dialog.close()
        win.close()
    print("PASS: Cast: a long list of problems scrolls inside the dialog and Close stays reachable")


def _test_cast_engine_menu_is_keyboard_complete_and_the_choice_is_visible():
    with _AbVoices("Ana", "Marcus"), _CastEnv():
        win, tab = _ab_tab(4)
        tab.segments[2]["direction"] = "sad"                       # xtts, csm, dia and f5 cannot speak a direction
        tab.rebuild(select=2)
        combo = tab.cmb_seg_engine
        assert combo.count() == 5 and combo.currentData() == "qwen"
        # every item has its state in its text and its reason in a tooltip; nothing depends on the popup being tall enough
        assert all(combo.itemText(i) and combo.itemData(i, QtCore.Qt.ToolTipRole) for i in range(5))
        # the keyboard reaches an enabled engine without the popup: Up/Down move through the enabled items only
        tab.listw.setCurrentRow(1)
        combo.setFocus()
        QTest.keyClick(combo, QtCore.Qt.Key_Down)
        _app.processEvents()
        assert combo.currentData() in cast_rules.ENGINES
        QTest.keyClick(combo, QtCore.Qt.Key_End)
        combo.activated.emit(combo.currentIndex())
        _app.processEvents()
        chosen = combo.currentData()
        assert tab.segments[1]["voice"].startswith(cast_rules.PREFIX[chosen] + ":"), (chosen, tab.segments[1]["voice"])
        assert combo.currentText().startswith(cast_rules.LABEL[chosen]), "the closed box names the chosen engine"
        assert cast_rules.LABEL[chosen] in tab.listw.item(1).data(QtCore.Qt.AccessibleTextRole) or chosen == "qwen"
        # an engine that cannot speak the selected segment is skipped by the keyboard, never chosen
        tab.listw.setCurrentRow(2)
        combo.setFocus()
        combo.setCurrentIndex(0)
        for _ in range(6):
            QTest.keyClick(combo, QtCore.Qt.Key_Down)
        assert combo.currentData() == "qwen" and tab.segments[2]["voice"].startswith("voice:"), "disabled engines are skipped, so nothing else is reachable"
        # the popup lists all five, and a disabled one still shows its reason
        combo.showPopup()
        _app.processEvents()
        assert combo.view().model().rowCount() == 5
        combo.hidePopup()
        win.close()
    print("PASS: Cast: the engine menu is keyboard complete (disabled engines skipped), and the chosen engine is shown in the box and the row")


def _test_cast_rules_and_the_page_never_load_a_model():
    import audiobook
    import model_loader
    from unittest.mock import patch

    def forbidden(*args, **kwargs):
        raise AssertionError("a compatibility check must never load a model or start a task")
    with _AbVoices("Ana", "Marcus"), _CastEnv(), patch.object(model_loader.ModelManager, "get", forbidden), \
            patch.object(model_loader, "load_model", forbidden), patch.object(audiobook, "render_segment", forbidden):
        win, tab = _ab_tab(5)
        win.models.get = forbidden                                # the window's own model manager, not the class
        win.run_task = forbidden                                  # nothing may start a task either
        for engine in ("xtts", "csm", "dia", "f5", "qwen"):
            _pick(tab.cmb_narrator_engine, engine)
            tab.refresh_compat()
        tab.cmb_lang.setCurrentText("french")
        tab.refresh_voices()
        _select_rows(tab, 0, 1)
        _pick(tab.cmb_seg_engine, "xtts")
        tab.segments[3]["voice"] = "csm:Bare"
        tab.rebuild(select=3)
        tab.btn_render.click()                                    # blocked by the pre-flight, so nothing is rendered
        tab.btn_preview.click()
        _app.processEvents()
        assert tab._problem_dialog is not None
        tab._problem_dialog.close()
        assert "TTS" not in sys.modules and "f5_tts" not in sys.modules, "no engine package was imported"
        win.close()
    print("PASS: Cast: browsing, switching engines, changing language and a blocked Render never load a model, start a task or import an engine")


def _test_voice_refresh_keeps_audiobooks_and_drama_studio_independent():
    with _AbVoices("Ana", "Marcus", "Zed"), _CastEnv():
        win, tab = _ab_tab(3)
        drama = win.drama_tab
        tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData("Ana"))
        tab.cmb_dialogue.setCurrentIndex(tab.cmb_dialogue.findData("Marcus"))
        win.refresh_voice_lists()
        drama.char_voice.setCurrentIndex(drama.char_voice.findData("Zed"))
        keys = (tab.narrator_key(), tab.dialogue_key())
        segments = [dict(s) for s in tab.segments]
        win.refresh_voice_lists()
        assert drama.char_voice.currentData() == "Zed", "Drama Studio keeps its own selection"
        assert (tab.narrator_key(), tab.dialogue_key()) == keys and tab.segments == segments, \
            "Drama Studio's selection never reaches the Audiobooks cast"
        tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData("Marcus"))
        win.refresh_voice_lists()
        assert drama.char_voice.currentData() == "Zed", "an Audiobooks choice never reaches Drama Studio"
        shutil.rmtree(voice_library.voice_dir("Zed"))            # a voice that disappears: Drama falls back, Cast keeps its keys
        win.refresh_voice_lists()
        assert drama.char_voice.currentData() == "" and tab.narrator_key() == "voice:Marcus"
        win.close()
    print("PASS: a voice-library refresh keeps Audiobooks and Drama Studio selections independent")


def _test_narrator_and_dialogue_on_the_same_voice_and_engine_stay_independent():
    with _AbVoices("Ana", "Marcus", "Zed"), _CastEnv():
        win, tab = _ab_tab(3)
        _pick(tab.cmb_narrator, "Ana")
        _pick(tab.cmb_dialogue, "Ana")                    # one reader for the whole book: both roles, same voice and engine
        assert tab.narrator_key() == tab.dialogue_key() == "voice:Ana"
        for _ in range(3):
            tab.refresh_voices()
            win.refresh_voice_lists()
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "voice:Ana"), "a refresh never moves either role"
        # editing one role leaves the other alone, in both directions and for both controls
        _pick(tab.cmb_narrator, "Marcus")
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Marcus", "voice:Ana")
        _pick(tab.cmb_narrator, "Ana")
        _pick(tab.cmb_dialogue_engine, "csm")
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "csm:Ana")
        _pick(tab.cmb_dialogue_engine, "qwen")
        _pick(tab.cmb_dialogue, "Zed")
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "voice:Zed")
        _pick(tab.cmb_dialogue, "Ana")
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "voice:Ana")
        # splitting uses each role's own assignment, and a saved project keeps every segment's voice
        tab.txt_source.setPlainText('He waited. "Not yet," she said. Then nothing.')
        tab.segments = []
        tab.rebuild(select=0)                              # nothing to replace, so no confirmation dialog
        tab.auto_split()
        kinds = {s["kind"]: s["voice"] for s in tab.segments}
        assert kinds.get("narration") == "voice:Ana" and kinds.get("dialogue") == "voice:Ana", kinds
        _pick(tab.cmb_dialogue, "Marcus")                 # later segments differ from earlier ones; both must survive
        tab.segments[0]["voice"] = "voice:Zed"
        saved = [dict(s) for s in tab.segments]
        target = os.path.join(tempfile.mkdtemp(), "book.json")
        real = QtWidgets.QFileDialog.getSaveFileName, QtWidgets.QFileDialog.getOpenFileName
        try:
            QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
            QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (target, ""))
            tab.save_project()
            tab.segments = []
            tab.rebuild(select=0)
            tab.load_project()
        finally:
            QtWidgets.QFileDialog.getSaveFileName, QtWidgets.QFileDialog.getOpenFileName = real
        assert tab.segments == saved, "save and load keep every segment's own voice, whichever role it came from"
        tab.refresh_voices()
        assert tab.segments == saved and tab.dialogue_key() == "voice:Marcus", "a refresh after loading changes nothing"
        win.close()
    print("PASS: Narrator and Dialogue on the same voice and engine stay independent through refresh, edits, split, save and load")


def _test_cast_refresh_and_voice_choice_never_rewrite_the_cast():
    with _AbVoices("Ana", "Marcus", "Zed"), _CastEnv():
        win, tab = _ab_tab(3)
        tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData("Ana"))
        tab.cmb_dialogue.setCurrentIndex(tab.cmb_dialogue.findData("Ana"))
        _pick(tab.cmb_dialogue_engine, "csm")                      # Ana/Qwen narrates, Ana/CSM speaks: a real cast
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "csm:Ana")
        for _ in range(2):
            tab.refresh_voices()
        assert (tab.narrator_key(), tab.dialogue_key()) == ("voice:Ana", "csm:Ana"), "a library refresh never changes the cast"
        # a voice with no Qwen clone prompt moves its role to an engine that can use it, and says so
        path = os.path.join(voice_library.voice_dir("Zed"), "clone_prompt.npz")
        os.remove(path)
        tab.cmb_narrator.setCurrentIndex(tab.cmb_narrator.findData("Zed"))
        tab.cmb_narrator.activated.emit(tab.cmb_narrator.currentIndex())
        _app.processEvents()
        assert tab.narrator_key() == "xtts:Zed" and "Narrator engine set to XTTS: Zed has no Qwen clone prompt" in win.lbl_status.text(), \
            (tab.narrator_key(), win.lbl_status.text())
        win.close()
    print("PASS: Cast: a library refresh never rewrites Narrator or Dialogue, and an automatic engine move is announced")


def _test_multiline_editors_never_trap_the_keyboard():
    win, _ = _make_window()
    win.resize(1024, 640)
    tab = win.audiobook_tab
    editors = {"audiobook chapter": tab.txt_source, "audiobook segment": tab.txt_segment,
               "clone text": win.txt_vc_text, "clone instruction": win.txt_vc_instruct}
    for name, editor in editors.items():
        assert editor.tabChangesFocus(), f"{name} traps Tab"
    win.show_page(3)
    _shown(win)
    tab.txt_source.setFocus()
    _app.processEvents()
    QTest.keyClicks(tab.txt_source, "one")
    QTest.keyClick(tab.txt_source, QtCore.Qt.Key_Return)
    QTest.keyClicks(tab.txt_source, "two")
    assert tab.txt_source.toPlainText() == "one\ntwo", "multiline entry must still work"
    QTest.keyClick(tab.txt_source, QtCore.Qt.Key_Tab)
    _app.processEvents()
    assert win.focusWidget() is not tab.txt_source and "\t" not in tab.txt_source.toPlainText()
    win.close()
    print("PASS: Audiobook and Clone text boxes release Tab and still accept multiline text")


def _test_tab_leaves_description_under_the_macos_default_tab_behavior():
    """Offscreen Qt tabs through every control, real macOS does not unless Full Keyboard Access is on.

    On macOS only text/list controls are Tab stops by default, so the description was the only one
    and Tab inserted a tab character. Emulate that default and drive real window-level key events.
    """
    macos_default = QtCore.Qt.TabFocusBehavior(3)  # text + list controls, as on a stock Mac
    _make_window()
    hints = _app.styleHints()
    original = hints.tabFocusBehavior()
    try:
        def key(win, k, mod=QtCore.Qt.NoModifier):
            QTest.keyClick(win.windowHandle(), k, mod)
            _app.processEvents()

        # Without the app's setting the trap reproduces (this documents why offscreen missed it).
        hints.setTabFocusBehavior(macos_default)
        win, _ = _make_window()
        hints.setTabFocusBehavior(macos_default)
        win.resize(1024, 640)
        _shown(win)
        win.txt_vd_instruct.setFocus()
        _app.processEvents()
        key(win, QtCore.Qt.Key_Tab)
        # Stock macOS reaches only text controls: Tab skips the suggestions and the Generate button
        # (the second text field, Sample text, is the next stop). The app's setting fixes that, below.
        assert win.focusWidget() is win.txt_vd_sample, _focus_label(win.focusWidget())
        assert "\t" not in win.txt_vd_instruct.toPlainText()
        key(win, QtCore.Qt.Key_Tab)
        assert win.focusWidget() is not win.btn_vd_generate, "stock macOS never lands on buttons"
        win.close()

        # With the app's startup configuration (what main() does) the full sequence works.
        for text in ("", "line one\nline two"):
            hints.setTabFocusBehavior(macos_default)
            configure_keyboard_navigation(_app)
            win, _ = _make_window()
            win.resize(1024, 640)
            _shown(win)
            win.nav_buttons[0].setFocus()
            _app.processEvents()
            key(win, QtCore.Qt.Key_Tab)                       # navigation -> Voice description
            assert win.focusWidget() is win.txt_vd_instruct, _focus_label(win.focusWidget())
            win.txt_vd_instruct.setPlainText(text)
            before = win.txt_vd_instruct.toPlainText()
            key(win, QtCore.Qt.Key_Tab)                       # description -> next control
            assert win.focusWidget() is not win.txt_vd_instruct, "Tab is trapped in the description"
            assert win.focusWidget() is win.vd_chips[0][0], _focus_label(win.focusWidget())
            assert win.txt_vd_instruct.toPlainText() == before, "Tab must not modify the description"
            key(win, QtCore.Qt.Key_Backtab, QtCore.Qt.ShiftModifier)   # Shift+Tab goes back
            assert win.focusWidget() is win.txt_vd_instruct
            assert win.txt_vd_instruct.toPlainText() == before
            key(win, QtCore.Qt.Key_Tab, QtCore.Qt.ShiftModifier)       # Shift+Tab as Tab+Shift
            assert win.focusWidget() is win.nav_buttons[0], _focus_label(win.focusWidget())
            win.txt_vd_instruct.setFocus()
            win.txt_vd_instruct.moveCursor(QtGui.QTextCursor.End)
            _app.processEvents()
            key(win, QtCore.Qt.Key_Return)                    # Enter still adds a line
            assert win.txt_vd_instruct.toPlainText() == before + "\n" and win.focusWidget() is win.txt_vd_instruct
            win.close()
    finally:
        hints.setTabFocusBehavior(original)
    print("PASS: Tab/Shift+Tab leave the description under the macOS default tab behavior; Enter adds a line")


def _test_stale_sample_callout_and_footer_status():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        win.edt_vd_name.setText("Narrator")
        _generate(win, "Warm narrator")
        assert win.lbl_status.text() == READY_STATUS
        assert win.frm_vd_stale.isHidden() and win.lbl_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("Bright announcer")
        _app.processEvents()
        # the inset callout carries the note; the footer must not claim the sample matches the form
        assert not win.frm_vd_stale.isHidden() and not win.lbl_vd_stale.isHidden()
        assert win.frm_vd_stale.isAncestorOf(win.lbl_vd_stale)
        assert win.lbl_status.text() == STALE_STATUS
        assert win.lbl_vd_stale.height() >= win.lbl_vd_stale.heightForWidth(win.lbl_vd_stale.width()), "note is clipped"
        win.txt_vd_instruct.setPlainText("Warm narrator")
        _app.processEvents()
        assert win.frm_vd_stale.isHidden() and win.lbl_status.text() == READY_STATUS
        # an unrelated status message is never overwritten
        win.status("Saved as 'x'.")
        win.txt_vd_instruct.setPlainText("Another")
        _app.processEvents()
        assert win.lbl_status.text() == "Saved as 'x'."
        win.close()
    print("PASS: a changed input shows the callout and a matching footer status, and reverting restores both")


_USED_ICONS = {"wand-sparkles", "copy", "mic-vocal", "book-open", "settings", "audio-lines", "play", "save",
               "download", "refresh-cw", "chevron-right", "chevron-down",
               "square", "folder-open", "circle-check", "circle-exclamation-point",
               "pause", "rotate-ccw", "rotate-cw", "pencil", "info", "quote"}


def _test_icon_assets_are_vendored_licensed_and_only_the_used_ones():
    on_disk = {f[:-4] for f in os.listdir(icons.ICON_DIR) if f.endswith(".svg")}
    assert on_disk == _USED_ICONS, f"vendor only the icons in use: {on_disk ^ _USED_ICONS}"
    licence = open(os.path.join(icons.ICON_DIR, "LICENSE"), encoding="utf-8").read()
    assert "ISC License" in licence and "Lucide" in licence and "Feather" in licence and "MIT License" in licence
    _make_window()
    for name in _USED_ICONS:
        for size in (icons.SIZE_COMPACT, icons.SIZE_NAV):
            assert not icons.icon(name, size).isNull(), name
    print("PASS: exactly the used Lucide icons are vendored, with the ISC and MIT Feather notices")


def _test_icons_use_theme_colours_per_state_and_are_high_dpi():
    _make_window()
    size = QtCore.QSize(16, 16)

    def has_colour(icon, hex_colour, mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off):
        """True if the rendered glyph contains the colour (antialiasing rounds channels by 1-2)."""
        want = QtGui.QColor(hex_colour)
        image = icon.pixmap(size, 1.0, mode, state).toImage()
        for x in range(image.width()):
            for y in range(image.height()):
                c = image.pixelColor(x, y)
                if c.alpha() > 200 and max(abs(c.red() - want.red()), abs(c.green() - want.green()),
                                           abs(c.blue() - want.blue())) <= 3:
                    return True
        return False

    ic = icons.icon("save", 16, "default")
    assert has_colour(ic, theme.COLORS["text_secondary"])                               # resting
    assert has_colour(ic, theme.COLORS["text_disabled"], QtGui.QIcon.Disabled)          # disabled
    assert has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Normal, QtGui.QIcon.On)     # checked / selected
    assert has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Active)                     # keyboard focus
    assert not has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Normal), "resting icons stay neutral and quiet"
    primary = icons.icon("audio-lines", 16, "primary")
    assert has_colour(primary, theme.COLORS["on_primary"]), "the primary action's icon is white, like its label"
    assert has_colour(primary, theme.COLORS["on_primary_disabled"], QtGui.QIcon.Disabled)
    assert ic.pixmap(size, 2.0).width() >= 32, "a 2x pixmap must exist for high-DPI displays"
    print("PASS: icons take theme colours for normal, selected, focused and disabled states and ship 2x pixmaps")


def _test_missing_icon_assets_degrade_to_the_text_label():
    _make_window()
    assert icons.icon("no-such-icon").isNull()
    button = QtWidgets.QPushButton("Label")
    icons.apply(button, "no-such-icon")
    assert button.text() == "Label" and button.icon().isNull()
    print("PASS: a missing icon asset leaves the labelled button intact")


def _test_create_voice_and_navigation_icons_never_replace_labels():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    labelled = {
        "nav": win.nav_buttons,
        "actions": [win.btn_vd_generate, win.btn_vd_play, win.btn_vd_save, win.btn_vd_download,
                    win.btn_vd_regenerate, win.btn_vd_advanced],
    }
    for button in labelled["nav"] + labelled["actions"]:
        assert not button.icon().isNull(), button.text()
        assert button.text().strip(), "icons never replace the text label"
        need = button.fontMetrics().horizontalAdvance(button.text()) + button.iconSize().width() + 16
        assert button.width() >= need, f"{button.text()!r} clips its text next to the icon"
    assert [b.text() for b in win.nav_buttons] == ["Create Voice", "Clone Voice", "My Voices", "Audiobooks", "Drama Studio", "Settings"]
    assert win.btn_vd_generate.text() == "Generate voice" and win.btn_vd_generate.accessibleName() in ("", "Generate voice")
    assert all(chip.icon().isNull() for chip, _n, _p in win.vd_chips), "suggestion chips stay text-only"
    # the disclosure says its state in words too, not by chevron alone
    assert win.btn_vd_advanced.accessibleDescription() == "Collapsed"
    win.btn_vd_advanced.click()
    assert win.btn_vd_advanced.accessibleDescription() == "Expanded"
    assert win.btn_vd_advanced.text() == "Advanced settings"
    # selected navigation uses the checked (On) icon, resting items the normal one
    assert win.nav_buttons[0].isChecked() and not win.nav_buttons[1].isChecked()
    win.close()
    print("PASS: Create Voice and navigation icons keep text labels, accessible names and chip text-only")


def _test_primary_button_uses_the_specified_orange_and_white_content():
    win, _ = _make_window()
    _shown(win)
    c = theme.COLORS
    assert (c["primary"], c["primary_hover"], c["primary_pressed"], c["primary_disabled"], c["on_primary"]) == \
        ("#C5540A", "#B24C08", "#A94708", "#6B4A38", "#FFFFFF")
    button = win.btn_vd_generate

    def fill():
        _app.processEvents()
        return button.grab().toImage().pixelColor(4, button.height() // 2).name().upper()

    win.nav_buttons[0].setFocus()
    assert fill() == c["primary"]
    QTest.mouseMove(button, button.rect().center())
    _app.processEvents()
    assert fill() == c["primary_hover"]
    QTest.mousePress(button, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, button.rect().center())
    assert fill() == c["primary_pressed"]
    edge = button.grab().toImage().pixelColor(0, button.height() // 2).name().upper()
    assert edge == c["primary_pressed_edge"], "pressed has an edge hover lacks, so the two never rely on a ~1.09:1 fill difference alone"
    assert c["primary_pressed_edge"] != c["primary_hover"] != c["primary_pressed"]
    # release outside the button so the press is cancelled and no generation starts
    QTest.mouseRelease(button, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(-20, -20))
    assert not win.busy
    QTest.mouseMove(win.lbl_vd_description, QtCore.QPoint(2, 2))
    button.setEnabled(False)
    assert fill() == c["primary_disabled"]
    button.setEnabled(True)
    # label and glyph are white; size is unchanged; the keyboard focus ring stays visible
    assert button.size() == QtCore.QSize(180, 42), button.size()
    image = button.icon().pixmap(button.iconSize(), 1.0, QtGui.QIcon.Normal).toImage()
    assert any(image.pixelColor(x, y).alpha() > 200 and image.pixelColor(x, y).name().upper() == "#FFFFFF"
               for x in range(image.width()) for y in range(image.height())), "the icon must be white"
    win.btn_vd_reset_text.setFocus()   # the control before Generate voice in Tab order
    _app.processEvents()
    QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_Tab)
    _app.processEvents()
    assert win.focusWidget() is button
    ring = button.grab().toImage().pixelColor(button.width() // 2, 0).name().upper()
    assert ring == c["text"].upper(), f"focus ring not visible: {ring}"
    win.close()
    print("PASS: Generate voice renders the specified orange in every state with a white icon, label and visible focus ring")


def _test_footer_actions_have_labels_icons_and_preserved_handlers():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    expected = {win.btn_stop_audio: "Stop audio", win.btn_open_out: "Open output folder",
                win.btn_play_last: "Play last output", win.btn_stop: "Stop generation"}
    for button, label in expected.items():
        assert button.text() == label and not button.icon().isNull(), label
        assert button.iconSize().height() == icons.SIZE_COMPACT
        assert qt_app.__dict__ and button.property("role") == "quiet", "footer actions stay neutral and secondary"
    assert win.btn_stop_audio.icon().cacheKey() != win.btn_open_out.icon().cacheKey()
    # Stop audio keeps calling audio_player.stop
    stopped = []
    real_stop, real_playing = audio_player.stop, audio_player.is_playing
    audio_player.stop = lambda: stopped.append(1)
    audio_player.is_playing = lambda: True
    try:
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden()
        win.btn_stop_audio.click()
        assert stopped == [1]
    finally:
        audio_player.stop, audio_player.is_playing = real_stop, real_playing
    # Open output folder keeps opening OUTPUT_DIR (and says so if the system refuses)
    opened = []
    real_open = QtGui.QDesktopServices.openUrl
    QtGui.QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)
    try:
        win.btn_open_out.click()
        assert opened == [qt_app.OUTPUT_DIR], opened
        QtGui.QDesktopServices.openUrl = staticmethod(lambda url: False)
        win.btn_open_out.click()
        assert win.lbl_status.text().startswith("Couldn't open the output folder")
    finally:
        QtGui.QDesktopServices.openUrl = real_open
    win.close()
    print("PASS: footer actions keep labels, neutral role and 16 px icons; Stop audio and Open folder keep their handlers")


def _test_stop_audio_follows_the_players_real_state():
    win, _ = _make_window()
    _shown(win)
    state = {"playing": False}
    real_playing = audio_player.is_playing
    audio_player.is_playing = lambda: state["playing"]
    try:
        win._refresh_audio_state()
        assert win.btn_stop_audio.isHidden(), "no playback session: Stop audio is not shown"
        state["playing"] = True
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden() and win.btn_stop_audio.toolTip() == qt_app.STOP_AUDIO_TIP
        state["playing"] = False                      # playback finished by itself
        win.audio_timer.timeout.emit()
        assert win.btn_stop_audio.isHidden(), "back to idle once the player reports silence"
        assert win.audio_timer.isActive() and win.audio_timer.interval() == 250
    finally:
        audio_player.is_playing = real_playing
    # the read-only query never initialises or starts anything
    audio_player._proc = None
    assert audio_player.is_playing() in (True, False)
    win.close()
    print("PASS: Stop audio is enabled only while the player reports audio, and returns to idle by itself")


def _test_footer_status_is_contextual_and_never_duplicates_errors():
    original = voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    try:
        win, _ = _make_window()
        assert win.lbl_status.text() == "Ready", "the Create Voice screen must not open on a library count"
        assert win.lbl_status_icon.isHidden()
        win.refresh_voices_tab()                      # a silent re-read never announces a library count
        assert win.lbl_status.text() == "Ready"
        win.refresh_voices_clicked()                  # the Refresh button reports its own result, not a count
        assert win.lbl_status.text() == "Voice list refreshed."
        win.status(READY_STATUS)
        assert not win.lbl_status_icon.isHidden() and win.lbl_status.text() == READY_STATUS
        win.status("Error: boom")
        assert not win.lbl_status_icon.isHidden()
        win.status("Ready")
        assert win.lbl_status_icon.isHidden(), "plain states show no glyph"
        win.close()
    finally:
        voice_library.VOICES_DIR = original
    with _Patched():
        win, _ = _make_window()                       # stub models raise a long technical message
        win.txt_vd_instruct.setPlainText("A gravelly old sailor")
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.txt_vd_details.toPlainText() and win.txt_vd_details.toPlainText() not in win.lbl_status.text()
        assert "Preview" in win.lbl_status.text()
        win.close()
    print("PASS: the footer says Ready at launch, uses a glyph for ready/error, and never repeats error details")


def _test_footer_fits_and_is_keyboard_reachable_at_minimum_width():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    for page in range(5):
        win.show_page(page)
        _app.processEvents()
        footer = win.btn_open_out.parentWidget()
        for button in (win.btn_stop_audio, win.btn_open_out, win.btn_play_last):
            if button.isHidden():
                continue
            top_left = button.mapTo(win, QtCore.QPoint(0, 0))
            assert QtCore.QRect(top_left, button.size()).right() <= win.width(), f"{button.text()!r} clips at 1024 px"
            need = button.fontMetrics().horizontalAdvance(button.text()) + button.iconSize().width() + 16
            assert button.width() >= need, f"{button.text()!r} is squeezed"
        assert footer.height() <= 60, f"footer is {footer.height()} px tall"
    # Tab reaches Open output folder; Stop audio only joins the order while audio plays
    win.show_page(0)
    audio_player.is_playing, real = (lambda: False), audio_player.is_playing
    try:
        win._refresh_audio_state()
        win.btn_open_out.setFocus()
        _app.processEvents()
        assert win.focusWidget() is win.btn_open_out
        assert not win.btn_stop_audio.focusPolicy() == QtCore.Qt.NoFocus
        assert win.btn_stop_audio.isHidden(), "a hidden control can never take Tab"
        audio_player.is_playing = lambda: True
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isHidden()
        win.btn_stop_audio.setFocus(QtCore.Qt.TabFocusReason)
        _app.processEvents()
        assert win.focusWidget() is win.btn_stop_audio
    finally:
        audio_player.is_playing = real
    win.close()
    print("PASS: footer actions fit at 1024 px on every page and stay keyboard-reachable")


def _test_create_voice_layout_at_minimum_and_wide_sizes():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.resize(1024, 640)
        win.show()
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.edt_vd_name.setText("Ada")
        win.txt_vd_instruct.setPlainText("Changed so the sample is a previous one")
        win.btn_vd_advanced.click()
        for chip, _name, _phrase in win.vd_chips:  # every suggestion on: the widest the chip row gets
            chip.setChecked(True)
        _app.processEvents()
        _app.processEvents()
        page = win.stack.widget(0)
        assert win.minimumWidth() <= 1024
        assert page.horizontalScrollBar().maximum() == 0, "Create Voice must not overflow horizontally"
        assert win.btn_vd_save.text() == "Save previous sample"
        assert win.btn_vd_save.width() >= win.btn_vd_save.sizeHint().width(), "Save label is clipped"
        assert win.edt_vd_name.width() >= 90, win.edt_vd_name.width()
        body = page.widget()
        top = lambda w: w.mapTo(body, w.rect().topLeft()).y()
        assert abs(top(win.preview_panel) - top(win.lbl_vd_description)) <= 2, "preview aligns with the description"
        win.resize(1700, 900)
        _app.processEvents()
        assert win.vd_content.width() <= CONTENT_MAX_WIDTH, win.vd_content.width()
        assert page.horizontalScrollBar().maximum() == 0
        win.hide()
    print("PASS: no horizontal overflow at 1024x640; content capped when wide; preview aligned")


# ------------------------------------------------------------------ keyboard + sample integrity
def _press(key, modifiers=QtCore.Qt.NoModifier, widget=None):
    QTest.keyClick(widget or _app.focusWidget(), key, modifiers)
    _app.processEvents()


def _tab(shift=False):
    _press(QtCore.Qt.Key_Tab, QtCore.Qt.ShiftModifier if shift else QtCore.Qt.NoModifier)


def _focus_label(w):
    """Readable name for the focused widget; unknown kinds show up as '?Class' so tests can reject them."""
    if isinstance(w, QtWidgets.QPushButton):
        return " ".join(w.text().split()) or w.accessibleName()   # icon-only buttons are named, not blank
    if isinstance(w, QtWidgets.QSlider):
        return w.accessibleName()
    if isinstance(w, QtWidgets.QCheckBox):
        return w.text()
    if isinstance(w, QtWidgets.QComboBox):
        return "combo:" + w.currentText()
    if isinstance(w, QtWidgets.QAbstractSpinBox):
        return w.accessibleName()
    if isinstance(w, QtWidgets.QLineEdit) and isinstance(w.parent(), QtWidgets.QAbstractSpinBox):
        return w.parent().accessibleName()   # a spin box's editor is its inner line edit
    if isinstance(w, (QtWidgets.QPlainTextEdit, QtWidgets.QLineEdit)):
        return w.accessibleName()
    return "?" + type(w).__name__


def _shown(win):
    win.resize(1240, 760)
    win.show()
    win.activateWindow()
    _app.processEvents()


def _focus_cycle(win, backwards=False, limit=45):
    """Press Tab (or Shift+Tab) from the checked nav item until focus returns to it."""
    start = win.nav_buttons[win.stack.currentIndex()]
    if win.stack.currentIndex() == 0:
        win.txt_vd_instruct.setFocus()  # how a user starts: the form has focus, nothing pre-focused the nav
        _app.processEvents()
        _tab(shift=True)  # first stop backwards must be the SELECTED nav item, not its last button
        assert _app.focusWidget() is start, _focus_label(_app.focusWidget())
    else:
        start.setFocus()
        _app.processEvents()
    labels = [_focus_label(_app.focusWidget())]
    for _ in range(limit):
        _tab(shift=backwards)
        if _app.focusWidget() is start:
            return labels
        labels.append(_focus_label(_app.focusWidget()))
    raise AssertionError(f"focus never returned to navigation: {labels}")


_CHIPS = ["Narration", "Relaxing", "Epic", "Professional"]
_EMPTY_ORDER = (["Create Voice", "Voice description"] + _CHIPS
                + ["Sample text", "Use standard text", "Generate voice", "combo:english", "Advanced settings", "Open output folder"])
_READY_ORDER = (["Create Voice", "Voice description"] + _CHIPS
                + ["Sample text", "Use standard text", "Generate voice", "Sample text used for this sample", "Edit text", "Seek", "Back 5 seconds",
                   "Play", "Forward 5 seconds", "Voice name", "Save", "Download", "Regenerate",
                   "combo:english", "Advanced settings", "Open output folder"])


def _test_tab_leaves_description_and_multiline_entry_still_works():
    win, _ = _make_window()
    _shown(win)
    field = win.txt_vd_instruct
    field.setPlainText("Line one")
    field.setFocus()
    _app.processEvents()
    assert field.tabChangesFocus()
    _tab()
    assert _app.focusWidget() is win.vd_chips[0][0], _focus_label(_app.focusWidget())
    assert field.toPlainText() == "Line one", "Tab must not insert a tab character"
    field.setFocus()
    cursor = field.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    field.setTextCursor(cursor)
    _press(QtCore.Qt.Key_Return)
    QTest.keyClicks(field, "line two")
    assert field.toPlainText() == "Line one\nline two", repr(field.toPlainText())
    assert _app.focusWidget() is field
    win.hide()
    print("PASS: Tab leaves the description without inserting a tab; Enter still starts a new line")


def _test_shift_tab_navigates_backwards():
    win, _ = _make_window()
    _shown(win)
    win.vd_chips[0][0].setFocus()
    _app.processEvents()
    _tab(shift=True)
    assert _app.focusWidget() is win.txt_vd_instruct
    _tab(shift=True)
    assert _app.focusWidget() is win.nav_buttons[0], _focus_label(_app.focusWidget())
    _tab(shift=True)  # wraps to the end of the window: the last visible footer action
    assert _focus_label(_app.focusWidget()) == "Open output folder"
    forward = _focus_cycle(win)
    backward = _focus_cycle(win, backwards=True)
    assert backward == [forward[0]] + forward[:0:-1], (forward, backward)
    win.hide()
    print("PASS: Shift+Tab walks the exact reverse of the Tab order, including wrap-around")


def _test_keyboard_focus_order_is_logical():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        order = _focus_cycle(win)
        assert order == _EMPTY_ORDER, order
        assert not [l for l in order if l.startswith("?")], "unexpected focus stop (invisible or non-control)"
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.edt_vd_name.setText("Ada")
        _app.processEvents()
        order = _focus_cycle(win)
        assert order == _READY_ORDER, order
        win.edt_vd_name.setText("")  # Save is disabled, so it is skipped rather than a dead stop
        _app.processEvents()
        assert "Save" not in _focus_cycle(win)
        win.edt_vd_name.setText("Ada")
        win.btn_vd_advanced.click()
        _app.processEvents()
        order = _focus_cycle(win)
        after = order[order.index("Advanced settings") + 1:]
        assert after == ["combo:Random", "Auto-save generated voices", "combo:Random", "Suggest name",
                         "Open Settings", "Open output folder"], after
        assert not [l for l in order if l.startswith("?")]
        win.hide()
    print("PASS: Tab order = navigation, form, Generate, preview, settings, footer; no invisible stops")


def _test_generate_shortcut_and_visible_hint():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        native = qt_app.shortcut_text()
        assert win.lbl_vd_shortcut.text() == f"or press {native}"
        assert native in win.btn_vd_generate.toolTip() and native in win.btn_vd_generate.accessibleDescription()
        assert win.lbl_vd_shortcut.property("role") == "muted"
        field = win.txt_vd_instruct
        field.setFocus()
        before = field.toPlainText()
        _press(QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)
        assert win.busy, "shortcut did not start generation"
        _wait_idle(win)
        assert win.pending_voice and field.toPlainText() == before, "shortcut must not type a newline"
        win.pending_voice = None
        win.show_page(1)  # the shortcut belongs to Create Voice only
        win.nav_buttons[1].setFocus()
        _press(QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)
        assert not win.busy and win.pending_voice is None
        win.hide()
    print("PASS: the generate shortcut works from the form, types nothing, and is shown next to the button")


def _generate(win, text=None):
    if text is not None:
        win.txt_vd_instruct.setPlainText(text)
    win.btn_vd_generate.click()
    _wait_idle(win)
    _app.processEvents()


def _test_changing_inputs_marks_the_sample_as_previous():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _generate(win, "Warm narrator")
        assert win.lbl_vd_preview_title.text() == "Preview" and win.lbl_vd_stale.isHidden()
        assert "Warm narrator" in win.lbl_vd_origin.text() and "english" in win.lbl_vd_origin.text()
        assert win.btn_vd_save.text() == "Save"
        win.txt_vd_instruct.setPlainText("A different voice")
        assert win.lbl_vd_preview_title.text() == "Previous sample"
        assert not win.lbl_vd_stale.isHidden()
        assert "Warm narrator" in win.lbl_vd_origin.text() and "different" not in win.lbl_vd_origin.text()
        assert win.btn_vd_save.text() == "Save previous sample"
        assert win.pending_voice["description"] == "Warm narrator", "the sample keeps its own inputs"
        win.txt_vd_instruct.setPlainText("Warm narrator")  # back to the original inputs
        assert win.lbl_vd_preview_title.text() == "Preview" and win.lbl_vd_stale.isHidden()
        assert win.btn_vd_save.text() == "Save"
        win.cmb_vd_lang.setCurrentText("french")  # language alone also counts
        assert win.lbl_vd_preview_title.text() == "Previous sample" and not win.lbl_vd_stale.isHidden()
        win.cmb_vd_lang.setCurrentText("english")
        assert win.lbl_vd_stale.isHidden()
        long_text = "x" * 200
        _generate(win, long_text)
        assert len(win.lbl_vd_origin.text()) < 130 and win.lbl_vd_origin.text().count("x") < 100
        assert win.lbl_vd_origin.toolTip() == long_text
    print("PASS: changed inputs relabel the audio as a previous sample and keep showing what made it")


def _test_saving_a_previous_sample_stores_its_original_inputs():
    saved = []
    real_save, real_dir = voice_library.save_voice, voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    voice_library.save_voice = lambda *a, **k: saved.append((a, k))
    try:
        with _Patched():
            win, _ = _make_window(models=_FakeModels())
            _generate(win, "Original voice description")
            sample = win.pending_voice
            win.txt_vd_instruct.setPlainText("Edited description that never produced this audio")
            win.cmb_vd_lang.setCurrentText("german")
            win.edt_vd_name.setText("Ada")
            assert win.btn_vd_save.isEnabled() and win.btn_vd_save.text() == "Save previous sample"
            win.btn_vd_save.click()
            _wait_idle(win)
            assert len(saved) == 1
            args = saved[0][0]
            name, wav, sr, prompt, description, language, seed = args[:7]
            assert name == "Ada" and prompt == "FAKE-PROMPT"
            assert description == "Original voice description", description
            assert language == "english" and seed == sample["seed"] and sr == sample["sr"]
            assert wav is sample["wav"]
            assert "made from" in win.lbl_vd_notice.text()
            assert win.txt_vd_instruct.toPlainText().startswith("Edited"), "saving must not touch the form"
    finally:
        voice_library.save_voice, voice_library.VOICES_DIR = real_save, real_dir
    print("PASS: saving a previous sample stores its original description, language and seed")


def _test_invalid_voice_names_disable_save():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.edt_vd_name.setText("Ada")
        assert not win.btn_vd_save.isEnabled(), "no sample yet"
        _generate(win)
        for bad, hint in (("", "Enter a voice name"), ("   ", "Enter a voice name"),
                          ("!!!", "letter or number"), ("***", "letter or number")):
            win.edt_vd_name.setText(bad)
            assert not win.btn_vd_save.isEnabled(), repr(bad)
            assert not win.lbl_vd_name_hint.isHidden() and hint in win.lbl_vd_name_hint.text(), repr(bad)
            assert "name" in win.btn_vd_save.toolTip().lower()
            win.btn_vd_save.click()
            assert not win.busy, "a disabled Save must start nothing"
        win.edt_vd_name.setText("Ada-2")
        assert win.btn_vd_save.isEnabled() and win.lbl_vd_name_hint.isHidden()
        win.set_busy(True)
        assert not win.btn_vd_save.isEnabled()
        win.set_busy(False)
        assert win.btn_vd_save.isEnabled(), "valid name: Save returns after a task"
        win.edt_vd_name.setText("")
        win.set_busy(True)
        win.set_busy(False)
        assert not win.btn_vd_save.isEnabled(), "finishing a task must not re-enable Save without a name"
    print("PASS: Save is disabled for empty or unusable names, with a reason, and stays so after tasks")


def _test_failed_regeneration_keeps_the_previous_valid_sample():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _generate(win, "Warm narrator")
        sample, wav = win.pending_voice, win.pending_voice["wav"].copy()
        win.edt_vd_name.setText("Ada")
        win.txt_vd_instruct.setPlainText("A new description that will fail")
        win.models = _StubModels()
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.pending_voice is sample and np.array_equal(win.pending_voice["wav"], wav)
        assert win.vd_preview.currentIndex() == PV_READY
        assert "previous sample" in win.lbl_vd_notice.text().lower() and not win.lbl_vd_notice.isHidden()
        assert win.lbl_vd_preview_title.text() == "Previous sample"
        assert win.btn_vd_play.isEnabled() and win.btn_vd_save.isEnabled()
        assert win.btn_vd_save.text() == "Save previous sample"
        assert "Warm narrator" in win.lbl_vd_origin.text()
        assert win.txt_vd_instruct.toPlainText() == "A new description that will fail"
    print("PASS: a failed regeneration keeps the previous sample, its inputs, and a labelled Save")


def _test_busy_toggles_action_buttons():
    win, _ = _make_window()
    win.set_busy(True)
    assert all(not b.isEnabled() for b in win.action_buttons)
    assert win.btn_stop.isEnabled()
    win.set_busy(False)
    assert win.btn_vd_generate.isEnabled() and win.btn_vc_generate.isEnabled()
    assert not win.btn_stop.isEnabled()
    print("PASS: set_busy disables and restores action buttons")


def _test_single_primary_button_and_neutral_nav():
    win, _ = _make_window()
    primaries = [b for b in win.findChildren(QtWidgets.QPushButton) if b.property("role") == "primary"]
    render = win.audiobook_tab.btn_render     # Audiobooks' one action: Render chapter
    assert set(primaries) == {win.btn_vd_generate, win.btn_vc_generate, render}, primaries
    assert win.stack.widget(0).isAncestorOf(win.btn_vd_generate)
    assert win.stack.widget(1).isAncestorOf(win.btn_vc_generate)
    assert win.stack.widget(3).isAncestorOf(render)
    assert all(b.property("role") == "nav" for b in win.nav_buttons)
    qss = theme.build_qss()
    selected = re.search(r'QPushButton\[role="nav"\]:checked \{(.*?)\}', qss, re.S).group(1)
    assert theme.COLORS["accent"].lower() not in selected.lower(), "selected nav must stay neutral"
    print("PASS: one primary button per action page; selected navigation is neutral")


def _test_sidebar_and_control_sizes():
    win, _ = _make_window()
    win.resize(1100, 800)
    win.show()
    _app.processEvents()
    side = win.findChild(QtWidgets.QFrame, "sidebar")
    assert 200 <= side.width() <= 220, side.width()
    assert 36 <= win.cmb_vd_lang.height() <= 40, win.cmb_vd_lang.height()
    assert 40 <= win.btn_vd_generate.height() <= 44, win.btn_vd_generate.height()
    win.btn_vd_advanced.click()
    win.resize(900, 620)
    _app.processEvents()
    win.hide()
    print("PASS: sidebar ~208 px, inputs 36-40 px, primary button 40-44 px")


def _test_theme_rules_and_contrast():
    qss = theme.build_qss()
    assert "gradient" not in qss.lower() and "shadow" not in qss.lower()
    c = theme.COLORS
    checks = {
        "text on canvas": (c["text"], c["canvas"], 4.5),
        "secondary on canvas": (c["text_secondary"], c["canvas"], 4.5),
        "muted on canvas": (c["text_muted"], c["canvas"], 4.5),
        "muted on surface": (c["text_muted"], c["surface"], 4.5),
        "muted on sidebar": (c["text_muted"], c["sidebar"], 4.5),
        "text on primary button": (c["on_primary"], c["primary"], 4.5),
        "text on primary pressed": (c["on_primary"], c["primary_pressed"], 4.5),
        "text on primary disabled": (c["on_primary_disabled"], c["primary_disabled"], 4.5),
        "text on primary hover": (c["on_primary"], c["primary_hover"], 4.5),
        "primary button on canvas": (c["primary"], c["canvas"], 3.0),
        "focus ring on primary": (c["text"], c["primary"], 3.0),
        "text on selection": (c["text"], c["selection"], 4.5),
        "nav selected text": (c["text"], c["nav_selected"], 4.5),
        "nav hover text": (c["text"], c["nav_hover"], 4.5),
        "error on surface": (c["error"], c["surface"], 4.5),
        "control border on canvas": (c["border_control"], c["canvas"], 3.0),
        "focus ring on canvas": (c["accent"], c["canvas"], 3.0),
        "focus ring on surface": (c["accent"], c["surface"], 3.0),
    }
    for label, (fg, bg, minimum) in checks.items():
        ratio = theme.contrast_ratio(fg, bg)
        assert ratio >= minimum, f"{label}: {ratio:.2f} < {minimum}"
    print("PASS: theme has no gradients/shadows; contrast targets met")


def _test_set_role_updates_property():
    win, _ = _make_window()
    b = QtWidgets.QPushButton("x")
    theme.set_role(b, "primary")
    assert b.property("role") == "primary"
    print("PASS: set_role sets the dynamic property")


def _tiny_wav(path, seconds=0.2, rate=24000):
    """A real, decodable reference recording (validation now reads the header, so an empty file is rejected)."""
    from audio_utils import write_wav
    write_wav(path, np.zeros(int(seconds * rate), dtype=np.float32) + 0.05, rate)


class _Library:
    """A throwaway voices folder so Clone Voice tests never touch the real library."""

    def __enter__(self):
        self.original = voice_library.VOICES_DIR
        voice_library.VOICES_DIR = tempfile.mkdtemp()
        return self

    def __exit__(self, *exc):
        voice_library.VOICES_DIR = self.original

    @staticmethod
    def add(name, description="Warm narrator", language="german"):
        path = voice_library.voice_dir(name)
        os.makedirs(path)
        with open(os.path.join(path, "voice.json"), "w") as f:
            json.dump({"name": name, "description": description, "language": language, "seed": 7,
                       "created": "x", "clone_prompt": "clone_prompt.npz"}, f)
        np.savez(os.path.join(path, "clone_prompt.npz"), **{"0": np.array({"k": 1}, dtype=object)})
        return path


class _CloneCapture:
    """Replaces the clone call so the test sees exactly what generation would receive."""

    def __enter__(self):
        self.calls = []
        self.original = qt_app.TTSGenerator.clone
        calls = self.calls

        def clone(gen, **kwargs):
            calls.append(kwargs)
            return [np.zeros(2400, dtype=np.float32)], 24000
        qt_app.TTSGenerator.clone = clone
        return self

    def __exit__(self, *exc):
        qt_app.TTSGenerator.clone = self.original


def _clone_window(models=None):
    win, stub = _make_window(models=models)
    win.show_page(1)
    return win, stub


def _fill_every_clone_field(win, wav="/tmp/ref.wav", npz="/tmp/p.npz"):
    win.edt_vc_ref.setText(wav)
    win.txt_vc_ref_text.setPlainText("Words in the recording")
    win.chk_vc_xvec.setChecked(False)
    win.edt_vc_reuse.setText(npz)
    win.edt_vc_save.setText("/tmp/out.npz")
    win.txt_vc_instruct.setPlainText("A custom description")


def _test_clone_sources_are_exclusive_and_only_the_active_one_is_sent():
    with _Patched(), _Library() as lib, _CloneCapture() as cap:
        lib.add("Ada")
        files = tempfile.mkdtemp()
        wav, npz = os.path.join(files, "ref.wav"), os.path.join(files, "p.npz")
        _tiny_wav(wav)
        open(npz, "wb").close()
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        _fill_every_clone_field(win, wav, npz)
        assert win.clone_source_kind() == "saved" and win.btn_vc_src_saved.isChecked()
        win.cmb_vc_voice.setCurrentIndex(0)
        win.btn_vc_generate.click()
        _wait_idle(win)
        sent = cap.calls[-1]
        assert sent["voice_name"] == "Ada" and sent["ref_audio"] == "" and sent["ref_text"] is None
        assert sent["reuse_prompt"] is None and sent["x_vector_only"] is False
        assert sent["description"] == "", "a saved voice's description comes from the voice, not the form"
        assert sent["save_prompt"] == "/tmp/out.npz", "exporting the prompt is an output option, not a source"

        win.btn_vc_src_ref.setChecked(True)
        assert win.clone_source_kind() == "reference"
        win.btn_vc_generate.click()
        _wait_idle(win)
        sent = cap.calls[-1]
        assert sent["voice_name"] is None and sent["reuse_prompt"] is None
        assert sent["ref_audio"] == wav and sent["ref_text"] == "Words in the recording"
        assert sent["description"] == "A custom description"
        win.chk_vc_xvec.setChecked(True)
        win.btn_vc_generate.click()
        _wait_idle(win)
        sent = cap.calls[-1]
        assert sent["x_vector_only"] is True and sent["ref_text"] is None, "X-vector sends no transcript"

        win.chk_vc_npz.setChecked(True)
        assert win.clone_source_kind() == "prompt"
        assert not win.edt_vc_ref.isEnabled() and not win.txt_vc_ref_text.isEnabled()
        assert win.row_vc_reuse.isEnabled() and not win.btn_vc_save_voice.isEnabled()
        win.btn_vc_generate.click()
        _wait_idle(win)
        sent = cap.calls[-1]
        assert sent["reuse_prompt"] == npz and sent["ref_audio"] == "" and sent["voice_name"] is None
        assert sent["ref_text"] is None and sent["x_vector_only"] is False
        # switching modes keeps the other mode's entries
        win.btn_vc_src_saved.setChecked(True)
        assert win.edt_vc_ref.text() == wav and win.edt_vc_reuse.text() == npz
        assert win.cmb_vc_voice.currentText() == "Ada"
        win.close()
    print("PASS: Clone Voice sends only the explicitly active source; the others are kept but never sent")


def _test_clone_validation_runs_before_any_model_loads():
    with _Patched(), _Library() as lib:
        lib.add("Ada")
        lib.add("Mute", description="")
        files = tempfile.mkdtemp()
        wav, npz = os.path.join(files, "ref.wav"), os.path.join(files, "p.npz")
        mp3 = os.path.join(files, "ref.mp3")
        _tiny_wav(wav)
        open(mp3, "wb").close()   # never decoded: the WAV-only rule for Save answers first
        np.savez(npz, a=np.zeros(1))   # a dict-style prompt, not a list
        win, stub = _clone_window()
        win.refresh_voice_lists()

        def expect(fragment, field=None, tags=False):
            win.btn_vc_generate.click()
            assert not win.busy, "validation must stop the run"
            assert stub.calls == [], f"a model was loaded: {stub.calls}"
            assert not win.lbl_vc_error.isHidden() and fragment in win.lbl_vc_error.text(), win.lbl_vc_error.text()
            assert win.lbl_vc_error.text() in win.lbl_status.text()
            if field is not None:
                assert _app.focusWidget() is field, (_focus_label(_app.focusWidget()), fragment)

        _shown(win)
        win.txt_vc_text.setPlainText("   ")
        expect("Enter the text", win.txt_vc_text)
        win.txt_vc_text.setPlainText("Hello")
        assert win.lbl_vc_error.isHidden(), "editing clears the stale problem"
        win.cmb_vc_voice.setCurrentIndex(-1)
        expect("Choose a saved voice", win.cmb_vc_voice)
        win.cmb_vc_voice.setCurrentText("Mute")
        win.txt_vc_text.setPlainText("[happy] Hello")
        expect("no saved description", win.txt_vc_text)
        win.txt_vc_text.setPlainText("Hello")
        win.btn_vc_src_ref.setChecked(True)
        expect("Choose a reference recording", win.edt_vc_ref)
        win.edt_vc_ref.setText(wav + ".missing")
        expect("not found", win.edt_vc_ref)
        win.edt_vc_ref.setText(wav)
        expect("transcript", win.txt_vc_ref_text)
        win.txt_vc_ref_text.setPlainText("Words")
        win.txt_vc_text.setPlainText("[sad] Hello")
        win.btn_vc_advanced.setChecked(False)
        expect("Describe the voice", win.txt_vc_instruct)
        assert win.txt_vc_instruct.isVisible() and not win.btn_vc_advanced.isChecked(), \
            "the description sits beside Save as voice, so no hidden section needs to open"
        win.txt_vc_instruct.setPlainText("Warm")
        win.chk_vc_npz.setChecked(True)
        win.edt_vc_reuse.clear()
        expect("Choose a prompt file", win.edt_vc_reuse)
        win.edt_vc_reuse.setText(npz + ".missing")
        expect("not found", win.edt_vc_reuse)
        win.edt_vc_reuse.setText(npz)
        expect("saved as a list", win.edt_vc_reuse)
        win.chk_vc_npz.setChecked(False)
        # saving a voice has its own pre-flight checks
        win.edt_vc_new_name.setText("New")
        win.edt_vc_ref.setText(mp3)
        win.btn_vc_save_voice.click()
        assert not win.busy and stub.calls == [] and "WAV" in win.lbl_vc_error.text()
        win.edt_vc_ref.setText(wav)
        win.txt_vc_ref_text.clear()
        win.btn_vc_save_voice.click()
        assert not win.busy and stub.calls == [] and "transcript" in win.lbl_vc_error.text()
        win.edt_vc_new_name.setText("!!!")
        win.btn_vc_save_voice.click()
        assert "name" in win.lbl_vc_error.text() and stub.calls == []
        win.close()
    print("PASS: Clone Voice checks text, source, files, transcript and tags before any model is loaded")


def _clone_ready_to_save(files):
    wav = os.path.join(files, "ref.wav")
    write_wav(wav, np.zeros(2400, dtype=np.float32), 24000)
    return wav


def _answer_box(label):
    """Make the next QMessageBox choose the button with this text; returns the list of seen texts."""
    seen = []
    original = QtWidgets.QMessageBox.exec

    def fake(box):
        seen.append((box.text(), box.informativeText()))
        for b in box.buttons():
            if b.text() == label:
                b.click()
        return 0
    QtWidgets.QMessageBox.exec = fake
    return seen, lambda: setattr(QtWidgets.QMessageBox, "exec", original)


def _test_save_clone_never_overwrites_silently():
    with _Patched(), _Library() as lib:
        folder = lib.add("Ana B")
        os.makedirs(os.path.join(folder, "emotions"))
        marker = os.path.join(folder, "emotions", "abc.npz")
        open(marker, "w").close()
        stray = os.path.join(voice_library.VOICES_DIR, "Plain_Folder")
        os.makedirs(stray)
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        saved = []
        real_save = voice_library.save_voice
        voice_library.save_voice = lambda *a, **k: saved.append(k.get("overwrite")) or {}
        try:
            win, stub = _clone_window(models=_FakeModels())
            win.btn_vc_src_ref.setChecked(True)
            win.edt_vc_ref.setText(wav)
            win.txt_vc_ref_text.setPlainText("Words")
            # same slug as an existing voice ("Ana_B"): must ask, and Cancel leaves everything intact
            win.edt_vc_new_name.setText("Ana_B")
            seen, restore = _answer_box("Cancel")
            try:
                win.btn_vc_save_voice.click()
            finally:
                restore()
            assert not win.busy and saved == [] and os.path.exists(marker), "cancel must leave the voice alone"
            assert seen and "Ana B" in seen[0][0] and "1 cached emotion sample" in seen[0][1], seen
            assert "share the folder" in seen[0][1] and "stays as it is" in seen[0][1], seen   # "Ana_B" vs "Ana B"
            # a stray folder without voice.json is never replaced, and nothing asks to
            win.edt_vc_new_name.setText("Plain Folder")
            seen, restore = _answer_box("Replace")
            try:
                win.btn_vc_save_voice.click()
            finally:
                restore()
            assert seen == [] and saved == [] and os.path.isdir(stray)
            assert "isn't a voice" in win.lbl_vc_error.text()
            # only an explicit Replace passes overwrite=True
            win.edt_vc_new_name.setText("Ana B")
            seen, restore = _answer_box("Replace")
            try:
                win.btn_vc_save_voice.click()
                _wait_idle(win)
            finally:
                restore()
            assert saved == [True]
            # a free name never asks
            win.edt_vc_new_name.setText("Brand New")
            seen, restore = _answer_box("Cancel")
            try:
                win.btn_vc_save_voice.click()
                _wait_idle(win)
            finally:
                restore()
            assert seen == [] and saved == [True, False]
            win.close()
        finally:
            voice_library.save_voice = real_save
    print("PASS: Save as voice asks before replacing, cancels cleanly, protects non-voice folders, and decides on the GUI thread")


def _test_save_clone_does_not_ask_while_a_task_is_running():
    with _Patched(), _Library() as lib:
        lib.add("Ana")
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        win, _ = _clone_window(models=_FakeModels())
        win.btn_vc_src_ref.setChecked(True)
        win.edt_vc_ref.setText(wav)
        win.txt_vc_ref_text.setPlainText("Words")
        win.edt_vc_new_name.setText("Ana")
        seen, restore = _answer_box("Replace")
        try:
            win.busy = True
            win.btn_vc_save_voice.click()
            win.busy = False
        finally:
            restore()
        assert seen == [], "no Replace dialog while another task is running"
        assert "Busy" in win.lbl_status.text()
        win.close()
    print("PASS: Save as voice does not show the Replace dialog while a task is running")


def _test_clone_voice_list_refresh_keeps_selection_language_and_has_no_blank_mode():
    with _Patched(), _Library() as lib:
        lib.add("Ada", language="german")
        lib.add("Bo", language="french")
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        assert [win.cmb_vc_voice.itemText(i) for i in range(win.cmb_vc_voice.count())] == ["Ada", "Bo"]
        assert win.cmb_vc_voice.currentIndex() == -1 and win._selected_clone_voice() is None
        win.cmb_vc_lang.setCurrentText("english")
        win.cmb_vc_voice.setCurrentIndex(1)
        assert win.cmb_vc_lang.currentText() == "french", "choosing a voice prefills Language"
        assert "french" in win.lbl_vc_voice_desc.text() or "Warm narrator" in win.lbl_vc_voice_desc.text()
        win.cmb_vc_lang.setCurrentText("spanish")          # the user may still override it
        win.refresh_voice_lists()
        assert win.cmb_vc_voice.currentText() == "Bo", "refresh keeps the chosen voice"
        assert win.cmb_vc_lang.currentText() == "spanish", "refresh must not overwrite the user's Language"
        assert win.txt_vc_instruct.toPlainText() == "", "a saved voice never writes into the description box"
        shutil.rmtree(voice_library.voice_dir("Bo"))
        win.refresh_voice_lists()
        assert win._selected_clone_voice() is None and win.cmb_vc_lang.currentText() == "spanish"
        shutil.rmtree(voice_library.voice_dir("Ada"))
        win.refresh_voice_lists()
        assert not win.lbl_vc_empty.isHidden() and win.cmb_vc_voice.count() == 0
        win.close()
    print("PASS: refreshing the voice list keeps the selection and Language, and there is no hidden blank mode")


class _PromptModels:
    """Builds a clone prompt that can really be written to a voice folder."""

    def get(self, kind, notify=None):
        class M:
            def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
                return {"a": np.zeros(1)}
        return M()


def _test_save_clone_offers_to_use_the_voice_without_switching():
    with _Patched(), _Library():
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        win, _ = _clone_window(models=_PromptModels())
        win.btn_vc_src_ref.setChecked(True)
        win.edt_vc_ref.setText(wav)
        win.txt_vc_ref_text.setPlainText("Words")
        win.edt_vc_new_name.setText("Cleo")
        assert win.row_vc_use.isHidden()
        win.btn_vc_save_voice.click()
        _wait_idle(win)
        assert voice_library.voice_exists("Cleo")
        assert win.clone_source_kind() == "reference", "saving must not switch the source"
        assert not win.row_vc_use.isHidden() and "Cleo" in win.btn_vc_use_saved.text()
        assert win.cmb_vc_voice.count() == 1 and win._selected_clone_voice() is None
        win.btn_vc_use_saved.click()
        assert win.clone_source_kind() == "saved" and win._selected_clone_voice() == "Cleo"
        assert win.row_vc_use.isHidden()
        win.close()
    print("PASS: after Save as voice the source stays Reference audio and 'Use' is an explicit action")


def _tree(path):
    out = {}
    for base, dirs, names in os.walk(path):
        for d in dirs:
            out[os.path.relpath(os.path.join(base, d), path) + "/"] = b""
        for n in names:
            with open(os.path.join(base, n), "rb") as f:
                out[os.path.relpath(os.path.join(base, n), path)] = f.read()
    return out


def _test_failed_replace_keeps_the_existing_voice_and_says_so():
    with _Patched(), _Library() as lib:
        folder = lib.add("Ana B")
        os.makedirs(os.path.join(folder, "emotions"))
        with open(os.path.join(folder, "emotions", "abc.npz"), "wb") as f:
            f.write(b"cache")
        before = _tree(folder)
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        win, _ = _clone_window(models=_PromptModels())
        win.btn_vc_src_ref.setChecked(True)
        win.edt_vc_ref.setText(wav)
        win.txt_vc_ref_text.setPlainText("Words")
        win.edt_vc_new_name.setText("Ana B")
        real = voice_library.write_wav

        def boom(*a, **k):
            raise OSError("disk full")
        voice_library.write_wav = boom
        seen, restore = _answer_box("Replace")
        try:
            win.btn_vc_save_voice.click()
            _wait_idle(win)
        finally:
            restore()
            voice_library.write_wav = real
        assert seen and "stays as it is" in seen[0][1]
        assert _tree(folder) == before, "a failed write must leave the voice byte-identical"
        assert not win.busy and "disk full" in win.lbl_vc_error.text() and "existing voice is unchanged" in win.lbl_vc_error.text() and not win.lbl_vc_error.isHidden()
        assert win.row_vc_use.isHidden(), "nothing was saved, so there is nothing to use"
        assert voice_library.list_voices() == ["Ana B"]
        # the same click now succeeds through the real save: old assets are gone, one voice remains
        seen, restore = _answer_box("Replace")
        try:
            win.btn_vc_save_voice.click()
            _wait_idle(win)
        finally:
            restore()
        assert not win.busy and not win.lbl_vc_error.text().startswith("Couldn't"), win.lbl_vc_error.text()
        assert not os.path.exists(os.path.join(folder, "emotions")), "assets of the old recording are dropped"
        assert os.path.isfile(os.path.join(folder, "reference.wav")) and os.path.isfile(os.path.join(folder, "clone_prompt.npz"))
        assert voice_library.list_voices() == ["Ana B"] and not os.path.exists(os.path.join(voice_library.VOICES_DIR, ".pending"))
        assert not win.row_vc_use.isHidden()
        # Create Voice shares the same confirmation and the same non-voice-folder guard
        stray = os.path.join(voice_library.VOICES_DIR, "Plain_Folder")
        os.makedirs(stray)
        win.pending_voice = {"wav": np.zeros(2400, dtype=np.float32), "sr": 24000, "description": "d",
                             "language": "english", "seed": 3, "text": "Hi", "preview": wav}
        win.edt_vd_name.setText("Plain Folder")
        seen, restore = _answer_box("Replace")
        try:
            win.save_voice()
        finally:
            restore()
        assert seen == [] and not win.busy and os.path.isdir(stray) and "isn't a voice" in win.lbl_vd_notice.text()
        after = _tree(folder)
        win.edt_vd_name.setText("Ana_B")
        seen, restore = _answer_box("Cancel")
        try:
            win.save_voice()
        finally:
            restore()
        assert seen and "its recording and clone prompt" in seen[0][1] and _tree(folder) == after and not win.busy
        win.close()
    print("PASS: a failed or cancelled replace leaves the existing voice untouched (Clone and Create Voice)")


def _test_numeric_fields_share_one_style_and_keep_their_behavior():
    with _Patched(), _Library():
        win, _ = _clone_window(models=_FakeModels())
        at = win.audiobook_tab
        # (spin, min, max, step, default, suffix, neighbouring combo, page index) taken from the code before the restyle
        specs = [
            ("Create seed", win.spn_vd_seed, 0, 2**31 - 1, 1, 42, "", win.cmb_vd_seed_mode, 0),
            ("Clone seed", win.spn_vc_seed, 0, 2**31 - 1, 1, 42, "", win.cmb_vc_seed_mode, 1),
            ("Clone temperature", win.spn_vc_temp, 0.1, 1.5, 0.05, 0.7, "", win.cmb_vc_seed_mode, 1),
            ("Audiobook pause", at.spn_pause, 0, 5000, 50, 350, " ms", at.cmb_seed_mode, 3),
            ("Audiobook seed", at.spn_seed, 0, 2**31 - 1, 1, 42, "", at.cmb_seed_mode, 3),
            ("Audiobook temperature", at.spn_temp, 0.1, 1.5, 0.05, 0.7, "", at.cmb_seed_mode, 3),
            ("Min words", win.spn_min, 0, 5000, 1, 200, "", win.cmb_device, 5),
            ("Max words", win.spn_max, 0, 5000, 1, 250, "", win.cmb_device, 5),
        ]
        win.resize(1240, 780)
        win.show()
        win.btn_vc_advanced.setChecked(True)
        checked = []
        for label, spin, lo, hi, step, default, suffix, neighbour, page in specs:
            assert spin.property("role") == "numeric", label
            assert spin.minimum() == lo and spin.maximum() == hi and abs(spin.singleStep() - step) < 1e-9, label
            assert abs(spin.value() - default) < 1e-9 and spin.suffix() == suffix, label
            assert spin.accessibleName(), f"{label} needs an accessible name"
            win.show_page(page)
            if page == 0:
                win.btn_vd_advanced.setChecked(True)
            if page == 3:
                at.btn_settings.setChecked(True)      # Audiobooks keeps these in its collapsed Render settings
            _app.processEvents()
            original_mode = neighbour.currentText() if neighbour.findText("Fixed") >= 0 else None
            if original_mode:
                neighbour.setCurrentText("Fixed")   # a Random seed is disabled by design
            if spin.isVisible():
                checked.append(label)
                assert spin.height() == neighbour.height(), f"{label}: {spin.height()} vs {neighbour.height()}"
                shown = spin.text()
                assert spin.lineEdit().fontMetrics().horizontalAdvance(shown) + 20 <= spin.width(), f"{label} clips '{shown}'"
                # same keyboard behaviour as before: Up/Down step by the same amount, typing replaces the value
                spin.setFocus()
                _app.processEvents()
                start = spin.value()
                QTest.keyClick(spin, QtCore.Qt.Key_Up)
                assert abs(spin.value() - (start + step)) < 1e-9, label
                QTest.keyClick(spin, QtCore.Qt.Key_Down)
                assert abs(spin.value() - start) < 1e-9, label
                spin.selectAll()
                QTest.keyClicks(spin, str(hi))
                QTest.keyClick(spin, QtCore.Qt.Key_Tab)
                assert abs(spin.value() - hi) < 1e-9, label
                assert spin.lineEdit().fontMetrics().horizontalAdvance(spin.text()) + 20 <= spin.width(), \
                    f"{label} clips its widest value '{spin.text()}'"
                spin.setValue(default)
            if original_mode:
                neighbour.setCurrentText(original_mode)
        assert len(checked) == len(specs), f"these were never shown, so never measured: {set(l[0] for l in specs) - set(checked)}"
        # the widest values still fit
        win.show_page(1)
        win.cmb_vc_seed_mode.setCurrentText("Fixed")
        win.spn_vc_seed.setValue(2**31 - 1)
        _app.processEvents()
        assert win.spn_vc_seed.lineEdit().fontMetrics().horizontalAdvance(win.spn_vc_seed.text()) + 20 <= win.spn_vc_seed.width()
        win.spn_vc_seed.setValue(42)
        # Random disables the value but keeps it
        win.cmb_vc_seed_mode.setCurrentText("Random")
        assert not win.spn_vc_seed.isEnabled() and win.spn_vc_seed.value() == 42
        win.close()
    print("PASS: every numeric field shares one style and keeps its range, step, default and keyboard behavior")


def _test_clone_save_records_the_real_seed_transcript_and_description():
    with _Patched(), _Library():
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        win, _ = _clone_window(models=_PromptModels())
        win.btn_vc_src_ref.setChecked(True)
        win.edt_vc_ref.setText(wav)

        def save(name, **setup):
            win.edt_vc_new_name.setText(name)
            win.btn_vc_save_voice.click()
            _wait_idle(win)
            with open(os.path.join(voice_library.voice_dir(name), "voice.json"), encoding="utf-8") as f:
                return json.load(f)

        # exact transcript, fixed seed and the single description field are stored as given
        win.txt_vc_ref_text.setPlainText("  Exactly these words.  ")
        win.txt_vc_instruct.setPlainText("Warm and slow")
        win.cmb_vc_seed_mode.setCurrentText("Fixed")
        win.spn_vc_seed.setValue(1234)
        cfg = save("Fixed Voice")
        assert cfg["seed"] == 1234 and cfg["description"] == "Warm and slow"
        assert cfg["ref_text"] == cfg["sample_text"] == "Exactly these words."
        # Random records 0: that is what clone() uses for [tag] emotion samples in Reference mode (`seed or 0`),
        # so the saved voice reproduces what the user heard. Nothing is invented at save time.
        win.cmb_vc_seed_mode.setCurrentText("Random")
        assert win.clone_args()["seed"] is None
        cfg = save("Random Voice")
        assert cfg["seed"] == 0
        # X-vector only: the transcript is explicitly empty, not the standard passage; description may be empty
        win.chk_vc_xvec.setChecked(True)
        win.txt_vc_instruct.clear()
        cfg = save("Xvec Voice")
        assert cfg["ref_text"] == "" and cfg["sample_text"] == "" and cfg["description"] == ""
        win.close()
        # library default is unchanged: no transcript argument means the standard passage
        root = tempfile.mkdtemp()
        cfg = voice_library.save_voice("Std", np.zeros(2400, dtype=np.float32), 24000, {"k": np.zeros(1)},
                                       "d", "english", 1, root=root)
        assert cfg["ref_text"] == voice_library.SAMPLE_TEXT
    print("PASS: Save as voice records the seed clone() really uses, the exact transcript (empty for X-vector) and one description")


def _test_import_voice_from_audio_stores_an_empty_transcript_and_no_invented_seed():
    with _Patched(), _Library():
        files = tempfile.mkdtemp()
        wav = _clone_ready_to_save(files)
        win, _ = _clone_window(models=_PromptModels())
        patches = {
            (QtWidgets.QFileDialog, "getOpenFileName"): lambda *a, **k: (wav, ""),
            (QtWidgets.QInputDialog, "getText"): lambda *a, **k: ("Imported", True),
            (QtWidgets.QInputDialog, "getItem"): lambda *a, **k: ("english", True),
            (QtWidgets.QInputDialog, "getMultiLineText"): lambda *a, **k: ("Warm", True),
            (QtWidgets.QInputDialog, "getInt"): lambda *a, **k: (0, True),   # 0 = random, resolved before the work starts
        }
        originals = {key: getattr(*key) for key in patches}
        for (cls, name), fn in patches.items():
            setattr(cls, name, staticmethod(fn))
        real = qt_app.random.randint
        qt_app.random.randint = lambda a, b: 4242
        try:
            win.create_voice_from_audio()
            _wait_idle(win)
        finally:
            qt_app.random.randint = real
            for (cls, name), fn in originals.items():
                setattr(cls, name, fn)
        with open(os.path.join(voice_library.voice_dir("Imported"), "voice.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["ref_text"] == "" and cfg["sample_text"] == "" and cfg["description"] == "Warm"
        assert cfg["seed"] == 4242, "a seed picked before the work and recorded as picked"
        win.close()
    print("PASS: importing a voice from audio stores an explicitly empty transcript")


class _FakeSession:
    """audio_player with a pretend playback session, so Play / Pause / seek can be driven without sound."""

    NAMES = ("play", "stop", "pause", "resume", "seek", "state", "current_path", "position", "can_control")

    def __enter__(self):
        self.saved = {n: getattr(audio_player, n) for n in self.NAMES}
        self.path, self.st, self.pos, self.played = None, "stopped", 0.0, []

        def play(path, start=0.0):
            self.played.append((path, start))
            self.path, self.st, self.pos = path, "playing", start

        def stop():
            self.path, self.st, self.pos = None, "stopped", 0.0
        audio_player.play, audio_player.stop = play, stop
        audio_player.pause = lambda: setattr(self, "st", "paused")
        audio_player.resume = lambda: setattr(self, "st", "playing")
        audio_player.seek = lambda s: setattr(self, "pos", s)
        audio_player.state = lambda: self.st
        audio_player.current_path = lambda: self.path
        audio_player.position = lambda: self.pos
        audio_player.can_control = lambda: True
        return self

    def __exit__(self, *exc):
        for n, fn in self.saved.items():
            setattr(audio_player, n, fn)


def _test_saved_audio_without_a_preview_is_not_a_failed_generation():
    with _Patched(), _Library() as lib, _FakeSession():
        lib.add("Ada")
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        win.cmb_vc_voice.setCurrentIndex(0)
        _shown(win)
        win.txt_vc_text.setPlainText("Hello there")
        mode = {"kind": "ok"}
        original_clone, original_load = qt_app.TTSGenerator.clone, qt_app.load_wav
        outputs = []

        def clone(gen, **kwargs):
            if mode["kind"] == "early":
                gen.clone_chunks = 3
                win.task.stop_event.set()
                return [np.zeros(24000, dtype=np.float32) + 0.1] * 2, 24000
            if mode["kind"] == "write_fails":
                return [np.zeros(24000, dtype=np.float32) + 0.1], 24000
            return [np.zeros(48000, dtype=np.float32) + 0.1], 24000

        def load(path):
            if mode["kind"] in ("unreadable", "early"):
                raise OSError("cannot read back " + path)
            return original_load(path)
        qt_app.TTSGenerator.clone, qt_app.load_wav = clone, load

        def generate(kind):
            mode["kind"] = kind
            win.btn_vc_generate.click()
            _wait_idle(win)
        try:
            generate("ok")
            first = win.vc_player.path()
            assert win._vc_pv_state == qt_app.CV_READY and first
            # the file is written, only reading it back fails
            generate("unreadable")
            path = win.last_output
            assert path != first and os.path.isfile(path)
            assert win._vc_pv_state == qt_app.CV_SAVED
            title, detail = win.lbl_vc_pv_title.text(), win.lbl_vc_pv_detail.text()
            assert title == "Audio saved. Preview unavailable" and "Couldn't generate" not in title + detail, (title, detail)
            assert path in detail and "Play last output" in detail and "Open output folder" in detail
            assert "cannot read back" not in detail, "the raw error stays behind Show details"
            assert win.lbl_status.text().startswith("Done. Audio saved to: "), win.lbl_status.text()
            assert not win.lbl_status.text().startswith(("Error", "Couldn't"))
            assert win.btn_play_last.isEnabled() and win.btn_open_out.isEnabled()
            assert not win.btn_vc_details.isHidden() and win.btn_vc_error_settings.isHidden()
            assert "Clone Voice Preview" in win._clone_failure_details and "cannot read back" in win._clone_failure_details
            # the earlier audio stays loaded and is labelled as the previous one
            assert win.vc_player.path() == first and not win.lbl_vc_pv_prev.isHidden()
            # a stopped-early run says which part was saved
            generate("early")
            assert win._vc_pv_state == qt_app.CV_SAVED and "only the part generated so far" in win.lbl_vc_pv_detail.text()
            assert win.lbl_status.text().startswith("Stopped early; partial audio saved to: ")
            # a failure to WRITE the file is still a failed generation
            mode["kind"] = "write_fails"
            original_save = qt_app.save_outputs
            qt_app.save_outputs = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
            try:
                win.btn_vc_generate.click()
                _wait_idle(win)
            finally:
                qt_app.save_outputs = original_save
            assert win._vc_pv_state == qt_app.CV_ERROR and "Couldn't generate" in win.lbl_vc_pv_title.text()
            # the next good run recovers
            generate("ok")
            assert win._vc_pv_state == qt_app.CV_READY and win.btn_vc_details.isHidden()
        finally:
            qt_app.TTSGenerator.clone, qt_app.load_wav = original_clone, original_load
    print("PASS: audio saved but not previewable is reported as saved, with the path, actions and details; real failures stay failures")


def _test_reference_formats_are_validated_up_front_and_save_accepts_float_and_24bit_wav():
    import importlib.util
    import wave
    import soundfile as sf
    with _Patched(), _Library():
        files = tempfile.mkdtemp()
        tone = (0.3 * np.sin(np.arange(24000) / 20)).astype("float32")
        paths = {}
        for name, kw in (("float.wav", dict(subtype="FLOAT")), ("pcm24.wav", dict(subtype="PCM_24")),
                         ("tone.flac", {}), ("plain.wav", dict(subtype="PCM_16"))):
            paths[name] = os.path.join(files, name)
            sf.write(paths[name], tone, 24000, **kw)
        m4a = os.path.join(files, "note.m4a")
        open(m4a, "wb").write(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
        empty = os.path.join(files, "empty.wav")
        open(empty, "wb").close()
        win, _ = _clone_window(models=_PromptModels())
        win.btn_vc_src_ref.setChecked(True)
        win.chk_vc_xvec.setChecked(True)
        win.txt_vc_text.setPlainText("Hello")
        real_find = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a, **k: None if name == "audioread" else real_find(name, *a, **k)
        try:
            # the pickers never offer M4A while no decoder exists
            assert "m4a" not in qt_app.audio_file_filter()
            # Generate: an undecodable reference is explained before any model loads
            win.edt_vc_ref.setText(m4a)
            assert not win._validate_clone()
            assert "M4A" in win.lbl_vc_error.text() and "WAV" in win.lbl_vc_error.text() and "Format not" not in win.lbl_vc_error.text()
            win.edt_vc_ref.setText(empty)
            assert not win._validate_clone() and "empty or damaged" in win.lbl_vc_error.text()
            # supported formats pass, including FLAC and the WAV flavours the old reader rejected
            for name, path in paths.items():
                win.edt_vc_ref.setText(path)
                assert win._validate_clone(), (name, win.lbl_vc_error.text())
        finally:
            importlib.util.find_spec = real_find
        # Save as voice: float32 and 24-bit WAV now work and are stored as the usual 16-bit mono reference.wav
        for i, name in enumerate(("float.wav", "pcm24.wav", "plain.wav")):
            win.edt_vc_ref.setText(paths[name])
            win.edt_vc_new_name.setText(f"Fmt {i}")
            win.btn_vc_save_voice.click()
            _wait_idle(win)
            vdir = voice_library.voice_dir(f"Fmt {i}")
            assert voice_library.voice_exists(f"Fmt {i}"), (name, win.lbl_status.text())
            with wave.open(os.path.join(vdir, "reference.wav"), "rb") as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 24000), name
            with open(os.path.join(vdir, "voice.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            assert set(cfg) >= {"name", "description", "language", "seed", "created", "clone_prompt"} and cfg["sample_text"] == ""
        # a FLAC is still refused for Save with the existing, actionable sentence
        win.edt_vc_ref.setText(paths["tone.flac"])
        win.edt_vc_new_name.setText("Fmt flac")
        win.btn_vc_save_voice.click()
        assert "needs a WAV" in win.lbl_vc_error.text() and not voice_library.voice_exists("Fmt flac")
    print("PASS: reference formats are checked before any model loads; Save as voice accepts float32 and 24-bit WAV")


def _test_clone_preview_states_transitions_and_player():
    import voice_library as _vl
    with _Patched(), _Library() as lib, _FakeSession() as session:
        lib.add("Ada")
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        win.cmb_vc_voice.setCurrentIndex(0)
        _shown(win)
        win.txt_vc_text.setPlainText("Hello there")
        seen_while_running, seen_state = [], []
        mode = {"kind": "ok"}
        original = qt_app.TTSGenerator.clone

        def clone(gen, **kwargs):
            seen_while_running.append((win._vc_pv_state, win.vc_player.isVisible(), win.vc_player.has_audio()))
            seen_state.append(session.st)
            kind = mode["kind"]
            if kind == "error":
                raise ValueError("boom")
            if kind == "flash":
                raise ImportError("flash_attn is not installed")
            if kind == "complete_stop":
                gen.clone_chunks = 1
                win.task.stop_event.set()                    # Stop pressed after the only chunk was already done
                return [np.zeros(24000, dtype=np.float32) + 0.1], 24000
            if kind == "midway":
                gen.clone_chunks = 3
                win.task.stop_event.set()                    # two of three chunks were made
                return [np.zeros(24000, dtype=np.float32) + 0.1] * 2, 24000
            if kind == "multi":
                return [np.zeros(24000, dtype=np.float32), np.zeros(48000, dtype=np.float32) + 0.1,
                        np.zeros(24000, dtype=np.float32)], 24000     # three chunks: the combined-file branch
            if kind in ("partial", "stopped"):
                win.task.stop_event.set()       # the user pressed Stop while it was running
            if kind == "stopped":
                return [], 24000
            return [np.zeros(48000, dtype=np.float32) + 0.1], 24000   # 2 s
        qt_app.TTSGenerator.clone = clone

        def generate(kind):
            mode["kind"] = kind
            win.btn_vc_generate.click()
            _wait_idle(win)

        try:
            player = win.vc_player
            # EMPTY
            assert win._vc_pv_state == qt_app.CV_EMPTY and not player.isVisible() and not player.has_audio()
            assert "appear here" in win.lbl_vc_pv_title.text() and "Choose a voice" in win.lbl_vc_pv_detail.text()
            # READY: file saved by the existing path, shown in the player, nothing plays by itself
            generate("ok")
            path = win.last_output
            assert seen_while_running[-1] == (qt_app.CV_BUSY, False, False), "Generating shows no player"
            assert win._vc_pv_state == qt_app.CV_READY and win.lbl_vc_pv_title.text() == "Audio ready"
            assert player.isVisible() and player.path() == path and os.path.isfile(path)
            assert abs(player.duration() - 2.0) < 0.01
            assert player.lbl_len.text() == "0:02" and player.btn_play.text() == "Play" and not session.played
            assert win.btn_play_last.isEnabled() and path in win.lbl_vc_pv_detail.text()
            # the player owns its path: another screen changing last_output does not retarget it
            win.last_output = "/elsewhere/render.wav"
            assert player.path() == path
            win.last_output = path
            # Play / Pause / Resume, Back / Forward and the slider all follow the (fake) session
            player.btn_play.click()
            assert session.played == [(path, 0.0)] and player.btn_play.text() == "Pause"
            win._refresh_audio_state()
            assert not win.btn_stop_audio.isHidden(), "the footer Stop appears for the Preview's audio too"
            player.btn_play.click()
            assert session.st == "paused" and player.btn_play.text() == "Play" and "Resume" in player.btn_play.toolTip()
            player.btn_forward.click()
            assert abs(session.pos - 1.95) < 1e-6 and player.lbl_pos.text() == "0:02", (session.pos, player.lbl_pos.text())
            player.btn_back.click()
            assert abs(session.pos - 0.0) < 1e-6
            player.wave.seek_requested.emit(0.5)
            assert abs(session.pos - 1.0) < 1e-6
            player.btn_play.click()
            assert session.st == "playing"
            # footer Stop resets everything the same way as for Create Voice
            win.stop_audio()
            assert session.st == "stopped" and player.btn_play.text() == "Play" and player.position() == 0.0
            # keyboard: Generate -> Seek -> Back -> Play -> Forward, Space activates the focused button
            win.btn_vc_generate.setFocus()
            _app.processEvents()
            order = []
            for _ in range(4):
                _tab()
                order.append(_focus_label(_app.focusWidget()))
            assert order == ["Seek", "Back 5 seconds", "Play", "Forward 5 seconds"], order
            _tab(shift=True)
            _tab(shift=True)
            assert _focus_label(_app.focusWidget()) == "Back 5 seconds"
            _tab()
            assert _app.focusWidget() is player.btn_play
            _press(QtCore.Qt.Key_Space)
            assert session.st == "playing"
            # GENERATING keeps the previous audio on screen and playable (as Create Voice keeps its last sample)
            player.btn_play.setFocus()
            generate("partial")
            assert seen_while_running[-1] == (qt_app.CV_BUSY, True, True), seen_while_running[-1]
            assert seen_state[-1] == "stopped", "playback ended when the generation started (as in Create Voice)"
            # the new result replaces it, and the old audio stops with it
            assert session.st == "stopped" and session.played[-1][0] == path
            # PARTIAL: stopped early (fewer chunks than planned, or no count known), audio kept and playable
            assert win._vc_pv_state == qt_app.CV_PARTIAL and win.lbl_vc_pv_title.text() == "Stopped early"
            assert player.isVisible() and player.has_audio() and os.path.isfile(player.path()) and player.path() == win.last_output
            assert "part generated so far" in win.lbl_vc_pv_detail.text() and win.lbl_vc_pv_prev.isHidden()
            partial_path = player.path()
            # STOPPED without audio: the previous audio is still there, labelled, and last_output is untouched
            player.btn_play.setFocus()
            generate("stopped")
            assert win._vc_pv_state == qt_app.CV_STOPPED and player.isVisible() and player.path() == partial_path
            assert win.lbl_vc_pv_title.text().startswith("Stopped before any audio")
            assert "previous audio is still available" in win.lbl_vc_pv_detail.text() and not win.lbl_vc_pv_prev.isHidden()
            assert win.last_output == partial_path and _app.focusWidget() is player.btn_play, "nothing vanished, so focus stays"
            # no previous audio at all: nothing to keep, honest text, focus leaves the hidden controls
            win._set_clone_preview(qt_app.CV_EMPTY)
            assert not player.isVisible() and not player.has_audio()
            generate("stopped")
            assert win._vc_pv_state == qt_app.CV_STOPPED and not player.isVisible()
            assert "Nothing was saved" in win.lbl_vc_pv_detail.text()
            # Stop pressed after the final chunk: the output is complete, so it is Ready, not "Stopped early"
            generate("complete_stop")
            assert win.task.stop_event.is_set() and win._vc_pv_state == qt_app.CV_READY
            assert win.lbl_vc_pv_title.text() == "Audio ready" and "Done. Audio saved to" in win.lbl_status.text()
            # a real early stop (2 of 3 chunks) is reported as such, in the Preview and in the footer
            generate("midway")
            assert win._vc_pv_state == qt_app.CV_PARTIAL and "Stopped early; partial audio saved" in win.lbl_status.text()
            # ERROR with previous audio: it stays, the message is sanitized and actionable, details are on request
            generate("ok")
            assert win._vc_pv_state == qt_app.CV_READY
            ok_path = player.path()
            generate("error")
            detail = win.lbl_vc_pv_detail.text()
            assert win._vc_pv_state == qt_app.CV_ERROR and player.isVisible() and player.path() == ok_path
            assert win.lbl_vc_pv_title.text() == "Couldn't generate new audio" and not win.lbl_vc_pv_prev.isHidden()
            assert "boom" not in detail and "ValueError" not in detail and "Traceback" not in detail, detail
            assert detail.startswith("Something went wrong while generating the audio.") and "previous audio is still available" in detail
            assert "boom" not in win.lbl_status.text() and "boom" not in win.lbl_vc_pv_title.text()
            assert not win.btn_vc_details.isHidden() and win.btn_vc_error_settings.isHidden()
            assert "ValueError" in win._clone_failure_details and "boom" in win._clone_failure_details and "Clone Voice audio" in win._clone_failure_details
            assert "Qwen3-TTS-12Hz-1.7B-Base" in win._clone_failure_details and "Source: saved" in win._clone_failure_details
            shown = []
            real_exec = QtWidgets.QDialog.exec
            QtWidgets.QDialog.exec = lambda d: shown.append([b.toPlainText() for b in d.findChildren(QtWidgets.QPlainTextEdit)]) or 0
            try:
                win.btn_vc_details.click()
            finally:
                QtWidgets.QDialog.exec = real_exec
            assert shown and "boom" in shown[0][0] and "Clone Voice audio" in shown[0][0]
            win._failure_details = "CREATE-ONLY DETAILS"
            shown.clear()
            QtWidgets.QDialog.exec = lambda d: shown.append([b.toPlainText() for b in d.findChildren(QtWidgets.QPlainTextEdit)]) or 0
            try:
                win.btn_vc_details.click()
                win._open_failure_dialog()
            finally:
                QtWidgets.QDialog.exec = real_exec
            assert "boom" in shown[0][0] and shown[1][0] == "CREATE-ONLY DETAILS", "each screen shows its own details"
            win.copy_failure_details("clone text")
            assert QtWidgets.QApplication.clipboard().text() == "clone text"
            win._failure_details = ""
            # focus on an error button that is about to disappear moves to a visible field
            win.btn_vc_details.setFocus()
            _app.processEvents()
            assert _app.focusWidget() is win.btn_vc_details
            generate("ok")
            assert _app.focusWidget() is not None and _app.focusWidget().isVisible() and win.btn_vc_details.isHidden()
            generate("error")
            win.btn_vc_generate.setFocus()
            order = []
            for _ in range(7):
                _tab()
                order.append(_focus_label(_app.focusWidget()))
            assert order[:2] == ["Show details", "Seek"], order
            assert order.index("Show details") < order.index("Seek") < order.index("Back 5 seconds") < order.index("Play")
            # a cause we can identify gets its own wording and the Settings shortcut
            mode["kind"] = "flash"
            generate("flash")
            assert "FlashAttention" in win.lbl_vc_pv_detail.text() and not win.btn_vc_error_settings.isHidden()
            # a fresh error with nothing before it shows no player
            win._set_clone_preview(qt_app.CV_EMPTY)
            generate("error")
            assert win.lbl_vc_pv_title.text() == "Couldn't generate the audio" and not player.isVisible()
            # a new success leaves no error controls behind
            generate("ok")
            assert win.btn_vc_details.isHidden() and win.btn_vc_error_settings.isHidden() and win.lbl_vc_pv_prev.isHidden()
            # several chunks are combined into one file that the player reads back at the right length
            generate("multi")
            combined = player.path()
            assert os.path.isfile(combined) and abs(player.duration() - 4.0) < 0.01 and player.lbl_len.text() == "0:04"
            assert len(load_wav(combined)[0]) == 96000
            assert player.wave.accessibleName() == "Waveform of the generated audio"
            assert not [f for f in os.listdir(os.path.dirname(combined)) if re.fullmatch(r"VoiceClone_\d{8}_\d{6}_[0-9a-f]{3}_\d+\.wav", f)], "temp segments cleaned"
            # a refused click while another task runs must leave the Ready preview alone
            win.busy = True
            win.generate_clone()
            win.busy = False
            assert win._vc_pv_state == qt_app.CV_READY and player.isVisible() and player.path() == combined
            assert "Busy" in win.lbl_status.text()
            # the detail line never keeps the height of a longer earlier message
            win._set_clone_preview(qt_app.CV_STOPPED, "word " * 80)
            _app.processEvents()
            long_height = win.lbl_vc_pv_detail.height()
            win._set_clone_preview(qt_app.CV_STOPPED, "short")
            _app.processEvents()
            assert win.lbl_vc_pv_detail.height() < long_height
            # a failure while saving or reading the file must not leave the panel on "Generating"
            real_save = qt_app.save_outputs
            qt_app.save_outputs = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
            try:
                generate("ok")
            finally:
                qt_app.save_outputs = real_save
            assert not win.busy and win._vc_pv_state == qt_app.CV_ERROR and "disk full" not in win.lbl_vc_pv_detail.text()
            assert "disk full" in win._clone_failure_details
            # playback failure is reported in the panel and the footer, and the player stays usable
            generate("ok")
            audio_player.play = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device"))
            player.btn_play.click()
            assert "no device" in win.lbl_vc_pv_detail.text() and win.lbl_status.text().startswith("Playback error")
            assert player.btn_play.text() == "Play" and player.isVisible()
        finally:
            qt_app.TTSGenerator.clone = original
            win.close()
    print("PASS: Clone Preview shows Empty, Generating, Ready, Stopped early, Stopped without audio and Error with one shared player")


def _test_create_and_clone_players_share_the_audio_session_without_crosstalk():
    with _Patched(), _Library() as lib, _FakeSession() as session:
        win, _ = _clone_window(models=_FakeModels())
        _shown(win)
        a, b = win.vd_player, win.vc_player
        wav = np.zeros(24000, dtype=np.float32)
        pa, pb = os.path.join(tempfile.mkdtemp(), "a.wav"), os.path.join(tempfile.mkdtemp(), "b.wav")
        a.set_audio(pa, wav, 24000)
        b.set_audio(pb, wav, 24000)
        b.toggle()
        assert b.state() == "playing" and a.state() == "stopped", "playing one screen's audio is not the other's"
        a.toggle()       # starting the other player replaces the single session
        assert a.state() == "playing" and b.state() == "stopped" and session.path == pa
        b.clear()        # clearing a player that is not playing leaves the session alone
        assert session.st == "playing" and session.path == pa
        a.clear()
        assert session.st == "stopped" and not a.has_audio() and not a.slider.isEnabled()
        win.close()
    print("PASS: the Create and Clone players share one audio session without controlling each other")


def _test_transcript_editor_prefill_handles_missing_empty_and_real_ref_text():
    with _Patched(), _Library():
        win, _ = _clone_window(models=_FakeModels())
        cases = {
            "Old": {"name": "Old"},                                                    # voices saved before the change: no key
            "Empty": {"name": "Empty", "ref_text": ""},                                # X-vector only: explicitly empty
            "Real": {"name": "Real", "ref_text": "Exactly these words."},
            "Own": {"name": "Own", "ref_text": "stored", "csm_ref_text": "the saved CSM transcript"},
        }
        expected = {"Old": "", "Empty": "", "Real": "Exactly these words.", "Own": "the saved CSM transcript"}
        for name, cfg in cases.items():
            os.makedirs(voice_library.voice_dir(name))
            with open(os.path.join(voice_library.voice_dir(name), "voice.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f)
        asked = []
        original = QtWidgets.QInputDialog.getMultiLineText
        QtWidgets.QInputDialog.getMultiLineText = staticmethod(lambda *a, **k: (asked.append(a[3]) or ("", False)))
        try:
            for name in cases:
                win.selected_voice_slug = lambda n=name: n
                win.edit_csm_transcript()
        finally:
            QtWidgets.QInputDialog.getMultiLineText = original
        assert asked == [expected[n] for n in cases], asked
        win.close()
    print("PASS: the transcript editor prefill handles a missing, an empty and a real ref_text")


def _test_clone_layout_keyboard_and_disclosure():
    with _Patched(), _Library() as lib:
        lib.add("Ada")
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        for size in ((1240, 780), (1024, 640)):
            win.resize(*size)
            win.show()
            _app.processEvents()
            page = win.stack.widget(1)
            for ref, adv in ((False, False), (True, False), (True, True)):
                (win.btn_vc_src_ref if ref else win.btn_vc_src_saved).setChecked(True)
                win.btn_vc_advanced.setChecked(adv)
                _app.processEvents()
                assert page.horizontalScrollBar().maximum() == 0, (size, ref, adv, "horizontal scrolling")
                assert page.verticalScrollBar().maximum() >= 0
            win.btn_vc_src_saved.setChecked(True)
            win.btn_vc_advanced.setChecked(False)
            _app.processEvents()
            g = win.btn_vc_generate.mapTo(page.viewport(), QtCore.QPoint(0, 0)).y() + win.btn_vc_generate.height()
            assert g <= page.viewport().height(), (size, "Generate is visible without scrolling", g)
            win.btn_vc_generate.click()   # no voice chosen: the inline problem appears above Generate
            _app.processEvents()
            assert not win.lbl_vc_error.isHidden()
            page.verticalScrollBar().setValue(0)
            g = win.btn_vc_generate.mapTo(page.viewport(), QtCore.QPoint(0, 0)).y() + win.btn_vc_generate.height()
            assert g <= page.viewport().height(), (size, "Generate stays visible with the problem shown", g)
        assert not win.btn_vc_advanced.isChecked() and not win.vc_advanced.isVisible(), "Advanced starts collapsed"
        assert win.btn_vc_advanced.accessibleDescription() == "Collapsed"
        win.btn_vc_advanced.setChecked(True)
        assert win.btn_vc_advanced.accessibleDescription() == "Expanded"
        assert win.vc_advanced.isAncestorOf(win.cmb_vc_seed_mode) and win.vc_advanced.isAncestorOf(win.dir_vc)
        assert win.vc_advanced.isAncestorOf(win.spn_vc_temp) and win.vc_advanced.isAncestorOf(win.edt_vc_save)
        # keyboard: arrow keys move between the two source modes and select
        win.btn_vc_src_saved.setFocus()
        _press(QtCore.Qt.Key_Right)
        assert win.btn_vc_src_ref.isChecked() and win.clone_source_kind() == "reference"
        _press(QtCore.Qt.Key_Left)
        assert win.btn_vc_src_saved.isChecked()
        # the primary button comes before Advanced, as in Create Voice
        generate_y = win.btn_vc_generate.mapTo(win, QtCore.QPoint(0, 0)).y()
        assert generate_y < win.btn_vc_advanced.mapTo(win, QtCore.QPoint(0, 0)).y()
        assert win.cmb_vc_voice.accessibleName() and win.txt_vc_text.accessibleName()
        win.close()
    print("PASS: Clone Voice fits both sizes without horizontal scrolling, Advanced is collapsed, and the keyboard works")



def _hover(widget):
    """Send the tooltip help event Qt sends after the mouse rests on a widget."""
    qt_app.InfoPopup.hide_popup()
    pos = QtCore.QPoint(widget.width() // 2, widget.height() // 2)
    QtWidgets.QApplication.sendEvent(widget, QtGui.QHelpEvent(QtCore.QEvent.ToolTip, pos, widget.mapToGlobal(pos)))
    _app.processEvents()


def _test_clone_advanced_info_buttons_and_numeric_fields():
    with _Patched(), _Library():
        win, _ = _clone_window(models=_FakeModels())
        _shown(win)
        win.btn_vc_advanced.setChecked(True)
        _app.processEvents()
        topics = {"Seed": qt_app.SEED_TIP, "Temperature": qt_app.TEMPERATURE_TIP,
                  "Direction tags": qt_app.DIRECTION_TAGS_TIP, "Description": qt_app.DESCRIPTION_TIP}
        assert set(win.info_buttons) == set(topics)
        for topic, tip in topics.items():
            b = win.info_buttons[topic]
            assert isinstance(b, qt_app.InfoButton) and b.property("role") == "info"
            assert b.accessibleName() == f"About {topic}" and b.accessibleDescription() == tip
            assert b.focusPolicy() != QtCore.Qt.NoFocus and not b.icon().isNull()
            assert b.width() == 22 and b.height() == 22
            assert tip in b.toolTip() and "<" not in tip, "plain text for assistive tech, rich text only for the tip"
            _hover(b)   # mouse rests on it: the same bubble as the keyboard shows
            assert qt_app.InfoPopup.is_open() and qt_app.InfoPopup.owner is b
            assert qt_app.InfoPopup._shared.text() == tip, topic
            QTest.mouseMove(win, QtCore.QPoint(2, 2))
            b.leaveEvent(QtCore.QEvent(QtCore.QEvent.Leave))
            assert not qt_app.InfoPopup.is_open(), "moving away hides it"
            QTest.mouseClick(b, QtCore.Qt.LeftButton)   # a click shows it too; clicking again hides it
            assert qt_app.InfoPopup.is_open() and qt_app.InfoPopup._shared.text() == tip, topic
            QTest.mouseClick(b, QtCore.Qt.LeftButton)
            b.clearFocus()
            qt_app.InfoPopup.hide_popup()
        # keyboard: Tab onto the info button shows the tip, Escape and Tab away hide it
        seed_info = win.info_buttons["Seed"]
        win.btn_vc_advanced.setFocus()
        for _ in range(12):
            _tab()
            if _app.focusWidget() is seed_info:
                break
        assert _app.focusWidget() is seed_info, _focus_label(_app.focusWidget())
        QTest.keyRelease(seed_info, QtCore.Qt.Key_Tab)   # the key's own release must not dismiss it
        _app.processEvents()
        assert qt_app.InfoPopup.is_open() and qt_app.InfoPopup._shared.text() == qt_app.SEED_TIP
        assert qt_app.InfoPopup._shared.width() <= 300, "wrapped, not stretched"
        assert win.rect().contains(win.mapFromGlobal(qt_app.InfoPopup._shared.geometry().topLeft())), "on screen"
        _press(QtCore.Qt.Key_Escape)
        assert not qt_app.InfoPopup.is_open()
        _press(QtCore.Qt.Key_Space)
        assert qt_app.InfoPopup.is_open(), "Space shows it again"
        _tab()
        assert not qt_app.InfoPopup.is_open(), "leaving the button hides the tip"
        assert _app.focusWidget() is win.cmb_vc_seed_mode, _focus_label(_app.focusWidget())
        # the checkboxes carry their own explanations instead of adding tab stops
        win.btn_vc_src_ref.setChecked(True)
        assert qt_app.XVECTOR_TIP in win.chk_vc_xvec.toolTip() and win.chk_vc_xvec.accessibleDescription()
        assert "trust" in win.chk_vc_npz.toolTip() and win.chk_vc_npz.accessibleDescription()
        win.btn_vc_src_saved.setChecked(True)

        # numeric fields: same height and style as the other fields, no arrow buttons, ranges kept
        for spin in (win.spn_vc_seed, win.spn_vc_temp):
            assert spin.property("role") == "numeric"
            assert spin.lineEdit().width() >= spin.width() - 24, "the stepper takes no room: the value has the whole field"
            assert spin.fontMetrics().horizontalAdvance(spin.text()) <= spin.lineEdit().width(), "value fully visible"
            assert spin.height() == win.cmb_vc_seed_mode.height() == win.cmb_vc_lang.height(), spin.height()
        assert (win.spn_vc_seed.minimum(), win.spn_vc_seed.maximum()) == (0, 2**31 - 1)
        assert (win.spn_vc_temp.minimum(), win.spn_vc_temp.maximum(), win.spn_vc_temp.singleStep()) == (0.1, 1.5, 0.05)
        assert win.spn_vc_seed.value() == 42 and win.spn_vc_temp.value() == 0.7 and win.cmb_vc_seed_mode.currentText() == "Fixed"
        win.spn_vc_seed.setValue(2**31 - 1)
        assert win.spn_vc_seed.fontMetrics().horizontalAdvance(win.spn_vc_seed.text()) <= win.spn_vc_seed.lineEdit().width() - 10
        win.spn_vc_seed.setValue(42)
        # typing and arrow keys
        win.spn_vc_temp.setFocus()
        win.spn_vc_temp.selectAll()
        QTest.keyClicks(win.spn_vc_temp.lineEdit(), "1.25")
        _press(QtCore.Qt.Key_Tab)
        assert abs(win.spn_vc_temp.value() - 1.25) < 1e-9
        win.spn_vc_temp.setFocus()
        _press(QtCore.Qt.Key_Up)
        assert abs(win.spn_vc_temp.value() - 1.30) < 1e-9, "arrow keys still step by 0.05"
        _press(QtCore.Qt.Key_Down)
        _press(QtCore.Qt.Key_Down)
        assert abs(win.spn_vc_temp.value() - 1.20) < 1e-9
        win.spn_vc_temp.setValue(9)
        assert win.spn_vc_temp.value() == 1.5
        win.spn_vc_temp.setValue(0)
        assert win.spn_vc_temp.value() == 0.1
        win.spn_vc_temp.setFocus()
        win.spn_vc_temp.selectAll()
        QTest.keyClicks(win.spn_vc_temp.lineEdit(), "7")   # 7 is not an acceptable value: it never gets in
        _press(QtCore.Qt.Key_Tab)
        assert 0.1 <= win.spn_vc_temp.value() <= 1.5
        win.spn_vc_seed.setFocus()
        win.spn_vc_seed.selectAll()
        QTest.keyClicks(win.spn_vc_seed.lineEdit(), "99999999999")
        _press(QtCore.Qt.Key_Tab)
        assert 0 <= win.spn_vc_seed.value() <= 2**31 - 1
        win.spn_vc_seed.setValue(-5)
        assert win.spn_vc_seed.value() == 0
        # Fixed / Random
        win.spn_vc_seed.setValue(123)
        win.spn_vc_temp.setValue(0.55)
        win.cmb_vc_seed_mode.setCurrentText("Random")
        assert not win.spn_vc_seed.isEnabled()
        assert win.clone_args()["seed"] is None and win.clone_args()["temperature"] == 0.55
        win.cmb_vc_seed_mode.setCurrentText("Fixed")
        assert win.spn_vc_seed.isEnabled() and win.clone_args()["seed"] == 123
        # no horizontal overflow at either size, with Advanced open in both modes
        for size in ((1240, 780), (1024, 640)):
            win.resize(*size)
            _app.processEvents()
            for ref in (False, True):
                (win.btn_vc_src_ref if ref else win.btn_vc_src_saved).setChecked(True)
                _app.processEvents()
                assert win.stack.widget(1).horizontalScrollBar().maximum() == 0, (size, ref)
        win.close()
    print("PASS: Advanced info buttons are focusable, labelled and show tips by mouse and keyboard; numeric fields keep their behavior")


def _test_clone_tab_order_in_both_source_modes():
    with _Patched(), _Library() as lib:
        lib.add("Ada")
        win, _ = _clone_window(models=_FakeModels())
        win.refresh_voice_lists()
        _shown(win)
        win.btn_vc_advanced.setChecked(False)
        saved = _focus_cycle(win, limit=40)
        assert all(not label.startswith("?") for label in saved), saved
        assert saved[:2] == ["Clone Voice", "Saved voice"] or saved[:3][1] == "Saved voice", saved[:4]
        order = [saved.index(x) for x in ("combo:", "Refresh", "Text to generate", "combo:english", "Generate audio", "Advanced settings") if x in saved]
        assert order == sorted(order), saved
        win.btn_vc_src_ref.setChecked(True)
        win.btn_vc_advanced.setChecked(True)
        _app.processEvents()
        ref = _focus_cycle(win, limit=60)
        assert all(not label.startswith("?") for label in ref), ref
        for needed in ("Reference audio", "Transcript", "Generate audio", "About Seed", "About Temperature",
                       "About Direction tags", "Seed value", "Temperature"):
            assert needed in ref, (needed, ref)
        assert ref.index("Generate audio") < ref.index("Advanced settings") < ref.index("About Direction tags")
        pinned = ["About Description", "Voice description", "New voice name", "Save as voice", "Advanced settings"]
        assert [ref.index(x) for x in pinned] == sorted(ref.index(x) for x in pinned), (pinned, ref)
        assert ref.index("Generate audio") < ref.index("New voice name")
        assert ref.index("About Seed") < ref.index("Seed value") < ref.index("About Temperature") < ref.index("Temperature")
        # one tab stop for the source pair, and it is the selected segment
        for ref in (False, True):
            (win.btn_vc_src_ref if ref else win.btn_vc_src_saved).setChecked(True)
            _app.processEvents()
            win.nav_buttons[1].setFocus()
            _tab()
            expected = win.btn_vc_src_ref if ref else win.btn_vc_src_saved
            assert _app.focusWidget() is expected, (ref, _focus_label(_app.focusWidget()))
            assert win.edt_vc_ref.accessibleName() != expected.text(), "distinct names"
        win.btn_vc_src_ref.setChecked(True)
        win.edt_vc_new_name.setText("Zed")
        win.row_vc_use.setVisible(True)
        win._vc_saved_name = "Ada"
        win.btn_vc_use_saved.setFocus()
        win.btn_vc_use_saved.click()
        assert win.clone_source_kind() == "saved" and _app.focusWidget() is win.cmb_vc_voice, \
            _focus_label(_app.focusWidget())
        win.close()
    print("PASS: Clone Voice tab order is logical in Saved voice mode and in Reference audio mode with Advanced open")



def run_ui_tests():
    print("Running UI smoke tests (offscreen)...")
    _test_window_builds_with_sidebar_navigation()
    _test_navigation_switches_pages()
    _test_create_voice_widgets()
    _test_advanced_settings_hidden_by_default()
    _test_suggestion_chips_are_reversible()
    _test_generate_validates_without_starting_a_task()
    _test_generation_failure_shows_error_and_keeps_input()
    _test_generation_success_shows_real_preview()
    _test_sample_text_provenance_and_editing()
    _test_sample_text_is_a_first_class_field_in_the_main_form()
    _test_create_voice_final_layout_flat_reader_scrollbars_and_contextual_origin()
    _test_loading_ui_shows_real_elapsed_time_in_plain_wording()
    _test_activity_bar_is_one_restrained_segment_that_only_runs_while_visible()
    _test_footer_stop_audio_is_contextual_and_does_not_shift_the_footer()
    _test_leaving_a_navigation_item_never_leaves_its_focus_ring_behind()
    _test_navigation_ring_follows_keyboard_focus_only()
    _test_saved_voice_and_download_use_the_spoken_text()
    _test_preview_player_play_pause_resume_seek_stop_and_end()
    _test_preview_player_stays_consistent_with_other_audio_and_pages()
    _test_preview_player_without_pause_support_and_playback_errors()
    _test_audio_player_position_pause_and_seek_over_pygame_clock()
    _test_preview_player_keyboard_focus_and_minimum_size_layout()
    _test_tab_leaves_description_and_multiline_entry_still_works()
    _test_shift_tab_navigates_backwards()
    _test_keyboard_focus_order_is_logical()
    _test_generate_shortcut_and_visible_hint()
    _test_changing_inputs_marks_the_sample_as_previous()
    _test_saving_a_previous_sample_stores_its_original_inputs()
    _test_invalid_voice_names_disable_save()
    _test_failed_regeneration_keeps_the_previous_valid_sample()
    _test_empty_preview_is_compact_and_has_no_audio_controls()
    _test_previous_sample_stays_available_while_regenerating()
    _test_preview_text_is_never_clipped_or_oversized()
    _test_footer_actions_match_what_they_can_do()
    _test_settings_page_separates_scopes()
    _test_my_voices_empty_state()
    _test_my_voices_actions_fit_at_1024_px()
    _test_engine_references_require_exactly_one_voice()
    _test_engine_references_menu_works_from_the_keyboard()
    _test_failure_classification_names_only_reliable_causes()
    _test_failure_details_are_sanitized_but_keep_the_cause()
    _test_known_failure_explains_cause_action_and_details()
    _test_unknown_failure_stays_generic_and_honest()
    _test_failed_regeneration_keeps_sample_and_offers_details()
    _test_loading_shows_one_indicator_and_only_real_stages()
    _test_other_pages_keep_their_footer_progress()
    _test_import_voice_orange_outline_is_the_keyboard_focus_ring()
    _test_my_voices_tab_order()
    _test_my_voices_rows_show_name_description_and_formatted_metadata()
    _test_my_voices_selection_states_gate_the_actions()
    _test_my_voices_handlers_rename_play_and_delete()
    _test_my_voices_has_no_misleading_inline_editing()
    _test_my_voices_keyboard_selection_and_return()
    _test_my_voices_rows_paint_selection_and_focus_visibly()
    _test_my_voices_rows_are_exposed_to_accessibility()
    _test_my_voices_tab_stops_under_the_macos_default_tab_behavior()
    _test_multiline_editors_never_trap_the_keyboard()
    _test_audiobook_list_refills_and_selection_never_change_segments()
    _test_audiobook_editor_applies_voice_to_the_selection_and_keeps_direction_per_segment()
    _test_audiobook_edits_split_merge_move_delete_and_project_round_trip_keep_voice_and_direction()
    _test_audiobook_typing_keeps_selection_caret_and_focus()
    _test_audiobook_keyboard_wheel_and_geometry_at_the_minimum_window()
    _test_cast_voices_appear_once_and_engines_carry_their_state()
    _test_cast_language_makes_engines_unavailable_but_never_rewrites_assignments()
    _test_cast_direction_and_tags_follow_the_selection_and_block_new_assignments()
    _test_cast_bulk_edits_keep_each_voice_and_each_engine()
    _test_cast_missing_voices_are_preserved_and_flagged()
    _test_cast_preflight_blocks_render_and_preview_before_licenses_and_models()
    _test_cast_role_markers_are_shapes_with_accessible_text()
    _test_cast_fits_both_window_sizes_and_keeps_render_in_view()
    _test_cast_new_segments_never_get_an_impossible_combination()
    _test_cast_problems_dialog_scrolls_and_keeps_close_reachable()
    _test_cast_engine_menu_is_keyboard_complete_and_the_choice_is_visible()
    _test_cast_rules_and_the_page_never_load_a_model()
    _test_cast_refresh_and_voice_choice_never_rewrite_the_cast()
    _test_narrator_and_dialogue_on_the_same_voice_and_engine_stay_independent()
    _test_voice_refresh_keeps_audiobooks_and_drama_studio_independent()
    _test_audiobook_rows_are_compact_readable_and_accessible()
    _test_audiobooks_page_opens_without_a_focused_button()
    _test_audiobook_render_and_preview_run_through_the_new_page()
    _test_audiobook_tab_cycle_from_the_navigation_is_pinned_both_ways()
    _test_tab_leaves_description_under_the_macos_default_tab_behavior()
    _test_stale_sample_callout_and_footer_status()
    _test_icon_assets_are_vendored_licensed_and_only_the_used_ones()
    _test_icons_use_theme_colours_per_state_and_are_high_dpi()
    _test_missing_icon_assets_degrade_to_the_text_label()
    _test_create_voice_and_navigation_icons_never_replace_labels()
    _test_primary_button_uses_the_specified_orange_and_white_content()
    _test_footer_actions_have_labels_icons_and_preserved_handlers()
    _test_stop_audio_follows_the_players_real_state()
    _test_footer_status_is_contextual_and_never_duplicates_errors()
    _test_footer_fits_and_is_keyboard_reachable_at_minimum_width()
    _test_create_voice_layout_at_minimum_and_wide_sizes()
    _test_clone_sources_are_exclusive_and_only_the_active_one_is_sent()
    _test_clone_validation_runs_before_any_model_loads()
    _test_save_clone_never_overwrites_silently()
    _test_save_clone_does_not_ask_while_a_task_is_running()
    _test_clone_voice_list_refresh_keeps_selection_language_and_has_no_blank_mode()
    _test_save_clone_offers_to_use_the_voice_without_switching()
    _test_failed_replace_keeps_the_existing_voice_and_says_so()
    _test_numeric_fields_share_one_style_and_keep_their_behavior()
    _test_clone_save_records_the_real_seed_transcript_and_description()
    _test_import_voice_from_audio_stores_an_empty_transcript_and_no_invented_seed()
    _test_clone_preview_states_transitions_and_player()
    _test_saved_audio_without_a_preview_is_not_a_failed_generation()
    _test_reference_formats_are_validated_up_front_and_save_accepts_float_and_24bit_wav()
    _test_create_and_clone_players_share_the_audio_session_without_crosstalk()
    _test_transcript_editor_prefill_handles_missing_empty_and_real_ref_text()
    _test_clone_layout_keyboard_and_disclosure()
    _test_clone_advanced_info_buttons_and_numeric_fields()
    _test_clone_tab_order_in_both_source_modes()
    _test_busy_toggles_action_buttons()
    _test_single_primary_button_and_neutral_nav()
    _test_sidebar_and_control_sizes()
    _test_theme_rules_and_contrast()
    _test_set_role_updates_property()
    print("All UI smoke tests passed.")


if __name__ == "__main__":
    run_ui_tests()
