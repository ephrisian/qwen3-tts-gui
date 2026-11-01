"""Audiobook tab: split text into segments, assign a voice to each, render one chapter file."""

import os

from PySide6 import QtCore, QtWidgets

import audiobook
from audio_utils import write_wav
from config import DEFAULT_LANGUAGE, LANGUAGES, OUTPUT_DIR
from voice_lock import EMOTION_PRESETS

COL_NUM, COL_TYPE, COL_VOICE, COL_DIR, COL_TEXT = range(5)


class AudiobookTab(QtWidgets.QWidget):
    def __init__(self, win, text_edit, combo, seed_row):
        super().__init__()
        self.win = win
        self.segments = []
        self._loading = False

        outer = QtWidgets.QVBoxLayout(self)

        # ---- source text + auto split
        outer.addWidget(QtWidgets.QLabel("Chapter / scene text:"))
        self.txt_source = text_edit(6, "Paste the chapter here, then click “Split into segments”.")
        self.txt_source.setMaximumHeight(130)
        outer.addWidget(self.txt_source)

        row = QtWidgets.QHBoxLayout()
        self.cmb_narrator = QtWidgets.QComboBox()
        self.cmb_dialogue = QtWidgets.QComboBox()
        self.cmb_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        btn_split = QtWidgets.QPushButton("Split into segments")
        btn_split.setToolTip("Quoted speech becomes dialogue segments; everything else is narration.")
        btn_split.clicked.connect(self.auto_split)
        for label, w in (("Narrator:", self.cmb_narrator), ("Dialogue:", self.cmb_dialogue),
                         ("Language:", self.cmb_lang)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(w)
        row.addWidget(btn_split)
        row.addStretch(1)
        outer.addLayout(row)

        # ---- segment table + editor
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Type", "Voice", "Direction", "Text"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_TEXT, QtWidgets.QHeaderView.Stretch)
        for c in (COL_NUM, COL_TYPE):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setColumnWidth(COL_VOICE, 190)
        self.table.setColumnWidth(COL_DIR, 130)
        self.table.verticalHeader().hide()
        self.table.itemSelectionChanged.connect(self.on_selection)
        ll.addWidget(self.table, 1)

        bulk = QtWidgets.QHBoxLayout()
        self.cmb_bulk = QtWidgets.QComboBox()
        btn_bulk = QtWidgets.QPushButton("Set voice for selected rows")
        btn_bulk.clicked.connect(self.apply_bulk_voice)
        bulk.addWidget(self.cmb_bulk)
        bulk.addWidget(btn_bulk)
        bulk.addStretch(1)
        ll.addLayout(bulk)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("Selected segment text:"))
        self.txt_segment = text_edit(8)
        self.txt_segment.textChanged.connect(self.on_segment_text_changed)
        rl.addWidget(self.txt_segment, 1)
        grid = QtWidgets.QGridLayout()
        buttons = [
            ("Split at cursor", self.split_at_cursor), ("Merge with next", self.merge_next),
            ("Add after", self.add_after), ("Delete", self.delete_selected),
            ("Move up", lambda: self.move(-1)), ("Move down", lambda: self.move(1)),
            ("▶ Preview segment", self.preview_segment),
        ]
        for i, (label, fn) in enumerate(buttons):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
            if label.startswith("▶"):
                self.btn_preview = b
        rl.addLayout(grid)
        split.addWidget(right)
        split.setSizes([700, 350])
        outer.addWidget(split, 3)

        # ---- render settings
        srow = QtWidgets.QHBoxLayout()
        self.spn_pause = QtWidgets.QSpinBox()
        self.spn_pause.setRange(0, 5000)
        self.spn_pause.setSingleStep(50)
        self.spn_pause.setSuffix(" ms")
        self.spn_pause.setValue(350)
        self.chk_loudness = QtWidgets.QCheckBox("Match loudness between voices")
        self.chk_loudness.setChecked(True)
        self.cmb_seed_mode, self.spn_seed = seed_row("Fixed", 42)
        self.spn_temp = QtWidgets.QDoubleSpinBox()
        self.spn_temp.setRange(0.1, 1.5)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(0.7)
        for label, w in (("Pause between segments:", self.spn_pause), ("", self.chk_loudness),
                         ("Seed:", self.cmb_seed_mode), ("", self.spn_seed),
                         ("Clone temperature:", self.spn_temp)):
            if label:
                srow.addWidget(QtWidgets.QLabel(label))
            srow.addWidget(w)
        srow.addStretch(1)
        outer.addLayout(srow)

        note = QtWidgets.QLabel(
            "Unchanged segments are cached, so re-rendering after an edit only regenerates what changed. "
            "Direction accepts any free text (same as [tag] in the other tabs).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        outer.addWidget(note)

        # ---- action bar
        bar = QtWidgets.QHBoxLayout()
        self.btn_render = QtWidgets.QPushButton("Render chapter")
        self.btn_render.clicked.connect(self.render_chapter)
        btn_save = QtWidgets.QPushButton("Save project…")
        btn_save.clicked.connect(self.save_project)
        btn_load = QtWidgets.QPushButton("Load project…")
        btn_load.clicked.connect(self.load_project)
        for b in (self.btn_render, btn_save, btn_load):
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        self.refresh_voices()

    # ------------------------------------------------------------------ voices
    def refresh_voices(self):
        choices = audiobook.voice_choices()
        for cmb in (self.cmb_narrator, self.cmb_dialogue, self.cmb_bulk):
            current = cmb.currentData()
            cmb.clear()
            for label, key in choices:
                cmb.addItem(label, key)
            idx = cmb.findData(current)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
        if self.cmb_dialogue.count() > 1 and self.cmb_dialogue.currentIndex() == self.cmb_narrator.currentIndex():
            self.cmb_dialogue.setCurrentIndex(1)
        self._loading = True
        for r in range(self.table.rowCount()):
            self._fill_voice_combo(self.table.cellWidget(r, COL_VOICE), self.segments[r]["voice"])
        self._loading = False

    def _fill_voice_combo(self, cmb, current):
        cmb.clear()
        for label, key in audiobook.voice_choices():
            cmb.addItem(label, key)
        idx = cmb.findData(current)
        if idx < 0 and current:
            cmb.addItem(f"Missing: {current}", current)
            idx = cmb.count() - 1
        cmb.setCurrentIndex(idx)

    # ------------------------------------------------------------------ table <-> segments
    def rebuild(self, select=None):
        self._loading = True
        self.table.setRowCount(len(self.segments))
        for r, seg in enumerate(self.segments):
            self.table.setItem(r, COL_NUM, QtWidgets.QTableWidgetItem(str(r + 1)))
            self.table.setItem(r, COL_TYPE, QtWidgets.QTableWidgetItem(seg["kind"]))
            self.table.setItem(r, COL_TEXT, QtWidgets.QTableWidgetItem(self._preview(seg["text"])))
            vc = QtWidgets.QComboBox()
            self._fill_voice_combo(vc, seg["voice"])
            vc.currentIndexChanged.connect(lambda _=0, c=vc: self.on_voice_changed(c))
            self.table.setCellWidget(r, COL_VOICE, vc)
            dc = QtWidgets.QComboBox()
            dc.setEditable(True)
            dc.addItems([""] + list(EMOTION_PRESETS))
            dc.setCurrentText(seg.get("direction", ""))
            dc.currentTextChanged.connect(lambda _t, c=dc: self.on_direction_changed(c))
            self.table.setCellWidget(r, COL_DIR, dc)
        self._loading = False
        if select is not None and self.segments:
            self.table.selectRow(max(0, min(select, len(self.segments) - 1)))
        else:
            self.on_selection()

    @staticmethod
    def _preview(text):
        flat = " ".join(text.split())
        return flat if len(flat) <= 140 else flat[:137] + "…"

    def _row_of(self, widget, col):
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, col) is widget:
                return r
        return -1

    def on_voice_changed(self, cmb):
        if self._loading:
            return
        r = self._row_of(cmb, COL_VOICE)
        if r >= 0:
            self.segments[r]["voice"] = cmb.currentData()

    def on_direction_changed(self, cmb):
        if self._loading:
            return
        r = self._row_of(cmb, COL_DIR)
        if r >= 0:
            self.segments[r]["direction"] = cmb.currentText().strip()

    def selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def current_row(self):
        rows = self.selected_rows()
        return rows[0] if rows else -1

    def on_selection(self):
        r = self.current_row()
        self._loading = True
        self.txt_segment.setPlainText(self.segments[r]["text"] if r >= 0 else "")
        self.txt_segment.document().clearUndoRedoStacks()
        self.txt_segment.setEnabled(r >= 0)
        self._loading = False

    def on_segment_text_changed(self):
        if self._loading:
            return
        r = self.current_row()
        if r >= 0:
            text = self.txt_segment.toPlainText()
            self.segments[r]["text"] = text
            self.table.item(r, COL_TEXT).setText(self._preview(text))

    # ------------------------------------------------------------------ editing actions
    def auto_split(self):
        text = self.txt_source.toPlainText().strip()
        if not text:
            self.win.status("Error: paste some text first.")
            return
        if self.segments:
            answer = QtWidgets.QMessageBox.question(
                self, "Replace segments?", "Replace the current segments with a new split?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        narrator = self.cmb_narrator.currentData()
        dialogue = self.cmb_dialogue.currentData() or narrator
        self.segments = audiobook.split_narration_dialogue(text, narrator, dialogue)
        self.rebuild(select=0)
        self.win.status(f"Split into {len(self.segments)} segments. Assign voices, then render.")

    def apply_bulk_voice(self):
        key = self.cmb_bulk.currentData()
        for r in self.selected_rows():
            self.segments[r]["voice"] = key
        keep = self.selected_rows()
        self.rebuild()
        sel = self.table.selectionModel()
        sel.clearSelection()
        for r in keep:
            sel.select(self.table.model().index(r, 0),
                       QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    def split_at_cursor(self):
        r = self.current_row()
        if r < 0:
            return
        pos = self.txt_segment.textCursor().position()
        parts = audiobook.split_segment_at(self.segments[r], pos)
        if len(parts) == 2:
            self.segments[r:r + 1] = parts
            self.rebuild(select=r + 1)

    def merge_next(self):
        r = self.current_row()
        if 0 <= r < len(self.segments) - 1:
            self.segments[r:r + 2] = [audiobook.merge_segments(self.segments[r], self.segments[r + 1])]
            self.rebuild(select=r)

    def add_after(self):
        r = self.current_row()
        voice = self.cmb_narrator.currentData() or ""
        self.segments.insert(r + 1, audiobook.new_segment("New segment.", voice))
        self.rebuild(select=r + 1)

    def delete_selected(self):
        rows = self.selected_rows()
        if not rows:
            return
        for r in reversed(rows):
            del self.segments[r]
        self.rebuild(select=rows[0])

    def move(self, delta):
        r = self.current_row()
        t = r + delta
        if r < 0 or not 0 <= t < len(self.segments):
            return
        self.segments[r], self.segments[t] = self.segments[t], self.segments[r]
        self.rebuild(select=t)

    # ------------------------------------------------------------------ rendering
    def render_settings(self):
        return dict(
            language=self.cmb_lang.currentText(),
            min_words=self.win.spn_min.value(), max_words=self.win.spn_max.value(),
            seed=self.spn_seed.value() if self.cmb_seed_mode.currentText() == "Fixed" else None,
            temperature=round(self.spn_temp.value(), 3),
        )

    def _check_segments(self, segs):
        if not segs:
            self.win.status("Error: there are no segments.")
            return False
        for s in segs:
            if not s["text"].strip():
                self.win.status("Error: a segment has no text.")
                return False
            if not s["voice"]:
                self.win.status("Error: every segment needs a voice.")
                return False
        return True

    def preview_segment(self):
        r = self.current_row()
        if r < 0:
            return
        seg = dict(self.segments[r])
        if not self._check_segments([seg]):
            return
        settings = self.render_settings()

        def work(task):
            return audiobook.render_segment(self.win.generator(task), seg, settings)

        def done(result):
            wav, sr = result
            path = os.path.join(OUTPUT_DIR, "_segment_preview.wav")
            write_wav(path, wav, sr)
            self.win.status(f"Previewing segment {r + 1}.")
            self.win.play(path)

        self.win.run_task(work, done)

    def render_chapter(self):
        segs = [dict(s) for s in self.segments]
        if not self._check_segments(segs):
            return
        settings = self.render_settings()
        pause, loudness = self.spn_pause.value(), self.chk_loudness.isChecked()

        def work(task):
            gen = self.win.generator(task)
            results = audiobook.render_chapter(
                gen, segs, settings, should_stop=task.stop_event.is_set, notify=task.notify.emit)
            if results is None:
                return None
            return audiobook.compile_audio(results, pause_ms=pause, match_loudness=loudness)

        def done(result):
            if result is None:
                self.win.status("Stopped. Finished segments are cached; render again to resume.")
                return
            wav, sr = result
            path = self.win.output_path("Audiobook")
            write_wav(path, wav, sr)
            self.win.last_output = path
            self.win.btn_play_last.setEnabled(True)
            self.win.status(f"Chapter rendered ({len(segs)} segments, {len(wav) / sr / 60:.1f} min): {path}")

        self.win.run_task(work, done)

    # ------------------------------------------------------------------ project files
    def project_meta(self):
        return dict(source=self.txt_source.toPlainText(), language=self.cmb_lang.currentText(),
                    pause_ms=self.spn_pause.value(), loudness=self.chk_loudness.isChecked())

    def save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "", "Project (*.json)")
        if path:
            audiobook.save_project(path, self.segments, self.project_meta())
            self.win.status(f"Project saved: {path}")

    def load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", "", "Project (*.json)")
        if not path:
            return
        self.segments, meta = audiobook.load_project(path)
        self.txt_source.setPlainText(meta.get("source", ""))
        self.cmb_lang.setCurrentText(meta.get("language", DEFAULT_LANGUAGE))
        self.spn_pause.setValue(meta.get("pause_ms", 350))
        self.chk_loudness.setChecked(meta.get("loudness", True))
        self.rebuild(select=0)
        self.win.status(f"Project loaded: {path}")
