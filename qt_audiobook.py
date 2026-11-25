"""Audiobook tab: a chapter script, a list of segments, and one editor for the selected segment(s); render one file.

Layout (Phase A1): a fixed header row (title, project actions, Render chapter) above a body with two columns.
Left: the chapter script, who reads it, and the render settings. Right: the segment list and the editor for what is
selected. The segments themselves are plain dicts (text, voice, kind, direction) exactly as before, so projects,
the cache and rendering are untouched.
"""

import os
import textwrap

from PySide6 import QtCore, QtGui, QtWidgets

import audiobook
import cast_rules
import icons
import voice_library
from audio_utils import write_wav
from config import DEFAULT_LANGUAGE, LANGUAGES, OUTPUT_DIR
from theme import COLORS, FONT_PX, set_role, style_numeric
from voice_list import NAME_ROLE, VoiceList
from voice_lock import EMOTION_PRESETS

DESC_ROLE, ROLE_ROLE, NUM_ROLE, STATE_ROLE = (QtCore.Qt.UserRole + 2, QtCore.Qt.UserRole + 3, QtCore.Qt.UserRole + 4,
                                              QtCore.Qt.UserRole + 5)
ROW_H = 30          # one line per segment: about twice the rows of the two-line My Voices anatomy, same type size
NO_TEXT = "(empty segment)"


class SegmentRowDelegate(QtWidgets.QStyledItemDelegate):
    """One compact line per segment:  12  N  Ana   [sad] The rain fell on the old harbour town…

    number | N (narration) or D (dialogue) | short voice | text excerpt. Everything is painted from the item's data, so
    the rows stay ordinary list items: selection, keyboard handling and the accessibility tree come from Qt, and the
    item's accessible text reads the whole row. The same type size as the rest of the app; only the second line is gone."""

    NUM_W, ROLE_W, VOICE_MAX, PAD, GLYPH = 30, 24, 130, 12, 16

    def sizeHint(self, option, index):
        return QtCore.QSize(1, ROW_H)

    def paint(self, painter, option, index):
        State = QtWidgets.QStyle
        selected = bool(option.state & State.State_Selected)
        hovered = bool(option.state & State.State_MouseOver)
        rect = option.rect
        painter.save()
        painter.fillRect(rect, QtGui.QColor(COLORS["selection"] if selected else COLORS["surface_hover"] if hovered
                                            else COLORS["surface"]))
        painter.fillRect(QtCore.QRect(rect.left(), rect.bottom(), rect.width(), 1), QtGui.QColor(COLORS["border"]))
        if selected:   # selection is shape as well as colour: an accent bar on the left edge
            painter.fillRect(QtCore.QRect(rect.left(), rect.top(), 3, rect.height() - 1), QtGui.QColor(COLORS["accent"]))
        font = QtGui.QFont(option.font)
        font.setPixelSize(FONT_PX["body"])
        bold = QtGui.QFont(font)
        bold.setWeight(QtGui.QFont.DemiBold)
        muted = QtGui.QColor(COLORS["text_secondary"] if selected else COLORS["text_muted"])
        text_color = QtGui.QColor(COLORS["text"])
        inner = rect.adjusted(self.PAD, 0, -self.PAD, -1)
        x = inner.left()

        def cell(width, align=QtCore.Qt.AlignLeft):
            return QtCore.QRect(x, rect.top(), width, rect.height() - 1), align | QtCore.Qt.AlignVCenter
        painter.setFont(font)
        painter.setPen(muted)
        box, align = cell(self.NUM_W - 6, QtCore.Qt.AlignRight)
        painter.drawText(box, align, str(index.data(NUM_ROLE) or ""))
        x += self.NUM_W
        role = index.data(ROLE_ROLE) or ""
        # Narration and dialogue differ by shape, not only by colour: an open book, or a quotation mark.
        glyph = icons.icon("quote" if role == "D" else "book-open", self.GLYPH, gap=0)
        glyph_box = QtCore.QRect(x + (self.ROLE_W - self.GLYPH) // 2, rect.top() + (rect.height() - 1 - self.GLYPH) // 2,
                                 self.GLYPH, self.GLYPH)
        glyph.paint(painter, glyph_box, QtCore.Qt.AlignCenter,
                    QtGui.QIcon.Active if role == "D" else QtGui.QIcon.Normal)
        x += self.ROLE_W
        metrics = QtGui.QFontMetrics(bold)
        voice_w = min(self.VOICE_MAX, max(60, inner.width() // 4))
        painter.setPen(text_color)
        box, align = cell(voice_w)
        painter.drawText(box, align, metrics.elidedText(index.data(NAME_ROLE) or "", QtCore.Qt.ElideRight, voice_w))
        x += voice_w + 10
        painter.setFont(font)
        painter.setPen(muted if selected else QtGui.QColor(COLORS["text_secondary"]))
        state = index.data(STATE_ROLE) or cast_rules.READY
        flag = 0 if state == cast_rules.READY else self.GLYPH + 8
        room = max(20, inner.right() - x - flag)
        box, align = cell(room)
        excerpt = QtGui.QFontMetrics(font).elidedText(index.data(DESC_ROLE) or "", QtCore.Qt.ElideRight, room)
        painter.drawText(box, align, excerpt)
        if flag:    # something to fix before this segment can render; the words are in the tooltip and accessible text
            warn = icons.icon("circle-exclamation-point", self.GLYPH, "default" if state == cast_rules.SETUP else "error", gap=0)
            warn.paint(painter, QtCore.QRect(inner.right() - self.GLYPH, rect.top() + (rect.height() - 1 - self.GLYPH) // 2,
                                             self.GLYPH, self.GLYPH), QtCore.Qt.AlignCenter)
        if option.state & State.State_HasFocus:
            painter.setPen(QtGui.QPen(QtGui.QColor(COLORS["accent"]), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect.adjusted(1, 1, -2, -2))
        painter.restore()


class _Combo(QtWidgets.QComboBox):
    """Never wider than the page asks, ignores the mouse wheel (so scrolling the page can't silently change a voice),
    and shows the full text as a tooltip when the box elides it."""

    def __init__(self, parent=None, min_chars=8, min_width=None):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(min_chars)
        if min_width:
            self.setMinimumWidth(min_width)     # the page may squeeze this combo to elide; it never makes the page wider
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.currentIndexChanged.connect(lambda _i: not self.isEditable() and self.setToolTip(self.currentText()))

    def showPopup(self):
        view = self.view()    # the list may be wider than the (elided) box, so full voice names stay readable
        view.setMinimumWidth(min(view.sizeHintForColumn(0) + 40, 640))
        super().showPopup()

    def wheelEvent(self, event):
        event.ignore()    # scrolling the page must never change a voice, focused or not; the keyboard and the popup do


class _WheelGuard(QtCore.QObject):
    """Event filter: the wheel over a setting never changes it (a changed setting would re-render every segment)."""

    def eventFilter(self, obj, event):
        return event.type() == QtCore.QEvent.Wheel


class _ElidedLabel(QtWidgets.QLabel):
    """One line that never widens the page: long text is elided, and the whole text stays in the tooltip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(20)

    def set_full_text(self, text):
        self._full = text
        self._elide()

    def minimumSizeHint(self):      # a label's own hint is as wide as its text; this one may always shrink
        return QtCore.QSize(20, super().minimumSizeHint().height())

    def sizeHint(self):
        return QtCore.QSize(60, super().sizeHint().height())

    def full_text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, max(20, self.width())))


class _StatusLine(QtWidgets.QWidget):
    """A glyph and one line of words about a voice and engine: never colour alone, never taller than one line."""

    GLYPHS = {cast_rules.READY: ("circle-check", "default"), cast_rules.SETUP: ("circle-exclamation-point", "default"),
              cast_rules.UNSUPPORTED: ("circle-exclamation-point", "error"), cast_rules.MISSING: ("circle-exclamation-point", "error")}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # it yields width, never asks for it
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.glyph = QtWidgets.QLabel()
        self.glyph.setFixedSize(16, 16)
        self.glyph.setAccessibleName("")
        self.text = _ElidedLabel()
        set_role(self.text, "caption")
        row.addWidget(self.glyph)
        row.addWidget(self.text, 1)
        self.state = cast_rules.READY
        self.set_status(None, "")

    def minimumSizeHint(self):
        return QtCore.QSize(0, 16)

    def sizeHint(self):
        return QtCore.QSize(120, 16)

    def set_status(self, state, text, detail=""):
        self.state = state
        name, role = self.GLYPHS.get(state, (None, None))
        self.glyph.setPixmap(icons.icon(name, 16, role, 0).pixmap(16, 16) if name and text else QtGui.QPixmap())
        self.text.set_full_text(text)
        self.setToolTip(detail or text)
        self.text.setAccessibleName(text)

    def full_text(self):
        return self.text.full_text()


def _field(text, buddy=None):
    label = QtWidgets.QLabel(text)
    set_role(label, "field")
    if buddy is not None:
        label.setBuddy(buddy)
    return label


class AudiobookTab(QtWidgets.QWidget):
    def __init__(self, win, text_edit, combo, seed_row, header=None):
        super().__init__()
        self.win = win
        self.segments = []
        self._loading = False
        self._problem_dialog = None
        self._problem_groups = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 12)
        outer.setSpacing(12)

        # ---- header: what this page is, and the chapter-level actions (never scrolls away)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        if header is not None:
            top.addWidget(header("Audiobooks", "Split a chapter, choose voices, render one audio file."), 1)
        else:
            top.addWidget(QtWidgets.QLabel("Audiobooks"), 1)
        self.btn_save = QtWidgets.QPushButton("Save project…")
        self.btn_save.clicked.connect(self.save_project)
        self.btn_load = QtWidgets.QPushButton("Load project…")
        self.btn_load.clicked.connect(self.load_project)
        self.btn_render = QtWidgets.QPushButton("Render chapter")
        set_role(self.btn_render, "primary")
        self.btn_render.setToolTip("Unchanged segments are cached, so rendering again after an edit only regenerates what changed.")
        self.btn_render.clicked.connect(self.render_chapter)
        for b in (self.btn_save, self.btn_load, self.btn_render):
            top.addWidget(b, 0, QtCore.Qt.AlignVCenter)
        outer.addLayout(top)

        # ---- body (scrolls only if the window is smaller than the minimum)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(body)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(24)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        cols.addLayout(self._build_script_column(text_edit, seed_row), 5)
        cols.addLayout(self._build_segment_column(text_edit), 6)

        self._chain()
        self.cmb_lang.currentTextChanged.connect(lambda _text: self.refresh_compat())
        self.refresh_voices()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_script_column(self, text_edit, seed_row):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        self.txt_source = text_edit(5, "Paste the chapter here, then choose Split into segments.")
        self.txt_source.setAccessibleName("Chapter script")
        self.txt_source.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        col.addWidget(_field("Chapter script", self.txt_source))
        col.addWidget(self.txt_source, 1)

        self.cmb_narrator = _Combo()
        self.cmb_dialogue = _Combo()
        self.cmb_narrator_engine = _Combo(min_chars=5)
        self.cmb_dialogue_engine = _Combo(min_chars=5)
        self.status_narrator = _StatusLine()
        self.status_dialogue = _StatusLine()
        self._role_engine = {"narrator": "qwen", "dialogue": "qwen"}
        self.cmb_lang = _Combo(min_chars=8)
        self.cmb_lang.addItems(LANGUAGES)
        self.cmb_lang.setCurrentText(DEFAULT_LANGUAGE)
        for w, name in ((self.cmb_narrator, "Narrator voice"), (self.cmb_dialogue, "Dialogue voice"),
                        (self.cmb_narrator_engine, "Narrator engine"), (self.cmb_dialogue_engine, "Dialogue engine"),
                        (self.cmb_lang, "Language")):
            w.setAccessibleName(name)
        self.cmb_narrator.activated.connect(lambda _i: self.on_role_changed("narrator"))
        self.cmb_dialogue.activated.connect(lambda _i: self.on_role_changed("dialogue"))
        self.cmb_narrator_engine.activated.connect(lambda i: self.on_role_engine_chosen("narrator", i))
        self.cmb_dialogue_engine.activated.connect(lambda i: self.on_role_engine_chosen("dialogue", i))
        cast = QtWidgets.QGridLayout()
        cast.setHorizontalSpacing(12)
        cast.setVerticalSpacing(4)
        # one role per row, voice then engine: the script column has height to spare, never width
        for row_index, (label, voice, engine, status) in enumerate((
                ("Narrator voice", self.cmb_narrator, self.cmb_narrator_engine, self.status_narrator),
                ("Dialogue voice", self.cmb_dialogue, self.cmb_dialogue_engine, self.status_dialogue))):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(4)
            names = QtWidgets.QHBoxLayout()
            names.setSpacing(6)
            names.addWidget(_field(label, voice), 3)
            names.addWidget(_field("Engine", engine), 2)
            box.addLayout(names)
            combos = QtWidgets.QHBoxLayout()
            combos.setSpacing(6)
            combos.addWidget(voice, 3)
            combos.addWidget(engine, 2)
            box.addLayout(combos)
            box.addWidget(status)
            cast.addLayout(box, row_index, 0, 1, 2)
        cast.addWidget(_field("Language", self.cmb_lang), 2, 0)
        cast.addWidget(self.cmb_lang, 3, 0)
        self.btn_split = QtWidgets.QPushButton("Split into segments")
        self.btn_split.setToolTip("Quoted speech becomes dialogue segments; everything else is narration.")
        self.btn_split.clicked.connect(self.auto_split)
        cast.addWidget(self.btn_split, 3, 1)
        cast.setColumnStretch(0, 1)
        cast.setColumnStretch(1, 1)
        col.addLayout(cast)
        hint = QtWidgets.QLabel("Quoted speech becomes dialogue.")   # one line: no height-for-width, so the page never grows with the window width
        set_role(hint, "caption")
        col.addWidget(hint)

        # ---- render settings (collapsed: the defaults are right for most chapters)
        self.btn_settings = QtWidgets.QPushButton("Render settings")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setAccessibleName("Render settings")
        set_role(self.btn_settings, "quiet")
        icons.apply(self.btn_settings, "chevron-right")
        self.btn_settings.setAccessibleDescription("Collapsed")
        self.btn_settings.toggled.connect(self._toggle_settings)
        col.addWidget(self.btn_settings, 0, QtCore.Qt.AlignLeft)

        self.pnl_settings = QtWidgets.QWidget()
        form = QtWidgets.QGridLayout(self.pnl_settings)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.spn_pause = QtWidgets.QSpinBox()
        self.spn_pause.setRange(0, 5000)
        self.spn_pause.setSingleStep(50)
        self.spn_pause.setSuffix(" ms")
        self.spn_pause.setValue(350)
        style_numeric(self.spn_pause, width="fit", name="Pause between segments")
        self.chk_loudness = QtWidgets.QCheckBox("Match loudness between voices")
        self.chk_loudness.setChecked(True)
        self.cmb_seed_mode, self.spn_seed = seed_row("Fixed", 42)
        self.cmb_seed_mode.setAccessibleName("Seed mode")
        self.spn_seed.setAccessibleName("Seed value")
        self.spn_temp = QtWidgets.QDoubleSpinBox()
        self.spn_temp.setRange(0.1, 1.5)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(0.7)
        style_numeric(self.spn_temp, width="fit", name="Clone temperature")
        form.addWidget(_field("Pause between segments", self.spn_pause), 0, 0)
        form.addWidget(self.spn_pause, 0, 1)
        form.addWidget(self.chk_loudness, 1, 0, 1, 2)
        form.addWidget(_field("Seed", self.cmb_seed_mode), 2, 0)
        seed = QtWidgets.QHBoxLayout()
        seed.addWidget(self.cmb_seed_mode)
        seed.addWidget(self.spn_seed)
        form.addLayout(seed, 2, 1)
        form.addWidget(_field("Clone temperature", self.spn_temp), 3, 0)
        form.addWidget(self.spn_temp, 3, 1)
        form.setColumnStretch(1, 1)
        self._wheel_guard = _WheelGuard(self)
        for w in (self.spn_pause, self.cmb_seed_mode, self.spn_seed, self.spn_temp):
            w.installEventFilter(self._wheel_guard)
            if hasattr(w, "lineEdit") and w.lineEdit() is not None:
                w.lineEdit().installEventFilter(self._wheel_guard)
        note = QtWidgets.QLabel("Unchanged segments are cached, so rendering again after an edit only regenerates what "
                                "changed. Direction accepts any free text, like [tag] in Clone Voice.")
        note.setWordWrap(True)
        set_role(note, "caption")
        form.addWidget(note, 4, 0, 1, 2)
        self.pnl_settings.setVisible(False)
        col.addWidget(self.pnl_settings)
        return col

    def _build_segment_column(self, text_edit):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(6)
        self.lbl_count = QtWidgets.QLabel("Segments")
        set_role(self.lbl_count, "field")
        col.addWidget(self.lbl_count)
        self.listw = VoiceList()
        set_role(self.listw, "voices")
        self.listw.setItemDelegate(SegmentRowDelegate(self.listw))
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listw.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.listw.setMouseTracking(True)
        self.listw.setAccessibleName("Segments")
        self.listw.setMinimumHeight(ROW_H * 3)
        self.listw.itemSelectionChanged.connect(self.on_selection)
        col.addWidget(self.listw, 1)

        title_row = QtWidgets.QHBoxLayout()
        self.lbl_seg_title = QtWidgets.QLabel("Selected segment")
        set_role(self.lbl_seg_title, "field")
        title_row.addWidget(self.lbl_seg_title)
        self.status_segment = _StatusLine()     # same line as the title: one row, so the list keeps its height
        title_row.addWidget(self.status_segment, 1)
        self.btn_preview = QtWidgets.QPushButton("▶ Preview segment")
        self.btn_preview.setAccessibleName("Preview segment")
        self.btn_preview.clicked.connect(self.preview_segment)
        title_row.addWidget(self.btn_preview)
        col.addLayout(title_row)
        self.txt_segment = text_edit(3)
        self.txt_segment.setMaximumHeight(self.txt_segment.fontMetrics().lineSpacing() * 3 + 16)   # the list gets the spare height
        self.txt_segment.setAccessibleName("Selected segment text")
        self.txt_segment.textChanged.connect(self.on_segment_text_changed)
        col.addWidget(self.txt_segment)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.cmb_seg_voice = _Combo(min_chars=6, min_width=100)
        self.cmb_seg_voice.setAccessibleName("Voice for the selected segments")
        self.cmb_seg_voice.setPlaceholderText("Choose a voice")
        self.cmb_seg_voice.activated.connect(self.on_voice_chosen)
        self.cmb_seg_engine = _Combo(min_chars=5, min_width=84)
        self.cmb_seg_engine.setAccessibleName("Engine for the selected segments")
        self.cmb_seg_engine.setPlaceholderText("Engine")
        self.cmb_seg_engine.activated.connect(self.on_engine_chosen)
        self.cmb_seg_direction = _Combo(min_chars=5, min_width=90)
        self.cmb_seg_direction.setEditable(True)
        self.cmb_seg_direction.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.cmb_seg_direction.addItems([""] + list(EMOTION_PRESETS))
        self.cmb_seg_direction.setAccessibleName("Direction for the selected segment")
        self.cmb_seg_direction.lineEdit().setPlaceholderText("e.g. whisper, sad, or any style")
        # editTextChanged also covers auto-completion and picking a preset; programmatic refills run under _loading
        self.cmb_seg_direction.editTextChanged.connect(self.on_direction_edited)
        for label, w, stretch in (("Voice", self.cmb_seg_voice, 3), ("Engine", self.cmb_seg_engine, 2),
                                  ("Direction", self.cmb_seg_direction, 3)):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(4)
            box.addWidget(_field(label, w))
            box.addWidget(w)
            row.addLayout(box, stretch)
        col.addLayout(row)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        specs = [
            ("btn_split_cursor", "Split here", "Split the segment at the text cursor", self.split_at_cursor),
            ("btn_merge", "Merge next", "Merge the segment with the next one", self.merge_next),
            ("btn_add", "Add after", "Add a new segment after the selection", self.add_after),
            ("btn_up", "Move up", "Move the segment up", lambda: self.move(-1)),
            ("btn_down", "Move down", "Move the segment down", lambda: self.move(1)),
            ("btn_delete", "Delete", "Delete the selected segments", self.delete_selected),
        ]
        for i, (attr, label, name, fn) in enumerate(specs):
            b = QtWidgets.QPushButton(label)
            b.setAccessibleName(name)
            b.setToolTip(name)
            b.clicked.connect(fn)
            setattr(self, attr, b)
            grid.addWidget(b, i // 3, i % 3)
        col.addLayout(grid)
        return col

    def _chain(self):
        order = [self.btn_save, self.btn_load, self.btn_render, self.txt_source, self.cmb_narrator,
                 self.cmb_narrator_engine, self.cmb_dialogue, self.cmb_dialogue_engine, self.cmb_lang, self.btn_split, self.btn_settings, self.spn_pause, self.chk_loudness,
                 self.cmb_seed_mode, self.spn_seed, self.spn_temp, self.listw, self.btn_preview, self.txt_segment,
                 self.cmb_seg_voice, self.cmb_seg_engine, self.cmb_seg_direction, self.btn_split_cursor, self.btn_merge, self.btn_add,
                 self.btn_up, self.btn_down, self.btn_delete]
        for first, second in zip(order, order[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def _toggle_settings(self, on):
        self.pnl_settings.setVisible(on)
        icons.apply(self.btn_settings, "chevron-down" if on else "chevron-right")
        self.btn_settings.setAccessibleDescription("Expanded" if on else "Collapsed")

    # ------------------------------------------------------------------ voices and engines
    # A voice is picked by name (once); the engine is a second control. The segment still stores "engine:name",
    # exactly as before, so projects, the cache and rendering are untouched. cast_rules decides what is possible.
    def _language(self):
        return self.cmb_lang.currentText()

    def _fill_names(self, cmb, current=None, first_by_default=False):
        """Every saved voice once; a voice that no longer exists stays visible as 'Missing: …'."""
        cmb.clear()
        for name in voice_library.list_voices():
            cmb.addItem(name, name)
        if current:
            idx = cmb.findData(current)
            if idx < 0:
                cmb.addItem(f"Missing: {current}", current)
                idx = cmb.count() - 1
            cmb.setCurrentIndex(idx)
        else:
            cmb.setCurrentIndex(0 if first_by_default and cmb.count() else -1)

    def _fill_engines(self, cmb, targets, current, memo, env):
        """All five engines, each with its state for these targets: [(voice name, direction, text, engine now)].
        Needs-setup engines stay selectable (the project can be prepared); an engine that cannot speak the combination
        is disabled for anything that does not already use it, so an existing invalid choice stays visible and editable."""
        cmb.clear()
        unavailable = []
        language = self._language()
        for engine in cast_rules.ENGINES:
            worst, first, blocked = cast_rules.READY, None, False
            for name, direction, text, own in targets:
                if name:
                    result = cast_rules.check(cast_rules.make_key(engine, name), language, direction, text, env=env, memo=memo)
                    state, issue = result.state, (result.issues[0] if result.issues else None)
                else:
                    found = cast_rules.combination_issues(engine, language, direction, text)
                    state, issue = (cast_rules.UNSUPPORTED, found[0]) if found else (cast_rules.READY, None)
                if cast_rules.SEVERITY[state] > cast_rules.SEVERITY[worst]:
                    worst, first = state, issue
                if state == cast_rules.UNSUPPORTED and own != engine:
                    blocked = True
            cmb.addItem(cast_rules.LABEL[engine] + (f" · {first.short}" if first else ""), engine)
            row = cmb.count() - 1
            cmb.setItemData(row, first.message if first else cast_rules.TRAITS[engine], QtCore.Qt.ToolTipRole)
            if blocked:
                cmb.model().item(row).setEnabled(False)
                unavailable.append(f"{cast_rules.LABEL[engine]} ({first.short})")
        cmb.setCurrentIndex(cmb.findData(current) if current else -1)
        cmb.setAccessibleDescription("Unavailable: " + ", ".join(unavailable) if unavailable else "")
        return unavailable

    @staticmethod
    def _describe(result):
        """(state, one line, tooltip) for a Result."""
        state = result.state
        if state == cast_rules.READY:
            return state, "Ready (local checks only)", " ".join(result.notes)
        return state, f"{cast_rules.STATE_TEXT[state]}: {result.message}", \
            "\n".join([i.message for i in result.issues] + list(result.notes))

    def _role_widgets(self, role):
        return ((self.cmb_narrator, self.cmb_narrator_engine, self.status_narrator) if role == "narrator"
                else (self.cmb_dialogue, self.cmb_dialogue_engine, self.status_dialogue))

    def _refresh_role(self, role, memo, env):
        voice, engine_combo, status = self._role_widgets(role)
        name, engine = voice.currentData(), self._role_engine[role]
        self._fill_engines(engine_combo, [(name, "", "", engine)], engine, memo, env)
        if not name:
            status.set_status(None, "Choose a voice to see which engines can use it.")
            detail = ""
        else:
            state, text, detail = self._describe(cast_rules.check(cast_rules.make_key(engine, name), self._language(),
                                                                 env=env, memo=memo))
            status.set_status(state, text, detail)
        voice.setAccessibleDescription(status.full_text())
        engine_combo.setAccessibleDescription(
            " ".join(part for part in (status.full_text(), engine_combo.accessibleDescription()) if part))

    def narrator_key(self):
        name = self.cmb_narrator.currentData()
        return cast_rules.make_key(self._role_engine["narrator"], name) if name else ""

    def dialogue_key(self):
        name = self.cmb_dialogue.currentData()
        return cast_rules.make_key(self._role_engine["dialogue"], name) if name else ""

    def on_role_changed(self, role):
        """The user picked a voice for Narrator or Dialogue. A voice without a Qwen clone prompt moves to the first
        engine that can use it; nothing else is rewritten."""
        if self._loading:
            return
        voice, _engine_combo, _status = self._role_widgets(role)
        name, memo, env = voice.currentData(), {}, cast_rules.environment()
        if name and cast_rules.check(cast_rules.make_key(self._role_engine[role], name), self._language(),
                                     env=env, memo=memo).state == cast_rules.MISSING:
            for engine in cast_rules.ENGINES:
                if cast_rules.check(cast_rules.make_key(engine, name), self._language(), env=env, memo=memo).state == cast_rules.READY:
                    self._role_engine[role] = engine
                    self.win.status(f"{role.capitalize()} engine set to {cast_rules.LABEL[engine]}: {name} has no Qwen clone prompt.")
                    break
        self.refresh_compat()

    def on_role_engine_chosen(self, role, index):
        if self._loading:
            return
        _voice, engine_combo, _status = self._role_widgets(role)
        engine = engine_combo.itemData(index)
        if engine:
            self._role_engine[role] = engine
        self.refresh_compat()

    def refresh_voices(self):
        self._loading = True
        try:
            had_dialogue = self.cmb_dialogue.currentData() is not None
            for cmb in (self.cmb_narrator, self.cmb_dialogue):
                current = cmb.currentData()
                self._fill_names(cmb, current, first_by_default=True)
            # Only the first fill picks a different default Dialogue voice. Once a person has chosen, a refresh never
            # moves either role, even when both name the same voice and engine (one reader for the whole book).
            if not had_dialogue and self.dialogue_key() == self.narrator_key():
                for index in range(self.cmb_dialogue.count()):
                    if self.cmb_dialogue.itemData(index) != self.cmb_narrator.currentData():
                        self.cmb_dialogue.setCurrentIndex(index)
                        break
        finally:
            self._loading = False
        self.refresh_compat()

    def refresh_compat(self):
        """Recompute everything derived from the library, the language, what this Python offers and the segments. Reads
        files only; never writes a segment, and never resets the text caret or undo history."""
        memo, env = {}, cast_rules.environment()
        was = self._loading
        self._loading = True
        try:
            for role in ("narrator", "dialogue"):
                self._refresh_role(role, memo, env)
            consistent = self.listw.count() == len(self.segments)   # not while a project is being swapped in
            if consistent:
                self._sync_voice_combo(memo=memo, env=env)
        finally:
            self._loading = was
        if consistent:
            self._refresh_items(memo, env)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_compat()     # a transcript or reference may have been set in My Voices meanwhile

    def _voice_label(self, key):
        engine, name = cast_rules.parse_key(key)
        if not key:
            return "No voice"
        if engine is None:
            return f"Unknown: {key}"
        return name if engine == "qwen" else f"{name} ({cast_rules.LABEL[engine]})"

    # ------------------------------------------------------------------ list <-> segments
    @staticmethod
    def _preview(text):
        flat = " ".join(text.split())
        return flat if len(flat) <= 140 else flat[:137] + "…"

    def _short_voice(self, key, result):
        """The voice as the row shows it: 'Ana', 'Ana (CSM)', or 'Missing: Ana'. The full label is in the tooltip."""
        if result.state == cast_rules.MISSING and result.issues[0].cause.startswith(("missing:voice", "missing:key")):
            return f"Missing: {cast_rules.parse_key(key)[1] or key}"
        return self._voice_label(key)

    def _style_item(self, item, i, memo=None, env=None):
        seg = self.segments[i]
        result = cast_rules.check_segment(seg, self._language(), env=env, memo=memo)
        label = self._short_voice(seg["voice"], result)
        engine = cast_rules.parse_key(seg["voice"])[0]
        direction = (seg.get("direction") or "").strip()
        preview = self._preview(seg["text"]) or NO_TEXT
        shown = f"[{direction}] {preview}" if direction else preview
        kind = seg["kind"].capitalize()
        status = "" if result.state == cast_rules.READY else f"{cast_rules.STATE_TEXT[result.state]}: {result.message}"
        item.setText(f"{i + 1} {label}")
        item.setData(NAME_ROLE, label)
        item.setData(DESC_ROLE, shown)
        item.setData(NUM_ROLE, i + 1)
        item.setData(STATE_ROLE, result.state)
        item.setData(ROLE_ROLE, "D" if seg["kind"] == "dialogue" else "N")
        engine_text = f", engine {cast_rules.LABEL[engine]}" if engine else ""
        item.setData(QtCore.Qt.AccessibleTextRole,
                     f"Segment {i + 1}, {seg['kind']}, voice {label}{engine_text}" + (f", direction {direction}" if direction else "")
                     + (f". {status}" if status else "") + f". {preview}")
        flat = " ".join(seg["text"].split())
        body = "\n".join(textwrap.wrap(flat[:400] + ("…" if len(flat) > 400 else ""), 70))
        item.setToolTip(f"Segment {i + 1} · {kind} · {label}" + (f" · {direction}" if direction else "")
                        + (f"\n{status}" if status else "") + f"\n{body}")

    def _refresh_items(self, memo=None, env=None):
        memo = {} if memo is None else memo
        env = cast_rules.environment() if env is None else env
        self.listw.blockSignals(True)
        try:
            for i in range(self.listw.count()):
                self._style_item(self.listw.item(i), i, memo, env)
        finally:
            self.listw.blockSignals(False)

    def rebuild(self, select=None):
        """Repopulate the list from self.segments (after split, merge, add, delete, move or load)."""
        self._loading = True
        self.listw.blockSignals(True)
        try:
            self.listw.clear()
            memo, env = {}, cast_rules.environment()
            for i in range(len(self.segments)):
                item = QtWidgets.QListWidgetItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                item.setSizeHint(QtCore.QSize(1, ROW_H))
                self.listw.addItem(item)
                self._style_item(item, i, memo, env)
            n = len(self.segments)
            self.lbl_count.setText(f"Segments ({n})" if n else "Segments")
            if select is not None and n:
                self.listw.setCurrentRow(max(0, min(select, n - 1)))
        finally:
            self.listw.blockSignals(False)
            self._loading = False
        self.on_selection()

    def selected_rows(self):
        return sorted({self.listw.row(i) for i in self.listw.selectedItems()})

    def current_row(self):
        rows = self.selected_rows()
        return rows[0] if len(rows) == 1 else -1

    def _set_enabled(self, widget, on):
        if not on and widget.hasFocus():
            self.listw.setFocus()     # never leave focus on a control that just went away
        widget.setEnabled(on)

    def on_selection(self):
        self._fill_editor()

    def _fill_editor(self):
        rows = self.selected_rows()
        single = len(rows) == 1
        self._loading = True
        try:
            r = rows[0] if single else -1
            self.txt_segment.setPlainText(self.segments[r]["text"] if single else "")
            self.txt_segment.document().clearUndoRedoStacks()
            self.txt_segment.setPlaceholderText(
                "" if single else "Split the chapter script, or add a segment, to start." if not self.segments
                else "Select one segment to edit its text.")
            self._sync_voice_combo(rows)
            self.cmb_seg_direction.setCurrentText(self.segments[r].get("direction", "") if single else "")
            if not rows:
                self.lbl_seg_title.setText("Selected segment")
            elif single:
                self.lbl_seg_title.setText(f"Segment {r + 1} · {self.segments[r]['kind'].capitalize()}")
            else:
                self.lbl_seg_title.setText(f"{len(rows)} segments selected")
        finally:
            self._loading = False
        self.refresh_actions()

    def _sync_voice_combo(self, rows=None, memo=None, env=None):
        """Show the selection's voice and engine; a mixed selection shows 'Mixed' and 'Keep each', and writes nothing."""
        memo = {} if memo is None else memo
        env = cast_rules.environment() if env is None else env
        was = self._loading
        self._loading = True
        try:
            rows = self.selected_rows() if rows is None else rows
            segs = [self.segments[i] for i in rows]
            parsed = [cast_rules.parse_key(seg["voice"]) for seg in segs]
            names = {name for _engine, name in parsed}
            engines = {engine for engine, _name in parsed}
            self._fill_names(self.cmb_seg_voice, next(iter(names)) if len(names) == 1 and "" not in names else None)
            self.cmb_seg_voice.setPlaceholderText("Mixed" if len(names) > 1 else "Choose a voice")
            self._fill_engines(self.cmb_seg_engine,
                               [(name, seg.get("direction", ""), seg.get("text", ""), engine)
                                for seg, (engine, name) in zip(segs, parsed)],
                               next(iter(engines)) if len(engines) == 1 and None not in engines else None, memo, env)
            self.cmb_seg_engine.setPlaceholderText("Keep each" if len(engines - {None}) > 1 else "Engine")
            self._refresh_segment_status(rows, memo, env)
        finally:
            self._loading = was

    def _refresh_segment_status(self, rows, memo, env):
        if not rows:
            self.status_segment.set_status(None, "Select segments to set their voice and engine.")
            return
        results = [cast_rules.check_segment(self.segments[i], self._language(), env=env, memo=memo) for i in rows]
        # the short reason fits beside the title; the whole sentence is in the tooltip and the accessible description
        if len(rows) == 1:
            state, _line, detail = self._describe(results[0])
            text = "Ready (local)" if state == cast_rules.READY else results[0].short[:1].upper() + results[0].short[1:]
            spoken = _line
        else:
            worst = max(results, key=lambda r: cast_rules.SEVERITY[r.state])
            bad = sum(r.state != cast_rules.READY for r in results)
            if bad:
                state = worst.state
                text = f"{bad} of {len(rows)}: {worst.short}"
                spoken = f"{bad} of {len(rows)} selected segments need attention. {cast_rules.STATE_TEXT[worst.state]}: {worst.message}"
                detail = spoken
            else:
                state = cast_rules.READY
                text = f"All {len(rows)} ready (local)"
                spoken = f"All {len(rows)} selected segments are ready (local checks only)"
                detail = spoken
        self.status_segment.set_status(state, text, detail or spoken)
        self.cmb_seg_voice.setAccessibleDescription(spoken)
        self.cmb_seg_engine.setAccessibleDescription(
            " ".join(part for part in (spoken, self.cmb_seg_engine.accessibleDescription()) if part))

    def refresh_actions(self):
        """Enable only what the current selection (and a running task) allows; never touches the editor contents."""
        rows = self.selected_rows()
        single = len(rows) == 1
        self._set_enabled(self.txt_segment, single)
        self._set_enabled(self.cmb_seg_voice, bool(rows))
        self._set_enabled(self.cmb_seg_engine, bool(rows))
        self._set_enabled(self.cmb_seg_direction, single)
        self._set_enabled(self.btn_split_cursor, single)
        self._set_enabled(self.btn_merge, single and rows[0] < len(self.segments) - 1)
        self._set_enabled(self.btn_up, single and rows[0] > 0)
        self._set_enabled(self.btn_down, single and rows[0] < len(self.segments) - 1)
        self._set_enabled(self.btn_delete, bool(rows))
        self._set_enabled(self.btn_preview, single and not getattr(self.win, "busy", False))

    def _apply_keys(self, rows, new_keys, what):
        """Write the new keys to the rows, or write nothing and say why. A change may not give any segment a
        combination its engine cannot speak, unless that segment was already in that state."""
        language = self._language()
        memo, env = {}, cast_rules.environment()
        blocked = []
        for r, key in zip(rows, new_keys):
            seg = self.segments[r]
            new = cast_rules.check(key, language, seg.get("direction", ""), seg.get("text", ""), env=env, memo=memo)
            old = cast_rules.check_segment(seg, language, env=env, memo=memo)
            same_engine = cast_rules.parse_key(key)[0] == cast_rules.parse_key(seg["voice"])[0]
            if new.state == cast_rules.UNSUPPORTED and (old.state != cast_rules.UNSUPPORTED or not same_engine):
                blocked.append((r, new))
        if blocked:
            r, new = blocked[0]
            self.win.status(f"Not changed: segment {r + 1} cannot use that {what}. {new.message}")
            self._sync_voice_combo(memo=memo, env=env)
            return False
        for r, key in zip(rows, new_keys):
            self.segments[r]["voice"] = key
        self.refresh_compat()
        return True

    def on_voice_chosen(self, index):
        """The user picked a voice: it applies to every selected segment, and each keeps its own engine ("Keep each")."""
        if self._loading:
            return
        name = self.cmb_seg_voice.itemData(index)
        rows = self.selected_rows()
        if not name or not rows:
            return
        keys = [cast_rules.make_key(cast_rules.parse_key(self.segments[r]["voice"])[0] or "qwen", name) for r in rows]
        self._apply_keys(rows, keys, "voice")

    def on_engine_chosen(self, index):
        """The user picked an engine: it applies to every selected segment, and each keeps its own voice."""
        if self._loading:
            return
        engine = self.cmb_seg_engine.itemData(index)
        rows = self.selected_rows()
        if not engine or not rows:
            return
        todo = [r for r in rows if cast_rules.parse_key(self.segments[r]["voice"])[1]]
        if not todo:
            self.win.status("Choose a voice first: these segments have none.")
            self._sync_voice_combo()
            return
        if self._apply_keys(todo, [cast_rules.make_key(engine, cast_rules.parse_key(self.segments[r]["voice"])[1]) for r in todo],
                            "engine") and len(todo) < len(rows):
            self.win.status(f"{len(rows) - len(todo)} selected segments have no voice and were left as they are.")

    def on_direction_edited(self, text):
        if self._loading:
            return
        r = self.current_row()
        if r >= 0:
            self.segments[r]["direction"] = text.strip()
            self._style_item(self.listw.item(r), r)
            self._sync_voice_combo()

    def on_segment_text_changed(self):
        if self._loading:
            return
        r = self.current_row()
        if r >= 0:
            self.segments[r]["text"] = self.txt_segment.toPlainText()
            self._style_item(self.listw.item(r), r)
            self._sync_voice_combo()

    # ------------------------------------------------------------------ editing actions
    def _role_refusal(self, *roles):
        """None, or why a role's voice and engine cannot speak the current language. New segments never get an
        impossible combination; the role itself stays as it is, visible and editable."""
        for role in roles:
            key = self.narrator_key() if role == "narrator" else self.dialogue_key()
            if not key:
                continue
            result = cast_rules.check(key, self._language())
            if result.state == cast_rules.UNSUPPORTED:
                return f"Nothing was added: the {role} engine cannot speak this. {result.message}"
        return None

    def auto_split(self):
        text = self.txt_source.toPlainText().strip()
        if not text:
            self.win.status("Error: paste some text first.")
            return
        refusal = self._role_refusal("narrator", "dialogue")
        if refusal:
            self.win.status(refusal)
            return
        if self.segments:
            answer = QtWidgets.QMessageBox.question(
                self, "Replace segments?", "Replace the current segments with a new split?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        narrator = self.narrator_key()
        dialogue = self.dialogue_key() or narrator
        self.segments = audiobook.split_narration_dialogue(text, narrator, dialogue)
        self.rebuild(select=0)
        self.win.status(f"Split into {len(self.segments)} segments. Assign voices, then render.")

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
        refusal = self._role_refusal("narrator")
        if refusal:
            self.win.status(refusal)
            return
        rows = self.selected_rows()
        r = rows[-1] if rows else len(self.segments) - 1
        voice = self.narrator_key()
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

    def _check_segments(self, segs, rows=None):
        """Stop before anything expensive: no license question, no model, no render until every segment can be
        spoken. Problems are shown grouped by cause, with a way to select the segments they affect."""
        if not segs:
            self.win.status("Error: there are no segments.")
            return False
        for s in segs:
            if not s["text"].strip():
                self.win.status("Error: a segment has no text.")
                return False
        groups = cast_rules.preflight(segs, self._language(), rows=rows)
        if groups:
            self._show_problems(groups)
            return False
        if any(s["voice"].startswith("xtts:") for s in segs) and os.environ.get("COQUI_TOS_AGREED") != "1":
            answer = QtWidgets.QMessageBox.question(
                self, "XTTS model license",
                "XTTS-v2 is subject to the Coqui Public Model License. "
                "Review https://tts-hub.github.io/cpml before continuing.\n\n"
                "Do you have a commercial license or agree to the non-commercial license?")
            if answer != QtWidgets.QMessageBox.Yes:
                self.win.status("XTTS license acceptance is required.")
                return False
            os.environ["COQUI_TOS_AGREED"] = "1"
        if any(s["voice"].startswith("f5:") for s in segs) and os.environ.get("F5_TTS_LICENSE_CONFIRMED") != "1":
            answer = QtWidgets.QMessageBox.question(
                self, "F5-TTS model license",
                "The pretrained F5-TTS weights use the CC BY-NC 4.0 license. "
                "Review https://creativecommons.org/licenses/by-nc/4.0/ before continuing.\n\n"
                "Do you have permission to use these weights for your intended purpose?")
            if answer != QtWidgets.QMessageBox.Yes:
                self.win.status("F5-TTS license confirmation is required.")
                return False
            os.environ["F5_TTS_LICENSE_CONFIRMED"] = "1"
        return True

    def select_segments(self, rows):
        """Select exactly these segments, show the first one, and leave the keyboard on the list."""
        self.listw.clearSelection()
        for r in rows:
            if 0 <= r < self.listw.count():
                self.listw.item(r).setSelected(True)
        if rows:
            self.listw.setCurrentRow(rows[0], QtCore.QItemSelectionModel.NoUpdate)
            self.listw.scrollToItem(self.listw.item(rows[0]))
        self.listw.setFocus()

    def _show_problems(self, groups):
        """Why Render (or Preview) did not start: one block per cause, worst first. Non-blocking, so the list and
        editor stay usable; the dialog never touches a segment."""
        total = sum(len(g.rows) for g in groups)
        self.win.status(f"Not rendered: {len(groups)} thing(s) to fix in {total} segment(s). Nothing was loaded or generated.")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fix these before rendering")
        dialog.setAccessibleName("Fix these before rendering")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)
        intro = QtWidgets.QLabel("Nothing was loaded or generated. Each problem below is grouped by its cause.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setAccessibleName("Problems")
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 8, 0)
        inner_layout.setSpacing(8)
        for group in groups:
            numbers = ", ".join(str(r + 1) for r in group.rows[:12]) + ("…" if len(group.rows) > 12 else "")
            heading = f"{cast_rules.STATE_TEXT[group.state]} · {len(group.rows)} segment{'s' if len(group.rows) != 1 else ''} ({numbers})"
            title = QtWidgets.QLabel(heading)
            set_role(title, "field")
            body = QtWidgets.QLabel(group.message)
            body.setWordWrap(True)
            body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            button = QtWidgets.QPushButton("Select these segments")
            button.setAccessibleName(f"Select the {len(group.rows)} segments: {group.message[:60]}")
            button.clicked.connect(lambda _checked=False, rows=list(group.rows): (dialog.close(), self.select_segments(rows)))
            inner_layout.addWidget(title)
            inner_layout.addWidget(body)
            inner_layout.addWidget(button, 0, QtCore.Qt.AlignLeft)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        scroll.setMinimumHeight(min(inner.sizeHint().height() + 4, 360))   # whole when short, scrolls when long
        layout.addWidget(scroll, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, 0, QtCore.Qt.AlignRight)
        dialog.setMinimumWidth(460)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._problem_dialog = dialog
        self._problem_groups = groups
        dialog.open()

    def preview_segment(self):
        r = self.current_row()
        if r < 0:
            return
        seg = dict(self.segments[r])
        if not self._check_segments([seg], rows=[r]):
            return
        settings = self.render_settings()

        def work(task):
            return audiobook.render_segment(self.win.generator(task), seg, settings)

        def done(result):
            wav, sr = result
            if wav is None:
                self.win.status("Stopped before the segment was finished.")
                return
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
            print(f"Audiobook generation and compilation complete: {path}", flush=True)
            QtWidgets.QApplication.beep()

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
