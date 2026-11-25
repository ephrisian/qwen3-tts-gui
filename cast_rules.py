"""Cast rules: can this voice, with this engine, speak this segment? Pure and read-only.

A segment's voice is the string "<prefix>:<voice name>" (Qwen uses the prefix "voice"). This module answers, for one
key, language, direction and text, which of four states applies, and why. It reads voice.json, checks that reference
files exist and, for the F5 limit, reads an audio header. It never imports an engine, loads a model, opens the network or
writes anything, and it changes no schema: keys, voice.json and projects are exactly what the engines already use.

    READY        every requirement that can be checked locally is met (model download and gated access are not)
    SETUP        something outside the segment is missing and can be provided: a confirmed transcript, a short F5
                 reference, an optional package. The assignment is valid for preparing a project; Render waits.
    UNSUPPORTED  the engine cannot speak this combination (language, direction, [emotion] tags). Never newly assigned.
    MISSING      the voice, or a file the engine reads, is gone.

The rules mirror the checks the engine methods in tts_generator.py and render_segment in audiobook.py make before they
load a model; tests.py runs both side by side so they cannot drift apart silently.
"""

import dataclasses
import functools
import importlib.util
import json
import os
import wave

import voice_library
from config import XTTS_LANGUAGES
from voice_lock import parse_emotion_segments

READY, SETUP, UNSUPPORTED, MISSING = "ready", "setup", "unsupported", "missing"
SEVERITY = {READY: 0, SETUP: 1, UNSUPPORTED: 2, MISSING: 3}
STATE_TEXT = {READY: "Ready", SETUP: "Needs setup", UNSUPPORTED: "Unsupported", MISSING: "Missing"}

ENGINES = ("qwen", "xtts", "csm", "dia", "f5")
PREFIX = {"qwen": "voice", "xtts": "xtts", "csm": "csm", "dia": "dia", "f5": "f5"}
ENGINE_OF_PREFIX = {prefix: engine for engine, prefix in PREFIX.items()}
LABEL = {"qwen": "Qwen", "xtts": "XTTS", "csm": "CSM", "dia": "Dia", "f5": "F5-TTS"}
TRAITS = {
    "qwen": "Clone prompt. All languages, direction and [emotion] tags.",
    "xtts": "10 languages. No direction or [emotion] tags.",
    "csm": "English only. Needs a confirmed transcript. No direction or tags.",
    "dia": "English only. Needs a confirmed transcript. No direction or tags.",
    "f5": "English and Chinese. Needs a transcript and a reference of 12 s or less. Non-commercial weights.",
}
LOCAL_ONLY_NOTE = "Local checks only: model download and access are not verified."
F5_MAX_SECONDS = 12


@dataclasses.dataclass(frozen=True)
class Issue:
    state: str
    cause: str      # same cause, same fix: segments sharing it are reported together
    short: str      # a few words for a menu item or a status line
    message: str    # a sentence that says what to do


@dataclasses.dataclass(frozen=True)
class Result:
    state: str
    issues: tuple = ()
    notes: tuple = ()

    @property
    def short(self):
        return self.issues[0].short if self.issues else ""

    @property
    def message(self):
        return self.issues[0].message if self.issues else ""


@dataclasses.dataclass(frozen=True)
class Group:
    """Segments that share one problem."""
    state: str
    cause: str
    message: str
    rows: tuple


# ------------------------------------------------------------------ keys
def parse_key(key):
    """'xtts:Ana' -> ('xtts', 'Ana'); 'voice:Ana' -> ('qwen', 'Ana'); '' or an unknown prefix -> (None, '')."""
    prefix, sep, name = (key or "").partition(":")
    engine = ENGINE_OF_PREFIX.get(prefix) if sep else None
    return (engine, name) if engine else (None, "")


def make_key(engine, name):
    return f"{PREFIX[engine]}:{name}"


# ------------------------------------------------------------------ environment
@functools.lru_cache(maxsize=None)
def _transformers_has(class_name):
    try:
        import transformers
        return hasattr(transformers, class_name)
    except Exception:
        return False


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class _Environment:
    """What this Python can offer, looked up only when asked (Transformers is not imported until a CSM or Dia key is
    checked) and without importing an engine or touching a model. 'f5' is 'ok' (importable here), 'unverified'
    (F5_TTS_PYTHON points at an interpreter we do not inspect) or 'missing'."""

    def __init__(self):
        self._found = {}

    def get(self, key, default=None):
        if key not in self._found:
            if key == "xtts":
                self._found[key] = _has_module("TTS")
            elif key == "csm":
                self._found[key] = _transformers_has("CsmForConditionalGeneration")
            elif key == "dia":
                self._found[key] = _transformers_has("DiaForConditionalGeneration")
            elif key == "f5":
                python = os.environ.get("F5_TTS_PYTHON")
                self._found[key] = (("unverified" if os.path.isfile(python) else "missing") if python
                                    else "ok" if _has_module("f5_tts") else "missing")
            else:
                return default
        return self._found[key]


def environment():
    return _Environment()


# ------------------------------------------------------------------ voice files
def _voice(name, root, memo):
    """(config, folder) of a saved voice, or None when voice.json is absent or unreadable."""
    key = ("voice", name, root)
    if memo is not None and key in memo:
        return memo[key]
    folder = voice_library.voice_dir(name, root)
    try:
        with open(os.path.join(folder, "voice.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        found = (config, folder) if isinstance(config, dict) else None
    except (OSError, ValueError):
        found = None
    if memo is not None:
        memo[key] = found
    return found


def _file_state(path, memo):
    key = ("file", path)
    if memo is not None and key in memo:
        return memo[key]
    found = os.path.isfile(path)
    if memo is not None:
        memo[key] = found
    return found


MIN_REFERENCE_SECONDS = 0.1


def _audio_seconds(path, memo):
    """Duration in seconds from the file header, or None when audio_utils.load_wav could not read the file.

    Mirrors load_wav without decoding it: the standard library first (8/16/32-bit PCM WAV), then soundfile for
    anything else (float or 24-bit WAV, FLAC, MP3, OGG). Only headers are read, so this stays cheap.
    """
    key = ("audio", path)
    if memo is not None and key in memo:
        return memo[key]
    seconds = None
    try:
        with open(path, "rb") as source, wave.open(source, "rb") as handle:
            if handle.getsampwidth() in (1, 2, 4):
                # The header's frame count is trusted by the standard library even when the file was cut short
                # (an interrupted copy), so count only the frames that are really there.
                frame_bytes = handle.getnchannels() * handle.getsampwidth()
                present = max(0, os.path.getsize(path) - source.tell()) // frame_bytes
                seconds = min(handle.getnframes(), present) / handle.getframerate()
            else:
                raise wave.Error("sample width load_wav hands to soundfile")
    except (wave.Error, EOFError, ValueError, ZeroDivisionError):
        seconds = None
        try:
            import soundfile
            info = soundfile.info(path)
            seconds = info.frames / info.samplerate
        except Exception:
            seconds = None
    except OSError:
        seconds = None
    if seconds is not None and seconds < MIN_REFERENCE_SECONDS:
        seconds = None   # empty or cut off to nothing: load_wav would hand the engine (almost) no samples
    if memo is not None:
        memo[key] = seconds
    return seconds


# ------------------------------------------------------------------ the rules
def _has_emotion_tags(text):
    try:
        return any(emotion for emotion, _ in parse_emotion_segments(text or ""))
    except Exception:
        return False


def _spoken(direction, text):
    """The text the generator receives: the direction becomes a leading [tag] (audiobook.apply_direction)."""
    direction = (direction or "").strip().strip("[]")
    return f"[{direction}] {text}" if direction else (text or "")


def _language_issue(engine, language):
    language = (language or "").lower()
    label = LABEL[engine]
    if engine == "xtts" and language not in XTTS_LANGUAGES:
        return Issue(UNSUPPORTED, "unsupported:xtts:language", "pick a language",
                     "XTTS needs an explicit language. Choose one other than auto, or use Qwen.")
    if engine in ("csm", "dia") and language != "english":
        return Issue(UNSUPPORTED, f"unsupported:{engine}:language", "English only",
                     f"{label} supports English only. Change Language to English, or use another engine.")
    if engine == "f5" and language not in ("english", "chinese"):
        return Issue(UNSUPPORTED, "unsupported:f5:language", "English or Chinese",
                     "F5-TTS supports English and Chinese. Change Language, or use another engine.")
    return None


def combination_issues(engine, language, direction="", text=""):
    """What this engine cannot do with this language, direction and text, whatever the voice."""
    found = []
    issue = _language_issue(engine, language)
    if issue:
        found.append(issue)
    if engine != "qwen":
        label = LABEL[engine]
        if (direction or "").strip().strip("[]"):
            found.append(Issue(UNSUPPORTED, f"unsupported:{engine}:direction", "no direction",
                               f"{label} does not support direction. Clear the direction, or use Qwen."))
        if _has_emotion_tags(text):
            found.append(Issue(UNSUPPORTED, f"unsupported:{engine}:tags", "no [tags]",
                               f"{label} does not support [emotion] tags. Remove them from the text, or use Qwen."))
    return found


def check(key, language, direction="", text="", root=None, env=None, memo=None):
    """The state of one assignment. `direction` and `text` are the segment's; pass '' to judge the engine and voice alone."""
    engine, name = parse_key(key)
    if not (key or "").strip():
        return Result(SETUP, (Issue(SETUP, "setup:none:voice", "no voice", "This segment has no voice. Choose one."),))
    if engine is None:
        return Result(MISSING, (Issue(MISSING, f"missing:key:{key}", "unknown voice",
                                      f"'{key}' is not a voice this version knows. Choose another voice."),))
    label = LABEL[engine]
    env = environment() if env is None else env
    issues, notes = [], [LOCAL_ONLY_NOTE]

    found = _voice(name, root, memo)
    if found is None:
        issues.append(Issue(MISSING, f"missing:voice:{name}", "voice missing",
                            f"The voice {name} no longer exists. Choose another voice, or restore its folder."))
        return Result(MISSING, tuple(issues), tuple(notes))
    config, folder = found

    # --- combination: independent of the voice, so never newly assigned
    issues.extend(combination_issues(engine, language, direction, text))
    if engine == "qwen" and _has_emotion_tags(_spoken(direction, text)) and not str(config.get("description") or "").strip():
        # render_segment sends "[direction] text", and Qwen reads a direction exactly like a tag in the text
        issues.append(Issue(UNSUPPORTED, f"unsupported:qwen:description:{name}", "needs description",
                            f"Direction and [emotion] tags need a voice description and {name} has none. "
                            "Use another voice, or clear the direction and the tags."))

    # --- files the engine reads
    reference_name = config.get("ref_audio", "reference.wav")
    if engine == "qwen":
        prompt = config.get("clone_prompt")
        if not prompt or not _file_state(os.path.join(folder, prompt), memo):
            issues.append(Issue(MISSING, f"missing:prompt:{name}", "prompt missing",
                                f"{name} has no Qwen clone prompt on disk. Choose another voice, or save it again from Clone Voice."))
    else:
        if engine == "f5" and config.get("f5_ref_audio"):
            reference_name = config["f5_ref_audio"]
        reference = os.path.join(folder, reference_name)
        if not _file_state(reference, memo):
            issues.append(Issue(MISSING, f"missing:reference:{name}", "reference missing",
                                f"The reference recording for {name} is missing. Choose another voice, or add it again in My Voices."))
        elif engine in ("csm", "dia", "f5"):
            seconds = _audio_seconds(reference, memo)
            if seconds is None:
                issues.append(Issue(SETUP, f"setup:{engine}:wav:{name}", "reference unreadable",
                                    f"{label} could not read the reference recording for {name}: it is empty or damaged, or in a format this computer can't decode. Add it again in My Voices."))
            elif engine == "f5" and seconds > F5_MAX_SECONDS:
                issues.append(Issue(SETUP, f"setup:f5:length:{name}", "reference over 12 s",
                                    f"F5-TTS needs a reference of {F5_MAX_SECONDS} s or less and {name}'s is {seconds:.0f} s. "
                                    f"In My Voices, select {name}, then Engine references > F5 reference…"))

    # --- transcript
    if engine in ("csm", "dia", "f5"):
        own = engine == "f5" and bool(config.get("f5_ref_audio"))
        transcript = str((config.get("f5_ref_text") if own else config.get("csm_ref_text")) or "").strip()
        if not transcript:
            where = "F5 reference…" if own else "Reference transcript…"
            issues.append(Issue(SETUP, f"setup:{engine}:transcript:{name}", "needs transcript",
                                f"{label} needs the exact reference transcript for {name}. "
                                f"In My Voices, select {name}, then Engine references > {where}"))

    # --- this Python
    if engine == "xtts" and not env.get("xtts"):
        issues.append(Issue(SETUP, "setup:xtts:package", "needs coqui-tts",
                            "XTTS needs the optional coqui-tts package. See the README, then restart the app."))
    elif engine == "csm" and not env.get("csm"):
        issues.append(Issue(SETUP, "setup:csm:package", "needs newer Transformers",
                            "CSM needs Transformers 4.52.1 or newer with CSM support. See the README."))
    elif engine == "dia" and not env.get("dia"):
        issues.append(Issue(SETUP, "setup:dia:package", "needs newer Transformers",
                            "Dia needs a Transformers version with Dia support. See the README."))
    elif engine == "f5":
        if env.get("f5") == "missing":
            issues.append(Issue(SETUP, "setup:f5:environment", "needs F5 environment",
                                "F5-TTS needs its own Python environment. Set F5_TTS_PYTHON to it (see the README)."))
        elif env.get("f5") == "unverified":
            notes.append("The F5 environment set in F5_TTS_PYTHON was not inspected.")

    if engine == "csm":
        notes.append("Access to the gated CSM model on Hugging Face is not verified.")
    if engine in ("xtts", "f5"):
        notes.append("A license confirmation is asked at the first render.")

    issues.sort(key=lambda item: -SEVERITY[item.state])
    state = issues[0].state if issues else READY
    return Result(state, tuple(issues), tuple(notes))


def check_segment(segment, language, root=None, env=None, memo=None):
    return check(segment.get("voice", ""), language, segment.get("direction", ""), segment.get("text", ""),
                 root=root, env=env, memo=memo)


def preflight(segments, language, rows=None, root=None, env=None, memo=None):
    """Problems that would stop a render, grouped by cause: [Group], worst first. `rows` names the segments' positions
    in the chapter when `segments` is only part of it. Empty when every segment is Ready."""
    rows = list(range(len(segments))) if rows is None else list(rows)
    env = environment() if env is None else env
    memo = {} if memo is None else memo
    groups = {}
    for row, segment in zip(rows, segments):
        for issue in check_segment(segment, language, root=root, env=env, memo=memo).issues:
            entry = groups.setdefault(issue.cause, [issue, []])
            entry[1].append(row)
    ordered = sorted(groups.values(), key=lambda pair: (-SEVERITY[pair[0].state], pair[1][0]))
    return [Group(issue.state, issue.cause, issue.message, tuple(found)) for issue, found in ordered]
