"""Multi-voice audiobook backend: segment text, render each segment, compile one file."""

import hashlib
import json
import os
import re
import uuid

import numpy as np

import voice_library
from config import OUTPUT_DIR

SEGMENT_CACHE_DIR = os.path.join(OUTPUT_DIR, "_segment_cache")
TARGET_RMS = 0.08  # about -22 dBFS

_QUOTE_RE = re.compile(r"\u201c[^\u201d]*\u201d|\"[^\"]*\"")


def saved_key(name):
    return f"voice:{name}"


def voice_choices():
    """[(label, key)] for every selectable voice: saved voices."""
    names = voice_library.list_voices()
    return ([(f"Qwen: {name}", saved_key(name)) for name in names] +
            [(f"XTTS: {name}", f"xtts:{name}") for name in names] +
            [(f"CSM: {name}", f"csm:{name}") for name in names] +
            [(f"F5-TTS: {name}", f"f5:{name}") for name in names] +
            [(f"Dia: {name}", f"dia:{name}") for name in names])


def new_segment(text, voice, kind="narration", direction=""):
    return {"text": text, "voice": voice, "kind": kind, "direction": direction}


def split_narration_dialogue(text, narrator, dialogue):
    """Split text into alternating narration / quoted-dialogue segments.

    Quotation marks stay with the dialogue so the model reads it as speech.
    Attribution such as 'she said' remains narration.
    """
    segments, pos = [], 0

    def add(chunk, kind, voice):
        chunk = chunk.strip()
        if chunk:
            segments.append(new_segment(chunk, voice, kind))

    for m in _QUOTE_RE.finditer(text):
        add(text[pos:m.start()], "narration", narrator)
        add(m.group(0), "dialogue", dialogue)
        pos = m.end()
    add(text[pos:], "narration", narrator)
    return segments


def split_segment_at(segment, index):
    """Split a segment's text at a character index into two segments (same voice/direction)."""
    left, right = segment["text"][:index].strip(), segment["text"][index:].strip()
    if not left or not right:
        return [segment]
    return [dict(segment, text=left), dict(segment, text=right)]


def merge_segments(a, b):
    return dict(a, text=f"{a['text']} {b['text']}".strip())


def apply_direction(segment):
    """Text with the segment's direction as a leading [tag] understood by the generator."""
    direction = (segment.get("direction") or "").strip().strip("[]")
    return f"[{direction}] {segment['text']}" if direction else segment["text"]


def _voice_fingerprint(voice_key):
    if voice_key.startswith(("voice:", "xtts:", "csm:", "f5:", "dia:")):
        name = voice_key.split(":", 1)[1]
        try:
            with open(os.path.join(voice_library.voice_dir(name), "voice.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            fingerprint = [cfg.get("created"), cfg.get("seed"), cfg.get("description")]
            if voice_key.startswith(("xtts:", "csm:", "f5:", "dia:")):
                ref_file_name = cfg.get("f5_ref_audio") if voice_key.startswith("f5:") else None
                ref_path = os.path.join(voice_library.voice_dir(name), ref_file_name or cfg.get("ref_audio", "reference.wav"))
                with open(ref_path, "rb") as ref_file:
                    fingerprint.append(hashlib.sha256(ref_file.read()).hexdigest())
            if voice_key.startswith(("csm:", "dia:")):
                fingerprint.append(cfg.get("csm_ref_text"))
            elif voice_key.startswith("f5:"):
                fingerprint.append(cfg.get("f5_ref_text") if cfg.get("f5_ref_audio") else cfg.get("csm_ref_text"))
            return fingerprint
        except OSError:
            return None
    return None


def segment_cache_key(segment, settings):
    payload = json.dumps(
        [segment["voice"], _voice_fingerprint(segment["voice"]), segment.get("direction", ""),
         segment["text"], settings], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _read_cached(path):
    """(wav, sr) from a cache file that carries the completion marker, else None.
    Files without the marker (written before it existed) are unverified: they may hold a truncated segment, so they
    are ignored and left on disk until a complete replacement is written over them."""
    try:
        with np.load(path) as data:
            if "complete" not in data.files or not bool(data["complete"]):
                return None
            return data["wav"], int(data["sr"])
    except Exception:
        return None


def _write_cached(path, wav, sr):
    """Write the finished segment with its completion marker. The data goes to a temporary file first and replaces the
    cache entry only once fully written, so an interrupted write never leaves a half file, and an older entry
    survives until its replacement is complete. A failed write still fails the render, as before."""
    tmp = f"{path[:-len('.npz')]}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.npz"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(tmp, wav=wav, sr=sr, complete=np.True_)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def render_segment(gen, segment, settings, cache_dir=SEGMENT_CACHE_DIR):
    """Render one segment to (wav, sr), reusing a cached result when nothing changed.

    settings: dict(language, min_words, max_words, seed, temperature)
    """
    path = os.path.join(cache_dir, segment_cache_key(segment, settings) + ".npz")
    cached = _read_cached(path)
    if cached is not None:
        return cached

    text = apply_direction(segment)
    voice = segment["voice"]
    if voice.startswith("voice:"):
        wavs, sr = gen.clone(
            text=text, language=settings["language"], min_words=settings["min_words"],
            max_words=settings["max_words"], seed=settings["seed"],
            temperature=settings["temperature"], voice_name=voice[len("voice:"):])
    elif voice.startswith("xtts:"):
        if segment.get("direction", "").strip():
            raise ValueError("XTTS does not support direction tags. Remove the segment direction.")
        wavs, sr = gen.xtts(
            text=text, language=settings["language"], seed=settings["seed"],
            temperature=settings["temperature"], voice_name=voice[len("xtts:"):])
    elif voice.startswith("csm:"):
        if segment.get("direction", "").strip():
            raise ValueError("CSM does not support direction tags. Remove the segment direction.")
        wavs, sr = gen.csm(
            text=text, language=settings["language"], seed=settings["seed"],
            temperature=settings["temperature"], voice_name=voice[len("csm:"):])
    elif voice.startswith(("f5:", "dia:")):
        engine, name = voice.split(":", 1)
        if (segment.get("direction") or "").strip():
            raise ValueError(f"{engine.upper()} does not support direction tags. Remove the segment direction.")
        wavs, sr = getattr(gen, engine)(
            text=text, language=settings["language"], seed=settings["seed"],
            temperature=settings["temperature"], voice_name=name)
    else:
        raise ValueError(f"Segment has no voice assigned: {segment['text'][:40]!r}")
    stopped = getattr(gen, "should_stop", None)
    if stopped is not None and stopped():
        # Stop was pressed while this segment rendered: Qwen's clone() then returns only the chunks finished so far.
        # That is not the segment. Return nothing and cache nothing, so a later render makes the whole segment.
        return None, sr
    if not wavs:
        return None, sr
    wav = np.concatenate([np.asarray(w, dtype=np.float32) for w in wavs])
    _write_cached(path, wav, sr)
    return wav, sr


def render_chapter(gen, segments, settings, should_stop=lambda: False, notify=print):
    """Render all segments, grouping by model so each model loads once.

    Returns [(wav, sr)] in segment order, or None if stopped.
    """
    order = sorted(range(len(segments)),
                   key=lambda i: (segments[i]["voice"].split(":", 1)[0], i))
    results = [None] * len(segments)
    for n, i in enumerate(order):
        if should_stop():
            return None
        seg = segments[i]
        notify(f"Segment {i+1}/{len(segments)} ({n+1} of {len(order)} rendered) — {seg['voice']}")
        wav, sr = render_segment(gen, seg, settings)
        if wav is None:
            return None
        results[i] = (wav, sr)
    return results


def compile_audio(results, pause_ms=300, match_loudness=True):
    """Join rendered segments into one waveform with silence between them."""
    if not results:
        raise ValueError("Nothing to compile.")
    sr = results[0][1]
    pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
    parts = []
    for k, (wav, wsr) in enumerate(results):
        if wsr != sr:
            raise ValueError("Segments have different sample rates.")
        wav = np.asarray(wav, dtype=np.float32)
        if match_loudness:
            rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
            if rms > 1e-6:
                wav = wav * (TARGET_RMS / rms)
        if k:
            parts.append(pause)
        parts.append(wav)
    out = np.concatenate(parts)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out, sr


def save_project(path, segments, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "meta": meta, "segments": segments}, f, indent=2, ensure_ascii=False)


def load_project(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"], data.get("meta", {})
