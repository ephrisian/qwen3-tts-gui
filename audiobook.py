"""Multi-voice audiobook backend: segment text, render each segment, compile one file."""

import hashlib
import json
import os
import re

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
    items = [(f"Saved: {n}", saved_key(n)) for n in voice_library.list_voices()]
    return items


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
    if voice_key.startswith("voice:"):
        name = voice_key[len("voice:"):]
        try:
            with open(os.path.join(voice_library.voice_dir(name), "voice.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            return [cfg.get("created"), cfg.get("seed"), cfg.get("description")]
        except OSError:
            return None
    return None


def segment_cache_key(segment, settings):
    payload = json.dumps(
        [segment["voice"], _voice_fingerprint(segment["voice"]), segment.get("direction", ""),
         segment["text"], settings], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def render_segment(gen, segment, settings, cache_dir=SEGMENT_CACHE_DIR):
    """Render one segment to (wav, sr), reusing a cached result when nothing changed.

    settings: dict(language, min_words, max_words, seed, temperature)
    """
    path = os.path.join(cache_dir, segment_cache_key(segment, settings) + ".npz")
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data["wav"], int(data["sr"])
        except Exception:
            pass

    text = apply_direction(segment)
    voice = segment["voice"]
    if voice.startswith("voice:"):
        wavs, sr = gen.clone(
            text=text, language=settings["language"], min_words=settings["min_words"],
            max_words=settings["max_words"], seed=settings["seed"],
            temperature=settings["temperature"], voice_name=voice[len("voice:"):])
    else:
        raise ValueError(f"Segment has no voice assigned: {segment['text'][:40]!r}")
    if not wavs:
        return None, sr
    wav = np.concatenate([np.asarray(w, dtype=np.float32) for w in wavs])
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, wav=wav, sr=sr)
    return wav, sr


def render_chapter(gen, segments, settings, should_stop=lambda: False, notify=print):
    """Render all segments, grouping by model so each model loads once.

    Returns [(wav, sr)] in segment order, or None if stopped.
    """
    order = sorted(range(len(segments)),
                   key=lambda i: (segments[i]["voice"].startswith("voice:"), i))
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
