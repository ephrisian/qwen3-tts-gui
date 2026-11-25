"""Unit tests for Qwen3-TTS GUI modules."""

import os
import shutil
import sys
import tempfile
import wave

import numpy as np

from config import OUTPUT_DIR
from text_utils import chunk_text, split_text
from audio_utils import write_wav, normalize_wavs, save_outputs, save_voice_clone_prompt, load_voice_clone_prompt


def _project_temp_path(suffix=".wav"):
    """Return a temp file path inside the project folder (output dir)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, dir=OUTPUT_DIR)
    os.close(fd)
    return path


# ------------------------------------------------------------------ audio utils

def _test_write_wav():
    """Test that write_wav produces a valid WAV file."""
    sr = 16000
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    path = _project_temp_path()

    try:
        write_wav(path, audio, sr)

        # Verify by reading back
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono channel"
            assert wf.getsampwidth() == 2, "Expected 16-bit samples"
            assert wf.getframerate() == sr, f"Expected sample rate {sr}"
            n_frames = wf.getnframes()
            expected_frames = int(sr * duration)
            assert abs(n_frames - expected_frames) < 10, (
                f"Expected ~{expected_frames} frames, got {n_frames}"
            )
        print("PASS: write_wav produces valid WAV file")
    finally:
        os.unlink(path)


def _test_write_wav_various_inputs():
    """Test write_wav handles various input types (scalar, 0-d, 1-d, multi-channel)."""
    sr = 16000

    test_cases = [
        ("scalar float", np.float32(0.5)),
        ("0-d array", np.array(0.5)),
        ("1-d array", 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))),
        ("multi-channel", np.stack([
            0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))),
            0.3 * np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, int(sr * 0.5))),
        ], axis=1)),
    ]

    for name, audio in test_cases:
        path = _project_temp_path()
        try:
            write_wav(path, audio, sr)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1, f"{name}: Expected mono channel"
                assert wf.getsampwidth() == 2, f"{name}: Expected 16-bit samples"
                assert wf.getframerate() == sr, f"{name}: Expected sample rate {sr}"
                n_frames = wf.getnframes()
                assert n_frames > 0, f"{name}: Expected non-empty audio"
            print(f"PASS: write_wav handles {name}")
        finally:
            os.unlink(path)


def _test_normalize_wavs():
    """Test normalize_wavs normalizes various wavs input types correctly."""

    # Test 1: numpy scalar
    wavs = normalize_wavs(np.float32(0.5))
    assert len(wavs) == 1, f"Expected 1, got {len(wavs)}"
    assert wavs[0].shape == (1,), f"Expected shape (1,), got {wavs[0].shape}"
    print("PASS: normalize_wavs normalizes numpy scalar")

    # Test 2: 0-d array
    wavs = normalize_wavs(np.array(0.5))
    assert len(wavs) == 1
    assert wavs[0].shape == (1,)
    print("PASS: normalize_wavs normalizes 0-d array")

    # Test 3: 1-d array
    wavs = normalize_wavs(np.array([0.1, 0.2, 0.3]))
    assert len(wavs) == 1
    assert wavs[0].shape == (3,)
    print("PASS: normalize_wavs normalizes 1-d array")

    # Test 4: list of arrays
    wavs = normalize_wavs([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    assert len(wavs) == 2
    print("PASS: normalize_wavs normalizes list of arrays")

    # Test 5: numpy generic scalar
    wavs = normalize_wavs(np.float64(0.75))
    assert len(wavs) == 1
    assert wavs[0].shape == (1,)
    print("PASS: normalize_wavs normalizes numpy generic scalar")


# ------------------------------------------------------------------ text utils

def _test_split_text():
    """Test split_text returns the text as a single-item list."""
    texts = split_text("Hello, World, Test")
    assert texts == ["Hello, World, Test"]
    print("PASS: split_text returns single-item list")


def _test_split_text_empty():
    """Test split_text returns empty list for empty input."""
    assert split_text("") == []
    assert split_text(None) == []
    print("PASS: split_text handles empty input")


def _test_chunk_text():
    """Test chunk_text splits long text at sentence boundaries."""
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
    chunks = chunk_text(text, min_words=5, max_words=10)
    assert len(chunks) >= 1
    # Each chunk should be a non-empty string
    for chunk in chunks:
        assert len(chunk.strip()) > 0
    print("PASS: chunk_text splits text correctly")


def _test_chunk_text_empty():
    """Test chunk_text returns empty list for empty input."""
    assert chunk_text("") == []
    assert chunk_text(None) == []
    print("PASS: chunk_text handles empty input")


def _test_chunk_text_single_sentence():
    """Test chunk_text keeps a single short sentence as one chunk."""
    chunks = chunk_text("Hello world.", min_words=5, max_words=10)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world."
    print("PASS: chunk_text keeps single short sentence intact")


# ------------------------------------------------------------------ NPZ / voice clone prompt loading

def _test_npz_load_allow_pickle():
    """Test that NPZ files containing object arrays load with allow_pickle=True."""
    path = _project_temp_path(suffix=".npz")
    try:
        # Save an object array (e.g. a dict with a numpy array inside)
        obj = {"ref_spk_embedding": np.array([0.1, 0.2, 0.3]), "other": "value"}
        np.savez(path, **obj)

        # Loading with allow_pickle=True should succeed
        data = np.load(path, allow_pickle=True)
        assert set(data.files) == {"ref_spk_embedding", "other"}
        print("PASS: NPZ object arrays load with allow_pickle=True")
    finally:
        os.unlink(path)


def _test_npz_numeric_keys_convert_to_list():
    """Test that NPZ saved from a list (numeric string keys) converts back to a list."""
    path = _project_temp_path(suffix=".npz")
    try:
        # Simulate saving a list (as gui_app.py does): convert to {str(i): v} then np.savez(**dict)
        prompt_list = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        prompt_dict = {str(i): v for i, v in enumerate(prompt_list)}
        np.savez(path, **prompt_dict)

        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}

        # Numeric keys should be detected and converted back to a list
        assert voice_clone_prompt and all(k.isdigit() for k in voice_clone_prompt.keys())
        converted = [voice_clone_prompt[k] for k in sorted(voice_clone_prompt.keys(), key=int)]
        assert len(converted) == 2
        assert np.array_equal(converted[0], np.array([0.1, 0.2]))
        assert np.array_equal(converted[1], np.array([0.3, 0.4]))
        print("PASS: NPZ numeric-string keys convert back to list")
    finally:
        os.unlink(path)


def _test_npz_named_keys_stay_dict():
    """Test that NPZ saved with named keys stays a dict (no conversion)."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt_dict = {"ref_spk_embedding": np.array([0.1, 0.2]), "other": np.array([0.3])}
        np.savez(path, **prompt_dict)

        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}

        # Named keys are not all digits, so no conversion
        assert not all(k.isdigit() for k in voice_clone_prompt.keys())
        assert "ref_spk_embedding" in voice_clone_prompt
        assert np.array_equal(voice_clone_prompt["ref_spk_embedding"], np.array([0.1, 0.2]))
        print("PASS: NPZ named keys stay as dict")
    finally:
        os.unlink(path)


def _test_npz_empty_files():
    """Test that an empty NPZ does not crash the conversion logic."""
    path = _project_temp_path(suffix=".npz")
    try:
        np.savez(path)  # empty
        data = np.load(path, allow_pickle=True)
        voice_clone_prompt = {k: data[k] for k in data.files}
        assert voice_clone_prompt == {}
        print("PASS: empty NPZ handled gracefully")
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ voice clone prompt helpers

class _FakePrompt:
    """Stand-in for objects returned by create_voice_clone_prompt (has ref_code attr)."""
    def __init__(self, embedding):
        self.ref_code = embedding


def _test_save_load_voice_clone_prompt_preserves_ref_code():
    """Test that save/load round-trips objects with ref_code attribute intact."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt = [_FakePrompt(np.array([0.1, 0.2, 0.3])), _FakePrompt(np.array([0.4, 0.5]))]
        save_voice_clone_prompt(path, prompt)

        loaded = load_voice_clone_prompt(path)
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert hasattr(loaded[0], "ref_code"), "Loaded object lost ref_code attribute"
        assert np.array_equal(loaded[0].ref_code, np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(loaded[1].ref_code, np.array([0.4, 0.5]))
        print("PASS: save/load voice clone prompt preserves ref_code")
    finally:
        os.unlink(path)


def _test_save_load_voice_clone_prompt_dict():
    """Test that save/load round-trips a dict of objects with ref_code intact."""
    path = _project_temp_path(suffix=".npz")
    try:
        prompt = {"ref_spk_embedding": _FakePrompt(np.array([0.1, 0.2]))}
        save_voice_clone_prompt(path, prompt)

        loaded = load_voice_clone_prompt(path)
        assert isinstance(loaded, dict)
        assert "ref_spk_embedding" in loaded
        assert hasattr(loaded["ref_spk_embedding"], "ref_code")
        assert np.array_equal(loaded["ref_spk_embedding"].ref_code, np.array([0.1, 0.2]))
        print("PASS: save/load voice clone prompt dict preserves ref_code")
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ save_outputs

def _test_save_outputs_single():
    """Test save_outputs writes a single WAV file."""
    sr = 16000
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
    path = _project_temp_path()
    try:
        result = save_outputs(audio, sr, path)
        assert result == path
        assert os.path.exists(path)
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() > 0
        print("PASS: save_outputs writes single WAV")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_save_outputs_stop_requested():
    """Test save_outputs skips writing when stop_requested is True."""
    path = _project_temp_path()
    os.unlink(path)  # remove pre-created file so we can verify nothing is written
    try:
        result = save_outputs(np.array([0.1, 0.2]), 16000, path, stop_requested=True)
        assert result == path
        assert not os.path.exists(path)
        print("PASS: save_outputs skips when stop_requested")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_audiobook_split():
    import audiobook
    segs = audiobook.split_narration_dialogue(
        'The rain fell. "Where?" she asked. \u201cHere,\u201d he said. Done.', "N", "D")
    assert [s["kind"] for s in segs] == ["narration", "dialogue", "narration", "dialogue", "narration"], segs
    assert segs[1]["text"] == '"Where?"' and segs[1]["voice"] == "D"
    assert audiobook.split_narration_dialogue("", "N", "D") == []
    a, b = audiobook.split_segment_at(segs[0], 8)
    assert a["text"] == "The rain" and b["text"] == "fell."
    assert audiobook.merge_segments(a, b)["text"] == "The rain fell."
    assert audiobook.apply_direction({"text": "Hi", "direction": "sad"}) == "[sad] Hi"
    print("PASS: audiobook split/merge/direction")


def _test_audiobook_compile():
    import audiobook
    sr = 1000
    quiet = (np.ones(500, dtype=np.float32) * 0.01, sr)
    loud = (np.ones(500, dtype=np.float32) * 0.5, sr)
    out, osr = audiobook.compile_audio([quiet, loud], pause_ms=100, match_loudness=True)
    assert osr == sr and len(out) == 500 + 100 + 500
    assert abs(out[0] - out[-1]) < 1e-4, "loudness should be matched"
    assert np.all(out[500:600] == 0), "pause should be silent"
    out2, _ = audiobook.compile_audio([quiet, loud], pause_ms=0, match_loudness=False)
    assert len(out2) == 1000
    print("PASS: audiobook compile")


def _test_audiobook_cache_marker_legacy_entries_and_interrupted_writes():
    """A cache entry counts only with its completion marker. Older entries (no marker) are unverified: regenerated
    when needed, never deleted, and kept until a complete replacement exists. Writes are all-or-nothing."""
    import shutil
    import audiobook
    from unittest.mock import patch

    calls = []

    class Gen:
        def clone(self, **kw):
            calls.append(1)
            return [np.full(10, 0.5, dtype=np.float32)], 24000
    settings = dict(language="english", min_words=0, max_words=0, seed=1, temperature=0.7)
    seg = audiobook.new_segment("Cache me.", "voice:X")
    cache = tempfile.mkdtemp()
    try:
        path = os.path.join(cache, audiobook.segment_cache_key(seg, settings) + ".npz")

        def entries():
            return sorted(os.listdir(cache))
        # a fresh render writes a marked entry through a temporary file that does not remain
        wav, _ = audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert entries() == [os.path.basename(path)] and len(calls) == 1
        with np.load(path) as data:
            assert bool(data["complete"]) and data["wav"].shape == (10,) and int(data["sr"]) == 24000
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert len(calls) == 1, "a marked entry is reused"
        # legacy entry (written before the marker, possibly truncated): not trusted, not deleted
        np.savez(path, wav=np.zeros(3, dtype=np.float32), sr=24000)
        legacy = open(path, "rb").read()
        # ... and while its replacement cannot be completed, it is still on disk, byte for byte
        for target, name, error in ((audiobook.os, "replace", OSError("disk full")),
                                    (audiobook.np, "savez", OSError("write failed"))):
            n = len(calls)
            with patch.object(target, name, side_effect=error):
                try:
                    audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
                except OSError:
                    pass   # a failed cache write fails the render, as it always did
                else:
                    raise AssertionError("a cache write failure must not be swallowed")
            assert len(calls) == n + 1, "the legacy entry must not be served"
            assert open(path, "rb").read() == legacy, "legacy data stays until a complete replacement exists"
            assert entries() == [os.path.basename(path)], "no temporary file is left behind"
        # once a replacement is written completely, it takes over and is reused
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        with np.load(path) as data:
            assert bool(data["complete"]) and data["wav"].shape == (10,)
        n = len(calls)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert len(calls) == n
        # unreadable, empty or explicitly incomplete entries are regenerated and replaced
        for damage in (b"", b"not an npz file at all"):
            open(path, "wb").write(damage)
            n = len(calls)
            audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
            assert len(calls) == n + 1
            with np.load(path) as data:
                assert bool(data["complete"])
        # a real, marked archive cut in half (a copy interrupted elsewhere) is regenerated, never half-served
        whole = open(path, "rb").read()
        open(path, "wb").write(whole[:len(whole) // 2])
        n = len(calls)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert len(calls) == n + 1
        with np.load(path) as data:
            assert bool(data["complete"]) and data["wav"].shape == (10,)
        np.savez(path, wav=np.zeros(3, dtype=np.float32), sr=24000, complete=np.False_)
        n = len(calls)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert len(calls) == n + 1
        # nothing is ever removed from the cache folder by rendering
        stray = os.path.join(cache, "legacy_other_entry.npz")
        np.savez(stray, wav=np.zeros(2, dtype=np.float32), sr=24000)
        audiobook.render_segment(Gen(), dict(seg, text="Another."), settings, cache_dir=cache)
        assert os.path.isfile(stray)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    print("PASS: cache entries need a completion marker; legacy entries are regenerated but kept until replaced; writes are all-or-nothing")


def _test_audiobook_stop_never_caches_or_returns_a_partial_segment():
    """Stop during a Qwen segment makes clone() return only the chunks done so far. That partial audio must neither
    be cached as if complete, nor let a chapter report success. Real TTSGenerator.clone and audiobook functions;
    only the model is a stand-in."""
    import shutil
    import audiobook
    import voice_library
    import tts_generator
    from unittest.mock import patch

    text = " ".join(f"Sentence number {i} is here." for i in range(12))
    settings = dict(language="english", min_words=10, max_words=12, seed=1, temperature=0.7)

    def make_generator(stop, stop_on_call=None):
        calls = {"n": 0}

        class Model:
            def generate_voice_clone(self, **kw):
                calls["n"] += 1
                if stop_on_call is not None and calls["n"] == stop_on_call:
                    stop["v"] = True   # the user presses Stop while this chunk renders
                return [np.full(1000, 0.1, dtype=np.float32)], 24000

        class Models:
            def get(self, kind, notify=None):
                return Model()
        return tts_generator.TTSGenerator(Models(), notify=lambda m: None, should_stop=lambda: stop["v"])

    with patch.object(voice_library, "load_voice", lambda name: ({"description": "", "seed": 1}, [object()], None)):
        full_chunks = len(make_generator({"v": False})._jobs(text, 10, 12))
        assert full_chunks > 2, full_chunks
        cache = tempfile.mkdtemp()
        try:
            # 1. Stop in the middle of one segment: nothing is returned and nothing is cached
            stop = {"v": False}
            segment = audiobook.new_segment(text, "voice:X")
            wav, _ = audiobook.render_segment(make_generator(stop, stop_on_call=1), segment, settings, cache_dir=cache)
            assert wav is None, "a partially rendered segment must not be returned"
            assert os.listdir(cache) == [], "a partially rendered segment must not be cached"
            # 2. The next clean render is complete (it must not find a truncated copy)
            wav, _ = audiobook.render_segment(make_generator({"v": False}), segment, settings, cache_dir=cache)
            assert len(wav) == 1000 * full_chunks and len(os.listdir(cache)) == 1
            # 3. A complete render without Stop is still cached and reused
            calls_before = len(os.listdir(cache))
            wav2, _ = audiobook.render_segment(make_generator({"v": False}), segment, settings, cache_dir=cache)
            assert len(wav2) == len(wav) and len(os.listdir(cache)) == calls_before
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        # 4. Stop during the LAST segment in render order: the chapter is stopped, not "rendered" with a truncated segment
        cache = tempfile.mkdtemp()
        try:
            stop = {"v": False}
            gen = make_generator(stop, stop_on_call=2)   # call 1 = the short segment, call 2 = first chunk of the long one
            segments = [audiobook.new_segment("Short one.", "voice:X"), audiobook.new_segment(text, "voice:X")]
            with patch.object(audiobook, "SEGMENT_CACHE_DIR", cache), \
                    patch.object(audiobook.render_segment, "__defaults__", (cache,)):
                results = audiobook.render_chapter(gen, segments, settings, should_stop=lambda: stop["v"],
                                                   notify=lambda m: None)
                assert results is None, "a chapter whose last segment was cut short must report a stop"
                assert len(os.listdir(cache)) == 1, "only the finished segment is cached"
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        # 5. Preview segment path: a generator with no stop flag at all behaves as before
        class Plain:
            def clone(self, **kw):
                return [np.zeros(10, dtype=np.float32)], 24000
        cache = tempfile.mkdtemp()
        try:
            wav, _ = audiobook.render_segment(Plain(), audiobook.new_segment("Hi.", "voice:X"), settings, cache_dir=cache)
            assert wav is not None and len(os.listdir(cache)) == 1
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    print("PASS: a stopped Qwen segment is never cached or returned, and a chapter stopped in its last segment reports a stop")


def _test_audiobook_cache_and_grouping():
    import audiobook
    calls = []

    class Gen:
        def clone(self, **kw):
            calls.append("clone")
            return [np.zeros(10, dtype=np.float32)], 24000

        def xtts(self, **kw):
            calls.append("xtts")
            return [np.ones(10, dtype=np.float32)], 24000

        def csm(self, **kw):
            calls.append("csm")
            return [np.full(10, 2, dtype=np.float32)], 24000

        def f5(self, **kw):
            calls.append("f5")
            return [np.full(10, 3, dtype=np.float32)], 24000

        def dia(self, **kw):
            calls.append("dia")
            return [np.full(10, 4, dtype=np.float32)], 44100

    cache = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = dict(language="english", min_words=0, max_words=0, seed=1, temperature=0.7)
        seg = audiobook.new_segment("Hi.", "voice:Ryan")
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert calls == ["clone"], "second render should hit the cache"
        audiobook.render_segment(Gen(), dict(seg, text="Changed."), settings, cache_dir=cache)
        assert calls == ["clone", "clone"]
        xtts_seg = dict(seg, voice="xtts:Ryan")
        wav, _ = audiobook.render_segment(Gen(), xtts_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts"] and np.all(wav == 1)
        audiobook.render_segment(Gen(), xtts_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts"]
        csm_seg = dict(seg, voice="csm:Ryan")
        wav, _ = audiobook.render_segment(Gen(), csm_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts", "csm"] and np.all(wav == 2)
        audiobook.render_segment(Gen(), csm_seg, settings, cache_dir=cache)
        assert calls == ["clone", "clone", "xtts", "csm"]
        for engine, value in (("f5", 3), ("dia", 4)):
            engine_seg = dict(seg, voice=f"{engine}:Ryan")
            wav, _ = audiobook.render_segment(Gen(), engine_seg, settings, cache_dir=cache)
            assert calls[-1] == engine and np.all(wav == value)
            count = len(calls)
            audiobook.render_segment(Gen(), engine_seg, settings, cache_dir=cache)
            assert len(calls) == count
            try:
                audiobook.render_segment(Gen(), dict(engine_seg, direction="sad"), settings, cache_dir=cache)
                assert False, "unsupported directions should be rejected"
            except ValueError as exc:
                assert "direction" in str(exc)
        try:
            audiobook.render_segment(Gen(), dict(csm_seg, direction="sad"), settings, cache_dir=cache)
            assert False, "CSM directions should be rejected"
        except ValueError as exc:
            assert "direction" in str(exc)
        try:
            audiobook.render_segment(Gen(), dict(xtts_seg, direction="sad"), settings, cache_dir=cache)
            assert False, "XTTS directions should be rejected"
        except ValueError as exc:
            assert "direction" in str(exc)
        from unittest.mock import patch
        order = []

        def fake_render(gen, segment, settings):
            order.append(segment["voice"])
            return np.zeros(2), 24000

        segments = [dict(seg, voice=voice) for voice in
                    ("csm:Ryan", "xtts:Ryan", "csm:Ryan", "voice:Ryan", "xtts:Ryan")]
        with patch.object(audiobook, "render_segment", fake_render):
            results = audiobook.render_chapter(Gen(), segments, settings, notify=lambda message: None)
        assert len(results) == 5
        assert order == ["csm:Ryan", "csm:Ryan", "voice:Ryan", "xtts:Ryan", "xtts:Ryan"]
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    print("PASS: audiobook segment cache")


def _test_xtts_generation():
    import json
    from unittest.mock import patch
    import voice_library
    import tts_generator

    class Synthesizer:
        output_sample_rate = 24000

        def split_into_sentences(self, text, language):
            assert language == "en"
            return ["First.", "Second."]

    class Model:
        synthesizer = Synthesizer()

        def tts(self, **kwargs):
            calls.append(kwargs)
            return [0.1, 0.2]

    class Models:
        def get(self, kind, notify):
            assert kind == "xtts"
            return Model()

    calls = []
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(path, "reference.wav")
        write_wav(reference, np.zeros(20), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_audio": "reference.wav"}, voice_file)
        with patch.object(voice_library, "VOICES_DIR", root), patch.object(tts_generator, "seed_everything"):
            import audiobook
            assert audiobook.voice_choices() == [("Qwen: Ryan", "voice:Ryan"), ("XTTS: Ryan", "xtts:Ryan"),
                                                 ("CSM: Ryan", "csm:Ryan"), ("F5-TTS: Ryan", "f5:Ryan"),
                                                 ("Dia: Ryan", "dia:Ryan")]
            segment = audiobook.new_segment("Hello.", "xtts:Ryan")
            settings = {"language": "english"}
            key_before = audiobook.segment_cache_key(segment, settings)
            write_wav(reference, np.ones(20) * 0.1, 24000)
            assert audiobook.segment_cache_key(segment, settings) != key_before
            project_path = os.path.join(root, "project.json")
            audiobook.save_project(project_path, [segment, dict(segment, voice="voice:Ryan")], {})
            restored, _ = audiobook.load_project(project_path)
            assert [item["voice"] for item in restored] == ["xtts:Ryan", "voice:Ryan"]
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            wavs, sr = generator.xtts("First. Second.", "english", 42, 0.7, "Ryan")
            assert len(wavs) == 2 and sr == 24000
            assert all(call["speaker_wav"] == reference and call["language"] == "en" for call in calls)
            assert all(call["temperature"] == 0.7 and call["enable_text_splitting"] for call in calls)
            with patch.object(Synthesizer, "split_into_sentences", lambda self, text: [text]):
                calls.clear()
                wavs, sr = generator.xtts("Legacy.", "english", 42, 0.7, "Ryan")
                assert len(wavs) == 1 and calls[0]["text"] == "Legacy." and sr == 24000
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.xtts("First. Second.", "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
            for text, language in (("Hello", "auto"), ("[sad] Hello", "english")):
                try:
                    generator.xtts(text, language, 42, 0.7, "Ryan")
                    assert False, "unsupported XTTS input should fail"
                except ValueError:
                    pass
    print("PASS: XTTS reference, language, and cancellation")


def _test_csm_generation():
    import contextlib
    import json
    import types
    from unittest.mock import patch
    import audiobook
    import voice_library
    import tts_generator

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def __getitem__(self, index):
            return Tensor(self.value[index])

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device, dtype):
            assert device == "cpu" and dtype == "float32"
            return self

    class Processor:
        input_length = 100

        def apply_chat_template(self, conversation, **kwargs):
            conversations.append(conversation)
            assert kwargs == {"tokenize": True, "return_dict": True, "sampling_rate": 24000}
            return Inputs(input_ids=Tensor(np.zeros((1, self.input_length))))

    class Model:
        device = "cpu"
        dtype = "float32"
        config = types.SimpleNamespace(max_position_embeddings=2048, codebook_eos_token_id=0)
        complete = True

        def generate(self, **kwargs):
            calls.append(kwargs)
            end = 0 if self.complete else 1
            return types.SimpleNamespace(sequences=Tensor([[[1, 1], [end, end]]]),
                                         audio=[Tensor([0.1, 0.2])])

    class Models:
        def get(self, kind, notify):
            assert kind == "csm"
            return processor, model

    calls = []
    conversations = []
    processor = Processor()
    model = Model()
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(path, "reference.wav")
        write_wav(reference, np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_text": "Unconfirmed text", "ref_audio": "reference.wav"}, voice_file)
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, from_numpy=Tensor)
        with (patch.object(voice_library, "VOICES_DIR", root),
              patch.object(tts_generator, "seed_everything"), patch.dict(sys.modules, {"torch": fake_torch})):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "an unconfirmed transcript should fail"
            except ValueError as exc:
                assert "transcript" in str(exc)
            segment = audiobook.new_segment("Hello.", "csm:Ryan")
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_csm_reference_text("Ryan", "  This matches the recording.  ")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            key_before = audiobook.segment_cache_key(segment, {})
            write_wav(reference, np.ones(2400) * 0.1, 24000)
            assert audiobook.segment_cache_key(segment, {}) != key_before
            wavs, sr = generator.csm("Hello there. Goodbye.", "english", 42, 0.7, "Ryan")
            assert sr == 24000 and len(wavs) == 1
            assert conversations[0][0]["content"][0]["text"] == "This matches the recording."
            assert len(conversations[0][0]["content"][1]["path"]) == 2400
            assert calls[0]["max_new_tokens"] == 600 and calls[0]["temperature"] == 0.7
            functional = types.ModuleType("torchaudio.functional")
            audio_module = types.ModuleType("torchaudio")
            audio_module.functional = functional

            def resample(audio, source_rate, target_rate):
                assert source_rate == 48000 and target_rate == 24000
                return Tensor(audio.value[::2])

            functional.resample = resample
            write_wav(reference, np.zeros(4800), 48000)
            with patch.dict(sys.modules, {"torchaudio": audio_module, "torchaudio.functional": functional}):
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
            assert len(conversations[-1][0]["content"][1]["path"]) == 2400
            write_wav(reference, np.zeros(2400), 24000)
            for text, language in (("Hello", "auto"), ("[sad] Hello", "english")):
                try:
                    generator.csm(text, language, 42, 0.7, "Ryan")
                    assert False, "unsupported CSM input should fail"
                except ValueError:
                    pass
            processor.input_length = 1800
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "oversized CSM context should fail"
            except ValueError as exc:
                assert "context" in str(exc)
            processor.input_length = 100
            model.complete = False
            try:
                generator.csm("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "truncated CSM audio should fail"
            except ValueError as exc:
                assert "audio limit" in str(exc)
            model.complete = True
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.csm(" ".join(["word"] * 25), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
    print("PASS: CSM reference, generation, context, and cancellation")


def _test_f5_generation_and_worker():
    import contextlib
    import io
    import json
    import types
    from unittest.mock import patch
    import audiobook
    import f5_worker
    import voice_library
    import tts_generator

    calls = []

    class Model:
        def infer(self, reference, transcript, text, seed):
            calls.append((reference, transcript, text, seed))
            return np.ones(10), 24000

    class Models:
        def get(self, kind, notify):
            assert kind == "f5"
            return Model()

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        reference = os.path.join(root, "short.wav")
        write_wav(reference, np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "ref_text": "Unconfirmed"}, voice_file)
        with patch.object(voice_library, "VOICES_DIR", root):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            try:
                generator.f5("Hello.", "english", 42, 0.7, "Ryan")
                assert False, "unconfirmed reference must fail"
            except ValueError as exc:
                assert "transcript" in str(exc)
            segment = audiobook.new_segment("Hello.", "f5:Ryan")
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_f5_reference("Ryan", reference, "Reference words.")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            wavs, sr = generator.f5("Hello.", "english", 42, 0.7, "Ryan")
            assert len(wavs) == 1 and sr == 24000
            assert calls[0] == (os.path.join(path, "f5_reference.wav"), "Reference words.", "Hello.", 42)
            key_before = audiobook.segment_cache_key(segment, {})
            voice_library.set_f5_reference("Ryan", reference, "Changed words.")
            assert audiobook.segment_cache_key(segment, {}) != key_before
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.f5(" ".join(["word"] * 50), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
            long_reference = os.path.join(root, "long.wav")
            write_wav(long_reference, np.zeros(12 * 24000 + 1), 24000)
            try:
                voice_library.set_f5_reference("Ryan", long_reference, "Words")
                assert False, "long reference must fail"
            except ValueError as exc:
                assert "12 seconds" in str(exc)

        class FakeF5:
            def __init__(self, device):
                assert device == "cpu"

            def infer(self, **kwargs):
                assert kwargs["ref_text"] == "Reference words." and kwargs["seed"] == 42
                return np.array([0.1, 0.2]), 24000, None

        package = types.ModuleType("f5_tts")
        api = types.ModuleType("f5_tts.api")
        api.F5TTS = FakeF5
        output_path = os.path.join(root, "result.npz")
        request = dict(reference=reference, transcript="Reference words.", text="Hello.",
                       seed=42, output_path=output_path)
        output = io.StringIO()
        with (patch.dict(sys.modules, {"f5_tts": package, "f5_tts.api": api}),
              patch.object(sys, "stdin", io.StringIO(json.dumps(request) + "\n")),
              contextlib.redirect_stdout(output)):
            f5_worker.run_worker("cpu")
        responses = [json.loads(line[len(f5_worker._PREFIX):]) for line in output.getvalue().splitlines()]
        assert responses == [{"ok": True}, {"ok": True, "sample_rate": 24000}]
        with np.load(output_path) as data:
            assert np.allclose(data["wav"], [0.1, 0.2])
        package_dir = os.path.join(root, "f5_tts")
        os.makedirs(package_dir)
        with open(os.path.join(package_dir, "__init__.py"), "w", encoding="utf-8") as package_file:
            package_file.write("")
        with open(os.path.join(package_dir, "api.py"), "w", encoding="utf-8") as api_file:
            api_file.write(
                "import numpy as np\n"
                "class F5TTS:\n"
                "    def __init__(self, device):\n"
                "        print('loading fake F5')\n"
                "    def infer(self, **kwargs):\n"
                "        return np.array([0.3, 0.4]), 24000, None\n"
            )
        with patch.dict(os.environ, {"F5_TTS_PYTHON": sys.executable, "PYTHONPATH": root}):
            worker = f5_worker.F5Worker("cpu")
            try:
                wav, rate = worker.infer(reference, "Reference words.", "Hello.", 42)
                assert rate == 24000 and np.allclose(wav, [0.3, 0.4])
            finally:
                worker.close()
    print("PASS: F5 reference, caching, cancellation, and worker protocol")


def _test_dia_generation():
    import contextlib
    import json
    import types
    from unittest.mock import patch
    import voice_library
    import tts_generator

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, **kwargs):
            prompts.append(kwargs)
            return Inputs(input_ids=Tensor(np.zeros((1, 50))),
                          decoder_input_ids=Tensor(np.zeros((1, 100, 9))),
                          decoder_attention_mask=Tensor(np.zeros((1, 115))))

        def get_audio_prompt_len(self, mask):
            return 100

        def batch_decode(self, output, audio_prompt_len):
            assert audio_prompt_len == 100
            return [Tensor(np.ones(441))]

    class Model:
        device = "cpu"
        config = types.SimpleNamespace(
            decoder_config=types.SimpleNamespace(max_position_embeddings=3072),
            encoder_config=types.SimpleNamespace(max_position_embeddings=1024),
            delay_pattern=[0, 15])

        def generate(self, **kwargs):
            calls.append(kwargs)
            return Tensor(np.zeros((1, 200, 9)))

    class Models:
        def get(self, kind, notify):
            assert kind == "dia"
            return Processor(), Model()

    calls, prompts, rates = [], [], []
    functional = types.ModuleType("torchaudio.functional")
    audio_module = types.ModuleType("torchaudio")
    audio_module.functional = functional

    def resample(audio, source_rate, target_rate):
        rates.append((source_rate, target_rate))
        return Tensor(np.zeros(round(len(audio.value) * target_rate / source_rate)))

    functional.resample = resample
    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, from_numpy=Tensor)
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Ryan")
        os.makedirs(path)
        write_wav(os.path.join(path, "reference.wav"), np.zeros(2400), 24000)
        with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
            json.dump({"name": "Ryan", "csm_ref_text": "Reference words."}, voice_file)
        with (patch.object(voice_library, "VOICES_DIR", root), patch.object(tts_generator, "seed_everything"),
              patch.dict(sys.modules, {"torch": fake_torch, "torchaudio": audio_module,
                                       "torchaudio.functional": functional})):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            wavs, sr = generator.dia("First sentence.", "english", 42, 0.7, "Ryan")
            assert sr == 24000 and len(wavs) == 1 and len(wavs[0]) == 240
            assert prompts[0]["text"] == "[S1] Reference words. [S1] First sentence."
            assert rates == [(24000, 44100), (44100, 24000)]
            assert calls[0]["max_new_tokens"] == 1400 and calls[0]["temperature"] == 0.7
            calls.clear()
            generator.should_stop = lambda: len(calls) == 1
            assert generator.dia(" ".join(["word"] * 20), "english", 42, 0.7, "Ryan") == ([], None)
            assert len(calls) == 1
    print("PASS: Dia reference, resampling, and cancellation")


def _test_engine_model_lifecycle():
    import importlib.util
    import types
    from unittest.mock import patch
    import f5_worker

    fake_torch = types.ModuleType("torch")
    fake_torch.float32, fake_torch.float16, fake_torch.bfloat16 = "float32", "float16", "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    qwen = types.ModuleType("qwen_tts")
    qwen.Qwen3TTSModel = object
    events = []

    class Worker:
        def __init__(self, device):
            events.append(("start", device))

        def close(self):
            events.append(("close",))

    class AudioTokenizer:
        def to(self, device):
            events.append(("codec", device))
            return self

        def eval(self):
            return self

    class Processor:
        audio_tokenizer = AudioTokenizer()

        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

    class DiaModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert kwargs["torch_dtype"] == "float32"
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.DiaForConditionalGeneration = DiaModel
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_loader.py")
    spec = importlib.util.spec_from_file_location("_test_engine_model_loader", module_path)
    module = importlib.util.module_from_spec(spec)
    with (patch.dict(sys.modules, {"torch": fake_torch, "qwen_tts": qwen, "transformers": transformers}),
          patch.object(f5_worker, "F5Worker", Worker)):
        spec.loader.exec_module(module)
        manager = module.ModelManager(lambda: ("cpu", "bfloat16", "sdpa"))
        with patch.object(module, "load_model", return_value=object()):
            manager.get("voice-clone")
            first = manager.get("f5")
            assert manager.get("f5") is first
            manager.get("dia")
            assert events == [("start", "cpu"), ("close",), ("codec", "cpu")]
            assert list(manager._models) == ["dia"]
            manager.get("voice-design")
            assert list(manager._models) == ["voice-design"]
    print("PASS: engine model lifecycle")


def _test_voice_library_roundtrip():
    import torch
    import voice_library
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        prompt = [_FakePrompt(torch.zeros(3))]
        wav = np.zeros(2400, dtype=np.float32)
        cfg = voice_library.save_voice("My Voice!", wav, 24000, prompt, "warm", "english", 7, root=root)
        assert voice_library.list_voices(root) == ["My Voice!"]
        loaded, p, path = voice_library.load_voice("My Voice!", root)
        assert loaded["seed"] == 7 and loaded["description"] == "warm" and cfg["sample_text"] == voice_library.SAMPLE_TEXT
        assert os.path.exists(os.path.join(path, "reference.wav"))
        try:
            voice_library.save_voice("My Voice!", wav, 24000, prompt, "x", "english", 1, root=root)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        voice_library.delete_voice("My Voice!", root)
        assert voice_library.list_voices(root) == []
        # transcript semantics: None = standard passage, "" = explicitly no transcript, text = stored exactly
        for given, stored in ((None, voice_library.SAMPLE_TEXT), ("", ""), ("Real words.", "Real words.")):
            cfg = voice_library.save_voice("T", wav, 24000, prompt, "", "english", 3, root=root,
                                           overwrite=True, sample_text=given)
            assert cfg["ref_text"] == cfg["sample_text"] == stored and cfg["seed"] == 3
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("PASS: voice library round trip")


def _snapshot_tree(path):
    """{relative path: bytes} for every file under path, plus directories, to prove nothing changed."""
    out = {}
    for base, dirs, files in os.walk(path):
        for d in dirs:
            out[os.path.relpath(os.path.join(base, d), path) + "/"] = b""
        for f in files:
            full = os.path.join(base, f)
            with open(full, "rb") as fh:
                out[os.path.relpath(full, path)] = fh.read()
    return out


def _test_voice_replace_is_failure_safe():
    import json
    import shutil
    import tempfile
    import numpy as np
    import torch
    from unittest.mock import patch
    import voice_library as vl

    wav = np.zeros(2400, dtype=np.float32)
    prompt = [_FakePrompt(torch.zeros(3))]
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))

    def listing():
        return sorted(os.listdir(root)), vl.list_voices(root), [d["name"] for d in vl.list_voice_details(root)]

    def make_old():
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        vl.save_voice("Ana", wav, 24000, prompt, "old description", "english", 11, root=root)
        folder = vl.voice_dir("Ana", root)
        os.makedirs(os.path.join(folder, "emotions"))
        with open(os.path.join(folder, "emotions", "abc.npz"), "wb") as f:
            f.write(b"emotion-cache")
        with open(os.path.join(folder, "f5_reference.wav"), "wb") as f:
            f.write(b"old-f5")
        cfg_path = os.path.join(folder, "voice.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg.update(csm_ref_text="old transcript", f5_ref_audio="f5_reference.wav", f5_ref_text="old f5")
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
        return folder, _snapshot_tree(folder)

    def replace(**kw):
        return vl.save_voice("Ana", wav, 22050, prompt, "new description", "english", 5,
                             root=root, overwrite=True, **kw)

    try:
        # --- success: the new version replaces everything bound to the old recording
        folder, before = make_old()
        assert vl.describe_voice_assets("Ana", root) == [
            "its recording and clone prompt", "1 cached emotion sample",
            "its F5 reference recording and transcript", "its CSM/Dia transcript"]
        cfg = replace()
        after = _snapshot_tree(folder)
        assert cfg["description"] == "new description" and cfg["sample_rate"] == 22050
        assert not os.path.exists(os.path.join(folder, "emotions")) and "f5_reference.wav" not in after
        assert "csm_ref_text" not in json.load(open(os.path.join(folder, "voice.json"), encoding="utf-8"))
        assert sorted(set(os.listdir(root)) - {".voices.lock"}) == ["Ana"], os.listdir(root)   # no staging or backup left behind

        # --- failure injection: every step leaves the old voice byte-identical and listed exactly once
        real_rename, real_rmtree = os.rename, shutil.rmtree
        stage_failures = {
            "write_wav": patch("voice_library.write_wav", side_effect=OSError("disk full")),
            "write_prompt": patch("voice_library.save_voice_clone_prompt", side_effect=PermissionError("denied")),
            "write_json": patch("voice_library._write_text_durably", side_effect=OSError("io error")),
        }
        for label, patcher in stage_failures.items():
            folder, before = make_old()
            with patcher:
                try:
                    replace()
                    raise AssertionError(f"{label}: expected a failure")
                except vl.VoiceSaveError as exc:
                    assert "existing voice is unchanged" in str(exc), (label, exc)
            assert _snapshot_tree(folder) == before, label
            assert listing()[1:] == (["Ana"], ["Ana"]), label
            pending_dir = os.path.join(root, ".pending")
            assert not os.path.isdir(pending_dir) or os.listdir(pending_dir) == [], label   # no staging left behind

        # second rename (staging -> final) fails: the backup is put back
        folder, before = make_old()
        calls = []

        def fail_second(src, dst):
            calls.append((src, dst))
            if len(calls) == 2:
                raise PermissionError("denied")
            return real_rename(src, dst)

        with patch("voice_library.os.rename", side_effect=fail_second):
            try:
                replace()
                raise AssertionError("expected VoiceSaveError")
            except vl.VoiceSaveError as exc:
                assert "previous version is unchanged" in str(exc), exc
        assert _snapshot_tree(folder) == before and vl.list_voices(root) == ["Ana"]

        # first rename (old -> backup) fails: nothing was touched
        folder, before = make_old()
        with patch("voice_library.os.rename", side_effect=PermissionError("denied")):
            try:
                replace()
                raise AssertionError("expected VoiceSaveError")
            except vl.VoiceSaveError as exc:
                assert "Nothing was changed" in str(exc), exc
        assert _snapshot_tree(folder) == before

        # second rename AND the rollback fail: the error names where the old voice is kept, and recovery restores it
        folder, before = make_old()
        calls = []

        def fail_after_first(src, dst):
            calls.append((src, dst))
            if len(calls) >= 2:
                raise PermissionError("denied")
            return real_rename(src, dst)

        with patch("voice_library.os.rename", side_effect=fail_after_first):
            try:
                replace()
                raise AssertionError("expected VoiceSaveError")
            except vl.VoiceSaveError as exc:
                assert "kept at" in str(exc) and ".pending" in str(exc), exc
        assert not os.path.exists(folder)
        assert vl.recover_interrupted_saves(root) == ["Ana"]
        assert _snapshot_tree(folder) == before and vl.list_voices(root) == ["Ana"]

        # an interrupt between the two renames still rolls back
        folder, before = make_old()
        calls = []

        def crash_after_first(src, dst):
            calls.append((src, dst))
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return real_rename(src, dst)

        with patch("voice_library.os.rename", side_effect=crash_after_first):
            try:
                replace()
                raise AssertionError("expected KeyboardInterrupt")
            except KeyboardInterrupt:
                pass
        assert _snapshot_tree(folder) == before   # even an interrupt rolls back

        # deleting the backup fails: the save still succeeded
        folder, before = make_old()
        def rmtree_fails_on_backup(path, *a, **k):
            if ".old-" in os.path.basename(path):
                if k.get("ignore_errors"):
                    return None   # what ignore_errors=True does with a "busy" folder
                raise OSError("busy")
            return real_rmtree(path, *a, **k)
        with patch("voice_library.shutil.rmtree", side_effect=rmtree_fails_on_backup):
            replace()
        assert json.load(open(os.path.join(folder, "voice.json"), encoding="utf-8"))["description"] == "new description"
        assert vl.list_voices(root) == ["Ana"]
        # that leftover is marked finished: deleting the voice afterwards must never bring the old copy back
        vl.delete_voice("Ana", root)
        assert vl.recover_interrupted_saves(root) == [] and vl.list_voices(root) == [] and not os.path.exists(folder)

        # a stale backup left next to a finished replace is discarded, a stale .new is removed
        folder, before = make_old()
        pending = os.path.join(root, ".pending")
        os.makedirs(os.path.join(pending, "Ana.old-zzz"))
        os.makedirs(os.path.join(pending, "Ana.new-zzz"))
        os.makedirs(os.path.join(pending, "Ana.done-zzz"))
        assert vl.recover_interrupted_saves(root) == [] and not os.path.exists(pending)

        # --- collisions and refusals
        folder, before = make_old()
        for other in ("Ana", "ana", " Ana "):          # same slug (and, on a case-insensitive disk, same folder)
            if other.strip().lower() == "ana" and os.path.exists(vl.voice_dir(other, root)):
                try:
                    vl.save_voice(other, wav, 24000, prompt, "x", "english", 1, root=root)
                    raise AssertionError("expected FileExistsError")
                except FileExistsError:
                    pass
        assert vl.voice_dir("Ana B", root) == vl.voice_dir("Ana_B", root)
        vl.save_voice("Ana B", wav, 24000, prompt, "x", "english", 1, root=root)
        try:
            vl.save_voice("Ana_B", wav, 24000, prompt, "y", "english", 1, root=root)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        long_a, long_b = "v" * 70 + "A", "v" * 70 + "B"   # both truncate to the same 60-character folder
        assert vl.voice_dir(long_a, root) == vl.voice_dir(long_b, root)
        vl.save_voice(long_a, wav, 24000, prompt, "x", "english", 1, root=root)
        try:
            vl.save_voice(long_b, wav, 24000, prompt, "y", "english", 1, root=root)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        assert _snapshot_tree(folder) == before
        # overwrite=True never replaces a folder that is not a voice
        os.makedirs(os.path.join(root, "Notes"))
        with open(os.path.join(root, "Notes", "keep.txt"), "w") as f:
            f.write("mine")
        try:
            vl.save_voice("Notes", wav, 24000, prompt, "x", "english", 1, root=root, overwrite=True)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        assert open(os.path.join(root, "Notes", "keep.txt")).read() == "mine"

        # --- two app instances: a save waits for, and never overlaps, another process holding the folder lock
        if os.name == "posix":
            import subprocess
            import time
            folder, before = make_old()
            holder = subprocess.Popen(
                [sys.executable, "-c",
                 "import fcntl,sys,time\n"
                 "f=open(sys.argv[1]+'/.voices.lock','a+'); fcntl.flock(f, fcntl.LOCK_EX)\n"
                 "print('held', flush=True); time.sleep(30)", root],
                stdout=subprocess.PIPE, text=True)
            try:
                assert holder.stdout.readline().strip() == "held"
                os.makedirs(os.path.join(root, ".pending", "Ana.old-qq"))   # recovery must not touch it while locked
                assert vl.recover_interrupted_saves(root) == [] and os.path.isdir(os.path.join(root, ".pending", "Ana.old-qq"))
                real_timeout, vl.LOCK_TIMEOUT = vl.LOCK_TIMEOUT, 0.3
                try:
                    replace()
                    raise AssertionError("expected VoiceSaveError while another process holds the lock")
                except vl.VoiceSaveError as exc:
                    assert "Another Qwen3-TTS window" in str(exc) and "Nothing was changed" in str(exc), exc
                finally:
                    vl.LOCK_TIMEOUT = real_timeout
                assert _snapshot_tree(folder) == before
            finally:
                holder.kill()
                holder.wait()
            replace()   # the lock dies with its process, so the next save goes through
            assert vl.list_voices(root) == ["Ana"]

        # --- read-only voices folder (POSIX): nothing is written and the voice survives
        if os.name == "posix" and os.geteuid() != 0:
            folder, before = make_old()
            os.remove(os.path.join(root, ".voices.lock"))   # a fresh folder has no lock file yet: that must not leak a raw error
            os.chmod(root, 0o555)
            try:
                try:
                    replace()
                    raise AssertionError("expected a permission failure")
                except vl.VoiceSaveError as exc:
                    assert "Nothing was changed" in str(exc) or "existing voice is unchanged" in str(exc), exc
            finally:
                os.chmod(root, 0o755)
            assert _snapshot_tree(folder) == before and vl.list_voices(root) == ["Ana"]
    finally:
        os.chmod(root, 0o755)
        shutil.rmtree(root, ignore_errors=True)
    print("PASS: voice replacement is failure-safe")


def _test_audio_loading_formats_and_capabilities():
    """load_wav reads what the model's own decoder reads, supported_audio_extensions never advertises a format
    that would fail, and unreadable files get a sentence the user can act on."""
    import importlib.util
    import soundfile as sf
    import audio_utils

    tmp = tempfile.mkdtemp()
    t = np.arange(24000) / 24000
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32")

    def check(name, **kw):
        path = os.path.join(tmp, name)
        sf.write(path, tone, 24000, **kw)
        data, rate = audio_utils.load_wav(path)
        assert rate == 24000 and data.ndim == 1 and data.dtype == np.float32, (name, rate, data.shape)
        assert len(data) == 24000, (name, len(data))
        assert np.abs(data - tone).max() < (0.02 if name.endswith(".mp3") else 1e-3), name
        return path
    # the standard-library path is unchanged for 16-bit PCM
    pcm = os.path.join(tmp, "pcm16.wav")
    write_wav(pcm, tone, 24000)
    with wave.open(pcm, "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    assert np.array_equal(audio_utils.load_wav(pcm)[0], raw), "16-bit PCM must read exactly as before"
    # what the old reader rejected
    check("float32.wav", subtype="FLOAT")
    check("pcm24.wav", subtype="PCM_24")
    check("flac24.flac", subtype="PCM_24")
    if "MP3" in sf.available_formats():
        check("tone.mp3")
    # stereo keeps the first channel, like the old reader
    stereo = os.path.join(tmp, "stereo_float.wav")
    sf.write(stereo, np.stack([tone, -tone], axis=1), 24000, subtype="FLOAT")
    left, _ = audio_utils.load_wav(stereo)
    assert np.abs(left - tone).max() < 1e-3
    # unreadable files: a usable sentence, never a raw decoder message
    empty, junk, m4a = (os.path.join(tmp, n) for n in ("empty.wav", "junk.wav", "voice.m4a"))
    open(empty, "wb").close()
    open(junk, "wb").write(b"not audio" * 40)
    open(m4a, "wb").write(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
    for bad in (empty, junk, m4a):
        try:
            audio_utils.load_wav(bad)
        except audio_utils.AudioReadError as exc:
            assert "Choose" in str(exc) and "Error" not in str(exc), str(exc)
        else:
            raise AssertionError(f"{bad} should not load")
    # capability: what is offered is exactly what soundfile decodes; M4A is never advertised or accepted, even if a
    # package named audioread happens to be installed (the model's loader would not use it)
    real_find = importlib.util.find_spec
    try:
        importlib.util.find_spec = lambda name, *a, **k: object() if name == "audioread" else real_find(name, *a, **k)
        exts = audio_utils.supported_audio_extensions()
        assert ".m4a" not in exts and {".wav", ".flac"} <= set(exts), exts
        assert "m4a" not in audio_utils.audio_file_filter()
        for e in exts:   # offered and rejected must never both be true
            probe = os.path.join(tmp, "probe" + e)
            sf.write(probe, tone, 24000, format=e[1:].upper()) if e != ".mp3" or "MP3" in sf.available_formats() else None
            if os.path.isfile(probe):
                assert audio_utils.audio_problem(probe) is None, e
        msg = audio_utils.audio_problem(m4a)
        assert "M4A" in msg and "can't be read on this computer" in msg and "WAV" in msg, msg
        assert "empty or damaged" in audio_utils.audio_problem(junk)
        assert audio_utils.audio_problem(pcm) is None and audio_utils.audio_problem(os.path.join(tmp, "float32.wav")) is None
    finally:
        importlib.util.find_spec = real_find
    # F5 reference: unreadable files keep failing as a ValueError naming the F5 reference; float WAV now works
    import json
    import voice_library
    root = tempfile.mkdtemp()
    os.makedirs(voice_library.voice_dir("F5 Check", root))
    with open(os.path.join(voice_library.voice_dir("F5 Check", root), "voice.json"), "w") as f:
        json.dump({"name": "F5 Check"}, f)
    try:
        voice_library.set_f5_reference("F5 Check", junk, "words", root)
    except ValueError as exc:
        assert str(exc).startswith("F5 reference: ") and "Choose" in str(exc), str(exc)
    else:
        raise AssertionError("a junk file must not become an F5 reference")
    voice_library.set_f5_reference("F5 Check", os.path.join(tmp, "float32.wav"), "words", root)
    assert os.path.isfile(os.path.join(voice_library.voice_dir("F5 Check", root), "f5_reference.wav"))
    print("PASS: audio loading reads float, 24-bit, FLAC and MP3, never advertises undecodable formats, and explains unreadable files")


def _test_clone_reports_how_many_chunks_it_planned():
    """The UI tells a real early stop from a Stop pressed after the last chunk by comparing with clone_chunks."""
    import numpy as np
    import tts_generator

    class Model:
        def create_voice_clone_prompt(self, **kw):
            return ["prompt"]

        def generate_voice_clone(self, **kw):
            return [np.zeros(240, dtype=np.float32)], 24000

    class Models:
        def get(self, kind, notify=None):
            return Model()

    text = "One two three. Four five six. Seven eight nine."
    seen = []
    generator = tts_generator.TTSGenerator(Models(), notify=seen.append)
    out, sr = generator.clone(text=text, language="english", min_words=1, max_words=3, ref_audio="x", ref_text="t")
    assert generator.clone_chunks == len(out) == len([m for m in seen if m.startswith("Chunk ")]) and len(out) >= 2
    planned = generator.clone_chunks
    stop = {"after": 1}
    stopper = tts_generator.TTSGenerator(Models(), notify=seen.append, should_stop=lambda: len(seen) > stop["after"] + 3)
    out2, _ = stopper.clone(text=text, language="english", min_words=1, max_words=3, ref_audio="x", ref_text="t")
    assert stopper.clone_chunks == planned and len(out2) < planned, "an early stop returns fewer chunks than planned"
    print("PASS: clone reports how many chunks it planned, so a real early stop can be told from a completed one")


def _test_emotion_tags_and_seed():
    import torch
    from voice_lock import parse_emotion_segments, parse_seed, seed_everything
    assert parse_emotion_segments("a [angry] b [whisper] c [1] d") == [
        (None, "a"), ("angry", "b"), ("whisper", "c [1] d")]
    assert parse_seed("Random", "5") is None and parse_seed("Fixed", "5") == 5
    seed_everything(3); x = torch.rand(2)
    seed_everything(3); assert torch.equal(x, torch.rand(2))
    print("PASS: emotion tags and seeding")


def _test_benchmark_helpers():
    from benchmark import (
        automatic_chunk_sizes, build_configurations, chunk_bounds, prepare_input_text,
        real_time_factor,
    )
    assert chunk_bounds(0) == (0, 0)
    assert chunk_bounds(250) == (200, 250)
    assert chunk_bounds(1) == (1, 1)
    assert real_time_factor(12, 60) == 0.2
    assert real_time_factor(1, 0) is None
    assert automatic_chunk_sizes("one two three four five six") == [0, 3, 2]
    sample = "First sentence has four words. Second sentence has five words."
    assert prepare_input_text(sample, 5) == "First sentence has four words."
    assert prepare_input_text(sample, 0) == sample
    one_factor = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                      [0, 3], order_seed=5)
    assert len(one_factor) == 4
    assert len(set(one_factor)) == 4
    backends = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                    [250, 400], matrix="backends", order_seed=5)
    assert len(backends) == 4
    assert {config[3] for config in backends} == {250}
    assert {(config[1], config[2]) for config in backends} == {
        ("bf16", "flash"), ("bf16", "sdpa"), ("fp16", "flash"), ("fp16", "sdpa")
    }
    chunking = build_configurations(["cuda"], ["bf16"], ["sdpa"], [0, 250, 400],
                                   matrix="chunking", order_seed=5)
    assert {config[3] for config in chunking} == {0, 250, 400}
    full = build_configurations(["cuda"], ["bf16", "fp16"], ["flash", "sdpa"],
                                [0, 3], matrix="full", order_seed=5)
    assert len(full) == 8
    print("PASS: benchmark helpers")


# ------------------------------------------------------------------ runner

def _cast_fixture(root, name, **config):
    """A voice folder with the files the engines read; `config` overrides or (None) removes voice.json keys."""
    import json
    path = os.path.join(root, name)
    os.makedirs(path)
    seconds = config.pop("seconds", 2)
    if seconds is not None:
        write_wav(os.path.join(path, "reference.wav"), np.zeros(int(24000 * seconds)), 24000)
    np.savez(os.path.join(path, "clone_prompt.npz"), x=np.zeros(1))
    data = {"name": name, "description": "calm", "ref_audio": "reference.wav", "clone_prompt": "clone_prompt.npz",
            "csm_ref_text": "Hello there."}
    data.update(config)
    data = {key: value for key, value in data.items() if value is not None}
    with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as voice_file:
        json.dump(data, voice_file)


_ALL_READY = {"xtts": True, "csm": True, "dia": True, "f5": "ok"}


def _test_cast_rules_states_and_reasons():
    import tempfile
    import cast_rules as cr

    assert cr.parse_key("voice:Ana") == ("qwen", "Ana") and cr.parse_key("f5:A:b") == ("f5", "A:b")
    assert cr.parse_key("") == (None, "") and cr.parse_key("Ana") == (None, "") and cr.parse_key("zzz:Ana") == (None, "")
    assert all(cr.parse_key(cr.make_key(engine, "Ana")) == (engine, "Ana") for engine in cr.ENGINES)
    with tempfile.TemporaryDirectory() as root:
        _cast_fixture(root, "Ana")
        _cast_fixture(root, "Bare", description="", csm_ref_text=None)
        _cast_fixture(root, "Long", seconds=13)
        _cast_fixture(root, "NoRef", seconds=None)
        _cast_fixture(root, "NoPrompt", clone_prompt=None)
        _cast_fixture(root, "OwnF5", f5_ref_audio="f5_reference.wav", f5_ref_text="Short words.")
        write_wav(os.path.join(root, "OwnF5", "f5_reference.wav"), np.zeros(24000), 24000)
        _cast_fixture(root, "OwnF5NoKey", f5_ref_audio="f5_reference.wav", f5_ref_text=None)
        write_wav(os.path.join(root, "OwnF5NoKey", "f5_reference.wav"), np.zeros(24000), 24000)

        def state(key, language="english", direction="", text="Hello.", env=_ALL_READY):
            return cr.check(key, language, direction, text, root=root, env=env)

        # Ready, with the honest note that only local facts were checked
        for engine in cr.ENGINES:
            result = state(cr.make_key(engine, "Ana"))
            assert result.state == cr.READY and result.issues == () and cr.LOCAL_ONLY_NOTE in result.notes, engine
        assert state("voice:Ana", language="auto", direction="whisper", text="[sad] Hi.").state == cr.READY
        # unsupported combinations
        assert state("xtts:Ana", language="auto").state == cr.UNSUPPORTED
        assert state("xtts:Ana", language="german").state == cr.READY
        for engine in ("csm", "dia"):
            assert state(f"{engine}:Ana", language="french").state == cr.UNSUPPORTED
        assert state("f5:Ana", language="chinese").state == cr.READY and state("f5:Ana", language="german").state == cr.UNSUPPORTED
        for engine in ("xtts", "csm", "dia", "f5"):
            direction = state(f"{engine}:Ana", direction="whisper")
            tags = state(f"{engine}:Ana", text="[sad] Hi.")
            assert direction.state == tags.state == cr.UNSUPPORTED and direction.issues[0].cause != tags.issues[0].cause
            assert state(f"{engine}:Ana", text="Chapter [1] begins.").state == cr.READY, "[1] is not an emotion tag"
        assert state("voice:Bare", text="[sad] Hi.").state == cr.UNSUPPORTED     # tags need a description
        assert state("voice:Bare", text="Hi.").state == cr.READY
        bare_direction = state("voice:Bare", direction="whisper", text="Hi.")  # a direction is sent as a leading [tag]
        assert bare_direction.state == cr.UNSUPPORTED and "need a voice description" in bare_direction.message
        assert state("voice:Bare", direction="[whisper]", text="Hi.").state == cr.UNSUPPORTED
        assert state("voice:Ana", direction="whisper", text="Hi.").state == cr.READY
        # needs setup: transcript, long F5 reference, packages; the assignment itself stays valid
        for engine in ("csm", "dia", "f5"):
            result = state(f"{engine}:Bare")
            assert result.state == cr.SETUP and "transcript" in result.issues[0].message and "My Voices" in result.issues[0].message
        assert state("xtts:Bare").state == cr.READY and state("voice:Bare").state == cr.READY
        assert state("f5:Long").state == cr.SETUP and "12 s" in state("f5:Long").short
        assert state("csm:Long").state == cr.READY, "the 12 s limit is F5's"
        assert state("f5:OwnF5").state == cr.READY, "F5 uses its own short reference and transcript when it has one"
        assert state("f5:OwnF5NoKey").state == cr.SETUP, "its own reference without its own transcript is not the CSM transcript"
        assert state("csm:OwnF5NoKey").state == cr.READY
        assert state("xtts:Ana", env={**_ALL_READY, "xtts": False}).state == cr.SETUP
        assert state("csm:Ana", env={**_ALL_READY, "csm": False}).state == cr.SETUP
        assert state("dia:Ana", env={**_ALL_READY, "dia": False}).state == cr.SETUP
        assert state("f5:Ana", env={**_ALL_READY, "f5": "missing"}).state == cr.SETUP
        unverified = state("f5:Ana", env={**_ALL_READY, "f5": "unverified"})
        assert unverified.state == cr.READY and any("not inspected" in note for note in unverified.notes)
        assert any("gated" in note for note in state("csm:Ana").notes) and any("license" in note for note in state("xtts:Ana").notes)
        # missing: the voice, its reference, its prompt, or an entry this version does not know
        assert state("voice:Gone").state == cr.MISSING and state("csm:Gone").state == cr.MISSING
        assert state("xtts:NoRef").state == cr.MISSING and state("voice:NoRef").state == cr.READY
        assert state("voice:NoPrompt").state == cr.MISSING and state("xtts:NoPrompt").state == cr.READY
        assert state("legacy").state == cr.MISSING
        assert state("").state == cr.SETUP and "no voice" in state("").short
        # the worst state wins, and several causes are all reported
        both = state("csm:Bare", language="french", direction="x")
        assert both.state == cr.UNSUPPORTED and len(both.issues) >= 3
        assert [cr.SEVERITY[item.state] for item in both.issues] == sorted((cr.SEVERITY[item.state] for item in both.issues), reverse=True)

        # pre-flight groups segments by cause and names their rows
        segments = [{"voice": "voice:Ana", "text": "Hi."}, {"voice": "csm:Bare", "text": "Hi."},
                    {"voice": "csm:Bare", "text": "Again."}, {"voice": "voice:Gone", "text": "Hi."},
                    {"voice": "dia:Bare", "text": "Hi."}, {"voice": "", "text": "Hi."}]
        groups = cr.preflight(segments, "english", root=root, env=_ALL_READY)
        by_cause = {group.cause: group for group in groups}
        assert by_cause["setup:csm:transcript:Bare"].rows == (1, 2)
        assert by_cause["setup:dia:transcript:Bare"].rows == (4,) and by_cause["missing:voice:Gone"].rows == (3,)
        assert by_cause["setup:none:voice"].rows == (5,) and len(groups) == 4
        assert groups[0].state == cr.MISSING, "worst first"
        assert cr.preflight(segments[:1], "english", root=root, env=_ALL_READY) == []
        assert cr.preflight(segments[1:3], "english", rows=[7, 9], root=root, env=_ALL_READY)[0].rows == (7, 9)
        # a memo is only a cache of file reads
        memo = {}
        assert cr.preflight(segments, "english", root=root, env=_ALL_READY, memo=memo) == groups and memo
    print("PASS: cast rules: ready, needs setup, unsupported and missing states, reasons, notes and grouped pre-flight")


def _test_cast_rules_reference_audio_agrees_with_load_wav():
    """cast_rules judges a reference from its header; audio_utils.load_wav is what the engines really call."""
    import struct
    import tempfile
    import soundfile as sf
    import audio_utils
    import cast_rules as cr

    tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)).astype("float32")
    with tempfile.TemporaryDirectory() as tmp:
        cases = {}                                           # file name -> should load_wav accept it

        def audio(name, data=tone, rate=24000, **kw):
            sf.write(os.path.join(tmp, name), data, rate, **kw)
            cases[name] = True
        write_wav(os.path.join(tmp, "pcm16.wav"), tone, 24000)
        cases["pcm16.wav"] = True
        audio("pcm8.wav", subtype="PCM_U8")
        audio("pcm24.wav", subtype="PCM_24")
        audio("pcm32.wav", subtype="PCM_32")
        audio("float32.wav", subtype="FLOAT")
        audio("float64.wav", subtype="DOUBLE")
        audio("stereo_float.wav", np.stack([tone, -tone], axis=1), subtype="FLOAT")
        audio("tone.flac", subtype="PCM_24")
        audio("tone.ogg")
        if "MP3" in sf.available_formats():
            audio("tone.mp3")
        audio("long.wav", np.zeros(24000 * 13, dtype="float32"), subtype="FLOAT")
        # inputs load_wav refuses: empty, damaged, wrong content, truncated header, missing
        audio("empty_frames.wav", np.zeros(0, dtype="float32"))
        cases["empty_frames.wav"] = True    # load_wav returns zero samples without raising (stdlib path); see below
        for name, payload in (("zero_bytes.wav", b""), ("text.wav", b"not audio at all"),
                              ("riff_only.wav", b"RIFF" + struct.pack("<I", 4) + b"WAVE"), ("voice.m4a", b"\x00" * 64)):
            with open(os.path.join(tmp, name), "wb") as handle:
                handle.write(payload)
            cases[name] = False
        with open(os.path.join(tmp, "pcm16.wav"), "rb") as handle:
            head = handle.read(300)
        with open(os.path.join(tmp, "cut_header.wav"), "wb") as handle:
            handle.write(head[:30])
        cases["cut_header.wav"] = False
        # cut off mid-data: the header is intact and still claims the full length (an interrupted copy)
        write_wav(os.path.join(tmp, "pcm16_long.wav"), np.zeros(24000 * 13, dtype="float32"), 24000)
        long_pcm = open(os.path.join(tmp, "pcm16_long.wav"), "rb").read()
        for name, size in (("cut_tiny.wav", 44 + 100), ("cut_half.wav", 44 + 24000 * 2 * 5)):
            with open(os.path.join(tmp, name), "wb") as handle:
                handle.write(long_pcm[:size])
            cases[name] = True
        cases["missing.wav"] = False
        assert any(ok for ok in cases.values()) and not all(cases.values()), "the fixtures must cover both outcomes"

        for name, ok in cases.items():
            path = os.path.join(tmp, name)
            try:
                audio_utils.load_wav(path)
                loads = True
            except (audio_utils.AudioReadError, OSError):
                loads = False
            assert loads == ok, f"{name}: load_wav {'read' if loads else 'refused'} it, the fixture says {ok}"
            seconds = cr._audio_seconds(path, None)
            if name in ("empty_frames.wav", "cut_tiny.wav"):
                # Deliberately stricter than load_wav, which hands the engine zero samples: a silent no-op reference
                # would only fail later, so the pre-flight says so now.
                assert len(audio_utils.load_wav(path)[0]) < 0.1 * 24000 and seconds is None
                continue
            assert (seconds is not None) == loads, f"{name}: cast_rules says {seconds}, load_wav {loads}"
            if loads:
                assert abs(seconds - len(audio_utils.load_wav(path)[0]) / audio_utils.load_wav(path)[1]) < 0.01, name
        # through check(): a readable float / 24-bit reference is Ready, an unreadable one needs setup
        for name, ready in (("float32.wav", True), ("pcm24.wav", True), ("tone.flac", True), ("text.wav", False),
                            ("empty_frames.wav", False), ("long.wav", True), ("cut_tiny.wav", False), ("cut_half.wav", True)):
            voice = "V_" + name.replace(".", "_")
            _cast_fixture(tmp, voice, seconds=None, ref_audio=name)
            shutil.copy(os.path.join(tmp, name), os.path.join(tmp, voice, name))
            for engine in ("csm", "dia", "f5"):
                result = cr.check(f"{engine}:{voice}", "english", "", "Hello.", root=tmp, env=_ALL_READY)
                if name == "long.wav" and engine == "f5":
                    assert result.state == cr.SETUP and "over 12" in result.short, result.short
                else:
                    assert (result.state == cr.READY) == ready, (engine, name, result.state, result.short)
                if not ready:
                    assert result.short == "reference unreadable", result.short
        # nothing was decoded: the header read must not call load_wav
        original = audio_utils.load_wav
        audio_utils.load_wav = lambda *a, **k: (_ for _ in ()).throw(AssertionError("pre-flight decoded a file"))
        try:
            assert cr._audio_seconds(os.path.join(tmp, "float32.wav"), None) is not None
        finally:
            audio_utils.load_wav = original
    print(f"PASS: cast rules read references the way load_wav does ({len(cases)} files: PCM 8/16/24/32-bit, float, "
          "FLAC, OGG, MP3, empty, damaged, missing), from the header only")


def _test_cast_rules_agree_with_the_engines_before_any_model_loads():
    """For every case the engines would reject on their own checks, cast_rules must not say Ready, and the other way
    round. The model manager raises a sentinel the moment an engine asks for a model, so the engine's own validation
    (the part that must match) is exactly what runs."""
    import itertools
    import tempfile
    from unittest.mock import patch
    import audiobook
    import cast_rules as cr
    import tts_generator
    import voice_library

    class Reached(Exception):
        pass

    class Models:
        def get(self, kind, notify=None):
            raise Reached(kind)

    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as cache:
        _cast_fixture(root, "Ana")
        _cast_fixture(root, "Bare", csm_ref_text=None)
        _cast_fixture(root, "Long", seconds=13)
        _cast_fixture(root, "NoRef", seconds=None)
        _cast_fixture(root, "OwnF5", f5_ref_audio="f5_reference.wav", f5_ref_text="Short words.")
        write_wav(os.path.join(root, "OwnF5", "f5_reference.wav"), np.zeros(24000), 24000)
        _cast_fixture(root, "OwnF5NoText", f5_ref_audio="f5_reference.wav", f5_ref_text="", csm_ref_text="Fine.")
        write_wav(os.path.join(root, "OwnF5NoText", "f5_reference.wav"), np.zeros(24000), 24000)
        _cast_fixture(root, "OwnF5NoKey", f5_ref_audio="f5_reference.wav", f5_ref_text=None, csm_ref_text="Fine.")
        write_wav(os.path.join(root, "OwnF5NoKey", "f5_reference.wav"), np.zeros(24000), 24000)
        _cast_fixture(root, "NoPrompt", clone_prompt=None)
        names = ("Ana", "Bare", "Long", "NoRef", "OwnF5", "OwnF5NoText", "OwnF5NoKey", "NoPrompt", "Gone")
        checked = refused_cases = accepted_cases = 0
        with patch.object(voice_library, "VOICES_DIR", root), patch.object(tts_generator, "seed_everything"):
            generator = tts_generator.TTSGenerator(Models(), notify=lambda message: None)
            for engine, name, language, direction, text in itertools.product(
                    cr.ENGINES, names, ("english", "chinese", "auto", "french"), ("", "whisper"),
                    ("Hello there.", "[sad] Hello there.")):
                if engine == "qwen" and (direction or "[" in text):
                    continue    # Qwen checks direction and tags only after its model loads: covered by the unit cases
                segment = audiobook.new_segment(text, cr.make_key(engine, name), direction=direction)
                settings = {"language": language, "seed": 1, "temperature": 0.7, "min_words": 1, "max_words": 50}
                try:
                    audiobook.render_segment(generator, segment, settings, cache_dir=cache)
                    refused = False
                except Reached:
                    refused = False
                except Exception:
                    refused = True
                state = cr.check_segment(segment, language, root=root, env=_ALL_READY).state
                assert (state != cr.READY) == refused, (engine, name, language, direction, text, state, refused)
                checked += 1
                refused_cases += refused
                accepted_cases += not refused
    assert refused_cases > 100 and accepted_cases > 50, (refused_cases, accepted_cases)    # the comparison is not vacuous
    print(f"PASS: cast rules agree with the real qwen (files), xtts, csm, dia and f5 validation in all {checked} cases "
          f"({refused_cases} refused, {accepted_cases} accepted; no model loaded)")


def run_tests():
    """Run all unit tests."""
    print("Running unit tests...")
    _test_write_wav()
    _test_write_wav_various_inputs()
    _test_normalize_wavs()
    _test_split_text()
    _test_split_text_empty()
    _test_chunk_text()
    _test_chunk_text_empty()
    _test_chunk_text_single_sentence()
    _test_npz_load_allow_pickle()
    _test_npz_numeric_keys_convert_to_list()
    _test_npz_named_keys_stay_dict()
    _test_npz_empty_files()
    _test_save_load_voice_clone_prompt_preserves_ref_code()
    _test_save_load_voice_clone_prompt_dict()
    _test_save_outputs_single()
    _test_save_outputs_stop_requested()
    _test_audiobook_split()
    _test_audiobook_compile()
    _test_audiobook_cache_and_grouping()
    _test_audiobook_stop_never_caches_or_returns_a_partial_segment()
    _test_audiobook_cache_marker_legacy_entries_and_interrupted_writes()
    _test_xtts_generation()
    _test_csm_generation()
    _test_dia_generation()
    _test_f5_generation_and_worker()
    _test_cast_rules_states_and_reasons()
    _test_cast_rules_reference_audio_agrees_with_load_wav()
    _test_cast_rules_agree_with_the_engines_before_any_model_loads()
    _test_engine_model_lifecycle()
    _test_voice_library_roundtrip()
    _test_voice_replace_is_failure_safe()
    _test_clone_reports_how_many_chunks_it_planned()
    _test_audio_loading_formats_and_capabilities()
    _test_emotion_tags_and_seed()
    _test_benchmark_helpers()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
