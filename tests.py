"""Unit tests for Qwen3-TTS GUI modules."""

import os
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


def _test_audiobook_cache_and_grouping():
    import audiobook
    calls = []

    class Gen:
        def clone(self, **kw):
            calls.append("clone")
            return [np.zeros(10, dtype=np.float32)], 24000

    cache = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = dict(language="english", min_words=0, max_words=0, seed=1, temperature=0.7)
        seg = audiobook.new_segment("Hi.", "voice:Ryan")
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        audiobook.render_segment(Gen(), seg, settings, cache_dir=cache)
        assert calls == ["clone"], "second render should hit the cache"
        audiobook.render_segment(Gen(), dict(seg, text="Changed."), settings, cache_dir=cache)
        assert calls == ["clone", "clone"]
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    print("PASS: audiobook segment cache")


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
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("PASS: voice library round trip")


def _test_emotion_tags_and_seed():
    import torch
    from voice_lock import parse_emotion_segments, parse_seed, seed_everything
    assert parse_emotion_segments("a [angry] b [whisper] c [1] d") == [
        (None, "a"), ("angry", "b"), ("whisper", "c [1] d")]
    assert parse_seed("Random", "5") is None and parse_seed("Fixed", "5") == 5
    seed_everything(3); x = torch.rand(2)
    seed_everything(3); assert torch.equal(x, torch.rand(2))
    print("PASS: emotion tags and seeding")


# ------------------------------------------------------------------ runner

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
    _test_voice_library_roundtrip()
    _test_emotion_tags_and_seed()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
