"""Unit tests for Qwen3-TTS GUI modules."""

import os
import sys
import tempfile
import wave

import numpy as np

from text_utils import chunk_text, split_text
from audio_utils import write_wav, normalize_wavs, save_outputs


def _test_write_wav():
    """Test that write_wav produces a valid WAV file."""
    sr = 16000
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
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


def _test_split_text():
    """Test comma-separated text splitting."""
    texts = split_text("Hello, World, Test")
    assert texts == ["Hello", "World", "Test"]
    print("PASS: batch text splitting")


def _test_save_outputs_normalization():
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


def run_tests():
    """Run all unit tests."""
    print("Running unit tests...")
    _test_write_wav()
    _test_write_wav_various_inputs()
    _test_split_text()
    _test_save_outputs_normalization()
    _test_chunk_text()
    _test_chunk_text_empty()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
