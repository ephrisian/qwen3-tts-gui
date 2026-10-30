"""Unit tests for Qwen3-TTS Gradio Studio modules."""

import os
import shutil
import tempfile
import wave
import numpy as np

from text_utils import chunk_text, split_text
from audio_utils import write_wav, normalize_wavs, save_outputs, get_wav_info, list_output_files
from tts_generator import TTSGenerator, generate_synthetic_audio


def test_write_wav():
    """Test that write_wav produces a valid WAV file with correct headers."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        write_wav(path, audio, sr)
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono channel"
            assert wf.getsampwidth() == 2, "Expected 16-bit samples"
            assert wf.getframerate() == sr, f"Expected sample rate {sr}"
            assert wf.getnframes() == int(sr * duration)
        print("PASS: write_wav produces valid WAV file")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_get_wav_info():
    """Test get_wav_info reads duration and sample rate correctly."""
    sr = 24000
    duration = 1.5
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        write_wav(path, audio, sr)
        info = get_wav_info(path)
        assert abs(info["duration"] - 1.5) < 0.1, f"Expected ~1.5s, got {info['duration']}"
        assert info["sample_rate"] == 24000
        print("PASS: get_wav_info correctly extracts WAV metadata")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_list_output_files():
    """Test that list_output_files finds files and sorts them newest first."""
    temp_dir = tempfile.mkdtemp()
    try:
        f1 = os.path.join(temp_dir, "CustomVoice_20261030_0001_aaa.wav")
        f2 = os.path.join(temp_dir, "VoiceDesign_20261030_0002_bbb.wav")

        write_wav(f1, np.zeros(1000), 16000)
        # Touch f2 with slightly newer timestamp
        write_wav(f2, np.zeros(2000), 16000)

        files = list_output_files(temp_dir)
        assert len(files) == 2, f"Expected 2 files, got {len(files)}"
        modes = [f["mode"] for f in files]
        assert "CustomVoice" in modes
        assert "VoiceDesign" in modes
        print("PASS: list_output_files correctly parses and sorts output directory")
    finally:
        shutil.rmtree(temp_dir)


def test_voice_design_seed_fixed_vs_random():
    """Test VoiceDesign seed handling: Fixed seed reproducibility and Random seed variation."""
    temp_dir = tempfile.mkdtemp()
    try:
        gen = TTSGenerator(output_dir=temp_dir)

        # Test 1: Fixed seed produces identical actual_seed and audio
        res_fixed_1 = gen.generate_voice_design(
            text="Hello fixed seed test.",
            instruct="Calm voice",
            seed_mode="Fixed",
            seed_value=12345,
        )
        res_fixed_2 = gen.generate_voice_design(
            text="Hello fixed seed test.",
            instruct="Calm voice",
            seed_mode="Fixed",
            seed_value=12345,
        )

        assert res_fixed_1["actual_seed"] == 12345
        assert res_fixed_2["actual_seed"] == 12345
        assert res_fixed_1["seed_mode"] == "Fixed"

        # Compare audio bytes
        with open(res_fixed_1["output_path"], "rb") as fa, open(res_fixed_2["output_path"], "rb") as fb:
            assert fa.read() == fb.read(), "Fixed seed audio outputs must match identically!"

        print("PASS: VoiceDesign Fixed seed is reproducible")

        # Test 2: Random seed produces varying seeds
        seeds = set()
        for _ in range(5):
            res_rnd = gen.generate_voice_design(
                text="Testing random seed generation.",
                instruct="Energetic voice",
                seed_mode="Random",
            )
            assert res_rnd["seed_mode"] == "Random"
            seeds.add(res_rnd["actual_seed"])

        assert len(seeds) > 1, f"Expected varied random seeds, got {seeds}"
        print("PASS: VoiceDesign Random seed produces varied unique seeds")

    finally:
        shutil.rmtree(temp_dir)


def test_chunk_text():
    """Test text chunking with sentences."""
    text = "Sentence one. Dr. Smith said sentence two! Is this sentence three? Yes, it certainly is."
    chunks = chunk_text(text, min_words=4, max_words=10)
    assert len(chunks) >= 1
    assert any("Dr. Smith" in c for c in chunks)
    print("PASS: chunk_text handles sentence packing without abbreviation breaks")


def run_tests():
    """Run all unit tests."""
    print("========================================")
    print("Running Qwen3-TTS Gradio Studio Unit Tests")
    print("========================================")
    test_write_wav()
    test_get_wav_info()
    test_list_output_files()
    test_voice_design_seed_fixed_vs_random()
    test_chunk_text()
    print("========================================")
    print("ALL TESTS PASSED! ✅")
    print("========================================")


if __name__ == "__main__":
    run_tests()
