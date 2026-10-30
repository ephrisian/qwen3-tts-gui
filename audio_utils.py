"""Audio WAV writing, output saving, and output file management utilities for Qwen3-TTS."""

import os
import struct
import time
from datetime import datetime
import numpy as np


def write_wav(path, audio, sr):
    """Write a numpy audio array to WAV using only the standard library.

    Ensures mono output (takes first channel if multi-channel),
    clips audio to [-1.0, 1.0], and converts to 16-bit PCM.
    """
    audio = np.asarray(audio)
    if audio.ndim == 0:
        audio = audio.reshape(1)

    # Ensure mono, 16-bit PCM
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    n_samples = len(pcm)
    byte_rate = sr * 2  # 1 channel * 2 bytes/sample
    data_size = n_samples * 2

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", 1))  # Mono
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm.tobytes())


def normalize_wavs(wavs):
    """Normalize various wavs input types into a list of 1-d numpy arrays.

    Handles: numpy scalar, 0-d array, 1-d array, list/tuple of arrays.
    """
    if isinstance(wavs, np.ndarray):
        if wavs.ndim == 0:
            return [wavs.reshape(1)]
        else:
            return [wavs]
    elif isinstance(wavs, (list, tuple)):
        normalized = []
        for w in wavs:
            if isinstance(w, np.ndarray) and w.ndim == 0:
                normalized.append(w.reshape(1))
            elif isinstance(w, np.generic):
                normalized.append(np.asarray(w).reshape(1))
            else:
                normalized.append(w)
        return normalized
    elif isinstance(wavs, np.generic):
        return [np.asarray(wavs).reshape(1)]
    else:
        return [wavs]


def get_wav_info(file_path):
    """Read basic WAV metadata (duration, sample rate, channels) without external dependencies."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 44:
        return {"duration": 0.0, "sample_rate": 0, "channels": 1}

    try:
        with open(file_path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return {"duration": 0.0, "sample_rate": 0, "channels": 1}
            f.seek(22)
            channels = struct.unpack("<H", f.read(2))[0]
            sample_rate = struct.unpack("<I", f.read(4))[0]
            f.seek(34)
            bits_per_sample = struct.unpack("<H", f.read(2))[0]
            
            # Find data chunk
            f.seek(36)
            subchunk2_id = f.read(4)
            while subchunk2_id and subchunk2_id != b"data":
                chunk_sz = struct.unpack("<I", f.read(4))[0]
                f.seek(chunk_sz, 1)
                subchunk2_id = f.read(4)

            if subchunk2_id == b"data":
                data_size = struct.unpack("<I", f.read(4))[0]
                bytes_per_sample = (bits_per_sample / 8) * channels
                if bytes_per_sample > 0 and sample_rate > 0:
                    duration = data_size / (sample_rate * bytes_per_sample)
                    return {
                        "duration": round(duration, 2),
                        "sample_rate": sample_rate,
                        "channels": channels,
                    }
    except Exception:
        pass

    # Fallback estimation based on 16-bit 24kHz mono
    size = os.path.getsize(file_path)
    est_dur = max(0.0, (size - 44) / (24000 * 2))
    return {"duration": round(est_dur, 2), "sample_rate": 24000, "channels": 1}


def format_file_size(size_bytes):
    """Format bytes into human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def list_output_files(output_dir):
    """List all audio files in the output directory with metadata, sorted newest first.

    Returns:
        List of dicts: [
            {
                "filename": "VoiceDesign_20261030_161500_abc.wav",
                "path": "/path/to/file.wav",
                "mode": "VoiceDesign",
                "size_str": "1.24 MB",
                "size_bytes": 1300000,
                "created_str": "2026-10-30 16:15:00",
                "created_ts": 1730333700.0,
                "duration": 3.45,
            }, ...
        ]
    """
    if not os.path.exists(output_dir):
        return []

    results = []
    valid_exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    try:
        filenames = os.listdir(output_dir)
    except OSError:
        return []

    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in valid_exts:
            continue

        full_path = os.path.join(output_dir, fn)
        try:
            stat = os.stat(full_path)
            size_bytes = stat.st_size
            mtime = stat.st_mtime
            created_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Determine mode from prefix
            if fn.startswith("CustomVoice"):
                mode = "CustomVoice"
            elif fn.startswith("VoiceDesign"):
                mode = "VoiceDesign"
            elif fn.startswith("VoiceClone"):
                mode = "VoiceClone"
            else:
                mode = "Other"

            info = get_wav_info(full_path)

            results.append({
                "filename": fn,
                "path": full_path,
                "mode": mode,
                "size_str": format_file_size(size_bytes),
                "size_bytes": size_bytes,
                "created_str": created_str,
                "created_ts": mtime,
                "duration": info["duration"],
                "sample_rate": info["sample_rate"],
            })
        except OSError:
            continue

    # Sort descending by creation timestamp (newest first)
    results.sort(key=lambda x: x["created_ts"], reverse=True)
    return results


def save_outputs(wavs, sr, output_path, stop_requested=False, notification_callback=None,
                 generation_saved_callback=None):
    """Save generated audio outputs to WAV file(s)."""
    if stop_requested:
        return output_path

    wavs = normalize_wavs(wavs)

    if len(wavs) == 1:
        w = wavs[0]
        write_wav(output_path, w, sr)
        if notification_callback:
            notification_callback(f"Audio saved to: {output_path}")
        if generation_saved_callback:
            generation_saved_callback(output_path)
    else:
        # Combine all batch outputs into a single WAV file
        combined = []
        temp_files = []
        base, ext = os.path.splitext(output_path)
        for i, w in enumerate(wavs):
            if stop_requested:
                break
            w_arr = w
            if isinstance(w_arr, np.ndarray) and w_arr.ndim > 1:
                w_arr = w_arr[:, 0]
            combined.append(w_arr)
            out = f"{base}_{i}{ext}"
            write_wav(out, w_arr, sr)
            temp_files.append(out)

        if combined and not stop_requested:
            full_audio = np.concatenate(combined)
            write_wav(output_path, full_audio, sr)

            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

            if notification_callback:
                notification_callback(
                    f"Batch complete. All outputs combined into: {output_path}"
                )
        else:
            if notification_callback:
                notification_callback(f"Audio saved to: {output_path}")

    return output_path
