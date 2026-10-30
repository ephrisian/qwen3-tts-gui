"""Audio WAV writing and output saving utilities for Qwen3-TTS."""

import os
import struct

import numpy as np


def write_wav(path, audio, sr):
    """Write a numpy audio array to WAV using only the standard library.

    Ensures mono output (takes first channel if multi-channel),
    clips audio to [-1.0, 1.0], and converts to 16-bit PCM.
    """
    # Ensure audio is at least 1-d numpy array
    audio = np.asarray(audio)
    if audio.ndim == 0:
        audio = audio.reshape(1)

    # Ensure mono, 16-bit PCM
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if multi-channel
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    n_samples = len(pcm)
    byte_rate = sr * 2  # 1 channel * 2 bytes/sample
    data_size = n_samples * 2

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # file size - 8
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # audio format (PCM)
        f.write(struct.pack("<H", 1))  # num channels
        f.write(struct.pack("<I", sr))  # sample rate
        f.write(struct.pack("<I", byte_rate))  # byte rate
        f.write(struct.pack("<H", 2))  # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm.tobytes())


def normalize_wavs(wavs):
    """Normalize various wavs input types into a list of 1-d numpy arrays.

    Handles: numpy scalar, 0-d array, 1-d array, list/tuple of arrays.
    """
    if isinstance(wavs, np.ndarray):
        # Single array (could be 0-d scalar or 1-d audio)
        if wavs.ndim == 0:
            return [wavs.reshape(1)]
        else:
            return [wavs]
    elif isinstance(wavs, (list, tuple)):
        # Already a list/tuple - ensure each element is at least 1-d
        normalized = []
        for w in wavs:
            if isinstance(w, np.ndarray) and w.ndim == 0:
                normalized.append(w.reshape(1))
            elif isinstance(w, np.generic):
                # numpy scalar (e.g. np.float32(0.5))
                normalized.append(np.asarray(w).reshape(1))
            else:
                normalized.append(w)
        return normalized
    elif isinstance(wavs, np.generic):
        # numpy scalar (e.g. np.float32(0.5))
        return [np.asarray(wavs).reshape(1)]
    else:
        # Scalar or other type - wrap in list
        return [wavs]


def save_outputs(wavs, sr, output_path, stop_requested=False, notification_callback=None,
                 generation_saved_callback=None):
    """Save generated audio outputs to WAV file(s).

    Args:
        wavs: List of audio arrays or single array.
        sr: Sample rate.
        output_path: Path to save the output WAV file.
        stop_requested: If True, skip saving.
        notification_callback: Callable to set notification message.
        generation_saved_callback: Callable to invoke when a single output is saved.

    Returns:
        The output path that was written.
    """
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
            # Ensure mono, 16-bit PCM compatible
            if isinstance(w_arr, np.ndarray) and w_arr.ndim > 1:
                w_arr = w_arr[:, 0]
            combined.append(w_arr)
            out = f"{base}_{i}{ext}"
            write_wav(out, w_arr, sr)
            temp_files.append(out)

        if combined and not stop_requested:
            # Concatenate all audio arrays into one
            full_audio = np.concatenate(combined)
            write_wav(output_path, full_audio, sr)

            # Clean up individual batch files, keep only the combined file
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
