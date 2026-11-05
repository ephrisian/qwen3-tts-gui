"""Benchmark Qwen3-TTS inference configurations with a fixed voice and input."""

import argparse
import csv
import gc
import hashlib
import itertools
import os
import statistics
import time
from datetime import datetime

import numpy as np

from audio_utils import write_wav
from config import ATTNS, DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DTYPES, OUTPUT_DIR
from tts_generator import TTSGenerator
import voice_library


def chunk_bounds(chunk_size):
    """Return generator min/max word bounds; zero means one whole-text call."""
    if chunk_size == 0:
        return 0, 0
    return max(1, int(chunk_size * 0.8)), chunk_size


def real_time_factor(elapsed_seconds, audio_seconds):
    return elapsed_seconds / audio_seconds if audio_seconds > 0 else None


def _csv_values(value, allowed, label):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated values from: {', '.join(allowed)}"
        )
    return values


def _chunk_sizes(value):
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk sizes must be comma-separated integers") from exc
    if not sizes or any(size < 0 for size in sizes):
        raise argparse.ArgumentTypeError("chunk sizes must be non-negative integers")
    return sizes


def _sync(device):
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _parser():
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-TTS speed for one saved voice, text, and seed."
    )
    parser.add_argument("--list-voices", action="store_true",
                        help="List saved voice names and exit")
    parser.add_argument("--voice", help="Saved voice name from the Voices tab")
    parser.add_argument("--text-file", help="UTF-8 file containing benchmark text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="english")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--devices",
                        help="Comma-separated: cuda,cpu (default: detected device)")
    parser.add_argument("--dtypes",
                        help=f"Comma-separated: {','.join(DTYPES)}")
    parser.add_argument("--attentions",
                        help=f"Comma-separated: {','.join(ATTNS)}")
    parser.add_argument("--chunk-sizes", type=_chunk_sizes, default=[250, 400, 600],
                        help="Max words per chunk; 0 sends the whole text in one call")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", help="Benchmark output directory (default: output/benchmark_*)")
    parser.add_argument("--no-save-audio", action="store_true",
                        help="Do not save one WAV per successful configuration")
    return parser


def _sync_generated(wavs, sample_rate):
    audio = np.concatenate([np.asarray(wav, dtype=np.float32) for wav in wavs])
    duration = len(audio) / sample_rate if sample_rate else 0.0
    return audio, duration


def _run_configuration(config, args, text, output_dir):
    import torch

    from model_loader import ModelManager

    device, dtype, attention, chunk_size = config
    min_words, max_words = chunk_bounds(chunk_size)
    settings = (device, dtype, attention)
    manager = ModelManager(lambda: settings, max_loaded=1)
    result = {
        "device": device,
        "dtype": dtype,
        "attention": attention,
        "chunk_size": chunk_size,
        "min_words": min_words,
        "max_words": max_words,
        "model_load_seconds": "",
        "warmup_seconds": "",
        "median_generation_seconds": "",
        "median_rtf": "",
        "audio_seconds": "",
        "chunks": "",
        "status": "error",
        "error": "",
        "wav_file": "",
    }
    try:
        load_start = time.perf_counter()
        manager.get("voice-clone")
        _sync(device)
        result["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
        generator = TTSGenerator(manager)

        def generate():
            return generator.clone(
                text=text,
                language=args.language,
                min_words=min_words,
                max_words=max_words,
                seed=args.seed,
                temperature=args.temperature,
                voice_name=args.voice,
            )

        warmup_times = []
        for _ in range(args.warmups):
            _sync(device)
            start = time.perf_counter()
            warmup_wavs, _ = generate()
            _sync(device)
            warmup_times.append(time.perf_counter() - start)
            if not warmup_wavs:
                raise RuntimeError("Generation returned no audio during warm-up")
        if warmup_times:
            result["warmup_seconds"] = round(statistics.median(warmup_times), 3)

        generation_times = []
        final_audio = None
        final_sample_rate = None
        final_chunks = 0
        audio_duration = 0.0
        for _ in range(args.repeats):
            _sync(device)
            start = time.perf_counter()
            wavs, sample_rate = generate()
            _sync(device)
            generation_times.append(time.perf_counter() - start)
            if not wavs:
                raise RuntimeError("Generation returned no audio")
            final_audio, audio_duration = _sync_generated(wavs, sample_rate)
            final_sample_rate = sample_rate
            final_chunks = len(wavs)

        median_generation = statistics.median(generation_times)
        result.update({
            "median_generation_seconds": round(median_generation, 3),
            "median_rtf": round(real_time_factor(median_generation, audio_duration), 4),
            "audio_seconds": round(audio_duration, 3),
            "chunks": final_chunks,
            "status": "ok",
        })
        if not args.no_save_audio:
            filename = f"{device}_{dtype}_{attention}_chunk{chunk_size}.wav"
            path = os.path.join(output_dir, filename)
            write_wav(path, final_audio, final_sample_rate)
            result["wav_file"] = filename
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_voices:
        voices = voice_library.list_voices()
        if voices:
            print("\n".join(voices))
        else:
            print("No saved voices found.")
        return 0

    if not args.voice or not args.text_file:
        parser.error("--voice and --text-file are required unless --list-voices is used")

    import torch

    cuda_available = torch.cuda.is_available()
    args.devices = _csv_values(args.devices or (DEFAULT_DEVICE if cuda_available else "cpu"),
                               ["cuda", "cpu"], "devices")
    args.dtypes = _csv_values(args.dtypes or (DEFAULT_DTYPE if cuda_available else "float32"),
                              DTYPES, "dtypes")
    args.attentions = _csv_values(args.attentions or (DEFAULT_ATTN if cuda_available else "sdpa"),
                                  ATTNS, "attentions")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be at least one")
    if "cuda" in args.devices and not cuda_available:
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")
    if not voice_library.voice_exists(args.voice):
        parser.error(f"Saved voice not found: {args.voice!r}")
    if not os.path.isfile(args.text_file):
        parser.error(f"Text file not found: {args.text_file}")
    with open(args.text_file, encoding="utf-8") as text_file:
        text = text_file.read().strip()
    if not text:
        parser.error("Benchmark text file is empty")

    configs = list(itertools.product(args.devices, args.dtypes, args.attentions, args.chunk_sizes))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, f"benchmark_{stamp}")
    os.makedirs(output_dir, exist_ok=True)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"Voice: {args.voice} | Seed: {args.seed} | Text SHA-256: {text_hash[:12]}")
    print(f"Configurations: {len(configs)} | Warmups: {args.warmups} | Repeats: {args.repeats}")

    results = []
    for index, config in enumerate(configs, 1):
        device, dtype, attention, chunk_size = config
        print(f"[{index}/{len(configs)}] {device} / {dtype} / {attention} / chunk {chunk_size}")
        result = _run_configuration(config, args, text, output_dir)
        result.update({
            "seed": args.seed,
            "language": args.language,
            "temperature": args.temperature,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "text_sha256": text_hash,
        })
        results.append(result)
        if result["status"] == "ok":
            print(f"  {result['median_generation_seconds']} s, RTF {result['median_rtf']}")
        else:
            print(f"  FAILED: {result['error']}")

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    successful = [item for item in results if item["status"] == "ok"]
    successful.sort(key=lambda item: item["median_rtf"])
    if successful:
        print("\nFastest successful configurations (lower RTF is faster):")
        for result in successful[:5]:
            print(
                f"  {result['device']} / {result['dtype']} / {result['attention']} / "
                f"chunk {result['chunk_size']}: RTF {result['median_rtf']}"
            )
    print(f"Results: {csv_path}")
    if not args.no_save_audio:
        print(f"Audio samples: {output_dir}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())