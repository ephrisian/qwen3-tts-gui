# Qwen3-TTS GUI

A desktop GUI for the Qwen3-TTS text-to-speech model, supporting VoiceDesign (voice creation), VoiceClone (voice cloning), and audiobook rendering.

## Requirements

- **Conda** (Miniconda or Anaconda) — required to create the isolated Python environment.
- **NVIDIA GPU** with CUDA support recommended (torch is installed from the cu121 wheel).
- **Windows**: use `install.bat` (see below).
- **Linux/macOS**: use `./manage_qwen.sh install` (see below).

## Installation

### Windows (Conda required)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) if you do not already have Conda.
2. Open a terminal (PowerShell or Command Prompt) in this project folder.
3. Run the Windows installer:

   ```bat
   install.bat
   ```

   This creates a Conda environment named `qwen3-tts` (Python 3.10) and installs all dependencies:
   - torch / torchvision / torchaudio (cu121)
   - qwen-tts
   - flash-attn
   - pygame
   - PySide6-Essentials

### Linux / macOS

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) if you do not already have Conda.
2. Open a terminal in this project folder.
3. Run:

   ```bash
   ./manage_qwen.sh install
   ```

   This performs the same steps as the Windows installer using the shell script.

## Usage

### Windows

After installation, activate the environment and launch the GUI:

```bat
conda activate qwen3-tts
python launch.py
```

Or run directly from any shell:

```bat
conda run -n qwen3-tts python launch.py
```

### Linux / macOS

```bash
./manage_qwen.sh run
```

Or:

```bash
conda activate qwen3-tts
python launch.py
```

## Project commands

| Command | Description |
|---------|-------------|
| `install` | Create the Conda environment and install dependencies |
| `run` | Launch the GUI (requires the environment to be installed) |
| `remove` | Remove the Conda environment |
| `cleanup` | Clean pip cache, temp files, and reset install checkpoints |
| `purge` | Remove the Conda environment and all project-generated files |

Windows: `install.bat` handles installation; use `conda activate qwen3-tts && python launch.py` to run.

Linux/macOS: `./manage_qwen.sh {install|run|remove|cleanup|purge}`

## Inference benchmark

List saved voices, then benchmark one voice against the same text and seed across
chunk sizes, dtypes, and attention backends:

```bash
python launch.py benchmark --list-voices
python launch.py benchmark --voice narrator --text-file benchmark.txt --seed 42 \
   --chunk-sizes 0,250,400,600 --dtypes bfloat16,float16 \
   --attentions flash_attention_2,sdpa --warmups 1 --repeats 3
```

The benchmark records model-load time separately from warmed generation time,
reports median real-time factor (lower is faster), and saves a WAV per successful
configuration for listening comparisons. Results are written to a timestamped
folder under `output/` as `results.csv`. Use a UTF-8 text file representative of
your normal workload; the full configuration grid can take a while to run.

## Notes

- Models are downloaded on first use into the project-local `.hf_cache` directory (via `HF_HOME`).
- Generated audio is written to `output/` and saved voices to `voices/`.
- At most two models stay in memory at once.
- The GUI is built with PySide6; all model logic is UI-independent (`tts_generator.py`, `audiobook.py`, `voice_library.py`, `voice_lock.py`, `model_loader.py`).
