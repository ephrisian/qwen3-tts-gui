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
   On Linux, double-click `Start Qwen3-TTS.desktop` in the project folder to launch
   the GUI without entering a command. The launcher opens a terminal window so startup
   messages and errors remain visible. If your file manager asks, choose “Allow
   Launching” or “Trust and Launch”.
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

List saved voices, then compare one voice, text, and seed across every backend
combination for the selected device, dtype, and attention options. The default
keeps chunk size fixed at 250 words so backend results are directly comparable:

```bash
python launch.py benchmark --list-voices
python launch.py benchmark --voice narrator --text-file test-text.md --seed 42 \
   --dtypes bfloat16,float16,float32 \
   --attentions flash_attention_2,sdpa,eager
python launch.py benchmark --voice narrator --text-file test-text.md --seed 42 \
   --matrix chunking --chunk-sizes 0,100,250,400
```

Backend mode defaults to the first 50 words, no warm-up passes, and one measured
pass per configuration to keep the nine-way sweep quick. Increase `--warmups` and
`--repeats` for more stable timing, or use `--max-input-words 0` to benchmark the
entire input. Chunking mode uses the full file by default. Use `--matrix chunking`
to compare chunk sizes using just the baseline backend; `--matrix full` tests all
backend and chunk-size combinations. Use `--devices`, `--dtypes`, and
`--attentions` to select the options to sweep; `--matrix backends` (the default)
tries every combination of those lists at the same chunk size. Unsupported
combinations are recorded as failures. The benchmark randomizes configuration
order, records model-load time separately, and reports median RTF, run-to-run
spread, and change versus baseline. Results and successful WAVs go to a timestamped
folder under `output/`. Since the default backend sweep measures each configuration
once, treat it as a screening run; use repeats for close comparisons.

## Notes

- Models are downloaded on first use into the project-local `.hf_cache` directory (via `HF_HOME`).
- Generated audio is written to `output/` and saved voices to `voices/`.
- At most two models stay in memory at once.
- The GUI is built with PySide6; all model logic is UI-independent (`tts_generator.py`, `audiobook.py`, `voice_library.py`, `voice_lock.py`, `model_loader.py`).
