# 🎙️ Qwen3-TTS Gradio Studio

This project migrates the original Tkinter desktop application to a modern **Gradio Web Application**, adds **output audio file listing and playback**, and introduces **Random & Fixed seed controls** for `VoiceDesign` voice persona generation.

---

## 🌟 Key Features Added

1. **Gradio Web Interface (`gradio_app.py` & `app.py`)**:
   - Replaced Tkinter desktop GUI with a modern Gradio web interface.
   - Accessible from any web browser on desktop, mobile, or remote servers.
   - Native HTML5 audio playback replacing the dependency on `pygame`.

2. **Output Files Listing & Playback Manager**:
   - Automatically scans and lists all generated `.wav` files in the `output/` directory.
   - Displays file metadata: file name, generation mode (CustomVoice, VoiceDesign, VoiceClone), creation date/time, duration, and file size.
   - Built-in interactive audio player for immediate playback of any output file.
   - One-click file refresh, download, and delete.

3. **VoiceDesign Seed Controls (Random & Fixed Seeds)**:
   - **Random Mode**: Selects a fresh random integer seed for every generation to yield diverse voice timbres. Logs the seed so you can keep it if you find a voice you like.
   - **Fixed Mode**: Locks the seed to a specific integer across generations to guarantee consistent voice persona timbre and prosody.
   - Quick **🎲 Roll Random Seed** button.

4. **Presets and Audio Generation**:
   - **CustomVoice**: 9 pre-configured speaker personas (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee).
   - **VoiceDesign**: Natural language voice description prompt.
   - **VoiceClone**: Zero-shot reference audio speaker cloning with prompt save and reuse.
   - Sentence-aware text chunking that prevents splits on abbreviations (e.g. `Dr.`, `U.S.`, `Jan.`).

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Gradio App
```bash
python launch.py
```
Or specify a custom port:
```bash
python launch.py --port 7860
```
Then open your browser at `http://localhost:7860`.

### 3. Run Unit Tests
```bash
python launch.py test
```

---

## 📂 Project Structure

- `gradio_app.py` - Complete Gradio Blocks application interface
- `app.py` - Standard Gradio launcher
- `tts_generator.py` - Synthesis logic with seed handling (Random & Fixed)
- `audio_utils.py` - WAV generation, output file listing, and metadata parsing
- `text_utils.py` - Sentence-aware chunking without breaking abbreviations
- `config.py` - Speaker presets, model maps, languages, seed defaults
- `model_loader.py` - PyTorch / Qwen3-TTS model loader
- `launch.py` - CLI launcher for Gradio application and unit tests
- `tests.py` - Comprehensive unit test suite
- `requirements.txt` - Python package requirements
- `server.ts` & `src/App.tsx` - Express and React Gradio Studio web app
