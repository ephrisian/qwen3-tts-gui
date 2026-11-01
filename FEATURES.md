# Qwen3-TTS Audiobook Tool

Run with `./manage_qwen.sh run` (or `python launch.py`). Tests: `python launch.py test`.
The GUI is Qt (PySide6); all model logic is UI-independent
(`tts_generator.py`, `audiobook.py`, `voice_library.py`, `voice_lock.py`, `model_loader.py`).

## Choosing a mode for one consistent, directable voice
| Mode | Voice identity | Direction (emotion / tone) |
|------|----------------|----------------------------|
| VoiceDesign | New every call (seed only fixes sampling noise) | `instruct` — used here to *create* voices and emotion references |
| VoiceClone (Base) | Any saved/cloned voice, stable | No instruct input; `[tag]`s are performed via VoiceDesign-generated emotion references anchored to the voice's speaker embedding |

## Tabs
- **Voice Creator (VoiceDesign)** – audition the fixed sample passage, play it back, then name and save the
  voice to `voices/<name>/` (`voice.json`, `reference.wav`, `clone_prompt.npz`).
- **VoiceClone** – pick a saved voice (or clone from reference audio), `[emotion]` tags, seed, temperature.
- **Audiobook** – paste a chapter, split into narration/dialogue segments, assign a voice and direction per
  segment, preview a segment, then render one compiled WAV (pauses, loudness matching, per-segment cache,
  save/load project JSON).

## Notes
- Emphasis on single words is not controllable; use wording and punctuation.
- Models load on first use; at most two stay in memory.
