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
  - "Auto-save generated voices" automatically saves each generated sample using a random name
    (gender selectable: Random / Male / Female). Use "🎲 Random name" to suggest a name manually.
  - "Save voice" saves the current preview under the name you type.
- **VoiceClone** – pick a saved voice (or clone from reference audio), `[emotion]` tags, seed, temperature.
  - "Save prompt to (NPZ)" exports the built clone prompt so it can be reused later via "Reuse prompt (NPZ)".
  - "Save this clone as a voice" builds a clone prompt from reference audio and registers it as a named voice,
    so it appears in the Audiobook's voice dropdowns (Narrator / Dialogue / per-segment Voice).
- **Voices** – browse all saved voices, play their reference clips, rename or delete them, import a voice
  folder (must contain `voice.json` + `reference.wav`), or create a new voice from any audio file
  (you must provide the transcript for that audio).
- **Audiobook** – paste a chapter, split into narration/dialogue segments, assign a voice and direction per
  segment, preview a segment, then render one compiled WAV (pauses, loudness matching, per-segment cache,
  save/load project JSON).

## Voice Manager (Tools → Voice Manager…)
View all saved voices, rename any voice, or delete voices you no longer want. Renaming updates the voice's
`voice.json` and its folder; deleting removes the entire voice directory.

## Voice Description box (VoiceClone)
The description is used **only when `[emotion]` tags are present**. It is combined with each tag's emotion
description to form the VoiceDesign instruct:

    instruct = f"{voice_description}. {emotion_description(tag)}"

If a saved voice is selected, its saved description overrides the box. Without emotion tags, the box has no
effect on the output.

## Emotion / direction tags
Put a `[tag]` before a passage; it applies until the next tag. Presets expand to richer descriptions
(e.g. `[sad]` → "sorrowful, subdued and heavy…"). Any free text is used verbatim, so you can write
`[frustrated]`, `[sad]`, `[angry]`, `[flustered]`, `[quiet]`, `[gritted teeth]`, or even
`[Sad Angry]` — the tag text is passed straight through as the style description.

## Notes
- Emphasis on single words is not controllable; use wording and punctuation.
- Models load on first use; at most two stay in memory.
