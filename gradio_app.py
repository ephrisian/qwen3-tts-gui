"""Gradio application for Qwen3-TTS with VoiceDesign Random/Fixed Seed and Output File Manager."""

import os
import random
import time
import gradio as gr

from config import (
    MODEL_MAP, SUPPORTED_SPEAKERS, LANGUAGES, DTYPES, ATTNS,
    DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_ATTN, DEFAULT_MIN_WORDS,
    DEFAULT_MAX_WORDS, DEFAULT_SPEAKER, DEFAULT_LANGUAGE,
    DEFAULT_SEED_MODE, DEFAULT_SEED, OUTPUT_DIR, GRADIO_TITLE,
)
from tts_generator import TTSGenerator
from model_loader import load_model, is_cuda_available, TORCH_AVAILABLE
from audio_utils import list_output_files

# Global TTS generator instance
generator = TTSGenerator(output_dir=OUTPUT_DIR)
current_loaded_model_id = None


def get_file_choices():
    """Format output files for the dropdown selector."""
    files = list_output_files(OUTPUT_DIR)
    if not files:
        return [("No audio files found in output directory", "")]
    choices = []
    for f in files:
        label = f"[{f['mode']}] {f['filename']} ({f['size_str']} • {f['duration']}s • {f['created_str']})"
        choices.append((label, f["path"]))
    return choices


def refresh_output_list():
    """Refresh the file list and return updated dropdown choices and summary."""
    choices = get_file_choices()
    first_path = choices[0][1] if choices and choices[0][1] else None
    files = list_output_files(OUTPUT_DIR)
    summary = f"**{len(files)} file(s)** available in `{OUTPUT_DIR}`"
    return gr.update(choices=choices, value=first_path), first_path, summary


def load_file_for_playback(selected_path):
    """Load the chosen audio file into the player and format file details."""
    if not selected_path or not os.path.exists(selected_path):
        return None, "No file selected or file does not exist."

    files = list_output_files(OUTPUT_DIR)
    matching = [f for f in files if f["path"] == selected_path]
    if matching:
        m = matching[0]
        details = (
            f"### Selected File Info\n"
            f"- **Filename:** `{m['filename']}`\n"
            f"- **Mode:** `{m['mode']}`\n"
            f"- **Duration:** {m['duration']} seconds\n"
            f"- **Size:** {m['size_str']} ({m['size_bytes']:,} bytes)\n"
            f"- **Created:** {m['created_str']}\n"
            f"- **Full Path:** `{m['path']}`"
        )
    else:
        details = f"Loaded file: `{os.path.basename(selected_path)}`"

    return selected_path, details


def delete_selected_output(selected_path):
    """Delete the selected audio file and refresh list."""
    if not selected_path or not os.path.exists(selected_path):
        choices = get_file_choices()
        return gr.update(choices=choices, value=None), None, "No file selected to delete.", ""

    filename = os.path.basename(selected_path)
    try:
        os.remove(selected_path)
        status_msg = f"Deleted `{filename}` successfully."
    except Exception as e:
        status_msg = f"Failed to delete `{filename}`: {e}"

    choices = get_file_choices()
    next_val = choices[0][1] if choices and choices[0][1] else None
    files = list_output_files(OUTPUT_DIR)
    summary = f"**{len(files)} file(s)** available in `{OUTPUT_DIR}`"
    return gr.update(choices=choices, value=next_val), next_val, summary, status_msg


def on_load_model_click(model_id, device, dtype, attn):
    """Handler for loading model weights."""
    global current_loaded_model_id
    if not model_id or not model_id.strip():
        return "⚠️ Error: Please enter a model ID.", "Not Loaded"

    try:
        m = load_model(model_id.strip(), device, dtype, attn)
        generator.set_model(m)
        current_loaded_model_id = model_id.strip()
        return f"Loaded: `{model_id.strip()}` on `{device}` ({dtype})", "Loaded"
    except Exception as e:
        err_msg = str(e)
        return (
            f"⚠️ Model load note: {err_msg}\n\n"
            f"*Note: Running with synthetic demonstration audio generator until GPU weights are loaded.*",
            "Demo Mode",
        )


def on_generate_custom_voice(text, language, speaker, instruct, min_words, max_words):
    """Generate audio for CustomVoice mode."""
    try:
        res = generator.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            min_words=int(min_words),
            max_words=int(max_words),
        )
        msg = f"Generated `{res['filename']}` ({res['chunks']} chunks) using speaker **{speaker}**."
        # Refresh file choices
        choices = get_file_choices()
        files = list_output_files(OUTPUT_DIR)
        summary = f"**{len(files)} file(s)** available in `{OUTPUT_DIR}`"
        return res["output_path"], msg, gr.update(choices=choices, value=res["output_path"]), res["output_path"], summary
    except Exception as e:
        err = f"Generation failed: {e}"
        return None, err, gr.update(), None, ""


def on_generate_voice_design(text, instruct, language, seed_mode, seed_val, min_words, max_words):
    """Generate audio for VoiceDesign mode with Random or Fixed seed."""
    try:
        res = generator.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            seed_mode=seed_mode,
            seed_value=seed_val,
            min_words=int(min_words),
            max_words=int(max_words),
        )
        seed_info = f"Seed used: `{res['actual_seed']}` (Mode: **{res['seed_mode']}**)"
        msg = (
            f"Generated `{res['filename']}` ({res['chunks']} chunks).\n"
            f"- **{seed_info}**\n"
            f"- Voice Prompt: *\"{instruct[:80]}...\"*"
        )
        choices = get_file_choices()
        files = list_output_files(OUTPUT_DIR)
        summary = f"**{len(files)} file(s)** available in `{OUTPUT_DIR}`"
        return res["output_path"], msg, gr.update(choices=choices, value=res["output_path"]), res["output_path"], summary, res["actual_seed"]
    except Exception as e:
        err = f"Generation failed: {e}"
        return None, err, gr.update(), None, "", seed_val


def on_generate_voice_clone(text, ref_audio, ref_text, x_vector_only, reuse_prompt, save_prompt,
                            language, min_words, max_words):
    """Generate audio for VoiceClone mode."""
    try:
        res = generator.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            reuse_prompt=reuse_prompt,
            save_prompt=save_prompt,
            language=language,
            min_words=int(min_words),
            max_words=int(max_words),
        )
        msg = f"Cloned voice saved to `{res['filename']}` ({res['chunks']} chunks)."
        choices = get_file_choices()
        files = list_output_files(OUTPUT_DIR)
        summary = f"**{len(files)} file(s)** available in `{OUTPUT_DIR}`"
        return res["output_path"], msg, gr.update(choices=choices, value=res["output_path"]), res["output_path"], summary
    except Exception as e:
        err = f"Generation failed: {e}"
        return None, err, gr.update(), None, ""


def update_speaker_desc(speaker):
    """Return markdown description for selected speaker."""
    desc = SUPPORTED_SPEAKERS.get(speaker, "")
    return f"**Speaker Profile:** {desc}"


def roll_random_seed():
    """Return a newly generated random integer for the seed field."""
    return random.randint(0, 2**31 - 1)


def create_gradio_ui():
    """Build the complete Gradio interface."""
    default_dev = "cuda" if is_cuda_available() else "cpu"

    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto; }
    .badge-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
    .seed-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    .file-manager-panel { background: #fdfdfd; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
    """

    with gr.Blocks(title=GRADIO_TITLE, theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css) as demo:
        gr.Markdown(
            f"""
            # 🎙️ Qwen3-TTS Gradio Studio
            *High-fidelity Text-to-Speech synthesis with CustomVoice, VoiceDesign (Random & Fixed seed), VoiceClone, and Output File Playback.*
            """
        )

        # Global Model Settings Accordion
        with gr.Accordion("⚙️ Global Settings & Model Configuration", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    inp_model_id = gr.Textbox(
                        value=MODEL_MAP["custom-voice"],
                        label="Model ID / Hugging Face Repository",
                        placeholder="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    )
                with gr.Column(scale=1):
                    inp_device = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value=default_dev,
                        label="Device",
                    )
                with gr.Column(scale=1):
                    inp_dtype = gr.Dropdown(
                        choices=DTYPES,
                        value=DEFAULT_DTYPE,
                        label="Data Type (Dtype)",
                    )
                with gr.Column(scale=1):
                    inp_attn = gr.Dropdown(
                        choices=ATTNS,
                        value=DEFAULT_ATTN,
                        label="Attention Implementation",
                    )

            with gr.Row():
                inp_min_words = gr.Number(value=DEFAULT_MIN_WORDS, label="Min Words Per Chunk", precision=0)
                inp_max_words = gr.Number(value=DEFAULT_MAX_WORDS, label="Max Words Per Chunk", precision=0)
                with gr.Column():
                    btn_load_model = gr.Button("🚀 Load Model Weights", variant="secondary")
                    out_model_status = gr.Markdown("Model status: *Ready (Demo / Standby)*")

        # Main Tabs
        with gr.Tabs() as tabs:
            # TAB 1: Custom Voice
            with gr.TabItem("🎭 CustomVoice", id="tab_cv"):
                with gr.Row():
                    with gr.Column(scale=3):
                        txt_cv_input = gr.Textbox(
                            value="Hello! Welcome to Qwen3-TTS running on Gradio. What text would you like me to speak today?",
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter text to synthesize...",
                        )
                        with gr.Row():
                            drp_cv_lang = gr.Dropdown(choices=LANGUAGES, value=DEFAULT_LANGUAGE, label="Language")
                            drp_cv_speaker = gr.Dropdown(
                                choices=list(SUPPORTED_SPEAKERS.keys()),
                                value=DEFAULT_SPEAKER,
                                label="Speaker Preset",
                            )
                        lbl_cv_desc = gr.Markdown(update_speaker_desc(DEFAULT_SPEAKER))
                        txt_cv_instruct = gr.Textbox(
                            value="",
                            label="Style / Emotion Instruction (Optional)",
                            placeholder="e.g., cheerful, calm, whisper, excited...",
                            lines=2,
                        )
                        with gr.Row():
                            btn_cv_generate = gr.Button("✨ Generate CustomVoice Audio", variant="primary", scale=2)
                            btn_cv_stop = gr.Button("🛑 Stop", variant="stop", scale=1)
                        status_cv = gr.Markdown("")

                    with gr.Column(scale=2):
                        audio_cv_out = gr.Audio(label="CustomVoice Audio Output", type="filepath")

            # TAB 2: Voice Design (with Random and Fixed Seed)
            with gr.TabItem("🎨 VoiceDesign (Voice Creation)", id="tab_vd"):
                gr.Markdown(
                    "Create a custom voice persona purely from a natural language description. "
                    "Use **Random** seed for endless variety or **Fixed** seed to keep the same voice persona consistent."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        txt_vd_input = gr.Textbox(
                            value="Hello, this is a newly designed voice created entirely from text prompt descriptions.",
                            label="Text to Synthesize",
                            lines=4,
                        )
                        with gr.Row():
                            drp_vd_lang = gr.Dropdown(choices=LANGUAGES, value=DEFAULT_LANGUAGE, label="Language")

                        txt_vd_instruct = gr.Textbox(
                            value="Warm, gentle female voice, calm and soothing tone, natural breathing.",
                            label="Voice Persona Description (Instruct - Required)",
                            lines=3,
                        )

                        # --- SEED CONTROL PANEL ---
                        with gr.Group():
                            gr.Markdown("#### 🎲 Voice Persona Seed Controls")
                            with gr.Row():
                                rad_seed_mode = gr.Radio(
                                    choices=["Random", "Fixed"],
                                    value=DEFAULT_SEED_MODE,
                                    label="Seed Mode",
                                    info="Random produces a fresh unique timbre every run. Fixed locks the persona so it sounds identical across generations.",
                                )
                                num_seed_val = gr.Number(
                                    value=DEFAULT_SEED,
                                    precision=0,
                                    label="Fixed Seed Number",
                                    info="Integer seed (0 to 2^31 - 1)",
                                )
                                btn_roll_seed = gr.Button("🎲 Roll Random Seed", scale=1)

                        with gr.Row():
                            btn_vd_generate = gr.Button("🎨 Generate VoiceDesign Audio", variant="primary", scale=2)
                            btn_vd_stop = gr.Button("🛑 Stop", variant="stop", scale=1)

                        status_vd = gr.Markdown("")

                    with gr.Column(scale=2):
                        audio_vd_out = gr.Audio(label="VoiceDesign Audio Output", type="filepath")

            # TAB 3: Voice Clone
            with gr.TabItem("🧬 VoiceClone", id="tab_vc"):
                gr.Markdown("Clone a speaker's timbre using a 3-10 second reference audio sample.")
                with gr.Row():
                    with gr.Column(scale=3):
                        txt_vc_input = gr.Textbox(
                            value="Hello, this synthesized speech is cloning the timbre and acoustic characteristics of your reference audio.",
                            label="Text to Synthesize",
                            lines=4,
                        )
                        with gr.Row():
                            drp_vc_lang = gr.Dropdown(choices=LANGUAGES, value=DEFAULT_LANGUAGE, label="Language")

                        audio_vc_ref = gr.Audio(label="Reference Audio Sample (Upload WAV/MP3)", type="filepath")
                        txt_vc_ref_text = gr.Textbox(
                            value="",
                            label="Reference Transcript (Optional if X-vector only is checked)",
                            placeholder="Transcript of the reference audio sample...",
                            lines=2,
                        )
                        chk_vc_xvector = gr.Checkbox(
                            value=False,
                            label="X-vector only mode (Speaker embedding only; reference transcript not needed)",
                        )
                        with gr.Row():
                            txt_vc_reuse_prompt = gr.Textbox(
                                value="",
                                label="Reuse Voice Clone Prompt (NPZ path)",
                                placeholder="path/to/prompt.npz",
                            )
                            txt_vc_save_prompt = gr.Textbox(
                                value="",
                                label="Save Generated Prompt To (NPZ path)",
                                placeholder="path/to/save.npz",
                            )

                        with gr.Row():
                            btn_vc_generate = gr.Button("🧬 Clone & Synthesize Audio", variant="primary", scale=2)
                            btn_vc_stop = gr.Button("🛑 Stop", variant="stop", scale=1)

                        status_vc = gr.Markdown("")

                    with gr.Column(scale=2):
                        audio_vc_out = gr.Audio(label="VoiceClone Audio Output", type="filepath")

            # TAB 4: Output File Manager & Playback
            with gr.TabItem("📁 Output Files & Playback", id="tab_files"):
                gr.Markdown(
                    "### 📂 Generated Audio Outputs\n"
                    "All generated audio files are automatically organized and listed below. "
                    "Select any file to inspect its metadata and listen to playback."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        lbl_output_summary = gr.Markdown(
                            f"**{len(list_output_files(OUTPUT_DIR))} file(s)** available in `{OUTPUT_DIR}`"
                        )
                        drp_output_files = gr.Dropdown(
                            choices=get_file_choices(),
                            value=get_file_choices()[0][1] if get_file_choices() and get_file_choices()[0][1] else None,
                            label="Select File from Output Folder",
                            interactive=True,
                        )
                        with gr.Row():
                            btn_refresh_files = gr.Button("🔄 Refresh File List", variant="secondary")
                            btn_delete_file = gr.Button("🗑️ Delete Selected File", variant="stop")

                        status_file_action = gr.Markdown("")
                        txt_file_details = gr.Markdown("Select a file above to inspect details.")

                    with gr.Column(scale=2):
                        audio_playback = gr.Audio(
                            label="🎧 Audio Playback",
                            type="filepath",
                            interactive=False,
                        )

        # --- EVENT WIRING ---
        drp_cv_speaker.change(fn=update_speaker_desc, inputs=[drp_cv_speaker], outputs=[lbl_cv_desc])
        btn_roll_seed.click(fn=roll_random_seed, outputs=[num_seed_val])

        # Model Loading
        btn_load_model.click(
            fn=on_load_model_click,
            inputs=[inp_model_id, inp_device, inp_dtype, inp_attn],
            outputs=[out_model_status],
        )

        # Stop handlers
        btn_cv_stop.click(fn=generator.request_stop)
        btn_vd_stop.click(fn=generator.request_stop)
        btn_vc_stop.click(fn=generator.request_stop)

        # Custom Voice Generation
        btn_cv_generate.click(
            fn=on_generate_custom_voice,
            inputs=[txt_cv_input, drp_cv_lang, drp_cv_speaker, txt_cv_instruct, inp_min_words, inp_max_words],
            outputs=[audio_cv_out, status_cv, drp_output_files, audio_playback, lbl_output_summary],
        )

        # Voice Design Generation (Passes seed_mode and seed_val!)
        btn_vd_generate.click(
            fn=on_generate_voice_design,
            inputs=[txt_vd_input, txt_vd_instruct, drp_vd_lang, rad_seed_mode, num_seed_val, inp_min_words, inp_max_words],
            outputs=[audio_vd_out, status_vd, drp_output_files, audio_playback, lbl_output_summary, num_seed_val],
        )

        # Voice Clone Generation
        btn_vc_generate.click(
            fn=on_generate_voice_clone,
            inputs=[
                txt_vc_input, audio_vc_ref, txt_vc_ref_text, chk_vc_xvector,
                txt_vc_reuse_prompt, txt_vc_save_prompt, drp_vc_lang, inp_min_words, inp_max_words,
            ],
            outputs=[audio_vc_out, status_vc, drp_output_files, audio_playback, lbl_output_summary],
        )

        # Output File Selection & Playback
        drp_output_files.change(
            fn=load_file_for_playback,
            inputs=[drp_output_files],
            outputs=[audio_playback, txt_file_details],
        )

        btn_refresh_files.click(
            fn=refresh_output_list,
            outputs=[drp_output_files, audio_playback, lbl_output_summary],
        )

        btn_delete_file.click(
            fn=delete_selected_output,
            inputs=[drp_output_files],
            outputs=[drp_output_files, audio_playback, lbl_output_summary, status_file_action],
        )

    return demo


# Create demo instance
app = create_gradio_ui()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Qwen3-TTS Gradio Studio")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print(f"Starting Qwen3-TTS Gradio Studio on {args.host}:{args.port}...")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
