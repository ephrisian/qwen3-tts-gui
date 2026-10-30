"""UI building functions for Qwen3-TTS GUI."""

import tkinter as tk
from tkinter import ttk, filedialog

from config import (
    LANGUAGES, DTYPES, ATTNS, MODEL_MAP, SUPPORTED_SPEAKERS,
    DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_ATTN, DEFAULT_MIN_WORDS,
    DEFAULT_MAX_WORDS, DEFAULT_SPEAKER, DEFAULT_LANGUAGE,
    WAV_FILETYPES, AUDIO_FILETYPES, NPZ_FILETYPES,
    TAB_MODE_MAP, MODE_TAB_MAP,
)


def build_common_controls(app, root):
    """Build the Global Settings panel shared across tabs.

    Args:
        app: The QwenTTSGUI instance.
        root: The tkinter root window.
    """
    frame = ttk.LabelFrame(root, text="Global Settings", padding=10)
    frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    # Model ID
    ttk.Label(frame, text="Model ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_model = tk.StringVar(value=MODEL_MAP["custom-voice"])
    app.entry_model = ttk.Entry(frame, textvariable=app.var_model, width=60)
    app.entry_model.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    # Device
    ttk.Label(frame, text="Device:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_device = tk.StringVar(value=DEFAULT_DEVICE if torch_cuda_available() else "cpu")
    ttk.Combobox(frame, textvariable=app.var_device, values=["cuda", "cpu"], width=10, state="readonly").grid(
        row=1, column=1, sticky=tk.W, padx=5, pady=2
    )

    # Dtype
    ttk.Label(frame, text="Dtype:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_dtype = tk.StringVar(value=DEFAULT_DTYPE)
    ttk.Combobox(frame, textvariable=app.var_dtype, values=DTYPES, width=10, state="readonly").grid(
        row=2, column=1, sticky=tk.W, padx=5, pady=2
    )

    # Attention
    ttk.Label(frame, text="Attention:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_attn = tk.StringVar(value=DEFAULT_ATTN)
    ttk.Combobox(frame, textvariable=app.var_attn, values=ATTNS, width=18, state="readonly").grid(
        row=3, column=1, sticky=tk.W, padx=5, pady=2
    )

    # Min / Max words per chunk
    ttk.Label(frame, text="Min words/chunk:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_min_words = tk.StringVar(value=DEFAULT_MIN_WORDS)
    ttk.Entry(frame, textvariable=app.var_min_words, width=10).grid(
        row=4, column=1, sticky=tk.W, padx=5, pady=2
    )
    ttk.Label(frame, text="Max words/chunk:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_max_words = tk.StringVar(value=DEFAULT_MAX_WORDS)
    ttk.Entry(frame, textvariable=app.var_max_words, width=10).grid(
        row=5, column=1, sticky=tk.W, padx=5, pady=2
    )

    # Auto output path (read-only, based on tab + timestamp)
    ttk.Label(frame, text="Output Path:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
    app.var_output = tk.StringVar(value=app._get_auto_output_path())
    app.entry_output = ttk.Entry(frame, textvariable=app.var_output, width=50, state="readonly")
    app.entry_output.grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)

    # Load model button
    ttk.Button(frame, text="Load Model", command=app._load_model).grid(
        row=7, column=0, columnspan=3, pady=10
    )

    app.lbl_model_status = ttk.Label(frame, text="Model not loaded", foreground="gray")
    app.lbl_model_status.grid(row=8, column=0, columnspan=3, pady=2)


def build_custom_voice_tab(app):
    """Build the CustomVoice tab.

    Args:
        app: The QwenTTSGUI instance.
    """
    frame = app.tab_cv
    pad = {"padx": 10, "pady": 5}

    # Text input
    ttk.Label(frame, text="Text (comma-separated for batch):").pack(anchor=tk.W, **pad)
    app.txt_cv_text = tk.Text(frame, height=6, wrap=tk.WORD)
    app.txt_cv_text.pack(fill=tk.BOTH, expand=True, **pad)
    app.txt_cv_text.insert("1.0", "Hello, this is Qwen3-TTS speaking.")

    # Language
    row = ttk.Frame(frame)
    row.pack(fill=tk.X, **pad)
    ttk.Label(row, text="Language:").pack(side=tk.LEFT, padx=5)
    app.var_cv_lang = tk.StringVar(value=DEFAULT_LANGUAGE)
    ttk.Combobox(row, textvariable=app.var_cv_lang, values=LANGUAGES, width=15, state="readonly").pack(
        side=tk.LEFT, padx=5
    )

    # Speaker
    row = ttk.Frame(frame)
    row.pack(fill=tk.X, **pad)
    ttk.Label(row, text="Speaker:").pack(side=tk.LEFT, padx=5)
    app.var_cv_speaker = tk.StringVar(value=DEFAULT_SPEAKER)
    app.cmb_cv_speaker = ttk.Combobox(
        row, textvariable=app.var_cv_speaker,
        values=list(SUPPORTED_SPEAKERS.keys()), width=20, state="readonly",
    )
    app.cmb_cv_speaker.pack(side=tk.LEFT, padx=5)
    app.lbl_cv_speaker_desc = ttk.Label(row, text=SUPPORTED_SPEAKERS[DEFAULT_SPEAKER], foreground="gray")
    app.lbl_cv_speaker_desc.pack(side=tk.LEFT, padx=10)
    app.cmb_cv_speaker.bind("<<ComboboxSelected>>", app._on_speaker_change)

    # Instruct
    ttk.Label(frame, text="Instruct (style/emotion, optional):").pack(anchor=tk.W, **pad)
    app.txt_cv_instruct = tk.Text(frame, height=3, wrap=tk.WORD)
    app.txt_cv_instruct.pack(fill=tk.X, **pad)

    # Buttons
    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, **pad)
    ttk.Button(btn_frame, text="Generate", command=lambda: app._start_generation("custom-voice")).pack(
        side=tk.LEFT, padx=5, pady=5
    )
    ttk.Button(btn_frame, text="List Speakers", command=app._list_speakers).pack(side=tk.LEFT, padx=5, pady=5)


def build_voice_design_tab(app):
    """Build the VoiceDesign tab.

    Args:
        app: The QwenTTSGUI instance.
    """
    frame = app.tab_vd
    pad = {"padx": 10, "pady": 5}

    ttk.Label(frame, text="Text (comma-separated for batch):").pack(anchor=tk.W, **pad)
    app.txt_vd_text = tk.Text(frame, height=6, wrap=tk.WORD)
    app.txt_vd_text.pack(fill=tk.BOTH, expand=True, **pad)
    app.txt_vd_text.insert("1.0", "Hello, this is a designed voice.")

    ttk.Label(frame, text="Language:").pack(anchor=tk.W, **pad)
    app.var_vd_lang = tk.StringVar(value=DEFAULT_LANGUAGE)
    ttk.Combobox(frame, textvariable=app.var_vd_lang, values=LANGUAGES, width=15, state="readonly").pack(
        anchor=tk.W, **pad
    )

    ttk.Label(frame, text="Instruct (voice description, required):").pack(anchor=tk.W, **pad)
    app.txt_vd_instruct = tk.Text(frame, height=6, wrap=tk.WORD)
    app.txt_vd_instruct.pack(fill=tk.X, **pad)
    app.txt_vd_instruct.insert(
        "1.0",
        "Warm, gentle female voice, calm and soothing tone.",
    )

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, **pad)
    ttk.Button(btn_frame, text="Generate", command=lambda: app._start_generation("voice-design")).pack(
        side=tk.LEFT, padx=5, pady=5
    )


def build_voice_clone_tab(app):
    """Build the VoiceClone tab.

    Args:
        app: The QwenTTSGUI instance.
    """
    frame = app.tab_vc
    pad = {"padx": 10, "pady": 5}

    ttk.Label(frame, text="Text (comma-separated for batch):").pack(anchor=tk.W, **pad)
    app.txt_vc_text = tk.Text(frame, height=6, wrap=tk.WORD)
    app.txt_vc_text.pack(fill=tk.BOTH, expand=True, **pad)
    app.txt_vc_text.insert("1.0", "Hello, I am cloning this voice.")

    ttk.Label(frame, text="Language:").pack(anchor=tk.W, **pad)
    app.var_vc_lang = tk.StringVar(value=DEFAULT_LANGUAGE)
    ttk.Combobox(frame, textvariable=app.var_vc_lang, values=LANGUAGES, width=15, state="readonly").pack(
        anchor=tk.W, **pad
    )

    # Reference audio
    row = ttk.Frame(frame)
    row.pack(fill=tk.X, **pad)
    ttk.Label(row, text="Reference Audio:").pack(side=tk.LEFT, padx=5)
    app.var_vc_ref_audio = tk.StringVar()
    ttk.Entry(row, textvariable=app.var_vc_ref_audio, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(row, text="Browse…", command=app._browse_ref_audio).pack(side=tk.LEFT, padx=5)

    # Reference text
    ttk.Label(frame, text="Reference Transcript (optional for x-vector only):").pack(anchor=tk.W, **pad)
    app.txt_vc_ref_text = tk.Text(frame, height=3, wrap=tk.WORD)
    app.txt_vc_ref_text.pack(fill=tk.X, **pad)

    # X-vector only
    app.var_vc_xvector = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        frame, text="X-vector only (speaker embedding only, ref_text not required)",
        variable=app.var_vc_xvector,
    ).pack(anchor=tk.W, **pad)

    # Prompt reuse / save
    row = ttk.Frame(frame)
    row.pack(fill=tk.X, **pad)
    ttk.Label(row, text="Reuse Prompt (NPZ):").pack(side=tk.LEFT, padx=5)
    app.var_vc_reuse_prompt = tk.StringVar()
    ttk.Entry(row, textvariable=app.var_vc_reuse_prompt, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(row, text="Browse…", command=app._browse_reuse_prompt).pack(side=tk.LEFT, padx=5)

    row = ttk.Frame(frame)
    row.pack(fill=tk.X, **pad)
    ttk.Label(row, text="Save Prompt To (NPZ):").pack(side=tk.LEFT, padx=5)
    app.var_vc_save_prompt = tk.StringVar()
    ttk.Entry(row, textvariable=app.var_vc_save_prompt, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(row, text="Browse…", command=app._browse_save_prompt).pack(side=tk.LEFT, padx=5)

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, **pad)
    ttk.Button(btn_frame, text="Generate", command=lambda: app._start_generation("voice-clone")).pack(
        side=tk.LEFT, padx=5, pady=5
    )


def build_status_bar(app, root):
    """Build the status bar and playback panel.

    Args:
        app: The QwenTTSGUI instance.
        root: The tkinter root window.
    """
    frame = ttk.Frame(root, padding=5)
    frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    app.progress = ttk.Progressbar(frame, mode="indeterminate")
    app.progress.pack(fill=tk.X, padx=5, pady=2)

    app.lbl_status = ttk.Label(frame, text="Ready", foreground="green")
    app.lbl_status.pack(anchor=tk.W, padx=5)

    # Passive notification message (replaces dialog boxes)
    app.lbl_notify = ttk.Label(frame, textvariable=app.notification_msg, foreground="blue")
    app.lbl_notify.pack(anchor=tk.W, padx=5)

    app.btn_stop = ttk.Button(frame, text="Stop Generation", command=app._request_stop, state=tk.DISABLED)
    app.btn_stop.pack(anchor=tk.E, padx=5, pady=2)

    # Prominent playback panel
    playback_panel = ttk.LabelFrame(root, text="Audio Playback", padding=10)
    playback_panel.pack(fill=tk.X, padx=10, pady=(0, 10))

    app.btn_play = ttk.Button(
        playback_panel, text="Play", command=app._play_audio, state=tk.DISABLED
    )
    app.btn_play.pack(side=tk.LEFT, padx=5)

    app.btn_play_stop = ttk.Button(
        playback_panel, text="Stop", command=app._stop_audio, state=tk.DISABLED
    )
    app.btn_play_stop.pack(side=tk.LEFT, padx=5)

    ttk.Button(playback_panel, text="Browse…", command=app._browse_playback_file).pack(side=tk.LEFT, padx=5)

    ttk.Label(playback_panel, text="Volume:").pack(side=tk.LEFT, padx=(15, 2))
    app.var_volume = tk.DoubleVar(value=0.8)
    app.scale_volume = ttk.Scale(
        playback_panel, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
        variable=app.var_volume, command=app._set_volume, length=150,
    )
    app.scale_volume.pack(side=tk.LEFT, padx=2)

    app.lbl_playback_status = ttk.Label(playback_panel, text="No audio loaded", foreground="gray")
    app.lbl_playback_status.pack(side=tk.LEFT, padx=10)


def build_notebook(app, root):
    """Build the notebook with tabs and bind tab change event.

    Args:
        app: The QwenTTSGUI instance.
        root: The tkinter root window.
    """
    app.notebook = ttk.Notebook(root)
    app.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    app.tab_cv = ttk.Frame(app.notebook)
    app.tab_vd = ttk.Frame(app.notebook)
    app.tab_vc = ttk.Frame(app.notebook)

    app.notebook.add(app.tab_cv, text="CustomVoice")
    app.notebook.add(app.tab_vd, text="VoiceDesign")
    app.notebook.add(app.tab_vc, text="VoiceClone")

    # Auto-update Model ID when switching tabs
    app.notebook.bind("<<NotebookTabChanged>>", app._on_tab_change)


def torch_cuda_available():
    """Check if CUDA is available via torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
