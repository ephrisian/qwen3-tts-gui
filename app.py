#!/usr/bin/env python3
"""Standard Hugging Face / Gradio entry point for Qwen3-TTS."""

from gradio_app import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
