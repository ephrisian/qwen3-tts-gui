#!/usr/bin/env python3
"""Launch script for Qwen3-TTS Gradio Application.

Usage:
    python launch.py              # Launch the Gradio web interface (default port 7860)
    python launch.py --port 8080  # Launch on custom port
    python launch.py --share      # Create temporary Gradio public share link
    python launch.py test         # Run test suite
"""

import sys
import argparse

from tests import run_tests


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
        return

    parser = argparse.ArgumentParser(description="Qwen3-TTS Gradio Studio Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Generate public shareable URL")

    # Filter out 'test' if present
    args = parser.parse_args([a for a in sys.argv[1:] if a != "test"])

    from gradio_app import app
    print(f"==================================================")
    print(f"🎙️ Launching Qwen3-TTS Gradio Studio")
    print(f"🌐 Host: {args.host} | Port: {args.port}")
    print(f"==================================================")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
