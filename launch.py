#!/usr/bin/env python3
"""Entry point for the Qwen3-TTS application.

Usage:
    python launch.py          # Launch the GUI
    python launch.py test     # Run unit tests
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from tests import run_tests
        run_tests()
    else:
        from qt_app import main as run_gui
        run_gui()


if __name__ == "__main__":
    main()
