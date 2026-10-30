#!/usr/bin/env python3
"""Entry point for the Qwen3-TTS GUI application.

Usage:
    python launch.py          # Launch the GUI
    python launch.py test     # Run unit tests
"""

import sys
import tkinter as tk

from gui_app import QwenTTSGUI
from tests import run_tests


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        root = tk.Tk()
        app = QwenTTSGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
