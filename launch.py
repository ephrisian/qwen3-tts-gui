#!/usr/bin/env python3
"""Entry point for the Qwen3-TTS application.

Usage:
    python launch.py          # Launch the GUI
    python launch.py test     # Run unit tests
    python launch.py benchmark --voice NAME --text-file PATH
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from tests import run_tests
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from benchmark import main as run_benchmark
        raise SystemExit(run_benchmark(sys.argv[2:]))
    else:
        from qt_app import main as run_gui
        run_gui()


if __name__ == "__main__":
    main()
