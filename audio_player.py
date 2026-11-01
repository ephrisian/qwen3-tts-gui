"""Minimal WAV playback: pygame if available, else a system player."""

import shutil
import subprocess

_proc = None


def _pygame():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame
    except Exception:
        return None


def stop():
    global _proc
    pg = _pygame()
    if pg:
        pg.mixer.music.stop()
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None


def play(path):
    """Start playing `path` without blocking. Raises RuntimeError if nothing can play it."""
    global _proc
    stop()
    pg = _pygame()
    if pg:
        pg.mixer.music.load(path)
        pg.mixer.music.play()
        return
    for cmd in (["pw-play"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            _proc = subprocess.Popen(cmd + [path])
            return
    raise RuntimeError("No audio player found (install pygame, or paplay/pw-play/ffplay).")
