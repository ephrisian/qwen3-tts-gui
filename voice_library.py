"""On-disk library of designed voices.

Each voice lives in voices/<slug>/ :
    voice.json        metadata (description, seed, sample text, file names, usage notes)
    reference.wav     the approved VoiceDesign sample
    clone_prompt.npz  prebuilt Base-model clone prompt (speaker embedding + ref codes + transcript)
    emotions/         cached per-emotion clone prompts (created on demand)
"""

import json
import os
import re
import shutil
from datetime import datetime
import random

import numpy as np

from audio_utils import load_voice_clone_prompt, save_voice_clone_prompt, write_wav

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

SAMPLE_TEXT = (
    "Are you certain the quick, brown fox jumped over the lazy, yawning dog? "
    "Honestly, I couldn't believe my own eyes! The majestic creature moved with "
    "such vivid, explosive power. Breathing deeply, we just watched the bizarre "
    "scene unfold, marveling at the quiet, azure sky above."
)

USAGE_NOTES = {
    "voice_clone": (
        "Pass clone_prompt.npz as voice_clone_prompt to Qwen3TTSModel.generate_voice_clone() on the "
        "Base model, or use reference.wav + ref_text with create_voice_clone_prompt()."
    ),
    "voice_design": (
        "description is the VoiceDesign instruct. Seed does not guarantee the same voice for new text; "
        "use the clone prompt for a stable identity."
    ),
}


def slugify(name):
    slug = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return slug[:60]


def voice_dir(name, root=None):
    return os.path.join(root or VOICES_DIR, slugify(name))


def list_voices(root=None):
    """Return sorted voice names that have a valid voice.json."""
    root = root or VOICES_DIR
    names = []
    if os.path.isdir(root):
        for entry in os.listdir(root):
            cfg = os.path.join(root, entry, "voice.json")
            if os.path.isfile(cfg):
                try:
                    with open(cfg, encoding="utf-8") as f:
                        names.append(json.load(f)["name"])
                except (OSError, ValueError, KeyError):
                    continue
    return sorted(names, key=str.lower)


def voice_exists(name, root=None):
    return os.path.isfile(os.path.join(voice_dir(name, root), "voice.json"))


def save_voice(name, wav, sr, prompt, description, language, seed,
               model_ids=None, root=None, overwrite=False):
    """Write a voice folder and return its config dict."""
    if not slugify(name):
        raise ValueError("Voice name must contain letters or digits.")
    path = voice_dir(name, root)
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"A voice named '{name}' already exists.")
        shutil.rmtree(path)
    os.makedirs(path)

    write_wav(os.path.join(path, "reference.wav"), np.asarray(wav), sr)
    save_voice_clone_prompt(os.path.join(path, "clone_prompt.npz"), prompt)

    config = {
        "schema": 1,
        "name": name.strip(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "description": description,
        "seed": int(seed),
        "sample_text": SAMPLE_TEXT,
        "ref_text": SAMPLE_TEXT,
        "ref_audio": "reference.wav",
        "clone_prompt": "clone_prompt.npz",
        "sample_rate": int(sr),
        "models": model_ids or {},
        "usage": USAGE_NOTES,
    }
    with open(os.path.join(path, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config


def load_voice(name, root=None):
    """Return (config, clone_prompt, voice_directory) for a saved voice."""
    path = voice_dir(name, root)
    with open(os.path.join(path, "voice.json"), encoding="utf-8") as f:
        config = json.load(f)
    prompt = load_voice_clone_prompt(os.path.join(path, config["clone_prompt"]))
    return config, prompt, path


def delete_voice(name, root=None):
    path = voice_dir(name, root)
    if os.path.isfile(os.path.join(path, "voice.json")):
        shutil.rmtree(path)


# ------------------------------------------------------------------ random name generation
_MALE_FIRST = [
    "James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kevin", "Brian", "George",
    "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
    "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
    "Benjamin", "Samuel", "Gregory", "Frank", "Alexander", "Raymond", "Jack",
    "Patrick", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Adam", "Henry",
    "Nathan", "Douglas", "Zachary", "Peter", "Kyle", "Walter", "Ethan", "Jeremy",
    "Harold", "Keith", "Christian", "Roger", "Noah", "Gerald", "Carl", "Keith",
]
_FEMALE_FIRST = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Sandra", "Margaret",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Helen", "Anna", "Brenda", "Pamela",
    "Nicole", "Emma", "Samantha", "Katherine", "Christine", "Janet", "Debra",
    "Rachel", "Catherine", "Heather", "Diane", "Julie", "Joyce", "Victoria",
    "Kelly", "Christina", "Joan", "Lauren", "Martha", "Victoria", "Alicia",
    "Julie", "Cheryl", "Megan", "Andrea", "Hannah", "Katie", "Natalie",
]
_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Flores",
    "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed",
    "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson",
    "Cox", "Howard", "Ward", "Patterson", "Hughes", "Flores", "Washington",
    "Butler", "Simmons", "Foster", "Gonzales", "Bryant", "Alexander", "Russell",
]


def random_voice_name(gender="random"):
    """Generate a random voice name.

    gender: "male", "female", or "random".
    Returns a full name like "James Smith".
    """
    if gender == "male":
        first = random.choice(_MALE_FIRST)
    elif gender == "female":
        first = random.choice(_FEMALE_FIRST)
    else:
        first = random.choice(_MALE_FIRST + _FEMALE_FIRST)
    last = random.choice(_LAST)
    return f"{first} {last}"


def unique_voice_name(base_name, root=None):
    """Return a unique variant of base_name by appending a number if needed."""
    root = root or VOICES_DIR
    if not voice_exists(base_name, root):
        return base_name
    i = 2
    while voice_exists(f"{base_name} {i}", root):
        i += 1
    return f"{base_name} {i}"


def list_voice_details(root=None):
    """Return list of voice detail dicts for all saved voices.

    Each dict: name, slug, description, language, seed, created, path.
    """
    root = root or VOICES_DIR
    details = []
    if not os.path.isdir(root):
        return details
    for entry in sorted(os.listdir(root)):
        cfg_path = os.path.join(root, entry, "voice.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            details.append({
                "name": cfg.get("name", entry),
                "slug": entry,
                "description": cfg.get("description", ""),
                "language": cfg.get("language", ""),
                "seed": cfg.get("seed", 0),
                "created": cfg.get("created", ""),
                "path": os.path.join(root, entry),
            })
        except (OSError, ValueError, KeyError):
            continue
    return details


def rename_voice(old_name, new_name, root=None):
    """Rename a saved voice directory and update voice.json name field."""
    root = root or VOICES_DIR
    old_slug = slugify(old_name)
    new_slug = slugify(new_name)
    if not new_slug:
        raise ValueError("New voice name must contain letters or digits.")
    old_path = os.path.join(root, old_slug)
    new_path = os.path.join(root, new_slug)
    if not os.path.isdir(old_path):
        raise FileNotFoundError(f"Voice '{old_name}' not found.")
    if os.path.exists(new_path) and old_slug != new_slug:
        raise FileExistsError(f"A voice named '{new_name}' already exists.")
    if old_slug != new_slug:
        os.rename(old_path, new_path)
    cfg_path = os.path.join(new_path, "voice.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["name"] = new_name.strip()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return new_name
