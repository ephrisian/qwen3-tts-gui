"""Text chunking and splitting utilities for Qwen3-TTS."""

import re


# Common abbreviations that should NOT trigger sentence splits.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m", "am", "pm", "u.s", "u.k", "u.n",
    "ph.d", "m.d", "b.a", "b.s", "m.a", "m.s", "ll.m", "j.d",
    "co", "corp", "inc", "ltd", "llc", "dept", "govt", "rev",
    "hon", "rep", "sen", "gen", "col", "capt", "sgt", "maj",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def _is_abbreviation(word):
    """Check if a word is a common abbreviation that should not end a sentence."""
    # Strip trailing punctuation and compare case-insensitively
    clean = word.rstrip(".,!?;:").lower()
    return clean in _ABBREVIATIONS


def _split_sentences(text):
    """Split text into sentences, avoiding false splits on abbreviations.

    Splits on . ! ? followed by whitespace, but does not split when the
    preceding token is a known abbreviation (e.g. "Dr.", "U.S.", "e.g.").
    """
    # First, split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If this part starts with a lowercase letter and the previous
        # sentence ended with an abbreviation, merge them back together.
        if sentences and part and part[0].islower():
            last_word = sentences[-1].split()[-1] if sentences[-1].split() else ""
            if _is_abbreviation(last_word):
                sentences[-1] = sentences[-1] + " " + part
                continue
        sentences.append(part)
    return sentences


def chunk_text(text, min_words=200, max_words=250):
    """Split long text into segments at sentence boundaries.

    Splits on sentence-ending punctuation (. ! ?) followed by whitespace, then
    greedily packs whole sentences into chunks targeting between min_words and
    max_words words per segment. Sentences are never broken up — if a single
    sentence exceeds max_words it is kept intact as its own chunk.
    Commas are never used as split points.

    After initial chunking, any chunk below min_words is merged with the
    previous chunk to avoid producing many tiny fragments.
    """
    if not text:
        return []

    sentences = _split_sentences(text)

    chunks = []
    current = ""
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        # If adding this sentence would exceed max_words and we already have
        # at least min_words, finalize the current chunk and start a new one.
        if current and current_words + sentence_words > max_words and current_words >= min_words:
            chunks.append(current)
            current = sentence
            current_words = sentence_words
        else:
            current = (current + " " + sentence).strip() if current else sentence
            current_words += sentence_words

    if current:
        chunks.append(current)

    # Post-process: merge any chunk below min_words with the previous chunk.
    if min_words > 0 and len(chunks) > 1:
        merged = []
        for chunk in chunks:
            if merged and len(chunk.split()) < min_words:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        chunks = merged

    return chunks


def split_text(text_arg):
    """Return the text as a single-item list. Commas are not used as separators."""
    return [text_arg] if text_arg else []
