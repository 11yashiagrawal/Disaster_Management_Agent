import re


def normalize_transcript(text: str) -> str:
    """
    Normalize raw emergency call transcript text into a consistent form
    for deterministic parsing.

    Current normalization steps:
    - trim surrounding whitespace
    - convert to lowercase
    - collapse repeated whitespace
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text