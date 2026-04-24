import re


def compute_panic_score(raw_text: str) -> float:
    text = raw_text or ""
    if not text.strip():
        return 0.0

    score = 0.0

    all_caps_tokens = re.findall(r"\b[A-Z]{2,}\b", text)
    if all_caps_tokens:
        score += 0.35

    alphabetic_chars = re.findall(r"[A-Za-z]", text)
    uppercase_chars = re.findall(r"[A-Z]", text)
    if alphabetic_chars:
        caps_ratio = len(uppercase_chars) / len(alphabetic_chars)
        if caps_ratio >= 0.25:
            score += 0.25
        elif caps_ratio >= 0.15:
            score += 0.15

    if re.search(r"[!?]{2,}", text):
        score += 0.10

    if re.search(r"\b(help|please|urgent|immediately|hurry|fast)\b", text, re.IGNORECASE):
        score += 0.20

    return round(min(score, 1.0), 2)
