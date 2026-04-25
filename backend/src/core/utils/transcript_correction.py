import re


COMMON_ASR_CORRECTIONS = {
    "smok": "smoke",
    "smokee": "smoke",
    "fyre": "fire",
    "burnng": "burning",
    "gass": "gas",
    "smel": "smell",
    "apartmant": "apartment",
    "faintng": "fainting",
    "breth": "breathe",
    "breathng": "breathing",
    "breathngg": "breathing",
    "injrd": "injured",
    "injurd": "injured",
    "kild": "killed",
    "colapsd": "collapsed",
    "colaps": "collapse",
    "bilding": "building",
    "rubbl": "rubble",
    "blockd": "blocked",
    "rood": "road",
    "kar": "car",
    "amblance": "ambulance",
    "markit": "market",
    "skul": "school",
    "brij": "bridge",
    "hospitl": "hospital",
    "hosptl": "hospital",
    "citi": "city",
    "sity": "city",
    "boz": "bus",
    "badli": "badly",
    "submrged": "submerged",
    "vhicles": "vehicles",
    "trap": "trapped",
    "cant": "cannot",
    "breathngg": "breathing",
    "unconsious": "unconscious",
    "properli": "properly",
    "grin": "green",
}


NOISE_TOKENS = {
    "aaaa",
    "aaaaa",
    "aaaaaa",
    "rhifh",
    "gibbersih",
    "gibbr",
    "xxxxx",
    "xxxx",
    "uhh",
    "umm",
    "aaa",
}


def repair_transcript(text: str) -> str:
    """
    Repair noisy ASR-like transcript issues using simple deterministic rules.

    Steps:
    - lowercase
    - remove repeated punctuation noise
    - remove standalone noise tokens
    - apply common ASR typo corrections
    - collapse whitespace
    """
    text = text.lower()

    text = re.sub(r"[.]{2,}", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    tokens = text.split()
    cleaned_tokens = []

    for token in tokens:
        if token in NOISE_TOKENS:
            continue

        corrected = COMMON_ASR_CORRECTIONS.get(token, token)
        cleaned_tokens.append(corrected)

    text = " ".join(cleaned_tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text