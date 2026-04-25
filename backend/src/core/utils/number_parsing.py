import re
from typing import Optional


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_leading_number_token(text: str) -> Optional[int]:
    """
    Extract a leading numeric quantity from a short phrase.

    Examples:
        "2 people injured" -> 2
        "two people injured" -> 2
        "an injured person" -> 1
    """
    match = re.match(r"^\s*(\d+)\b", text)
    if match:
        return int(match.group(1))

    match = re.match(r"^\s*([a-z]+)\b", text.lower())
    if not match:
        return None

    token = match.group(1)
    return NUMBER_WORDS.get(token)