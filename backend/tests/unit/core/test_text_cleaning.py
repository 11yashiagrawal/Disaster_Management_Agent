from core.utils.text_cleaning import normalize_transcript


def test_normalize_transcript_strips_outer_whitespace() -> None:
    text = "   There is a fire near the hospital.   "

    result = normalize_transcript(text)

    assert result == "there is a fire near the hospital."


def test_normalize_transcript_lowercases_text() -> None:
    text = "FIRE NEAR GREEN PARK"

    result = normalize_transcript(text)

    assert result == "fire near green park"


def test_normalize_transcript_collapses_repeated_whitespace() -> None:
    text = "There   is \n\n a    fire\tnear   the station"

    result = normalize_transcript(text)

    assert result == "there is a fire near the station"


def test_normalize_transcript_handles_already_clean_text() -> None:
    text = "there is smoke near city hospital"

    result = normalize_transcript(text)

    assert result == "there is smoke near city hospital"