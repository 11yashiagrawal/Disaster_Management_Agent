from core.utils.transcript_correction import repair_transcript


def test_repair_transcript_fixes_common_asr_typos() -> None:
    text = "smok near citi hospitl and road blockd"
    result = repair_transcript(text)

    assert result == "smoke near city hospital and road blocked"


def test_repair_transcript_removes_noise_tokens() -> None:
    text = "aaaaa rhifh fire near metro station xxxxx"
    result = repair_transcript(text)

    assert result == "fire near metro station"


def test_repair_transcript_fixes_emergency_words() -> None:
    text = "gass smel in apartmant people faintng cant breth"
    result = repair_transcript(text)

    assert result == "gas smell in apartment people fainting cannot breathe"


def test_repair_transcript_collapses_ellipsis_and_symbols() -> None:
    text = "bus.... kar hit badli near sity hospital!!!"
    result = repair_transcript(text)

    assert result == "bus car hit badly near city hospital"