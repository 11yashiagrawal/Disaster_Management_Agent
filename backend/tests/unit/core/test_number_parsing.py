from core.utils.number_parsing import extract_leading_number_token


def test_extracts_digit_number() -> None:
    assert extract_leading_number_token("2 people injured") == 2


def test_extracts_word_number() -> None:
    assert extract_leading_number_token("two people injured") == 2


def test_extracts_article_as_one() -> None:
    assert extract_leading_number_token("an injured person") == 1


def test_returns_none_for_missing_number() -> None:
    assert extract_leading_number_token("people injured") is None