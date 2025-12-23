import pytest

from app.scripts.generate_timeline_round import (
    _is_truncated,
    _validate_headline,
    _unique_headlines,
    _finalize_headline,
    _normalize_choice,
)


def test_is_truncated_detection():
    assert _is_truncated("City officials outline next steps for")
    assert _is_truncated("Plan announced by council:")
    assert not _is_truncated("City officials outline next steps for commuters.")


def test_validate_headline_bounds():
    good = "City transit board approves expanded late-night service after public hearings conclude."
    assert _validate_headline(good)
    too_short = "Short headline."
    assert not _validate_headline(too_short)
    truncated = "Council approves plan for"
    assert not _validate_headline(truncated)


def test_uniqueness_checks():
    a = "Port authority unveils revised harbor safety rules after inspections."
    b = "City council approves revised harbor safety rules after inspections."
    c = "National museum opens new exhibit featuring modern artists."
    assert not _unique_headlines([a, b, c])  # first 10 words overlap
    assert _unique_headlines([a, c, "Regional rail agency expands evening service after public feedback."])


def test_generation_like_loop():
    headlines = []
    for i in range(50):
        real = _finalize_headline(_normalize_choice(f"Historic waterfront event draws crowds {i}", 14, 22))
        fake1 = _finalize_headline(_normalize_choice(f"City council reviews transit updates after public input {i}", 14, 22))
        fake2 = _finalize_headline(_normalize_choice(f"Regional board announces phased upgrades to commuter routes {i}", 14, 22))
        assert _validate_headline(real)
        assert _validate_headline(fake1)
        assert _validate_headline(fake2)
        assert _unique_headlines([real, fake1, fake2])
        headlines.append((real, fake1, fake2))
