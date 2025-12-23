import re

from app.utils.preview import build_preview_from_text


def test_removes_photo_credit_and_bullets():
    credit, preview = build_preview_from_text(
        """
        Photo by Gria on Unsplash
        ## Overview
        What happened: Attackers exploited a zero-day.
        - bullet one
        - bullet two
        What to do: Patch now and rotate keys.
        """
    )
    assert credit and "Photo by Gria" in credit
    assert "bullet" not in preview
    assert "What to do" not in preview
    assert preview.endswith((".", "!", "?"))


def test_truncates_to_sentence_boundary():
    _, preview = build_preview_from_text(
        "What happened: A long incident with many details. More text without bullet markers. Final sentence ends here without trailing tokens"
    )
    assert len(preview) <= 520
    assert preview.endswith((".", "!", "?"))


def test_limits_to_first_what_happened():
    _, preview = build_preview_from_text(
        "Intro text. What happened: breach in system A and B. More details. What to do: steps and patches."
    )
    assert "What to do" not in preview
    assert "breach in system" in preview
