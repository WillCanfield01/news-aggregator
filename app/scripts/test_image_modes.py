from __future__ import annotations

import random

from app.scripts.generate_timeline_round import (
    pick_image_for_choice,
    _abstract_fallback_image_for_category,
    _photo_fallback_image,
)


def test_image_modes():
    rng = random.Random(123)
    used: set[str] = set()
    meta = {"text": "City parade", "category": "culture"}

    photo_url, _, _ = pick_image_for_choice(meta, rng, used, None, mode="real", allow_abstract=False)
    assert "/static/roulette/fallbacks/" not in photo_url, "headline/photo should not use abstract fallback"

    abstract = _abstract_fallback_image_for_category("general", set(), random.Random(1))
    assert "/static/roulette/fallbacks/" in abstract, "quote fallback should be abstract"

    photo_fallback = _photo_fallback_image(set(), random.Random(2))
    assert photo_fallback.startswith("http"), "photo fallback should be photoreal URL"


if __name__ == "__main__":
    test_image_modes()
    print("image mode tests passed")
