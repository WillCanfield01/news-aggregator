from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from flask import current_app


@dataclass
class AffiliatePick:
    asin: str
    name: str
    blurb: str
    approx_price: str  # e.g., "$24.99"
    reviews_hint: str  # e.g., "5,800+ reviews"


def amazon_url(asin: str) -> str:
    """
    Build a tagged Amazon URL for a product detail page.
    """
    tag = current_app.config.get("AMAZON_ASSOC_TAG")
    base = f"https://www.amazon.com/dp/{asin}"
    if tag:
        return f"{base}?tag={tag}"
    return base


# NOTE: replace placeholder ASINs with real ones later.
TOOL_PICKS: Dict[str, List[AffiliatePick]] = {
    "resume": [
        AffiliatePick(
            asin="REPLACE_MIC_ASIN",
            name="Mini Mic Pro",
            blurb="Crisp audio for Zoom and interview calls.",
            approx_price="$24.99",
            reviews_hint="5,800+ reviews",
        ),
        AffiliatePick(
            asin="REPLACE_KEYBOARD_ASIN",
            name="Low-profile wireless keyboard",
            blurb="More comfortable typing for long writing sessions.",
            approx_price="$19.99",
            reviews_hint="10,000+ reviews",
        ),
    ],
    "habit": [
        AffiliatePick(
            asin="B085PHGTB8",
            name="Magnetic dry erase calendar (12\" x 17\")",
            blurb="Stick it on the fridge or wall to see your streaks daily.",
            approx_price="$16.93",
            reviews_hint="3,800+ reviews",
        ),
    ],
    "trip": [
        AffiliatePick(
            asin="B014VBI3HK",
            name="Packing cubes set (4-piece)",
            blurb="Keep your suitcase organized with breathable mesh cubes.",
            approx_price="$17.80",
            reviews_hint="42,000+ reviews",
        ),
        AffiliatePick(
            asin="B0D63657GY",
            name="Tile Mate Bluetooth tracker",
            blurb="Find keys and bags fast with iOS or Android.",
            approx_price="$14.00",
            reviews_hint="9,900+ reviews",
        ),
    ],
    "grocery": [
        AffiliatePick(
            asin="B0BRTJJ9F5",
            name="Bentgo Prep 60-piece meal prep set",
            blurb="Microwave/freezer-safe containers for batch cooking.",
            approx_price="$34.99",
            reviews_hint="339+ reviews",
        ),
    ],
    "language": [
        AffiliatePick(
            asin="B0CHCCGQ24",
            name="Spanish conversational flash cards (75 phrases)",
            blurb="Learn daily phrases with IPA and audio for quick practice.",
            approx_price="$14.99",
            reviews_hint="139+ reviews",
        ),
    ],
    "countdown": [
        AffiliatePick(
            asin="B09TCLK2TC",
            name="AIMILAR 999-day digital countdown timer",
            blurb="Dedicated countdown display for long timelines and big events.",
            approx_price="$15.99",
            reviews_hint="2,000+ reviews",
        ),
    ],
}


def get_tool_picks(tool_key: str) -> List[AffiliatePick]:
    return TOOL_PICKS.get(tool_key, [])
