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
            asin="REPLACE_WHITEBOARD_ASIN",
            name="Magnetic whiteboard calendar",
            blurb="Track your streaks where you can see them.",
            approx_price="$17.99",
            reviews_hint="4,000+ reviews",
        ),
    ],
    "trip": [
        AffiliatePick(
            asin="REPLACE_PACKING_CUBES_ASIN",
            name="Packing cubes set",
            blurb="Keep your suitcase organized for any trip.",
            approx_price="$21.99",
            reviews_hint="30,000+ reviews",
        ),
        AffiliatePick(
            asin="REPLACE_AIRTAG_ASIN",
            name="Bluetooth luggage tracker",
            blurb="Find your bag quickly at the airport.",
            approx_price="$29.99",
            reviews_hint="40,000+ reviews",
        ),
    ],
    "grocery": [
        AffiliatePick(
            asin="REPLACE_CONTAINERS_ASIN",
            name="Meal prep container set",
            blurb="Make weekly food planning easier.",
            approx_price="$23.99",
            reviews_hint="12,000+ reviews",
        ),
    ],
    "language": [
        AffiliatePick(
            asin="REPLACE_FLASHCARDS_ASIN",
            name="Language flash cards",
            blurb="Reinforce new words away from the screen.",
            approx_price="$9.99",
            reviews_hint="3,500+ reviews",
        ),
    ],
}


def get_tool_picks(tool_key: str) -> List[AffiliatePick]:
    return TOOL_PICKS.get(tool_key, [])
