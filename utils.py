from typing import Any

from bs4 import Tag


def format_candidate(elem: Tag, score: float) -> dict[str, Any]:
    """Format a candidate element with its tag, text, score, and classes."""
    if elem.name == "img":
        text = elem.get("src", "")[:80]
    elif elem.name == "a":
        text = f"{elem.get_text(strip=True)[:50]} -> {elem.get('href', '')[:30]}"
    else:
        text = elem.get_text(strip=True)[:80]

    return {
        "tag": elem.name,
        "text": text,
        "score": round(score, 3),
        "classes": elem.get("class", []),
    }
