"""Simple CLI evaluation harness for retrieval coverage."""

from __future__ import annotations

import sys


def keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = text.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python evaluate.py "retrieved text" "keyword1,keyword2"')
        raise SystemExit(1)
    text = sys.argv[1]
    keywords = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
    print(f"Keyword coverage: {keyword_coverage(text, keywords):.0%}")
