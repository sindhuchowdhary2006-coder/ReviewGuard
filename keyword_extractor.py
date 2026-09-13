"""
ReviewGuard - Negative Review Keyword Extractor
Given a list of predicted-negative reviews, returns the top N most frequent keywords.
"""

import re
from collections import Counter
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

# Additional domain-specific noise words to suppress
EXTRA_NOISE = {'product', 'item', 'one', 'got', 'get', 'also', 'would', 'could',
               'even', 'like', 'just', 'really', 'much', 'still', 'use', 'used',
               'using', 'buy', 'bought', 'ordered', 'received', 'came'}


def extract_negative_keywords(
    negative_reviews: List[str],
    top_n: int = 10,
) -> List[Tuple[str, int]]:
    """
    Given a list of raw/cleaned negative review strings, returns the top_n
    most frequently occurring keywords (excluding stopwords).

    Returns a list of (keyword, count) tuples sorted by count descending.
    """
    all_tokens: List[str] = []

    for text in negative_reviews:
        if not isinstance(text, str):
            continue
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in STOP_WORDS
            and t not in EXTRA_NOISE
            and len(t) > 2
        ]
        all_tokens.extend(tokens)

    counter = Counter(all_tokens)
    return counter.most_common(top_n)


def format_keywords(keyword_counts: List[Tuple[str, int]]) -> List[dict]:
    """Converts (word, count) tuples to a list of dicts for JSON serialisation."""
    return [{"keyword": word, "count": count} for word, count in keyword_counts]
