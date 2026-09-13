"""
ReviewGuard - Data Preparation
Loads Amazon Reviews dataset, cleans text, labels sentiment, and creates TF-IDF vectors.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans a single review string:
      1. Lowercase
      2. Remove punctuation and digits
      3. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and digits, keep letters and spaces only
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenise and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def rating_to_sentiment(rating) -> str:
    """Maps a numeric star rating to a sentiment label."""
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return "neutral"

    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Loads the Amazon Reviews CSV, cleans text, and assigns sentiment labels.

    Flexible column name mapping:
      - review text  : 'reviewText' | 'review_text' | 'text' | 'review'
      - rating       : 'overall'    | 'rating'       | 'stars'
      - date         : 'reviewTime' | 'date'         | 'review_date'
      - product name : 'asin'       | 'product_name' | 'product'
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # --- Normalise column names ---
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ('reviewtext', 'review_text', 'text', 'review'):
            col_map[col] = 'review_text'
        elif lc in ('overall', 'rating', 'stars'):
            col_map[col] = 'rating'
        elif lc in ('reviewtime', 'date', 'review_date', 'reviewdate'):
            col_map[col] = 'date'
        elif lc in ('asin', 'product_name', 'product', 'productname'):
            col_map[col] = 'product_name'

    df.rename(columns=col_map, inplace=True)

    # Ensure required columns exist
    for required in ('review_text', 'rating'):
        if required not in df.columns:
            raise ValueError(
                f"Column '{required}' not found. Available columns: {list(df.columns)}"
            )

    # Optional columns with defaults
    if 'date' not in df.columns:
        df['date'] = pd.NaT
    if 'product_name' not in df.columns:
        df['product_name'] = 'Unknown Product'

    # Drop rows missing critical fields
    df.dropna(subset=['review_text', 'rating'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Clean text
    print("Cleaning review text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)

    # Assign sentiment labels
    df['sentiment'] = df['rating'].apply(rating_to_sentiment)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    print(f"\nDataset loaded: {len(df)} reviews")
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())

    return df


def build_tfidf_vectors(df: pd.DataFrame, max_features: int = 10000):
    """
    Fits a TF-IDF vectorizer on the cleaned text.
    Returns the sparse feature matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,             # ignore very rare terms
        sublinear_tf=True     # log normalization
    )

    X = vectorizer.fit_transform(df['cleaned_text'])
    print(f"TF-IDF matrix shape: {X.shape}")
    return X, vectorizer


def save_vectorizer(vectorizer, path: str = "models/tfidf_vectorizer.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved → {path}")


if __name__ == "__main__":
    # Smoke-test with a small synthetic dataset
    sample = pd.DataFrame({
        'review_text': [
            "Amazing product, battery life is great!",
            "Screen broke after one week, terrible quality.",
            "It's okay, nothing special.",
            "Best phone I've ever bought, highly recommend.",
            "Completely dead on arrival, waste of money.",
        ],
        'rating':       [5,          1,                             3,              5,                              1],
        'date':         ['2023-01-10','2023-02-15',                '2023-03-05',   '2023-04-20',                  '2023-05-01'],
        'product_name': ['PhoneX',   'PhoneX',                    'TabletY',      'PhoneX',                       'TabletY'],
    })

    os.makedirs("data", exist_ok=True)
    sample.to_csv("data/sample_reviews.csv", index=False)

    df = load_and_prepare("data/sample_reviews.csv")
    X, vectorizer = build_tfidf_vectors(df)
    save_vectorizer(vectorizer)
    print("\nData preparation complete.")
