"""
ReviewGuard - Model Training
Trains a Logistic Regression classifier on TF-IDF vectors.
Prints full evaluation metrics and saves model + vectorizer to disk.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preparation import load_and_prepare, build_tfidf_vectors, save_vectorizer

MODEL_PATH      = "models/sentiment_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
LABEL_CLASSES   = ["negative", "neutral", "positive"]


def train(data_path: str):
    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    df = load_and_prepare(data_path)

    # ------------------------------------------------------------------
    # 2. Build TF-IDF features
    # ------------------------------------------------------------------
    X, vectorizer = build_tfidf_vectors(df)
    y = df['sentiment'].values

    # ------------------------------------------------------------------
    # 3. Train / test split  (80 / 20, stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain samples : {X_train.shape[0]}")
    print(f"Test  samples : {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Train Logistic Regression
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',   # handle class imbalance
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

    print("\n" + "=" * 50)
    print("  REVIEWGUARD - MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (weighted)")
    print(f"  Recall    : {recall:.4f}  (weighted)")
    print(f"  F1-Score  : {f1:.4f}  (weighted)")
    print("=" * 50)

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_CLASSES, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_CLASSES)
    cm_df = pd.DataFrame(cm, index=LABEL_CLASSES, columns=LABEL_CLASSES)
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())

    # Interpretation hint
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    if off_diag.max() > 0:
        row, col = np.unravel_index(off_diag.argmax(), off_diag.shape)
        print(
            f"\n⚠  Most common confusion: actual '{LABEL_CLASSES[row]}' "
            f"predicted as '{LABEL_CLASSES[col]}' ({off_diag[row, col]} times)"
        )

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved     → {MODEL_PATH}")
    save_vectorizer(vectorizer, VECTORIZER_PATH)

    return model, vectorizer


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_reviews.csv"

    # If the sample CSV doesn't exist yet, create it
    if not os.path.exists(data_path) and data_path == "data/sample_reviews.csv":
        import data_preparation  # runs __main__ block via import trick? No — call directly.
        sample = pd.DataFrame({
            'review_text': [
                "Amazing battery life, very happy with this purchase!",
                "Screen cracked after two days, terrible build quality.",
                "Decent product for the price, nothing extraordinary.",
                "Absolutely love this phone, best purchase of the year.",
                "Completely useless, stopped working after one week.",
                "Fast delivery and good packaging, product is fine.",
                "Battery drains within 2 hours, very disappointed.",
                "Great camera, smooth performance, highly recommend.",
                "Not as described, feels cheap and flimsy.",
                "Average experience, does what it is supposed to do.",
                "Excellent sound quality and comfortable to wear.",
                "Broke on first use, total waste of money.",
                "Works perfectly, no issues at all so far.",
                "Sluggish performance and constant freezing issues.",
                "Pretty good for the price range honestly.",
            ],
            'rating':       [5, 1, 3, 5, 1, 4, 1, 5, 2, 3, 5, 1, 5, 2, 4],
            'date': [
                '2023-01-10', '2023-01-15', '2023-02-05',
                '2023-02-20', '2023-03-01', '2023-03-15',
                '2023-04-10', '2023-04-25', '2023-05-05',
                '2023-05-20', '2023-06-01', '2023-06-15',
                '2023-07-10', '2023-07-25', '2023-08-01',
            ],
            'product_name': [
                'PhoneX', 'PhoneX', 'TabletY', 'PhoneX', 'TabletY',
                'PhoneX', 'EarBudsZ', 'PhoneX', 'TabletY', 'EarBudsZ',
                'EarBudsZ', 'TabletY', 'PhoneX', 'TabletY', 'EarBudsZ',
            ],
        })
        os.makedirs("data", exist_ok=True)
        sample.to_csv(data_path, index=False)
        print(f"Sample dataset created at {data_path}")

    train(data_path)
