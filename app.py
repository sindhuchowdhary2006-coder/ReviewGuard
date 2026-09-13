"""
ReviewGuard - Flask API
Routes:
  POST /predict          - single review sentiment prediction
  POST /analyze-batch    - CSV upload, batch predictions + per-product breakdown
  GET  /negative-keywords - top complaint keywords from negative reviews
  GET  /sentiment-trend  - monthly sentiment trend
"""

import os
import io
import json

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

from data_preparation import clean_text, rating_to_sentiment
from keyword_extractor import extract_negative_keywords, format_keywords

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upload limit

MODEL_PATH      = "models/sentiment_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

# Load artefacts once at startup
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Model and vectorizer loaded.")
else:
    model      = None
    vectorizer = None
    print("⚠  Model artefacts not found. Run train_model.py first.")


def _check_model():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    return None


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: {"review": "Battery drains within 2 hours, very disappointed"}
    Returns: {"sentiment": "negative", "confidence": 0.91}
    """
    err = _check_model()
    if err:
        return err

    data = request.get_json(force=True, silent=True)
    if not data or "review" not in data:
        return jsonify({"error": "Request body must include a 'review' field."}), 400

    raw_review = data["review"]
    if not isinstance(raw_review, str) or not raw_review.strip():
        return jsonify({"error": "'review' must be a non-empty string."}), 400

    cleaned   = clean_text(raw_review)
    X         = vectorizer.transform([cleaned])
    proba     = model.predict_proba(X)[0]
    classes   = model.classes_

    predicted_idx = int(np.argmax(proba))
    sentiment     = classes[predicted_idx]
    confidence    = round(float(proba[predicted_idx]), 4)

    # Build full probability breakdown
    all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    return jsonify({
        "sentiment":   sentiment,
        "confidence":  confidence,
        "probabilities": all_probs,
    })


# ---------------------------------------------------------------------------
# POST /analyze-batch
# ---------------------------------------------------------------------------
@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    """
    Accepts: multipart/form-data with a CSV file (field name: 'file').
    CSV must have columns: review_text (or review / text), product_name (optional).

    Returns JSON with:
      - total_counts          : {positive: N, neutral: N, negative: N}
      - per_product_breakdown : {product: {positive: N, ...}, ...}
      - download_url          : link to download the enriched CSV

    Also stores the enriched CSV in memory for the /download-batch endpoint.
    """
    err = _check_model()
    if err:
        return err

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {}
    for col in df.columns:
        if col in ('reviewtext', 'review_text', 'text', 'review'):
            rename_map[col] = 'review_text'
        elif col in ('product_name', 'product', 'asin', 'productname'):
            rename_map[col] = 'product_name'
        elif col in ('overall', 'rating', 'stars'):
            rename_map[col] = 'rating'
        elif col in ('reviewtime', 'date', 'review_date', 'reviewdate'):
            rename_map[col] = 'date'
    df.rename(columns=rename_map, inplace=True)

    if 'review_text' not in df.columns:
        return jsonify({
            "error": "CSV must have a review text column named one of: "
                     "review_text, review, text, reviewText"
        }), 400

    if 'product_name' not in df.columns:
        df['product_name'] = 'Unknown Product'

    # Clean & predict
    df['cleaned_text'] = df['review_text'].astype(str).apply(clean_text)
    X                  = vectorizer.transform(df['cleaned_text'])
    probas             = model.predict_proba(X)
    classes            = model.classes_

    predicted_idx      = np.argmax(probas, axis=1)
    df['sentiment']    = [classes[i] for i in predicted_idx]
    df['confidence']   = [round(float(probas[r, predicted_idx[r]]), 4)
                          for r in range(len(df))]

    # Total counts
    total_counts = df['sentiment'].value_counts().to_dict()
    for label in ('positive', 'neutral', 'negative'):
        total_counts.setdefault(label, 0)

    # Per-product breakdown
    per_product = {}
    for product, group in df.groupby('product_name'):
        counts = group['sentiment'].value_counts().to_dict()
        for label in ('positive', 'neutral', 'negative'):
            counts.setdefault(label, 0)
        counts['total'] = int(group.shape[0])
        per_product[str(product)] = counts

    # Top negative keywords
    negative_reviews = df.loc[df['sentiment'] == 'negative', 'review_text'].tolist()
    top_keywords = format_keywords(extract_negative_keywords(negative_reviews))

    # Store enriched CSV in app state for download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    app.config['LAST_BATCH_CSV'] = csv_buffer.getvalue()

    return jsonify({
        "total_counts":         total_counts,
        "per_product_breakdown": per_product,
        "top_negative_keywords": top_keywords,
        "download_url":          "/download-batch",
        "rows_processed":        len(df),
    })


# ---------------------------------------------------------------------------
# GET /download-batch
# ---------------------------------------------------------------------------
@app.route("/download-batch", methods=["GET"])
def download_batch():
    """Returns the last batch-processed CSV as a downloadable file."""
    csv_data = app.config.get('LAST_BATCH_CSV')
    if not csv_data:
        return jsonify({"error": "No batch results available. Call /analyze-batch first."}), 404

    buffer = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='reviewguard_predictions.csv',
    )


# ---------------------------------------------------------------------------
# GET /negative-keywords
# ---------------------------------------------------------------------------
@app.route("/negative-keywords", methods=["GET"])
def negative_keywords():
    """
    Reads the last batch CSV (if available) or a standalone 'reviews_data'
    passed in the request and returns top-10 negative keywords.

    Query params:
      top_n (int, default 10)
    """
    err = _check_model()
    if err:
        return err

    top_n = int(request.args.get('top_n', 10))

    csv_data = app.config.get('LAST_BATCH_CSV')
    if not csv_data:
        return jsonify({"error": "No batch data available. Call /analyze-batch first."}), 404

    df = pd.read_csv(io.StringIO(csv_data))
    if 'sentiment' not in df.columns or 'review_text' not in df.columns:
        return jsonify({"error": "Batch data missing expected columns."}), 500

    negative_reviews = df.loc[df['sentiment'] == 'negative', 'review_text'].tolist()
    keywords = format_keywords(extract_negative_keywords(negative_reviews, top_n=top_n))

    return jsonify({
        "total_negative_reviews": len(negative_reviews),
        "top_keywords": keywords,
    })


# ---------------------------------------------------------------------------
# GET /sentiment-trend
# ---------------------------------------------------------------------------
@app.route("/sentiment-trend", methods=["GET"])
def sentiment_trend():
    """
    Groups reviews by month and returns the percentage of
    positive / neutral / negative reviews per month.

    Requires a CSV uploaded via /analyze-batch that includes a 'date' column,
    OR accepts a CSV file upload directly via multipart/form-data (field: 'file').
    """
    err = _check_model()
    if err:
        return err

    # Try multipart upload first, then fall back to last batch CSV
    if 'file' in request.files:
        try:
            df = pd.read_csv(request.files['file'])
        except Exception as e:
            return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

        # Normalise columns
        df.columns = df.columns.str.strip().str.lower()
        rename_map = {}
        for col in df.columns:
            if col in ('reviewtext', 'review_text', 'text', 'review'):
                rename_map[col] = 'review_text'
            elif col in ('reviewtime', 'date', 'review_date', 'reviewdate'):
                rename_map[col] = 'date'
            elif col in ('overall', 'rating', 'stars'):
                rename_map[col] = 'rating'
        df.rename(columns=rename_map, inplace=True)

        if 'review_text' not in df.columns:
            return jsonify({"error": "CSV must contain a review text column."}), 400
        if 'date' not in df.columns:
            return jsonify({"error": "CSV must contain a 'date' column for trend analysis."}), 400

        df['cleaned_text'] = df['review_text'].astype(str).apply(clean_text)
        X                  = vectorizer.transform(df['cleaned_text'])
        classes            = model.classes_
        predicted_idx      = np.argmax(model.predict_proba(X), axis=1)
        df['sentiment']    = [classes[i] for i in predicted_idx]

    else:
        csv_data = app.config.get('LAST_BATCH_CSV')
        if not csv_data:
            return jsonify({
                "error": "No data available. Either upload a CSV or call /analyze-batch first."
            }), 404
        df = pd.read_csv(io.StringIO(csv_data))

    if 'date' not in df.columns:
        return jsonify({
            "error": "No 'date' column found in the data. "
                     "Include a date column in your CSV for trend analysis."
        }), 400

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'sentiment'], inplace=True)

    if df.empty:
        return jsonify({"error": "No valid dated rows found after parsing."}), 400

    df['month'] = df['date'].dt.to_period('M').astype(str)

    trend = []
    for month, group in sorted(df.groupby('month')):
        total  = len(group)
        counts = group['sentiment'].value_counts().to_dict()
        trend.append({
            "month":    month,
            "total":    total,
            "positive": counts.get('positive', 0),
            "neutral":  counts.get('neutral',  0),
            "negative": counts.get('negative', 0),
            "positive_pct": round(counts.get('positive', 0) / total * 100, 1),
            "neutral_pct":  round(counts.get('neutral',  0) / total * 100, 1),
            "negative_pct": round(counts.get('negative', 0) / total * 100, 1),
        })

    return jsonify({"trend": trend})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
