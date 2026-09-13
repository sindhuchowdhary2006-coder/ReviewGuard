# ReviewGuard — Deployment Guide (Render Free Tier)

## Project Structure

```
ReviewGuard/
├── app.py                  # Flask API (all routes)
├── data_preparation.py     # Text cleaning + TF-IDF
├── train_model.py          # Model training + evaluation
├── keyword_extractor.py    # Negative keyword analysis
├── wsgi.py                 # Gunicorn entry point
├── requirements.txt        # Python dependencies
├── render.yaml             # Render config (auto-deploy)
├── templates/
│   └── dashboard.html      # Frontend dashboard
└── data/
    └── sample_reviews.csv  # Created on first run
```

---

## Step 1 — Run Locally First

You have Anaconda installed. Use a conda environment:

```bash
conda create -n reviewguard python=3.11 -y
conda activate reviewguard
pip install -r requirements.txt
```

Download NLTK data once:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

Train the model (creates models/ folder with .joblib files):

```bash
python train_model.py
```

Start the app:

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Step 2 — Use Your Real Dataset

The app accepts any CSV with these columns (flexible naming):

| Column        | Accepted names                                      |
|---------------|-----------------------------------------------------|
| Review text   | `review_text`, `reviewText`, `text`, `review`       |
| Star rating   | `rating`, `overall`, `stars`                        |
| Date          | `date`, `reviewTime`, `review_date`                 |
| Product name  | `product_name`, `product`, `asin`                   |

Train on it:

```bash
python train_model.py data/your_amazon_reviews.csv
```

The `models/` folder will contain:
- `sentiment_model.joblib`
- `tfidf_vectorizer.joblib`

---

## Step 3 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial ReviewGuard commit"
git remote add origin https://github.com/YOUR_USERNAME/reviewguard.git
git push -u origin main
```

> The `.gitignore` excludes `models/` and `data/` by default.
> You have two options for the trained models:
>
> **Option A (recommended for free tier):** Remove `models/` from `.gitignore`,
> commit the .joblib files, so Render doesn't need to retrain on every deploy.
>
> **Option B:** Let Render retrain on each deploy using the build command in
> `render.yaml` (slower cold start, but keeps repo clean).

---

## Step 4 — Deploy to Render

1. Go to https://render.com and sign up / log in (free account).

2. Click **New → Web Service**.

3. Connect your GitHub account and select the `reviewguard` repository.

4. Render will auto-detect `render.yaml`. Confirm these settings:
   - **Environment:** Python
   - **Build Command:**
     ```
     pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" && python train_model.py data/sample_reviews.csv
     ```
   - **Start Command:**
     ```
     gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
     ```
   - **Instance Type:** Free

5. Click **Create Web Service**. Render builds and deploys automatically.

6. Your live URL will be something like:
   `https://reviewguard.onrender.com`

---

## Step 5 — Test the Live API

Single review prediction:

```bash
curl -X POST https://reviewguard.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Battery drains within 2 hours, very disappointed"}'
```

Expected response:

```json
{
  "sentiment": "negative",
  "confidence": 0.91,
  "probabilities": {
    "negative": 0.91,
    "neutral": 0.06,
    "positive": 0.03
  }
}
```

Health check:

```bash
curl https://reviewguard.onrender.com/health
```

---

## Free Tier Notes

- **Cold starts:** Free Render instances spin down after 15 min of inactivity.
  The first request after idle takes ~30 seconds to wake up. This is normal.
- **Disk:** Render's free tier has ephemeral storage — the `models/` folder is
  recreated on every deploy. Commit your .joblib files to avoid retraining on
  each deploy (Option A above).
- **Memory:** The free tier provides 512 MB RAM. The sample model is well within
  this. Large Amazon datasets (>100k reviews) may need the Starter plan.

---

## API Reference

| Method | Route               | Description                                      |
|--------|---------------------|--------------------------------------------------|
| GET    | `/`                 | Dashboard UI                                     |
| GET    | `/health`           | Health check                                     |
| POST   | `/predict`          | Single review → sentiment + confidence           |
| POST   | `/analyze-batch`    | CSV upload → bulk predictions + product breakdown |
| GET    | `/download-batch`   | Download last batch result as CSV                |
| GET    | `/negative-keywords`| Top complaint keywords from last batch           |
| GET    | `/sentiment-trend`  | Monthly sentiment % trend from last batch        |
