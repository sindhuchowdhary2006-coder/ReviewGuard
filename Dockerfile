# Use official Python slim image — has glibc matching PyPI manylinux wheels
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy rest of the app
COPY . .

# Train the model at build time so it's ready on startup
RUN python train_model.py

# Expose port
EXPOSE 10000

# Start with gunicorn
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120"]
