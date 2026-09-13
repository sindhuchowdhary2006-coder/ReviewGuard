"""
WSGI entry point for Gunicorn / Render deployment.
"""
from app import app

if __name__ == "__main__":
    app.run()
