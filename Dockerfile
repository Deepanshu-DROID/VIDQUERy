FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render will set the PORT env var; this is just documentation
EXPOSE 10000

# Use gunicorn to serve the Flask app, binding to Render's PORT (default 10000 locally)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} app:app"]
