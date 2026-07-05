FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Flask default port (informational; Render assigns the real port via $PORT)
EXPOSE 5000

# Use gunicorn in production instead of Flask's dev server.
# Render injects the actual port via the $PORT env var at runtime, so we
# can't hardcode 5000 in an exec-form CMD -- shell form lets $PORT expand.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} backend.api:app
