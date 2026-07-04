FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Flask default port
EXPOSE 5000

# Use gunicorn in production instead of Flask's dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.api:app"]
