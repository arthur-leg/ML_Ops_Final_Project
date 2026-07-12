FROM python:3.12-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY services/ ./services/
COPY train.py promote.py ./

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:${PORT:-5000} backend.api:app
