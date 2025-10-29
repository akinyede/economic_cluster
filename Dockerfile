FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements_mvp.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default to production settings
ENV USE_SQLITE=false \
    FLASK_ENV=production \
    ENFORCE_CSRF=true

EXPOSE 5000

CMD ["python", "run_web.py"]

