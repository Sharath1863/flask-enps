# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variables
ENV TRANSFORMERS_CACHE=/models
ENV FLASK_DEBUG=false
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /app

# Copy only necessary files to leverage Docker caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get autoremove -y build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the model during build to avoid runtime delay
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')"

# Copy the application code
COPY app.py .

# Expose port for Flask
EXPOSE 5000

# Run the app with Gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
