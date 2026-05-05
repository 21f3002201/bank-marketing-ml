FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal) and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create output/model directories to avoid permission issues
RUN mkdir -p /app/models /app/outputs /app/data

# Expose a port for the Flask API
EXPOSE 8080

ENV PORT=8080
# Run the Flask app via Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
