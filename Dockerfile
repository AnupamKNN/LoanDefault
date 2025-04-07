# Use official Python image as base
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Flask port
EXPOSE 8000

# Run Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
