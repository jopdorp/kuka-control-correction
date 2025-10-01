FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    samba-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir Flask==2.3.2 Werkzeug==2.3.6

# Copy application code
COPY kuka_file_upload_service.py .
COPY templates/ templates/
COPY static/ static/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["python3", "kuka_file_upload_service.py", "--kuka-ip", "krcpc", "--port", "8000"]
