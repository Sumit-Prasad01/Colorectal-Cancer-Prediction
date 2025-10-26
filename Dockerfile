FROM python:3.10-slim

WORKDIR /app
COPY . /app

#  Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install your package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 5000
CMD ["python", "application.py"]
