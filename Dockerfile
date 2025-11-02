FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY preprocess.py predict.py run.sh ./
COPY model.pkl features.json threshold.txt ./

# Ensure run script is executable
RUN chmod +x run.sh

# Create input and output directories expected for mounting
RUN mkdir -p /app/input /app/output /app/work

# Default command
ENTRYPOINT ["/bin/bash", "run.sh"]