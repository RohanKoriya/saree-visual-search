FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow's webp support and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch to keep the image small and free-tier friendly.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# The index must already be built (see README "Setup") -- either bake it
# into the image by running build_index.py before `docker build`, or mount
# a volume at /app/index containing a pre-built index.faiss + metadata.json.

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
