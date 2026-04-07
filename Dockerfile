FROM python:3.11-slim

WORKDIR /app

# System deps for Playwright + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates \
        libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libdbus-1-3 libexpat1 libxcb1 libxkbcommon0 \
        libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 \
        libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium
RUN playwright install chromium --with-deps

# Pre-download the embedding model so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Persist data outside the container
VOLUME ["/app/data"]

ENV DATA_DIR=/app/data

ENTRYPOINT ["python", "main.py"]
CMD ["--subreddits", "wallstreetbets", "investing", "technology", "--sort", "hot", "--limit", "25"]
