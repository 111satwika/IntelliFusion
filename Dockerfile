# IntelliFusion - see DEPLOYMENT.md for how to build/run this via
# docker compose (the app also needs a reachable Ollama instance -
# docker-compose.yml runs one as a sidecar container).

FROM python:3.12-slim

# libgl1/libglib2.0-0: needed at *runtime* (not just build) by
# opencv-python-headless and EasyOCR, which both link against them
# even though the pip wheels are "headless".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

# Created up front so a fresh checkout with nothing ingested yet still
# starts cleanly (server.py/store.py also create these lazily, but the
# bind-mounted ./data volume from docker-compose.yml may be empty on
# first run).
RUN mkdir -p data/chroma_db data/media/frames data/media/sources data/graphs data/eval/reports

EXPOSE 8000

# Single worker, deliberately: every heavy model (MiniLM/CLIP/CLAP/
# Whisper/EasyOCR) is a lazy-loaded singleton per process, so multiple
# workers would multiply memory several times over for a single-user
# tool with no throughput benefit.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
