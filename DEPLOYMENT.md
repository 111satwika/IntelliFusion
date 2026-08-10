# Deploying IntelliFusion with Docker

This runs the app as two containers: `app` (the FastAPI server +
webapp) and `ollama` (the local LLM, as a sidecar). Everything stays
on your own machine/network - no cloud API calls, same as running it
locally.

## 1. Prerequisites

- Docker + Docker Compose.
- Enough disk/RAM for a 7B-class local LLM (`qwen2.5:7b-instruct`,
  ~4.7 GB) plus a vision model (`llava`) and the embedding/OCR/
  transcription models this app already uses locally (MiniLM, CLIP,
  CLAP, EasyOCR, Whisper). No GPU is required - everything here runs
  on CPU, just slower than on GPU.

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and, at minimum, decide on `APP_PASSWORD` (see "Auth"
below) and `BIND_ADDRESS` (see "Network exposure" below). Everything
else has a working default.

## 3. Build and start

```bash
docker compose up -d --build
```

This starts `ollama` and `app` - `app` waits for Ollama's healthcheck
(`ollama list` succeeding, not just the container starting) before it
starts, so a slow-to-boot Ollama can't cause an early request to fail.
The app won't be able to *answer* anything yet, though - Ollama starts
with no models pulled.

## 4. Pull the models (one-time, after first start)

```bash
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec ollama ollama pull llava
```

These are stored in the `ollama-data` named volume, so you only need
to do this once - it survives container rebuilds/restarts.

## 5. Open it

`http://<this-machine's-IP>:8000` (or `http://localhost:8000` from
the same machine).

## Network exposure

`docker-compose.yml` publishes port 8000 on `${BIND_ADDRESS:-0.0.0.0}`
- by default that's **every network interface on the host**, i.e.
reachable from any device on your LAN, not just the host machine
itself. That's the point for most setups ("deploy so my other devices
can reach it"), but it also means if this host has any port-forwarding
or UPnP active on its router, port 8000 could become reachable from
the open internet without you doing anything else. If you only want
this reachable from the host machine (e.g. because a reverse proxy
running elsewhere on the host is the real entry point), set
`BIND_ADDRESS=127.0.0.1` in `.env`.

If you start it with no `APP_PASSWORD` at all, the app logs a warning
on startup (`docker compose logs app`) reminding you it has no login -
check that log line if you're ever unsure whether auth is active.

## Auth

If `APP_PASSWORD` is set in `.env`, every page and API route (except
the login page itself) requires signing in first with that password -
see `api/auth.py`. If it's left blank, the app has **no login at
all**, identical to running it locally today.

This is a single shared password for one trusted user/household, not
a real multi-user account system - no rate limiting, no lockout, no
per-user permissions. That's an intentional scope decision matching
this app's design (a personal RAG tool), not an oversight.

**If this will ever be reachable from outside your own LAN** (a public
IP, a cloud VM, a domain name pointed at it), put a real reverse proxy
in front of it with TLS - e.g. [Caddy](https://caddyserver.com/) (a
couple of lines of config for automatic HTTPS) or a
[Tailscale](https://tailscale.com/) network so it's never on the open
internet at all - and set `COOKIE_SECURE=1` in `.env` once you have
real HTTPS, so the login cookie is never sent over plain HTTP.

## Persistent data

`./data` on the host is bind-mounted into the container at `/app/data`
and holds everything that makes the app "yours": the Chroma vector
database, ingested media files, GitHub code graphs, and session
settings. `docker compose down && docker compose up -d` does **not**
lose this data. `docker compose down -v` would (it also deletes the
named volumes, including your pulled Ollama models) - only do that
intentionally.

## Updating

```bash
git pull
docker compose up -d --build
```

Your `./data` and the `ollama-data`/model-cache volumes are untouched
by a rebuild.

## Troubleshooting

- **Chat questions time out / never answer**: check the models were
  actually pulled (`docker compose exec ollama ollama list`).
- **First ingestion of an image/audio/video-heavy source is very
  slow**: EasyOCR/Whisper/CLIP/CLAP all download their weights on
  first use, into the `model-cache`/`easyocr-cache` volumes - this is
  a one-time cost per volume, not per container restart.
- **Website crawling with `render_js` fails**: confirms the Playwright
  Chromium install step in the `Dockerfile` completed; check
  `docker compose logs app` for the actual Playwright error.
