---
title: Cosmic Observer
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: Real-time space situational awareness dashboard
---

# Cosmic Observer

Real-time space situational awareness dashboard: live satellite tracking
(SGP4), conjunction / collision-probability screening, launch-window and
COLA analysis, NEO risk lists, space weather, and a computer-vision module
(YOLOv8 + OWL-ViT) running on NASA imagery and the live ISS feed.

Built with FastAPI, NumPy/SciPy, Astropy, PyTorch and OpenCV.

## Run locally

```bash
cp .env.example .env   # add your own NASA_API_KEY (optional)
docker compose up --build
# → http://localhost:8000
```

The hosted demo runs with `PUBLIC_DEMO=1` (read-only: video upload and
alert acknowledgement are disabled).
