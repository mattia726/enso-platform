# Prompt: INFRA agent — GCP deploy + reproducibility

Goal:
Provide a deploy path for a GPU VM (GCP L4) and a clean demo endpoint.

Tasks:
- Add `docker/Dockerfile.backend` (optional)
- Add a `scripts/gcp_setup.sh` with:
  - python env setup
  - install deps
  - run `uvicorn` with proper host/port
  - recommend `tmux` usage
- Add `.env.example` and config loading
- Add guidance for CORS and reverse proxy

Deliverables:
- docs/deploy_gcp.md
- scripts and configs
