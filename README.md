## Sampling Tool (CLI + API + Web)

End-to-end sampling pipeline with a Typer CLI, FastAPI backend, and Next.js web UI. Supports SURF and VALEO datasets, optional overlay generation, and Dockerized development.

### Features
- SURF and VALEO pipelines with shared utilities
- Stride-based sampling (ratio → stride)
- Optional WEBP overlay generation (distance/intensity, alpha, radius)
- Typer CLI, FastAPI API, Next.js web frontend
- Task DB (SQLite) with create/list/delete, live status, results viewer
- Docker/Docker Compose with dynamic `${HOME}` mount and `/mnt` read-only

---

## Quick Start (Docker Compose)

```bash
# From repo root
docker compose -p sampling up -d --build

# Web UI
#   http://localhost:3050  (Tasks table is the landing page)
# API
#   http://localhost:8000
#   http://localhost:8000/docs (OpenAPI)
```

Volumes (already configured in compose):
- `${PWD}:/app` (source live mount)
- `/mnt:/mnt:ro`
- `${HOME}:${HOME}` (host home mounted into container)

The web UI has a file picker with allowed roots: `${HOME}`, `/mnt` and `EXTRA_FS_ROOTS` if provided. Hidden entries (dotfiles) are filtered out.

---

## CLI Usage (Typer)

```bash
# Help
python sampling_tool.py --help

# Run (current version: no subcommand)
python sampling_tool.py \
  --path /path/to/dataset \
  --data-type [surf|valeo] \
  --stride 10 \
  --compress \
  --delete \
  --export /path/to/export \
  --overlay-every 5 \
  --overlay-intensity \
  --overlay-point-radius 2 \
  --overlay-alpha 1.0 \
  [SURF only] --cam-num 5 \
  [SURF only] --cam-info /path/to/cam_info.json

# If your local CLI shows a `run` command in --help, use this instead:
# python sampling_tool.py run [same options as above]
```

Notes:
- `stride` must be ≥ 1. `ratio` is deprecated.
- For SURF, `--cam-num` and `--cam-info` are required.
- If `--export` is omitted, defaults to `consts.SAMPLING_DIR`.

---

## API (FastAPI)

Key endpoints:
- `POST /run` → start a job (body mirrors CLI options). Returns `{id, status,...}`
- `GET /jobs/{id}` → job status (completed/failed/running, export path, timestamps)
- `GET /tasks` → task list (SQLite)
- `DELETE /tasks/{id}` → remove task from DB
- `GET /fs/roots` → allowed root directories
- `GET /fs/entries?path=...&only_dirs=true|false` → list directory entries
- `GET /files?path=...` → stream a file from allowed roots
- `GET /health`

Security note: file access is restricted to allowed roots.

---

## Web (Next.js)

- Landing page: `/tasks` (task table)
- Create task: `/tasks/create`
  - SURF requires Camera number and Camera info file
  - Path/Export pickers select directories; Camera info picker selects files
- Task detail: `/tasks/[id]`
  - Live status polling; when completed, results are shown inline
- Results view: `/results/[id]?export=/abs/export/dir`
  - Scans export dir, shows only pairs with overlay present (JPG/PNG + WEBP)
  - Pagination controls (page size selectable)

Environment:
- Web port: 3050
- API port: 8000

---

## Local Development (without Docker)

Backend:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd web
npm install
npm run dev -- -p 3050
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` if you run web outside compose.

---

## Troubleshooting

- 500 on POST /run with enums: fixed by using FastAPI `jsonable_encoder` (already in code).
- OpenCV/numpy build issues: versions are relaxed in `requirements.txt`.
- Debian `libgl1-mesa-glx` missing: using `libgl1` in Docker base image.
- Permissions: ensure export path is writable within the mounted `${HOME}`.

---

## License

Internal project. All rights reserved.
