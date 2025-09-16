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

---

## Helper script: run sampling via Docker Compose

`run_sampling.sh` 예시 스크립트(옵션 래핑):

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT="sampling"
PATH_ARG=""
DATA_TYPE=""
STRIDE="1"
EXPORT_ARG=""
CAM_NUM=""
CAM_INFO=""
COMPRESS=0
DELETE=0
OVERLAY_EVERY="0"
OVERLAY_INTENSITY=0
OVERLAY_POINT_RADIUS="2"
OVERLAY_ALPHA="1.0"

usage(){
  echo "Usage: $0 -p <path> -t <surf|valeo> [-s N] [-e DIR] [--cam-num N --cam-info FILE] [--compress --delete] [--overlay-every N --overlay-intensity --overlay-point-radius N --overlay-alpha F] [--project NAME]"; exit 1;
}

while [[ $# -gt 0 ]]; do case "$1" in
  -p|--path) PATH_ARG="$2"; shift 2;;
  -t|--data-type) DATA_TYPE="$2"; shift 2;;
  -s|--stride) STRIDE="$2"; shift 2;;
  -e|--export) EXPORT_ARG="$2"; shift 2;;
  --cam-num) CAM_NUM="$2"; shift 2;;
  --cam-info) CAM_INFO="$2"; shift 2;;
  -z|--compress) COMPRESS=1; shift;;
  -d|--delete) DELETE=1; shift;;
  --overlay-every) OVERLAY_EVERY="$2"; shift 2;;
  --overlay-intensity) OVERLAY_INTENSITY=1; shift;;
  --overlay-point-radius) OVERLAY_POINT_RADIUS="$2"; shift 2;;
  --overlay-alpha) OVERLAY_ALPHA="$2"; shift 2;;
  --project) PROJECT="$2"; shift 2;;
  -h|--help) usage;;
  *) echo "Unknown arg: $1"; usage;;
esac; done

[[ -z "$PATH_ARG" || -z "$DATA_TYPE" ]] && usage

CMD=(docker compose -p "$PROJECT" run --rm sampling-api
  python sampling_tool.py
  --path "$PATH_ARG" --data-type "$DATA_TYPE" --stride "$STRIDE"
  --overlay-every "$OVERLAY_EVERY" --overlay-point-radius "$OVERLAY_POINT_RADIUS" --overlay-alpha "$OVERLAY_ALPHA")

[[ -n "$EXPORT_ARG" ]] && CMD+=(--export "$EXPORT_ARG")
[[ "$OVERLAY_INTENSITY" -eq 1 ]] && CMD+=(--overlay-intensity)
[[ "$COMPRESS" -eq 1 ]] && CMD+=(--compress)
[[ "$DELETE" -eq 1 ]] && CMD+=(--delete)
[[ "$DATA_TYPE" == "surf" && -n "$CAM_NUM" ]] && CMD+=(--cam-num "$CAM_NUM")
[[ "$DATA_TYPE" == "surf" && -n "$CAM_INFO" ]] && CMD+=(--cam-info "$CAM_INFO")

echo "${CMD[@]}"
exec "${CMD[@]}"
```

사용 예시:

```bash
# VALEO
bash run_sampling.sh -p /mnt/valeo/20250915_172030 -t valeo -s 10 --overlay-every 5 -e $HOME/valeo_out

# SURF (cam-num, cam-info 필수)
bash run_sampling.sh -p /mnt/surf/2025-09-15 -t surf -s 10 --cam-num 5 --cam-info $HOME/calib/cam_info.json -z -e $HOME/surf_out

# 실행 권한 부여
chmod +x run_sampling.sh
```
