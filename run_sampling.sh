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

usage() {
  echo "Usage: $0 -p <path> -t <surf|valeo> [-s N] [-e <export_dir>] [--cam-num N --cam-info FILE] [--compress --delete] [--overlay-every N --overlay-intensity --overlay-point-radius N --overlay-alpha F] [--project NAME]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
  esac
done

[[ -z "$PATH_ARG" || -z "$DATA_TYPE" ]] && usage

CMD=(docker compose -p "$PROJECT" run --rm sampling-api
  python sampling_tool.py
  --path "$PATH_ARG"
  --data-type "$DATA_TYPE"
  --stride "$STRIDE"
  --overlay-every "$OVERLAY_EVERY"
  --overlay-point-radius "$OVERLAY_POINT_RADIUS"
  --overlay-alpha "$OVERLAY_ALPHA"
)

[[ -n "$EXPORT_ARG" ]] && CMD+=(--export "$EXPORT_ARG")
[[ "$OVERLAY_INTENSITY" -eq 1 ]] && CMD+=(--overlay-intensity)
[[ "$COMPRESS" -eq 1 ]] && CMD+=(--compress)
[[ "$DELETE" -eq 1 ]] && CMD+=(--delete)
# SURF 전용
[[ "$DATA_TYPE" == "surf" && -n "$CAM_NUM" ]] && CMD+=(--cam-num "$CAM_NUM")
[[ "$DATA_TYPE" == "surf" && -n "$CAM_INFO" ]] && CMD+=(--cam-info "$CAM_INFO")

echo "${CMD[@]}"
exec "${CMD[@]}"