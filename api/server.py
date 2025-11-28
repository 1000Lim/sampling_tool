import os
import uuid
import time
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from consts import DataType, SAMPLING_DIR
from api.db import init_db, create_task, update_task, delete_task, get_task, list_tasks
from sampling_tool import main as run_sampling


class RunRequest(BaseModel):
    path: str = Field(..., description="Root path of the raw dataset to process")
    data_type: DataType = Field(DataType.SURF, description="Dataset type: surf, valeo, or aclm")
    stride: int = Field(10, ge=1, description="Sample every Nth LiDAR frame (1 = all)")
    cam_num: Optional[int] = Field(None, description="SURF only: reference camera number")
    export: Optional[str] = Field(None, description="Export directory (default: SAMPLING_DIR)")
    compress: bool = Field(False, description="Compress images and lidar to tar")
    delete: bool = Field(False, description="Delete images/lidar after compression")
    cam_info: Optional[str] = Field(None, description="SURF only: cam info file path")
    overlay_every: int = Field(0, ge=0, description="Generate WEBP overlay every N pairs (0=off)")
    overlay_intensity: bool = Field(False, description="Color overlay by intensity instead of distance")
    overlay_point_radius: int = Field(2, ge=1, description="Overlay point radius in pixels")
    overlay_alpha: float = Field(1.0, ge=0.0, le=1.0, description="Overlay alpha blending")
    skip_head: int = Field(0, ge=0, description="Skip N LiDAR frames from the beginning after sampling")
    skip_tail: int = Field(0, ge=0, description="Skip N LiDAR frames from the end after sampling")
    convert_raw_to_jpg: bool = Field(False, description="ACLM only: convert .raw to .jpg")
    raw_output_format: str = Field('gray', description="ACLM only: raw conversion format (gray or rgb)")
    raw_dgain: float = Field(1.5, ge=0.1, le=10.0, description="ACLM only: digital gain for RGB conversion")
    enable_multithreading: bool = Field(False, description="ACLM only: enable multithreading for raw conversion")
    num_workers: int = Field(10, ge=1, le=30, description="ACLM only: number of worker threads (default: 10)")


class JobStatus(BaseModel):
    id: str
    status: str
    started_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None
    export: Optional[str] = None
    progress: Optional[Dict] = None


app = FastAPI(title="Sampling Tool API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_jobs: Dict[str, JobStatus] = {}


@app.on_event("startup")
def on_startup():
    init_db()


# --- Simple filesystem browser (restricted roots) ---
class FsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


def _allowed_roots() -> List[Path]:
    roots: List[Path] = []
    # Prefer host home if provided via docker-compose (HOST_HOME=${HOME})
    host_home = os.environ.get("HOST_HOME")
    if host_home and os.path.isdir(host_home):
        roots.append(Path(host_home).resolve())
    # Container's own home (usually /root)
    container_home = os.path.expanduser("~")
    if os.path.isdir(container_home):
        roots.append(Path(container_home).resolve())
    # Common mount point
    if os.path.isdir("/mnt"):
        roots.append(Path("/mnt").resolve())
    # Optional extra roots (colon-separated)
    extra = os.environ.get("EXTRA_FS_ROOTS")
    if extra:
        for p in extra.split(":"):
            if p and os.path.isdir(p):
                roots.append(Path(p).resolve())
    # Deduplicate
    dedup: List[Path] = []
    for r in roots:
        if r not in dedup:
            dedup.append(r)
    return dedup


def _is_path_allowed(p: Path) -> bool:
    try:
        rp = p.resolve()
    except Exception:
        return False
    for root in _allowed_roots():
        try:
            rp.relative_to(root)
            return True
        except Exception:
            continue
    return False


@app.get("/fs/roots")
def fs_roots() -> List[FsEntry]:
    roots = _allowed_roots()
    return [FsEntry(name=str(r), path=str(r), is_dir=True) for r in roots]


@app.get("/fs/entries")
def fs_entries(path: Optional[str] = Query(None), only_dirs: bool = Query(True)):
    # If no path, default to first allowed root
    roots = _allowed_roots()
    if not roots:
        raise HTTPException(status_code=500, detail="No allowed roots configured")
    base = Path(path).expanduser() if path else roots[0]
    if not _is_path_allowed(base):
        raise HTTPException(status_code=400, detail="Path not allowed")
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail="Path not found or not a directory")
    entries: List[FsEntry] = []
    try:
        for child in sorted(base.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if only_dirs and not child.is_dir():
                continue
            entries.append(FsEntry(name=child.name, path=str(child.resolve()), is_dir=child.is_dir()))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    parent = str(base.parent.resolve()) if base.parent != base else None
    return {"cwd": str(base.resolve()), "parent": parent, "entries": [e.dict() for e in entries]}


@app.post("/fs/mkdir")
def fs_mkdir(parent_path: str, dir_name: str):
    """Create a new directory"""
    parent = Path(parent_path).expanduser()
    if not _is_path_allowed(parent):
        raise HTTPException(status_code=400, detail="Parent path not allowed")
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(status_code=404, detail="Parent directory not found")

    # Sanitize directory name
    dir_name = dir_name.strip()
    if not dir_name or '/' in dir_name or '\\' in dir_name or dir_name in ('.', '..'):
        raise HTTPException(status_code=400, detail="Invalid directory name")

    new_dir = parent / dir_name
    if new_dir.exists():
        raise HTTPException(status_code=400, detail="Directory already exists")

    try:
        new_dir.mkdir(parents=False, exist_ok=False)
        return {"ok": True, "path": str(new_dir.resolve())}
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create directory: {str(e)}")


@app.get("/files")
def serve_file(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    # Basic guard: only allow within allowed roots
    if not _is_path_allowed(p.parent):
        raise HTTPException(status_code=400, detail="Path not allowed")
    return FileResponse(str(p))


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_job(job_id: str, req: RunRequest):
    job = _jobs[job_id]
    try:
        export_dir = req.export or SAMPLING_DIR
        run_sampling(
            stride=req.stride,
            work_path=req.path,
            cam_num=req.cam_num,
            export=export_dir,
            compress=req.compress,
            remove=req.delete,
            cam_info=req.cam_info,
            data_type=req.data_type.value,
            overlay_every=req.overlay_every,
            overlay_intensity=req.overlay_intensity,
            overlay_point_radius=req.overlay_point_radius,
            overlay_alpha=req.overlay_alpha,
            skip_head=req.skip_head,
            skip_tail=req.skip_tail,
            convert_raw_to_jpg=req.convert_raw_to_jpg,
            raw_output_format=req.raw_output_format,
            raw_dgain=req.raw_dgain,
            enable_multithreading=req.enable_multithreading,
            num_workers=req.num_workers,
        )
        job.status = "completed"
        job.finished_at = time.time()
        job.export = export_dir
        update_task(job_id, status="completed", export_dir=export_dir, finished_at=job.finished_at)
    except Exception as e:
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(e)
        update_task(job_id, status="failed", error=job.error, finished_at=job.finished_at)


@app.post("/run", response_model=JobStatus)
def run(req: RunRequest, tasks: BackgroundTasks):
    # Basic validation for SURF
    if req.data_type == DataType.SURF and req.cam_num is None:
        raise HTTPException(status_code=400, detail="cam_num is required for SURF")
    job_id = str(uuid.uuid4())
    job = JobStatus(id=job_id, status="running", started_at=time.time(), export=req.export)
    _jobs[job_id] = job
    create_task(job_id, status="running", params=jsonable_encoder(req), created_at=job.started_at, export_dir=req.export)
    tasks.add_task(_run_job, job_id, req)
    return job


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        dbj = get_task(job_id)
        if dbj:
            import json as json_module
            progress = None
            if dbj.get("progress"):
                try:
                    progress = json_module.loads(dbj["progress"])
                except Exception:
                    pass
            return JobStatus(id=dbj["id"], status=dbj["status"], started_at=dbj["created_at"], finished_at=dbj.get("finished_at"), error=dbj.get("error"), export=dbj.get("export_dir"), progress=progress)
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/tasks")
def tasks_list():
    return list_tasks()


@app.post("/tasks/{task_id}/cancel")
def tasks_cancel(task_id: str):
    """Cancel a running task"""
    job = _jobs.get(task_id)
    if job and job.status == "running":
        job.status = "cancelled"
        job.finished_at = time.time()
        job.error = "Task cancelled by user"
        update_task(task_id, status="cancelled", error=job.error, finished_at=job.finished_at)
        return {"ok": True, "message": "Task cancellation requested"}

    # Check database
    dbj = get_task(task_id)
    if dbj and dbj["status"] == "running":
        update_task(task_id, status="cancelled", error="Task cancelled by user", finished_at=time.time())
        return {"ok": True, "message": "Task cancelled"}

    raise HTTPException(status_code=400, detail="Task is not running or does not exist")


@app.delete("/tasks/{task_id}")
def tasks_delete(task_id: str):
    """Delete a task (will cancel if running)"""
    job = _jobs.get(task_id)
    if job and job.status == "running":
        # Cancel first
        job.status = "cancelled"
        job.finished_at = time.time()
        job.error = "Task cancelled during deletion"
        update_task(task_id, status="cancelled", error=job.error, finished_at=job.finished_at)

    delete_task(task_id)
    _jobs.pop(task_id, None)
    return {"ok": True}



