"""
main.py - FastAPI app.

  POST /api/analyze            upload a CSV (+ optional question) -> analysis JSON + PDF link
  GET  /api/jobs/{id}          fetch a finished analysis again
  GET  /api/jobs/{id}/report.pdf
  GET  /                       the upload + dashboard page (static/index.html)

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import run_pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
JOBS_DIR = Path(os.environ.get("JOBS_DIR", ROOT / "jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
JOB_TTL_HOURS = int(os.environ.get("JOB_TTL_HOURS", "24"))
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

JOBS_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Data Analyst Agent")


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):  # blocks path traversal like ../../etc
        raise HTTPException(404, "Report not found.")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise HTTPException(404, "Report not found or expired.")
    return d


def _save_upload(upload: UploadFile, dest: Path) -> None:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "The uploaded file is empty.")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), question: str = Form("")):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    question = question.strip()[:300]

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    csv_path = job_dir / "data.csv"
    try:
        _save_upload(file, csv_path)
        return run_pipeline(csv_path, job_dir, job_id, question, filename=Path(file.filename).name)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except ValueError as exc:  # readable problems with the CSV itself
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, str(exc))
    except Exception:
        log.exception("Analysis failed")
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, "Something went wrong while analysing this file.")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    path = _job_dir(job_id) / "result.json"
    if not path.exists():
        raise HTTPException(404, "Report not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/report.pdf")
def get_pdf(job_id: str):
    path = _job_dir(job_id) / "report.pdf"
    if not path.exists():
        raise HTTPException(404, "Report not found.")
    return FileResponse(path, media_type="application/pdf", filename="analysis-report.pdf")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
