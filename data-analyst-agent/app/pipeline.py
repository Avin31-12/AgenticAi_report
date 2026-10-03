"""
pipeline.py
-----------
One function that runs the whole flow for an uploaded CSV:
analyze -> charts -> narrative -> PDF -> result.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .analysis import analyze_csv
from .charts import build_charts
from .narrative import generate_narrative
from .pdf_report import build_pdf


def run_pipeline(csv_path, job_dir, job_id: str, question: str = "", filename: str | None = None) -> dict:
    job_dir = Path(job_dir)
    result = analyze_csv(csv_path, question)
    if filename:
        result["dataset"]["filename"] = filename

    charts = build_charts(result, job_dir / "charts")
    narrative = generate_narrative(result, question)
    build_pdf(result, narrative, charts, job_dir / "report.pdf", question)

    payload = {
        "job_id": job_id,
        "question": question,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf_url": f"/api/jobs/{job_id}/report.pdf",
        "result": result,
        "narrative": narrative,
    }
    (job_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload
