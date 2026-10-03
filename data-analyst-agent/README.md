# Data Analyst Agent

Upload a CSV, get a dashboard, a written summary and a downloadable PDF.

## Run it (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    copy .env.example .env        # then put your Groq key in .env
    uvicorn app.main:app --reload

Open http://127.0.0.1:8000 and upload `sample_data/sample_quarterly_sales.csv`.
Without a Groq key the app still works; the summary is written automatically
from the statistics instead of by the AI.

## Run the tests

    pip install -r requirements-dev.txt
    python -m pytest

## How it works

    CSV upload -> analysis.py (pandas, no AI) -> charts.py (PNG for the PDF)
               -> narrative.py (one Groq call, fallback if it fails)
               -> pdf_report.py (PDF) -> JSON for the dashboard (static/index.html)

* `analysis.py`  all numbers: KPIs, trend, breakdowns, outliers, correlations, data quality
* `narrative.py` the only place the AI is used; it sees aggregate facts, never raw rows
* `main.py`      upload endpoint, size limit, job folders, auto-cleanup after 24h
* Reports live in `jobs/<id>/` and are deleted after `JOB_TTL_HOURS`
