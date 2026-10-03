import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.pop("GROQ_API_KEY", None)  # tests must work without a key

from app.analysis import analyze_csv  # noqa: E402
from app.main import app  # noqa: E402
from app.narrative import fallback_narrative, generate_narrative  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_quarterly_sales.csv"
client = TestClient(app)


def test_analysis_finds_core_things():
    r = analyze_csv(SAMPLE)
    assert r["dataset"]["metric"] == "revenue"
    assert r["trend"] and len(r["trend"]["series"]) == 12
    assert r["breakdowns"] and r["breakdowns"][0]["items"]
    assert any(o["column"] == "revenue" for o in r["outliers"])
    json.dumps(r)  # must be JSON-safe


def test_question_can_change_the_metric():
    r = analyze_csv(SAMPLE, "How does marketing_spend change over time?")
    assert r["dataset"]["metric"] == "marketing_spend"


def test_works_on_a_file_with_no_sales_columns(tmp_path):
    p = tmp_path / "people.csv"
    pd.DataFrame({"name": list("abcdefghij") * 3, "city": ["X", "Y"] * 15, "age": range(20, 50)}).to_csv(p, index=False)
    r = analyze_csv(p)
    assert r["dataset"]["rows"] == 30
    assert r["insights"]


def test_currency_text_columns_become_numbers(tmp_path):
    p = tmp_path / "money.csv"
    pd.DataFrame({"amount": ["$1,200", "$300", "$450", "$99"] * 5, "who": ["a", "b"] * 10}).to_csv(p, index=False)
    r = analyze_csv(p)
    assert r["dataset"]["metric"] == "amount"


def test_fallback_narrative_without_api_key():
    r = analyze_csv(SAMPLE)
    n = generate_narrative(r, "what drives sales?")
    assert n["source"] == "fallback" and n["executive_summary"] and n["key_findings"]
    assert fallback_narrative(r)["recommendations"]


def test_full_upload_creates_dashboard_data_and_pdf():
    with open(SAMPLE, "rb") as f:
        res = client.post("/api/analyze", files={"file": ("sales.csv", f, "text/csv")}, data={"question": "trend?"})
    assert res.status_code == 200, res.text
    data = res.json()
    pdf = client.get(data["pdf_url"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    again = client.get(f"/api/jobs/{data['job_id']}")
    assert again.json()["result"]["dataset"]["rows"] == data["result"]["dataset"]["rows"]


def test_rejects_non_csv_and_empty_and_bad_ids():
    assert client.post("/api/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("x.csv", b"", "text/csv")}).status_code == 400
    assert client.get("/api/jobs/../../etc/passwd").status_code in (404, 422)
    assert client.get("/api/jobs/" + "z" * 32).status_code == 404


def test_homepage_is_served():
    assert "Data Analyst Agent" in client.get("/").text


def test_histogram_ignores_extreme_values():
    r = analyze_csv(SAMPLE)
    h = r["histogram"]
    assert h["excluded"] > 0
    # with the 3 huge rows trimmed, the data should spread over several bins
    assert sum(1 for b in h["bins"] if b["count"] > 0) >= 5


class _FakeClient:
    """Stands in for the Groq/OpenAI client so the AI path can be tested offline."""
    def __init__(self, content=None, error=None, **_):
        self._content, self._error = content, error
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        if self._error:
            raise self._error
        msg = type("M", (), {"content": self._content})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


def test_ai_narrative_is_used_when_groq_answers(monkeypatch):
    import openai
    good = json.dumps({
        "executive_summary": "Revenue is steady.",
        "key_findings": ["North leads.", "Outliers exist."],
        "recommendations": ["Review outliers."],
        "caveats": ["Small sample."],
    })
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient(content=good))
    n = generate_narrative(analyze_csv(SAMPLE), "q")
    assert n["source"] == "ai" and n["executive_summary"] == "Revenue is steady."


def test_falls_back_when_groq_fails_or_returns_junk(monkeypatch):
    import openai
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("app.narrative.time.sleep", lambda s: None)
    r = analyze_csv(SAMPLE)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient(error=RuntimeError("Error code: 429 rate limit")))
    n = generate_narrative(r)
    assert n["source"] == "fallback" and "rate limit" in n["caveats"][-1]
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient(content="not json"))
    assert generate_narrative(r)["source"] == "fallback"
