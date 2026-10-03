"""
analysis.py
-----------
Deterministic CSV analysis. Pure pandas/numpy: no LLM calls and no exec().

analyze_csv() returns ONE JSON-safe dict. The charts, the PDF, the AI
narrative and the web dashboard all read from that same dict, so the numbers
can never disagree between the PDF and the dashboard.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MAX_ROWS = 200_000
MAX_COLUMNS = 60

# Column-name hints used to guess the "main" numeric column (earlier = stronger).
METRIC_HINTS = [
    "revenue", "sales", "amount", "profit", "income", "total",
    "price", "cost", "spend", "value", "units", "quantity", "qty",
]
# Metrics where averaging makes more sense than summing.
MEAN_HINTS = ("price", "rate", "avg", "average", "score", "age", "ratio", "percent", "pct", "temp")

_NUM_STRIP = re.compile(r"[,\$€£₹%\s]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fmt_num(x) -> str:
    """Compact human-friendly number: 1234567 -> 1.23M."""
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x) or math.isinf(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e4:
        return f"{x / 1e3:.1f}K"
    if x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.2f}"


def sanitize(obj):
    """Make any nested structure JSON-safe (numpy types, NaN, Timestamps)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


# ---------------------------------------------------------------------------
# Loading + cleaning
# ---------------------------------------------------------------------------
def load_csv(path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, nrows=MAX_ROWS, encoding="utf-8-sig", encoding_errors="replace")
        if df.shape[1] == 1:  # maybe a ; or tab separated file
            try:
                alt = pd.read_csv(
                    path, nrows=MAX_ROWS, sep=None, engine="python",
                    encoding="utf-8-sig", encoding_errors="replace",
                )
                if alt.shape[1] > 1:
                    df = alt
            except Exception:
                pass
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}")
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    df = df.dropna(how="all").dropna(axis=1, how="all")

    for col in list(df.columns):
        s = df[col]
        if (
            pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue

        # "$1,234" / "12%" / "1 000" -> numbers
        stripped = non_null.astype(str).str.replace(_NUM_STRIP, "", regex=True)
        if pd.to_numeric(stripped, errors="coerce").notna().mean() >= 0.95:
            df[col] = pd.to_numeric(
                s.astype(str).str.replace(_NUM_STRIP, "", regex=True), errors="coerce"
            )
            continue

        # date-looking text -> datetime
        sample = non_null.astype(str).head(200)
        if sample.str.contains(r"\d").mean() >= 0.9 and _to_datetime(sample).notna().mean() >= 0.9:
            df[col] = _to_datetime(s)
    return df


def _kind(s: pd.Series, n_rows: int) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nun = s.nunique(dropna=True)
    if nun <= 30 or (nun <= 100 and nun / max(n_rows, 1) <= 0.2):
        return "categorical"
    return "text"


def _is_id_like(name: str, s: pd.Series, n_rows: int) -> bool:
    low = name.lower().strip()
    if low in {"id", "index", "idx", "row", "rowid", "row_id"} or low.endswith(("_id", " id")):
        return True
    return bool(
        pd.api.types.is_integer_dtype(s)
        and n_rows > 20
        and s.nunique() == n_rows
        and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    )


def _norm(text: str) -> str:
    return re.sub(r"[_\s]+", " ", text.lower()).strip()


def _pick_metric(numeric: list[str], question: str) -> str | None:
    if not numeric:
        return None
    q = _norm(question)
    for c in numeric:  # a column named in the question wins
        if _norm(c) and _norm(c) in q:
            return c
    for hint in METRIC_HINTS:
        for c in numeric:
            if hint in c.lower():
                return c
    return numeric[0]


def _agg_for(metric: str | None) -> str:
    if metric is None:
        return "count"
    return "mean" if any(h in metric.lower() for h in MEAN_HINTS) else "sum"


def _agg_word(agg: str) -> str:
    return {"sum": "Total", "mean": "Average", "count": "Rows"}[agg]


# ---------------------------------------------------------------------------
# Analysis pieces
# ---------------------------------------------------------------------------
def _profile(df, kinds, id_like) -> list[dict]:
    n = max(len(df), 1)
    out = []
    for c in list(df.columns)[:MAX_COLUMNS]:
        s = df[c]
        miss = int(s.isna().sum())
        item = {
            "name": c,
            "kind": "id" if c in id_like else kinds[c],
            "missing": miss,
            "missing_pct": round(100 * miss / n, 1),
            "unique": int(s.nunique(dropna=True)),
        }
        k = kinds[c]
        if k == "numeric" and s.notna().any():
            item.update(min=s.min(), max=s.max(), mean=s.mean(), median=s.median())
        elif k == "datetime" and s.notna().any():
            item.update(min=s.min().date().isoformat(), max=s.max().date().isoformat())
        elif k == "categorical":
            item["top"] = [str(i) for i in s.value_counts().head(3).index]
        out.append(item)
    return out


def _trend(df, date_col, metric, agg):
    cols = [date_col] + ([metric] if metric else [])
    d = df[cols].dropna(subset=[date_col])
    if len(d) < 3:
        return None
    span = (d[date_col].max() - d[date_col].min()).days
    if span >= 120:
        freq, unit, pfmt = "MS", "month", "%Y-%m"
    elif span >= 21:
        freq, unit, pfmt = "W", "week", "%Y-%m-%d"
    else:
        freq, unit, pfmt = "D", "day", "%Y-%m-%d"

    g = d.set_index(date_col)
    s = g.resample(freq).size() if metric is None else g[metric].resample(freq).agg(agg)
    s = s.dropna()
    if len(s) < 2:
        return None

    values = s.to_numpy(dtype=float)
    periods = [i.strftime(pfmt) for i in s.index]
    mean = float(np.mean(np.abs(values))) or 1e-9
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) * len(values) / mean
    direction = "upward" if slope > 0.1 else "downward" if slope < -0.1 else "flat"
    first, last = float(values[0]), float(values[-1])
    title_metric = metric or "rows"
    return {
        "title": f"{_agg_word(agg)} {title_metric} per {unit}" if metric else f"Rows per {unit}",
        "unit": unit,
        "series": [{"period": p, "value": float(v)} for p, v in zip(periods, values)],
        "direction": direction,
        "first": {"period": periods[0], "value": first},
        "last": {"period": periods[-1], "value": last},
        "change_pct": ((last - first) / first * 100) if first else None,
        "peak": {"period": periods[int(values.argmax())], "value": float(values.max())},
        "low": {"period": periods[int(values.argmin())], "value": float(values.min())},
    }


def _breakdowns(df, cats, metric, agg, limit=3):
    out = []
    candidates = [c for c in cats if 2 <= df[c].nunique(dropna=True) <= 25]
    candidates.sort(key=lambda c: df[c].nunique(dropna=True))
    for c in candidates[:limit]:
        if metric is None:
            g = df.groupby(c, observed=True).size()
        else:
            g = df.groupby(c, observed=True)[metric].agg(agg)
        g = g.dropna().sort_values(ascending=False)
        if g.empty:
            continue
        total = float(g.sum())
        out.append({
            "column": c,
            "metric": metric,
            "agg": agg,
            "title": f"Rows by {c}" if metric is None else f"{_agg_word(agg)} {metric} by {c}",
            "groups": int(len(g)),
            "items": [
                {
                    "label": str(label),
                    "value": float(v),
                    "share": (float(v) / total) if (agg != "mean" and total) else None,
                }
                for label, v in g.head(10).items()
            ],
        })
    return out


def _histogram(df, metric):
    """Histogram of the middle 98% of values, so a few extreme rows don't flatten it."""
    if metric is None:
        return None
    vals = df[metric].dropna().to_numpy(dtype=float)
    if len(vals) < 8 or np.ptp(vals) == 0:
        return None
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    inside = vals[(vals >= lo) & (vals <= hi)]
    bins = int(min(12, max(5, math.sqrt(len(inside)))))
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    trimmed = len(vals) - len(inside)
    return {
        "title": f"Distribution of {metric}" + (" (middle 98% of rows)" if trimmed else ""),
        "metric": metric,
        "excluded": int(trimmed),
        "bins": [
            {"label": f"{fmt_num(edges[i])} to {fmt_num(edges[i + 1])}", "count": int(counts[i])}
            for i in range(len(counts))
        ],
    }


def _correlations(df, numeric):
    cols = numeric[:25]
    if len(cols) < 2:
        return [], None
    corr = df[cols].corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    top = [
        {
            "a": a, "b": b, "r": round(r, 3),
            "strength": "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak",
            "direction": "positive" if r > 0 else "negative",
        }
        for a, b, r in pairs[:6]
    ]
    scatter = None
    if top and abs(top[0]["r"]) >= 0.3:
        a, b = top[0]["a"], top[0]["b"]
        pts = df[[a, b]].dropna()
        pts = pts.sample(min(300, len(pts)), random_state=0)
        scatter = {
            "title": f"{b} vs {a}",
            "x": a, "y": b, "r": top[0]["r"],
            "points": [{"x": float(x), "y": float(y)} for x, y in zip(pts[a], pts[b])],
        }
    return top, scatter


def _outliers(df, numeric):
    out = []
    for c in numeric[:12]:
        s = df[c].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        count = int(mask.sum())
        if not count:
            continue
        ext = s[mask].to_numpy(dtype=float)
        order = np.argsort(-np.abs(ext - float(s.median())))
        out.append({
            "column": c, "count": count,
            "pct": round(100 * count / len(s), 1),
            "lower": float(lo), "upper": float(hi),
            "examples": [float(v) for v in ext[order][:3]],
        })
    out.sort(key=lambda o: o["count"], reverse=True)
    return out[:5]


def _quality(df):
    n = max(len(df), 1)
    miss = df.isna().sum()
    missing = [
        {"column": c, "missing": int(v), "missing_pct": round(100 * v / n, 1)}
        for c, v in miss.sort_values(ascending=False).items() if v > 0
    ][:8]
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing,
        "high_missing_columns": [m["column"] for m in missing if m["missing_pct"] > 20],
    }


def _insights(n, m, metric, agg, trend, breakdowns, corr, outliers, quality) -> list[str]:
    out = [
        f"The file has {n:,} rows and {m} columns"
        + (f"; the analysis focuses on '{metric}'." if metric else "; no numeric column was found, so rows are counted.")
    ]
    if trend:
        t = trend
        change = f" ({t['change_pct']:+.1f}%)" if t["change_pct"] is not None else ""
        phrase = {"upward": "is trending upward", "downward": "is trending downward", "flat": "is roughly flat"}[t["direction"]]
        out.append(
            f"{trend['title']} {phrase}: {fmt_num(t['first']['value'])} in {t['first']['period']} "
            f"to {fmt_num(t['last']['value'])} in {t['last']['period']}{change}. "
            f"The peak was {fmt_num(t['peak']['value'])} in {t['peak']['period']}."
        )
    for b in breakdowns[:2]:
        top = b["items"][0]
        if top["share"] is not None:
            out.append(f"'{top['label']}' leads {b['column']} with {top['share']:.0%} of the total.")
        else:
            out.append(f"'{top['label']}' has the highest value for {b['column']} ({fmt_num(top['value'])}).")
    if corr:
        c = corr[0]
        out.append(f"{c['a']} and {c['b']} show a {c['strength']} {c['direction']} correlation (r = {c['r']:.2f}).")
    for o in outliers[:2]:
        out.append(
            f"'{o['column']}' has {o['count']} unusual values ({o['pct']}% of rows) "
            f"outside {fmt_num(o['lower'])} to {fmt_num(o['upper'])}."
        )
    if quality["duplicate_rows"]:
        out.append(f"{quality['duplicate_rows']:,} duplicate rows were found.")
    if quality["high_missing_columns"]:
        out.append("Columns with more than 20% missing values: " + ", ".join(quality["high_missing_columns"][:5]) + ".")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_csv(path, question: str = "") -> dict:
    df = load_csv(path)
    n, m = len(df), df.shape[1]
    if n == 0 or m == 0:
        raise ValueError("The CSV has no usable rows or columns.")

    kinds = {c: _kind(df[c], n) for c in df.columns}
    id_like = {c for c in df.columns if kinds[c] == "numeric" and _is_id_like(c, df[c], n)}
    numeric = [c for c in df.columns if kinds[c] == "numeric" and c not in id_like]
    cats = [c for c in df.columns if kinds[c] == "categorical"]
    dates = [c for c in df.columns if kinds[c] == "datetime"]

    metric = _pick_metric(numeric, question or "")
    agg = _agg_for(metric)
    date_col = max(dates, key=lambda c: df[c].notna().sum()) if dates else None

    trend = _trend(df, date_col, metric, agg) if date_col else None
    breakdowns = _breakdowns(df, cats, metric, agg)
    histogram = _histogram(df, metric)
    corr, scatter = _correlations(df, numeric)
    outliers = _outliers(df, numeric)
    quality = _quality(df)

    kpis = {"rows": n, "columns": m, "metric": metric, "agg": agg}
    cards = [{"label": "Rows", "value": f"{n:,}"}, {"label": "Columns", "value": str(m)}]
    if metric:
        s = df[metric].dropna()
        kpis.update(total=s.sum(), mean=s.mean(), median=s.median(), max=s.max(), min=s.min())
        main = s.sum() if agg == "sum" else s.mean()
        cards += [
            {"label": f"{_agg_word(agg)} {metric}", "value": fmt_num(main)},
            {"label": f"Median {metric}", "value": fmt_num(s.median())},
            {"label": f"Max {metric}", "value": fmt_num(s.max())},
        ]
    if date_col:
        lo, hi = df[date_col].min(), df[date_col].max()
        kpis.update(date_from=lo.date().isoformat(), date_to=hi.date().isoformat())
        long_span = (hi - lo).days >= 60
        f = "%b %Y" if long_span else "%d %b %Y"
        cards.append({"label": "Date range", "value": f"{lo:{f}} to {hi:{f}}"})

    result = {
        "dataset": {
            "filename": Path(path).name, "rows": n, "columns": m,
            "metric": metric, "metric_agg": agg, "date_column": date_col,
        },
        "kpis": kpis,
        "kpi_cards": cards[:6],
        "trend": trend,
        "breakdowns": breakdowns,
        "histogram": histogram,
        "correlations": corr,
        "scatter": scatter,
        "outliers": outliers,
        "quality": quality,
        "columns": _profile(df, kinds, id_like),
        "insights": _insights(n, m, metric, agg, trend, breakdowns, corr, outliers, quality),
    }
    return sanitize(result)
