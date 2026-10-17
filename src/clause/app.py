"""Clause — contract review. Web UI over the pipeline.

    uv run python -m clause.app --port 8820
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from . import report as report_mod
from .corpus import Cuad, Edgar
from .pipeline import EXPECTED_BY_TYPE, review
from .web import App, Raw

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def build_app(cuad: Cuad, edgar: Edgar) -> App:
    app = App(UI_DIR, title="clause")
    benchmark_path = Path("C:/d drive data/clause-120k/benchmark_cuad.json")
    sweep_path = Path("C:/d drive data/clause-120k/sweep_retrieval.json")
    # The last review is kept so a report can be exported without re-reviewing a
    # 15,000-word contract on every download click.
    last: dict = {}

    @app.route("/api/stats")
    def _stats(_payload: dict) -> dict:
        return {
            "cuad": cuad.stats,
            "edgar": edgar.stats,
            "contract_types_known": sorted(k for k in EXPECTED_BY_TYPE if not k.startswith("_")),
        }

    @app.route("/api/benchmark")
    def _benchmark(_payload: dict) -> dict:
        out: dict = {}
        if benchmark_path.exists():
            out["cuad"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if sweep_path.exists():
            out["retrieval_sweep"] = json.loads(sweep_path.read_text(encoding="utf-8"))
        return out or {"note": "No benchmark yet. Run: python -m clause.evaluate --contracts 25"}

    @app.route("/api/sample")
    def _sample(payload: dict) -> dict:
        if not edgar.manifest:
            return {"error": "no contracts fetched yet - run scripts/fetch_edgar.py"}
        wanted = (payload.get("type") or "").lower()
        pool = [m for m in edgar.manifest if not wanted or m["query"] == wanted] or edgar.manifest
        entry = random.choice(pool)
        return {
            "contract_type": entry["query"],
            "text": edgar.text(entry),
            "words": entry["words"],
            "source_url": entry["url"],
            "doc_id": entry["doc_id"],
        }

    @app.route("/api/review")
    def _review(payload: dict) -> dict:
        text = payload.get("text") or ""
        if len(text.split()) < 50:
            return {"error": "That looks too short to be a contract (need at least 50 words)."}
        result = review(text, cuad, contract_type=payload.get("contract_type", ""))
        result["name"] = payload.get("name", "")
        last.clear()
        last.update(result)
        return result

    def _report(payload: dict, kind: str) -> Raw:
        data = last if last else None
        if data is None:
            return Raw(b"Run a review first.", "text/plain; charset=utf-8")
        name = payload.get("name") or data.get("contract_type") or "review"
        stem = SAFE_NAME.sub("-", f"clause-review-{name}").strip("-")[:60]
        agree = int(payload.get("min_agreement", 2))
        if kind == "pdf":
            return Raw(report_mod.pdf(data, name=name, min_agreement=agree),
                       "application/pdf", f"{stem}.pdf")
        if kind == "html":
            return Raw(report_mod.html(data, name=name, min_agreement=agree).encode("utf-8"),
                       "text/html; charset=utf-8", f"{stem}.html")
        return Raw(report_mod.markdown(data, name=name, min_agreement=agree).encode("utf-8"),
                   "text/markdown; charset=utf-8", f"{stem}.md")

    @app.route("/api/report.pdf")
    def _pdf(payload: dict) -> Raw:
        return _report(payload, "pdf")

    @app.route("/api/report.md")
    def _md(payload: dict) -> Raw:
        return _report(payload, "md")

    @app.route("/api/report.html")
    def _html(payload: dict) -> Raw:
        return _report(payload, "html")

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Clause review desk")
    ap.add_argument("--port", type=int, default=8820)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    cuad, edgar = Cuad(), Edgar()
    print(f"CUAD: {len(cuad.exemplars)} exemplar spans, {len(cuad.holdout)} held-out contracts", flush=True)
    print(f"EDGAR: {len(edgar.manifest)} contracts", flush=True)
    build_app(cuad, edgar).serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
