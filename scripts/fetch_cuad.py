"""Fetch CUAD's expert-annotated clauses as labelling exemplars.

CUAD (510 contracts, 13,101 clauses, 41 labels, CC BY 4.0) is the incumbent this
dataset is measured against, and its expert spans are the exemplars for the
retrieval pass. Pulled through the public datasets-server rows API, which serves
JSON in small pages - the right shape for a throttled connection.

    python scripts/fetch_cuad.py --out "C:/d drive data/clause-120k/raw/cuad"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

# Built with an f-string, not %-formatting: the URL itself contains %2F, which
# a %-format call reads as a conversion and rejects.
ROWS_BASE = "https://datasets-server.huggingface.co/rows?dataset=theatticusproject%2Fcuad-qa"


def rows_url(offset: int, length: int) -> str:
    return f"{ROWS_BASE}&config=default&split=train&offset={offset}&length={length}"
UA = "clause-research mhammas1100@gmail.com"


def get(url: str, *, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {exc}")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--max-rows", type=int, default=30000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "cuad_clauses.jsonl"

    have = 0
    if path.exists():
        have = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"already have {have} rows")

    kept = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        offset = have
        while offset < args.max_rows:
            try:
                payload = json.loads(get(rows_url(offset, args.page)))
            except Exception as exc:  # noqa: BLE001
                print(f"page at {offset} failed: {exc}")
                break
            rows = payload.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r["row"]
                answers = row.get("answers") or {}
                texts = answers.get("text") or []
                if not texts:
                    continue  # CUAD includes negatives; exemplars need a span
                # The label is the leading clause of CUAD's question text.
                question = row.get("question", "")
                label = question.split("related to")[-1].strip(' "?')
                fh.write(json.dumps({
                    "contract": row.get("title", ""),
                    "label": label,
                    "question": question,
                    "spans": texts,
                }, ensure_ascii=False) + "\n")
                kept += 1
            fh.flush()
            offset += len(rows)
            if offset % 1000 < args.page:
                print(f"  offset {offset}, exemplars kept {kept}")
            time.sleep(0.3)
    print(f"done. new exemplars: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
