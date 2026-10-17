"""Fetch material-contract exhibits (EX-10*) from SEC EDGAR full-text search.

Public data, no key. SEC asks for a declared User-Agent and <=10 requests/second;
this stays well under that and the connection here is the binding limit anyway.

    python scripts/fetch_edgar.py --out "C:/d drive data/clause-120k/raw/edgar" --per-query 40
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "clause-research mhammas1100@gmail.com"
SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%s&forms=8-K,10-K,10-Q,S-1&dateRange=custom&startdt=%s&enddt=%s"

# One query per contract type, so the corpus is sampled for type balance rather
# than for whatever EDGAR happens to serve first. CUAD's thin types (as few as 3
# non-competes) are exactly what this is meant to avoid.
QUERIES = [
    "master services agreement",
    "employment agreement",
    "license agreement",
    "non-competition agreement",
    "consulting agreement",
    "supply agreement",
    "distribution agreement",
    "credit agreement",
    "lease agreement",
    "asset purchase agreement",
    "joint venture agreement",
    "confidentiality agreement",
    "development agreement",
    "manufacturing agreement",
    "transition services agreement",
    "indemnification agreement",
    "security agreement",
    "subscription agreement",
    "franchise agreement",
    "settlement agreement",
]
WINDOWS = [("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
           ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]

TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
ANY_TAG_RE = re.compile(r"<[^>]+>")


def get(url: str, *, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip

                    raw = gzip.decompress(raw)
                return raw
        except Exception as exc:  # noqa: BLE001 - any network error is a retry
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}: {exc}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def to_text(raw: bytes) -> str:
    s = raw.decode("utf-8", errors="replace")
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", s, flags=re.IGNORECASE)
    s = ANY_TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-query", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["doc_id"])
    print(f"already fetched: {len(done)}")

    saved = 0
    with manifest_path.open("a", encoding="utf-8", newline="\n") as man:
        for query in QUERIES:
            for start, end in WINDOWS:
                url = SEARCH % (urllib.parse.quote(f'"{query}"'), start, end)
                try:
                    hits = json.loads(get(url))["hits"]["hits"]
                except Exception as exc:  # noqa: BLE001
                    print(f"  search failed [{query} {start}]: {exc}")
                    continue
                print(f"  {query} {start}: {len(hits)} hits")
                for h in hits[: args.per_query]:
                    doc_id = h["_id"]
                    if doc_id in done:
                        continue
                    accession, _, filename = doc_id.partition(":")
                    if not re.search(r"ex-?10", filename, re.IGNORECASE):
                        continue  # EX-10 is the material-contract exhibit
                    cik = h["_source"]["ciks"][0].lstrip("0")
                    acc = accession.replace("-", "")
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{filename}"
                    try:
                        text = to_text(get(doc_url))
                    except Exception as exc:  # noqa: BLE001
                        print(f"    fetch failed {filename}: {exc}")
                        continue
                    if len(text.split()) < 500:
                        continue  # a stub or a cover page, not a contract
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)[:120]
                    (args.out / f"{safe}.txt").write_text(text, encoding="utf-8")
                    man.write(json.dumps({
                        "doc_id": doc_id, "query": query, "window": [start, end],
                        "cik": cik, "url": doc_url, "words": len(text.split()),
                        "file": f"{safe}.txt",
                    }, ensure_ascii=False) + "\n")
                    man.flush()
                    done.add(doc_id)
                    saved += 1
                    if saved % 10 == 0:
                        print(f"    saved {saved} (total {len(done)})")
                    time.sleep(0.2)
    print(f"done. new: {saved}, total: {len(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
