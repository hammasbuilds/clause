"""Corpora for Clause: CUAD (labelled) and EDGAR (scale, unlabelled).

Two corpora with two different jobs:

* **CUAD** - 510 contracts, 41 labels, 13,823 expert answer spans, CC BY 4.0.
  It is the gold standard *and* the exemplar source, so it is split by contract:
  the exemplar half feeds retrieval, the held-out half is never retrieved from
  and is what the pipeline is scored on. Without that split, the retrieval pass
  would be looking up the answer.
* **EDGAR** - material-contract exhibits (EX-10) fetched by
  ``scripts/fetch_edgar.py``, one query per contract type so the corpus is
  sampled for type balance rather than for whatever EDGAR serves first. This is
  the production corpus: no labels, and the thing the product actually reviews.

BM25 is vendored here rather than imported from a sibling repo: these products
stay independently installable.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("C:/d drive data/clause-120k/raw")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
# CUAD phrases every question the same way: ... related to "<Label>" that should
# be reviewed ... Splitting on "related to " and taking the tail kept the whole
# rest of the sentence as the label, which then showed up in the UI as a label
# called 'Ip Ownership Assignment" that should be reviewed by a lawyer. Details: ...'.
CUAD_LABEL_RE = re.compile(r'related to "([^"]+)"')


def cuad_label(question: str) -> str:
    m = CUAD_LABEL_RE.search(question)
    return m.group(1).strip() if m else question.strip()


def tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


class BM25:
    """Okapi BM25. Stdlib only; built once per corpus and queried per request."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in collections.Counter(doc).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: list[str], limit: int = 8) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in query:
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


@dataclass
class Exemplar:
    label: str
    text: str
    contract: str


def split_of(name: str, *, holdout: float = 0.5) -> str:
    """Contract-level split, hashed so it is stable across runs."""
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "holdout" if h < holdout else "exemplar"


class Cuad:
    """CUAD, split by contract into an exemplar half and a held-out half."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_ROOT / "cuad_zip" / "CUAD_v1" / "CUAD_v1.json")
        self.exemplars: list[Exemplar] = []
        self.holdout: list[dict] = []
        self.labels: list[str] = []
        if not self.path.exists():
            self.bm25 = BM25([])
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        labels: set[str] = set()
        for contract in payload["data"]:
            title = contract["title"]
            which = split_of(title)
            for para in contract["paragraphs"]:
                context = para["context"]
                gold: dict[str, list[str]] = collections.defaultdict(list)
                for qa in para["qas"]:
                    label = cuad_label(qa["question"])
                    labels.add(label)
                    for ans in qa["answers"]:
                        text = ans["text"].strip()
                        if not text:
                            continue
                        if which == "exemplar":
                            self.exemplars.append(Exemplar(label=label, text=text, contract=title))
                        gold[label].append(text)
                if which == "holdout":
                    self.holdout.append({"contract": title, "context": context, "gold": dict(gold)})
        self.labels = sorted(labels)
        self.bm25 = BM25([tokens(e.text) for e in self.exemplars])

    @property
    def stats(self) -> dict:
        by_label = collections.Counter(e.label for e in self.exemplars)
        return {
            "path": str(self.path),
            "available": self.path.exists(),
            "labels": len(self.labels),
            "exemplar_spans": len(self.exemplars),
            "holdout_contracts": len(self.holdout),
            "thinnest_labels": by_label.most_common()[-5:] if by_label else [],
        }


class Edgar:
    """The EX-10 exhibits fetched from EDGAR, with their manifest."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or DEFAULT_ROOT / "edgar")
        self.manifest: list[dict] = []
        self.duplicate_manifest_lines = 0
        man = self.root / "manifest.jsonl"
        if man.exists():
            seen: set[str] = set()
            for line in man.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                # One entry per document. A repeated doc_id means the same contract
                # would be read, labelled and written twice.
                if row["doc_id"] in seen:
                    self.duplicate_manifest_lines += 1
                    continue
                seen.add(row["doc_id"])
                self.manifest.append(row)

    def text(self, entry: dict) -> str:
        path = self.root / entry["file"]
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @property
    def stats(self) -> dict:
        by_query = collections.Counter(m["query"] for m in self.manifest)
        words = [m.get("words", 0) for m in self.manifest]
        return {
            "root": str(self.root),
            "contracts": len(self.manifest),
            "duplicate_manifest_lines_skipped": self.duplicate_manifest_lines,
            "contract_types": len(by_query),
            "by_type": dict(by_query.most_common()),
            "median_words": sorted(words)[len(words) // 2] if words else 0,
            "total_words": sum(words),
        }
