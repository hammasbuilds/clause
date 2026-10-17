"""Build clause-120k: every clause of every contract in the corpus, labelled.

    python -m clause.build --out "C:/d drive data/clause-120k/v1"

One row is one clause, labelled or not. Unlabelled clauses are kept on purpose:
a corpus of only the clauses a labeller recognised would teach a model that every
clause has a label, and it would hide the labeller's recall from anyone reading
the dataset.

Each label on a row carries **which passes agreed on it**, because that is the
signal the benchmark says to trust: on held-out CUAD contracts, a label two
passes agree on pointed at the right text 78% of the time against 24% for
retrieval alone. A consumer of this dataset can filter on it; a consumer who
ignores it is choosing noise knowingly, not by accident.

Splits are by contract, hashed, so no contract contributes clauses to two splits.
The build is resumable: contracts already written are skipped.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from .corpus import Cuad, Edgar
from .pipeline import label_clause, obligations, segment

DEFAULT_OUT = Path("C:/d drive data/clause-120k/v1")
SPLITS = (("train", 0.90), ("val", 0.07), ("test", 0.03))


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return SPLITS[-1][0]


def row_for(entry: dict, clause, labels, duties) -> dict:
    payload = f"{entry['doc_id']}\x1f{clause.index}\x1f{clause.text[:200]}"
    return {
        "row_id": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "contract_id": entry["doc_id"],
        "cik": entry.get("cik", ""),
        "contract_type": entry.get("query", ""),
        "source_url": entry.get("url", ""),
        "clause_index": clause.index,
        "number": clause.number,
        "heading": clause.heading,
        "text": clause.text,
        "words": len(clause.text.split()),
        "labels": [
            {
                "label": f.label,
                "passes": sorted(set(f.passes)),
                "agreement": len(set(f.passes)),
                "confidence": round(f.confidence, 3),
                "span": f.span,
                "rule_evidence": f.rule_evidence,
                "nearest_expert_span": (f.neighbours[0]["exemplar"] if f.neighbours else ""),
            }
            for f in labels
        ],
        "obligations": [o.to_dict() for o in duties],
        "licence": "contract text: public (SEC EDGAR); labels: derived, CC BY 4.0 exemplars from CUAD",
        "model_in_label_path": False,
    }


def build(out: Path, *, limit: int | None = None, resume: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cuad, edgar = Cuad(), Edgar()
    if not cuad.exemplars:
        raise SystemExit("CUAD not found - unzip CUAD_v1.zip first (see scripts/fetch_cuad.py)")
    if not edgar.manifest:
        raise SystemExit("no contracts - run scripts/fetch_edgar.py first")

    done_path = out / "contracts_done.txt"
    done: set[str] = set()
    if resume and done_path.exists():
        done = {ln.strip() for ln in done_path.read_text(encoding="utf-8").splitlines() if ln.strip()}

    handles = {
        name: (out / f"{name}.jsonl").open("a" if done else "w", encoding="utf-8", newline="\n")
        for name, _ in SPLITS
    }
    done_fh = done_path.open("a" if done else "w", encoding="utf-8", newline="\n")

    todo = [m for m in edgar.manifest if m["doc_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} contracts already done, {len(todo)} to go", flush=True)

    started = time.time()
    counts = collections.Counter()
    try:
        for n, entry in enumerate(todo, 1):
            text = edgar.text(entry)
            if not text:
                continue
            split = split_of(entry["doc_id"])
            for clause in segment(text):
                labels = label_clause(clause, cuad)
                duties = obligations(clause)
                handles[split].write(
                    json.dumps(row_for(entry, clause, labels, duties), ensure_ascii=False) + "\n"
                )
                counts["clauses"] += 1
                counts["labelled_clauses"] += 1 if labels else 0
                counts["labels"] += len(labels)
                counts["obligations"] += len(duties)
                counts[f"split_{split}"] += 1
            counts["contracts"] += 1
            done_fh.write(entry["doc_id"] + "\n")
            if n % 25 == 0:
                for h in handles.values():
                    h.flush()
                done_fh.flush()
                rate = (time.time() - started) / n
                print(f"  {n}/{len(todo)} contracts · {counts['clauses']} clauses · "
                      f"{rate:.2f}s each · {(len(todo) - n) * rate / 60:.0f} min left", flush=True)
    finally:
        for h in handles.values():
            h.close()
        done_fh.close()

    return dict(counts)


def stats(out: Path) -> dict:
    by_label: collections.Counter = collections.Counter()
    by_agreement: collections.Counter = collections.Counter()
    by_type: collections.Counter = collections.Counter()
    by_split: collections.Counter = collections.Counter()
    contracts_in_split: dict[str, set] = collections.defaultdict(set)
    clause_words: list[int] = []
    labelled = total = labels_total = duties_total = 0
    row_ids: set[str] = set()
    duplicate_ids = 0

    for name, _ in SPLITS:
        path = out / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                by_split[name] += 1
                contracts_in_split[name].add(row["contract_id"])
                by_type[row["contract_type"]] += 1
                clause_words.append(row["words"])
                if row["row_id"] in row_ids:
                    duplicate_ids += 1
                row_ids.add(row["row_id"])
                if row["labels"]:
                    labelled += 1
                for lab in row["labels"]:
                    by_label[lab["label"]] += 1
                    by_agreement[lab["agreement"]] += 1
                    labels_total += 1
                duties_total += len(row["obligations"])

    straddling = set()
    names = [n for n, _ in SPLITS]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            straddling |= contracts_in_split[a] & contracts_in_split[b]

    clause_words.sort()
    return {
        "clauses": total,
        "clauses_with_at_least_one_label": labelled,
        "label_coverage": round(labelled / total, 4) if total else None,
        "labels": labels_total,
        "labels_two_or_more_passes": by_agreement[2] + by_agreement[3],
        "obligations": duties_total,
        "contracts": len(set().union(*contracts_in_split.values())) if contracts_in_split else 0,
        "by_split": dict(by_split),
        "contracts_in_two_splits": len(straddling),
        "duplicate_row_ids": duplicate_ids,
        "median_clause_words": clause_words[len(clause_words) // 2] if clause_words else 0,
        "by_agreement": {str(k): v for k, v in sorted(by_agreement.items())},
        "by_contract_type": dict(by_type.most_common()),
        "top_labels": dict(by_label.most_common(20)),
        "thinnest_labels": dict(by_label.most_common()[-10:]),
        "unlabelled_clauses_are_kept": True,
        "model_in_label_path": False,
        "measured_label_quality": {
            "source": "python -m clause.evaluate on held-out CUAD contracts",
            "document_f1": 0.575,
            "span_precision_all": 0.440,
            "span_precision_two_passes_agreeing": 0.779,
            "span_precision_retrieval_alone": 0.236,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build clause-120k")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="only this many new contracts")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.stats_only:
        counts = build(args.out, limit=args.limit)
        print(json.dumps(counts, indent=2))
    s = stats(args.out)
    (args.out / "stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("by_contract_type", "top_labels", "thinnest_labels")}, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
