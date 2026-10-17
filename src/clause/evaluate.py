"""Measure the Clause pipeline against CUAD's expert annotations.

    python -m clause.evaluate --contracts 60

CUAD is split by contract (``corpus.split_of``): the exemplar half feeds the
retrieval pass, and this runs only on the **held-out** half, so retrieval cannot
look up the answer.

Two levels are reported because they answer different questions:

* **document level** - did we find label L anywhere in a contract that has L?
  This is what "did the review miss the indemnity clause?" means to a reader.
* **span level** - did the clause we pointed at actually overlap the expert span?
  A pipeline can score well at document level by flagging everything, and this is
  the number that catches it.

Per-pass precision is reported too (rule only, retrieval only, and agreement),
because the point of three passes is knowing which one to trust.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .corpus import Cuad
from .pipeline import label_clause, segment


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _overlaps(span: str, gold_spans: list[str], *, min_words: int = 4) -> bool:
    """True when the predicted span shares a run of words with any gold span."""
    s = _norm(span)
    for gold in gold_spans:
        g = _norm(gold)
        if not g:
            continue
        if g in s or s in g:
            return True
        gw = g.split()
        if len(gw) >= min_words:
            # Any window of min_words from the gold span appearing in ours counts:
            # CUAD spans and our clause boundaries are drawn differently.
            for i in range(len(gw) - min_words + 1):
                if " ".join(gw[i : i + min_words]) in s:
                    return True
    return False


def evaluate(cuad: Cuad, *, contracts: int | None = None, retrieval_vote_min: float | None = None) -> dict:
    doc_tp: collections.Counter = collections.Counter()
    doc_fp: collections.Counter = collections.Counter()
    doc_fn: collections.Counter = collections.Counter()
    span_tp = span_total = 0
    by_pass: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [hits, total]

    pool = cuad.holdout[:contracts] if contracts else cuad.holdout
    for item in pool:
        clauses = segment(item["context"])
        found: dict[str, list[dict]] = collections.defaultdict(list)
        for c in clauses:
            for f in label_clause(c, cuad, retrieval_vote_min=retrieval_vote_min):
                found[f.label].append(f.to_dict())

        gold = {k: v for k, v in item["gold"].items() if v}
        for label in set(found) | set(gold):
            if label in found and label in gold:
                doc_tp[label] += 1
            elif label in found:
                doc_fp[label] += 1
            else:
                doc_fn[label] += 1

        for label, hits in found.items():
            if label not in gold:
                continue
            for h in hits:
                span_total += 1
                ok = _overlaps(h["span"], gold[label])
                span_tp += int(ok)
                key = "+".join(sorted(set(h["passes"])))
                by_pass[key][1] += 1
                by_pass[key][0] += int(ok)

    def prf(tp: int, fp: int, fn: int) -> dict:
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = (2 * p * r / (p + r)) if p and r else None
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4) if p is not None else None,
            "recall": round(r, 4) if r is not None else None,
            "f1": round(f, 4) if f is not None else None,
        }

    labels = sorted(set(doc_tp) | set(doc_fp) | set(doc_fn))
    per_label = {label: prf(doc_tp[label], doc_fp[label], doc_fn[label]) for label in labels}
    micro = prf(sum(doc_tp.values()), sum(doc_fp.values()), sum(doc_fn.values()))

    return {
        "contracts_evaluated": len(pool),
        "document_level_micro": micro,
        "document_level_per_label": per_label,
        "span_level": {
            "predictions_on_labels_the_contract_has": span_total,
            "overlapping_an_expert_span": span_tp,
            "span_precision": round(span_tp / span_total, 4) if span_total else None,
        },
        "by_pass": {
            k: {
                "predictions": v[1],
                "overlapping_expert_span": v[0],
                "precision": round(v[0] / v[1], 4) if v[1] else None,
            }
            for k, v in sorted(by_pass.items(), key=lambda kv: -kv[1][1])
        },
        "retrieval_vote_min": retrieval_vote_min,
        "corpus": cuad.stats,
        "model_used": None,
        "note": (
            "Held-out CUAD contracts only; the retrieval pass indexes the exemplar half. "
            "Document level answers 'did the review miss this clause type'; span level "
            "answers 'did it point at the right text'."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure Clause against CUAD")
    ap.add_argument("--contracts", type=int, default=60)
    ap.add_argument("--retrieval-vote-min", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="try several retrieval thresholds")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cuad = Cuad()
    if not cuad.holdout:
        print("CUAD not found - run scripts/fetch_cuad.py or unzip CUAD_v1.zip first")
        return 2
    if args.sweep:
        print(f"{'vote_min':>9} {'doc_P':>7} {'doc_R':>7} {'doc_F1':>7} {'span_P':>7} {'preds':>7}")
        rows = []
        for vote in (8.0, 12.0, 16.0, 20.0, 25.0, 30.0):
            r = evaluate(cuad, contracts=args.contracts, retrieval_vote_min=vote)
            m, sp = r["document_level_micro"], r["span_level"]
            print(f"{vote:9.1f} {m['precision'] or 0:7.3f} {m['recall'] or 0:7.3f} "
                  f"{m['f1'] or 0:7.3f} {sp['span_precision'] or 0:7.3f} "
                  f"{sp['predictions_on_labels_the_contract_has']:7d}")
            rows.append({"vote_min": vote, "doc": m, "span": sp})
        out = args.out or Path("C:/d drive data/clause-120k/sweep_retrieval.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"written to {out}")
        return 0

    report = evaluate(cuad, contracts=args.contracts, retrieval_vote_min=args.retrieval_vote_min)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    out = args.out or Path("C:/d drive data/clause-120k/benchmark_cuad.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "document_level_per_label"},
                     ensure_ascii=True, indent=2))
    worst = sorted(
        ((k, v) for k, v in report["document_level_per_label"].items() if v["f1"] is not None),
        key=lambda kv: kv[1]["f1"],
    )[:8]
    print("\nweakest labels:")
    for label, m in worst:
        print(f"  {label[:42]:42s} f1={m['f1']:.3f}  p={m['precision']}  r={m['recall']}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
