> **This repository has moved.** Clause is now project [`projects/p02_clause`](https://github.com/hammasbuilds/nlp-legal-agentops/tree/main/projects/p02_clause) of [**nlp-legal-agentops**](https://github.com/hammasbuilds/nlp-legal-agentops), together with the other two products it was
> built alongside. That repository has the current code, the measured numbers, a
> screenshot of every feature and a recorded walkthrough. This one is kept for its
> history and is no longer updated.

# Clause

Contract review for the firms that nobody sells contract AI to: solo practitioners, boutiques, and the
small-business owner holding a 40-page agreement. Paste a contract, get the obligations, the unusual
terms, the protections that are missing, and a client-ready report — with every finding anchored to the
exact text it came from.

Product 2 of three in [`THREE-PRODUCTS-PLAN.md`](../THREE-PRODUCTS-PLAN.md).

```
CONTACT_EMAIL=you@example.com uv run python scripts/fetch_edgar.py --out <dir>   # SEC asks for a contact
uv run python -m clause.app          # the review desk on http://127.0.0.1:8820
uv run python -m clause.evaluate     # score the pipeline against expert annotations
```

## Why it exists

| Resource | Size | Note |
|---|---|---|
| CUAD | 510 contracts, 13,823 expert answer spans, 41 clause types | the expert benchmark, CC BY 4.0 — and as few as 3 examples of some contract types |
| **clause-120k (ours)** | **3,649 contracts, 25.0M words, 20 contract types** | built from SEC EDGAR EX-10 exhibits, one query per contract type so the corpus is sampled for type balance |

CUAD's contracts came from EDGAR, which is public and effectively unbounded — the corpus was never the
bottleneck, the annotation was. So this fetches contracts at scale and labels them without a model.

**The dataset goes to Kaggle, not here** - this repository is the code that builds, measures and
reads it. (clause-120k is still building; the link lands here when it is done.)

## How it was built

**We built clause-120k** — the contract collection, the segmentation, the three labelling passes, the
obligation extraction and the splits are ours, and the whole dataset rebuilds from this repository.

It is built on two public corpora, named here because CUAD's licence asks for attribution:

| Built on | What it provides | Licence |
|---|---|---|
| **SEC EDGAR** material-contract exhibits (EX-10) | 3,649 commercial contracts, 25.0M words, 20 contract types — we fetched one query per type so the corpus is balanced by type | public (US government filings) |
| **CUAD** (The Atticus Project) | 510 contracts with 13,823 expert-annotated clause spans across 41 clause types | CC BY 4.0 |

Every row names its contract, its CIK and the EDGAR URL it came from.

## What it is for

- **Research:** contract-review work in this portfolio — clause typing, obligation extraction and
  measuring where a labelling pipeline can and cannot be trusted.
- **Product:** the **Clause** desk in this repository; `python -m clause.app`.
- **Also usable for:** clause classification, obligation extraction (party, trigger, deadline, remedy),
  contract-structure parsing, and retrieval over legal language.

## Citing it

If you use clause-120k, or anything derived from it, cite this repository:

```bibtex
@misc{clause120k,
  author = {Muhammad Hammas},
  title  = {clause-120k: clause-level annotations over SEC EDGAR material contracts},
  year   = {2026},
  url    = {https://github.com/hammasbuilds/clause}
}
```

CUAD is CC BY 4.0, so attribution to The Atticus Project travels with anything built on the exemplars.

## How the review works

**Structure first, retrieval second, no model anywhere.** The 2026 evidence on retrieval architectures
points this way: GraphRAG's advantage over plain dense retrieval concentrates in multi-hop questions,
and for contracts the useful structure is explicit — numbering, defined terms, parties,
cross-references. A vector index alone also pulls text from *irrelevant* contracts, because legal
language is formulaic and the same wording recurs everywhere.

1. **Segment** — the contract's own numbering and run-in headings become clauses.
2. **Label** — three independent passes per clause, and the *agreement between them is kept*:
   a rule pass (29 cue patterns keyed to CUAD's label names), a retrieval pass (BM25 k-NN over CUAD's
   expert spans, from a disjoint half of CUAD), and a structure pass (the clause's own heading).
3. **Obligations** — who owes what, on what trigger, by when, with what remedy. This is the part CUAD
   does not carry, and the reason the product is more than a clause classifier.
4. **Missing protections** — what a contract of this type should contain and does not.

A cross-encoder reranker is the documented next step (one 2026 benchmark put hybrid+rerank at 0.816
Recall@5 against 0.644 for BM25 alone). It needs a model, and the GPU is reserved, so the slot is
explicit in `pipeline.rerank_slot` rather than implied.

## Measured against expert annotations

25 held-out CUAD contracts. The retrieval pass only indexes the *other* half of CUAD, split by
contract, so it cannot look up the answer.

| | |
|---|---|
| Found a clause type the contract really has (document recall) | **0.585** |
| A reported clause type was really there (document precision) | **0.579** |
| Document F1 | **0.582** |
| Pointed at text overlapping the expert span (span precision) | **0.456** |

Those are the numbers after a **plausibility gate**: a finding from retrieval alone must show a word
the label is named for (an Insurance finding needs "insur" somewhere in the clause). Without it the
pipeline reported an insurance clause in a contract that never mentions insurance. The gate trades
recall for precision and wins on both the headline and the one that matters most:
precision 0.458 → 0.579, span precision 0.270 → **0.456**, F1 0.551 → 0.582, with the number of
predictions cut from 1,205 to 472.

Only labels whose cue is *diagnostic* belong in that table. Cues for Parties and Document Name were
tried and removed: a clause can name the parties without using the word "between", and requiring it
cost recall (0.585 → 0.538) for no gain worth having.

**The finding that shaped the product:** agreement between passes predicts correctness far better than
any single pass.

| Passes that agreed | Predictions | Pointed at the right text |
|---|---|---|
| retrieval + structure | 8 | **1.000** |
| retrieval + rule + structure | 2 | **1.000** |
| retrieval + rule | 77 | **0.779** |
| rule alone | 27 | 0.444 |
| retrieval alone | 1,018 | 0.236 |

So the UI shows the agreement for every finding and can filter to the ones two passes agree on.
Strongest clause types: Governing Law F1 0.930, Anti-Assignment 0.878, Expiration Date 0.872. Weakest:
ROFR/ROFO/ROFN 0.143, Joint IP Ownership 0.182 — reported rather than averaged away.

A threshold sweep over the retrieval vote, re-run on the current pipeline, moves document F1 from
0.551 (vote ≥ 8) to 0.550 (vote ≥ 30) and span precision from 0.270 to 0.276, while predictions fall
only from 1,205 to 1,176: **the threshold is not the lever, agreement is.** That negative result is why
the knob sits at its measured default instead of being tuned for a nicer headline.

| vote ≥ | precision | recall | F1 | span precision | predictions |
|---|---|---|---|---|---|
| 8 | 0.458 | 0.691 | 0.551 | 0.270 | 1,205 |
| 16 | 0.459 | 0.691 | 0.552 | 0.271 | 1,201 |
| 20 | 0.459 | 0.691 | 0.552 | 0.275 | 1,183 |
| 30 | 0.459 | 0.688 | 0.550 | 0.276 | 1,176 |

## Input → Output

```
IN   an EX-10 employment agreement, 4,709 words
OUT  61 clauses · 89 findings · 21 clause types · 80 obligations
     missing protections: (none for this type)
     unusual terms: Change Of Control (clause 11), Non-Compete (clause 8)
     obligations: Employee — "be Beaverton, Oregon"; Employer — "pay the Executive a base salary ..."
     report: clause-review-Browser-Test-MSA.pdf (7,050 bytes), .md, printable .html
```

## The report, and why it is a PDF

Clause produces the one artefact of these three products that leaves the building: a review sent to a
client or attached to a file. It needs fixed pagination and no reflow, so `report.py` writes a real PDF
with no dependency — the content is Latin-script, so it uses the standard PDF fonts every reader has
built in. Markdown and printable HTML come out of the same report model.

## Problems hit while building this

- **CUAD's label names were parsed with the whole question tail attached**, so the UI showed a clause
  type called `Ip Ownership Assignment" that should be reviewed by a lawyer. Details: ...`. Every CUAD
  question has the form `... related to "<Label>" that should be reviewed ...`; splitting on the phrase
  instead of matching the quoted group fixed 41 labels at once.
- **"Each Party shall …" is the most common subject in a commercial contract**, and the party list did
  not contain the word *Party*, so nearly every obligation came back as `(unnamed party)`.
- **Definitions were being reported as obligations** ("X shall have the meaning set forth in …"). The
  filter that fixes it was written twice: the first version had its word boundaries eaten by the tool
  that wrote the file, so the pattern silently matched nothing.
- **Confidentiality came back "missing" from an agreement that clearly had it**, because the rule
  required its supporting cue in the *same clause* — contracts define the term in one clause and impose
  the duty in the next.
- **Agreement between passes was almost never reached** (one finding in a 61-clause contract), because
  the retrieval pass only ever compared its single top vote. It now confirms any label the rule found,
  over 20 neighbours instead of 5 — agreement went from 52 to 77 predictions and document F1 from 0.537
  to 0.552.
- **Two servers can bind the same port on Windows.** A "restarted" server kept serving the old code
  while the new process sat beside it, which looked exactly like a fix that did not work.

## What it does NOT do

- It is not legal advice, and it does not decide anything. Every finding points at text for a person to read.
- It does not claim a model's judgement: there is no model in the pipeline. The numbers above are what
  rules, structure and lexical retrieval achieve.
- It does not hide its precision. Document precision is 0.463: roughly half of what it reports on a
  single pass is not what CUAD's experts marked, which is why agreement is shown on every row.
- It does not read PDFs or Word files yet — plain text in, for now.
- Its corpus skews to the large commercial agreements that get filed with the SEC, while the buyer signs
  smaller ones. A held-out slice from public small-business material is still to come.
