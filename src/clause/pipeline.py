"""Clause review: segment, label, extract obligations, flag what is missing.

Methodology, and why this shape
-------------------------------
The 2026 evidence on retrieval architectures says two things that decide this
design. First, GraphRAG's advantage over plain dense retrieval is concentrated in
multi-hop questions, and for contracts the useful structure is explicit:
clause numbering, defined terms, party names and cross-references. Second, a
vector index alone pulls text from *irrelevant* contracts, because legal
language is formulaic and similar wording recurs everywhere.

So the pipeline is **structure first, retrieval second, and a model nowhere**:

1. ``segment`` - the contract's own numbering and headings become clause nodes.
2. ``label`` - three independent passes over each clause, with agreement
   recorded per clause rather than collapsed into one score:
   a **rule** pass (legal cues are formulaic, which is what makes rules work),
   a **retrieval** pass (BM25 k-NN over CUAD's expert spans, from the exemplar
   half only), and a **structure** pass (defined terms and cross-references).
3. ``obligations`` - who owes what, on what trigger, by when, with what remedy.
4. ``missing`` - labels usual for this contract type that are absent here.

A cross-encoder reranker over the retrieval candidates is the documented next
step (hybrid + rerank reached 0.816 Recall@5 against 0.644 for BM25 alone in a
2026 benchmark) and it needs a model, so ``rerank_slot`` marks where it goes
instead of implying it is there.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

from .corpus import Cuad, tokens

# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------

# "7.", "7.1", "(a)", "ARTICLE V", "Section 7.1" - a contract tells you where its
# clauses begin, and using that beats a sentence splitter that cuts mid-proviso.
NUM_RE = re.compile(
    r"^\s*(?:(?:ARTICLE|Article|SECTION|Section)\s+[IVXLC0-9]+[.:]?|"
    r"\d+(?:\.\d+){0,3}[.)]?|\([a-z]\)|\([ivx]+\))\s+",
)
HEADING_RE = re.compile(r"^\s*([A-Z][A-Z \-/&,']{3,60})\s*[.:]?\s*$")

# EDGAR filing furniture, not contract terms. The redaction legend that modern
# exhibits carry - "CERTAIN PORTIONS ... HAVE BEEN EXCLUDED ... THE REGISTRANT
# TREATS AS PRIVATE AND CONFIDENTIAL" - contains the word confidential, so it was
# being labelled a Confidentiality clause in nearly every contract in the corpus.
# Left in, it would have been the single most common label in the dataset.
FILING_BOILERPLATE = re.compile(
    r"certain\s+(?:confidential\s+)?(?:portions?|information)\s+(?:of\s+this\s+)?exhibit"
    r"|ha(?:s|ve)\s+been\s+(?:omitted|excluded)\s+(?:from\s+this\s+exhibit\s+)?"
    r"(?:because|pursuant)"
    r"|treats?\s+as\s+private\s+and\s+confidential"
    r"|omitted\s+pursuant\s+to\s+item\s+601"
    r"|\[\*{2,}\]\s*(?:indicates|designates)"
    r"|^\s*exhibit\s+\d+(?:\.\d+)*\s*$"
    r"|^\s*page\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_filing_boilerplate(text: str) -> bool:
    """True for EDGAR cover text that is part of the filing, not the agreement."""
    return bool(FILING_BOILERPLATE.search(text))
SENT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


@dataclass
class Clause:
    index: int
    number: str
    heading: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "number": self.number,
            "heading": self.heading,
            "text": self.text,
            "words": len(self.text.split()),
            "start": self.start,
            "end": self.end,
        }


def segment(contract: str, *, min_words: int = 6, max_words: int = 400) -> list[Clause]:
    # The floor was 12 words, which silently dropped short clauses that carry real
    # terms: "This Agreement shall be governed by the laws of X." is 11 words, and
    # a definition clause is often shorter. A test over a two-clause contract is
    # what surfaced it - Confidentiality was reported "missing" from a contract
    # whose only confidentiality clause was 6 words long.
    """Split a contract into clauses, using its own numbering and headings."""
    lines = contract.splitlines(keepends=True)
    blocks: list[tuple[str, str, str, int]] = []  # number, heading, text, offset
    offset = 0
    cur_num = cur_head = ""
    buf: list[str] = []
    cur_start = 0

    def flush() -> None:
        text = " ".join(" ".join(buf).split())
        if text:
            blocks.append((cur_num, cur_head, text, cur_start))

    for line in lines:
        stripped = line.strip()
        m_head = HEADING_RE.match(stripped) if stripped else None
        m_num = NUM_RE.match(line) if stripped else None
        if m_head and len(stripped.split()) <= 9:
            flush()
            buf, cur_num, cur_head, cur_start = [], "", stripped.title(), offset
        elif m_num:
            flush()
            buf, cur_num, cur_start = [line[m_num.end():]], m_num.group(0).strip(), offset
        else:
            buf.append(line)
        offset += len(line)
    flush()

    clauses: list[Clause] = []
    for number, heading, text, start in blocks:
        parts = [text]
        if len(text.split()) > max_words:  # an unnumbered wall of text: fall back to sentences
            parts, acc = [], []
            for sent in SENT_RE.split(text):
                acc.append(sent)
                if len(" ".join(acc).split()) >= max_words // 2:
                    parts.append(" ".join(acc))
                    acc = []
            if acc:
                parts.append(" ".join(acc))
        for part in parts:
            if len(part.split()) < min_words:
                continue
            if is_filing_boilerplate(part):
                continue
            clauses.append(
                Clause(
                    index=len(clauses),
                    number=number,
                    heading=heading,
                    text=part,
                    start=start,
                    end=start + len(part),
                )
            )
    return clauses


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

# Rule cues, keyed to CUAD label names so the two passes are directly comparable.
# Each entry is (label, required pattern, optional supporting pattern).
RULES: list[tuple[str, re.Pattern, re.Pattern | None]] = [
    ("Governing Law", re.compile(r"govern(?:ed|ing)\s+by\s+.{0,40}laws?\s+of", re.IGNORECASE), None),
    ("Agreement Date", re.compile(r"\b(?:dated\s+as\s+of|entered\s+into\s+(?:as\s+of\s+)?this)\b", re.IGNORECASE), None),
    ("Expiration Date", re.compile(r"\b(?:expire|expiration|terminate)s?\s+on\b", re.IGNORECASE), None),
    ("Renewal Term", re.compile(r"\b(?:automatically\s+renew|renewal\s+term|successive\s+(?:one|two|three|\d+)[- ]year)\b", re.IGNORECASE), None),
    ("Notice Period To Terminate Renewal", re.compile(r"notice\s+(?:of\s+)?(?:non-?renewal|intent\s+not\s+to\s+renew)", re.IGNORECASE), None),
    ("Termination For Convenience", re.compile(r"terminate\s+this\s+agreement\s+(?:at\s+any\s+time\s+)?(?:for\s+any\s+reason|without\s+cause|for\s+convenience)", re.IGNORECASE), None),
    ("Non-Compete", re.compile(r"\b(?:shall\s+not|will\s+not|agrees?\s+not\s+to)\b.{0,80}\b(?:compet(?:e|ing|ition)|engage\s+in\s+any\s+business)\b", re.IGNORECASE | re.DOTALL), None),
    ("Exclusivity", re.compile(r"\b(?:exclusive|sole\s+and\s+exclusive)\b.{0,60}\b(?:right|licen[cs]e|distributor|supplier|provider)\b", re.IGNORECASE | re.DOTALL), None),
    ("No-Solicit Of Employees", re.compile(r"\b(?:not\s+(?:to\s+)?solicit|no[nt]-?solicit)\b.{0,60}\bemployee", re.IGNORECASE | re.DOTALL), None),
    ("No-Solicit Of Customers", re.compile(r"\b(?:not\s+(?:to\s+)?solicit|no[nt]-?solicit)\b.{0,60}\b(?:customer|client)", re.IGNORECASE | re.DOTALL), None),
    # The supporting cue used to be required in the *same clause*, which missed
    # agreements that define "Confidential Information" in one clause and impose the
    # duty in the next: an employment agreement came back with "Confidentiality
    # missing" while its no-solicit and IP clauses were found.
    ("Confidentiality", re.compile(r"\bconfidential(?:ity)?\b", re.IGNORECASE), None),
    ("Ip Ownership Assignment", re.compile(r"\b(?:hereby\s+assigns?|shall\s+own(?:ership)?|all\s+right,?\s+title\s+and\s+interest)\b.{0,80}\b(?:intellectual\s+property|inventions?|work\s+product)\b", re.IGNORECASE | re.DOTALL), None),
    ("License Grant", re.compile(r"\bhereby\s+grants?\b.{0,60}\blicen[cs]e\b", re.IGNORECASE | re.DOTALL), None),
    ("Cap On Liability", re.compile(r"\b(?:in\s+no\s+event|aggregate\s+liability|total\s+liability)\b.{0,120}\b(?:exceed|limited\s+to)\b", re.IGNORECASE | re.DOTALL), None),
    ("Uncapped Liability", re.compile(r"\b(?:unlimited|no\s+limit(?:ation)?\s+on)\s+liability\b", re.IGNORECASE), None),
    ("Insurance", re.compile(r"\b(?:maintain|carry|procure)\b.{0,60}\binsurance\b", re.IGNORECASE | re.DOTALL), None),
    ("Audit Rights", re.compile(r"\bright\s+to\s+(?:audit|inspect)\b|\baudit\s+rights?\b", re.IGNORECASE), None),
    ("Warranty Duration", re.compile(r"\bwarrant(?:s|y|ies)\b.{0,80}\b(?:period\s+of|for\s+a\s+period|\d+\s+(?:days|months|years))\b", re.IGNORECASE | re.DOTALL), None),
    ("Anti-Assignment", re.compile(r"\b(?:not\s+(?:be\s+)?assign|may\s+not\s+assign|no\s+assignment)\b.{0,80}\b(?:without\s+the\s+(?:prior\s+)?(?:written\s+)?consent)\b", re.IGNORECASE | re.DOTALL), None),
    ("Change Of Control", re.compile(r"\bchange\s+(?:of|in)\s+control\b", re.IGNORECASE), None),
    ("Most Favored Nation", re.compile(r"\bmost\s+favo(?:u)?red\s+(?:nation|customer)\b", re.IGNORECASE), None),
    ("Minimum Commitment", re.compile(r"\bminimum\s+(?:purchase|order|commitment|quantity|volume)\b", re.IGNORECASE), None),
    ("Price Restrictions", re.compile(r"\b(?:price|pricing)\b.{0,60}\b(?:shall\s+not\s+(?:be\s+)?increase|fixed\s+for|subject\s+to\s+adjustment)\b", re.IGNORECASE | re.DOTALL), None),
    ("Revenue/Profit Sharing", re.compile(r"\b(?:revenue|profit|royalt(?:y|ies))\b.{0,60}\b(?:shar(?:e|ing)|percentage|\d+%)\b", re.IGNORECASE | re.DOTALL), None),
    ("Post-Termination Services", re.compile(r"\b(?:following|after)\s+(?:the\s+)?termination\b.{0,80}\b(?:continue|provide|transition)\b", re.IGNORECASE | re.DOTALL), None),
    ("Indemnification", re.compile(r"\b(?:indemnify|indemnification|hold\s+harmless)\b", re.IGNORECASE), None),
    ("Force Majeure", re.compile(r"\bforce\s+majeure\b|\bacts?\s+of\s+god\b", re.IGNORECASE), None),
    ("Dispute Resolution", re.compile(r"\b(?:arbitration|mediat(?:e|ion)|exclusive\s+jurisdiction|venue)\b", re.IGNORECASE), None),
]

# Which labels a buyer should expect to see, by contract type. Absence is the
# finding: a services agreement with no liability cap is worth saying out loud.
EXPECTED_BY_TYPE = {
    "master services agreement": ["Governing Law", "Cap On Liability", "Indemnification", "Confidentiality", "Termination For Convenience", "Dispute Resolution"],
    "employment agreement": ["Governing Law", "Confidentiality", "Non-Compete", "No-Solicit Of Employees", "Ip Ownership Assignment"],
    "license agreement": ["Governing Law", "License Grant", "Cap On Liability", "Indemnification", "Ip Ownership Assignment"],
    "consulting agreement": ["Governing Law", "Confidentiality", "Ip Ownership Assignment", "Cap On Liability"],
    "supply agreement": ["Governing Law", "Minimum Commitment", "Price Restrictions", "Warranty Duration", "Indemnification"],
    "distribution agreement": ["Governing Law", "Exclusivity", "Minimum Commitment", "Termination For Convenience"],
    "lease agreement": ["Governing Law", "Insurance", "Anti-Assignment", "Indemnification"],
    "confidentiality agreement": ["Governing Law", "Confidentiality", "Dispute Resolution"],
    "_default": ["Governing Law", "Confidentiality", "Indemnification", "Dispute Resolution", "Cap On Liability"],
}

DEFINED_TERM_RE = re.compile(r'"([A-Z][A-Za-z \-]{2,40})"')
XREF_RE = re.compile(r"\b(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
# "Each Party shall ..." is the most common subject in a commercial contract and
# the first version of this list did not contain the word Party at all, so almost
# every obligation came back as "(unnamed party)".
PARTY_RE = re.compile(
    r'(?:^|[\s("])(?:the\s+|each\s+|either\s+|such\s+)?'
    r'((?:Receiving|Disclosing|Indemnifying|Indemnified|Defaulting|Non-Defaulting|Purchasing|Selling)\s+Part(?:y|ies)'
    r'|Part(?:y|ies)|Compan(?:y|ies)|Executive|Employee|Employer|Contractor|Subcontractor|Consultant'
    r'|Supplier|Vendor|Licensor|Licensee|Franchisor|Franchisee|Purchaser|Seller|Buyer|Lessor|Lessee'
    r'|Landlord|Tenant|Distributor|Reseller|Client|Customer|Borrower|Lender|Guarantor|Agent)',
    re.IGNORECASE,
)

# A definition is not a duty. "X shall have the meaning set forth in ..." and
# "X means ..." were being reported as obligations, which is noise in the one
# panel a reader checks first.
DEFINITION_RE = re.compile(
    r"\bshall\s+(?:have\s+the\s+meaning|mean)\b"
    r"|\bmeans\b"
    r"|\bis\s+defined\s+(?:as|to\s+mean)\b"
    r"|\bhas\s+the\s+meaning\b"
    r"|\brefers\s+to\b",
    re.IGNORECASE,
)


INLINE_HEAD_RE = re.compile(r"^((?:[A-Z][\w'’/-]*\s+){0,5}[A-Z][\w'’/-]*)[.:—-]\s")


def inline_heading(text: str) -> str:
    """The run-in heading a clause starts with, if it has one."""
    m = INLINE_HEAD_RE.match(text.strip())
    if not m:
        return ""
    candidate = m.group(1).strip()
    words = candidate.split()
    if not (1 <= len(words) <= 6):
        return ""
    # Must look like a heading: most words capitalised, no sentence verbs.
    caps = sum(1 for w in words if w[:1].isupper())
    return candidate if caps >= max(1, len(words) - 1) else ""


def rerank_slot(candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Where a cross-encoder reranker goes once a GPU is free. Identity for now."""
    return candidates


# Words that make a label's own name recognisable in the text. A retrieval-only
# finding must show at least one of them in the clause; without this the pass
# reported an Insurance clause in a contract that never mentions insurance, which
# is exactly the 0.236 precision the benchmark measures for retrieval alone.
LABEL_CUES = {
    "Insurance": ("insur",),
    "Audit Rights": ("audit", "inspect"),
    "Cap On Liability": ("liabilit", "damages"),
    "Uncapped Liability": ("liabilit",),
    "Indemnification": ("indemnif", "hold harmless"),
    "Confidentiality": ("confidential", "non-disclosure", "nondisclosure"),
    "Non-Compete": ("compet",),
    "Exclusivity": ("exclusiv",),
    "Anti-Assignment": ("assign",),
    "Change Of Control": ("change of control", "change in control"),
    "Most Favored Nation": ("most favo",),
    "Minimum Commitment": ("minimum",),
    "Force Majeure": ("force majeure", "act of god", "acts of god"),
    "Dispute Resolution": ("arbitrat", "mediat", "jurisdiction", "venue", "dispute"),
    "Renewal Term": ("renew",),
    "Post-Termination Services": ("terminat",),
    "Price Restrictions": ("price", "pricing"),
    "Revenue/Profit Sharing": ("revenue", "profit", "royalt"),
    "Ip Ownership Assignment": ("intellectual property", "inventions", "work product", "title and interest"),
    "License Grant": ("licen",),
    "Insurance Proceeds": ("insur",),
    "No-Solicit Of Employees": ("solicit",),
    "No-Solicit Of Customers": ("solicit",),
    "Notice Period To Terminate Renewal": ("notice", "renew"),
    "Expiration Date": ("expir", "terminates on"),
    "Agreement Date": ("dated", "entered into"),
    # Added after the first full build, where Rofr/Rofo/Rofn came out as the most
    # common label in the corpus (11,626 of 75,127) purely from retrieval, with no
    # cue to stop it. A right of first refusal says so in the text.
    #
    # Only labels whose cue is *diagnostic* belong here. Cues for Parties and
    # Document Name were tried and removed: a clause can name the parties without
    # the word "between", so requiring it cost recall (0.628 -> 0.538) and the
    # measurement did not justify it.
    "Governing Law": ("govern", "laws of", "law of"),
    "Warranty Duration": ("warrant",),
    "Termination For Convenience": ("terminat",),
    "Competitive Restriction Exception": ("compet",),
    "Rofr/Rofo/Rofn": ("first refusal", "first offer", "first negotiation", "preemptive right",
                       "pre-emptive right", "right of first"),
    "Effective Date": ("effective date", "effective as of"),
    "Non-Disparagement": ("disparag",),
    "Third Party Beneficiary": ("third party beneficiar", "third-party beneficiar"),
    "Volume Restriction": ("volume", "quantity", "units"),
    "Irrevocable Or Perpetual License": ("perpetual", "irrevocable"),
    "Unlimited/All-You-Can-Eat-License": ("unlimited", "all-you-can-eat"),
    "Liquidated Damages": ("liquidated damages",),
    "Covenant Not To Sue": ("not to sue", "covenant not"),
    "Source Code Escrow": ("escrow", "source code"),
}


def label_is_plausible(label: str, text: str) -> bool:
    """True when the clause carries a word that the label is named for.

    Labels with no cue listed are let through: the point is to stop confident
    nonsense, not to silently drop every label this table does not know about.
    """
    cues = LABEL_CUES.get(label)
    if not cues:
        return True
    low = text.lower()
    return any(cue in low for cue in cues)


def _heading_matches(head: str, key: str) -> bool:
    """A heading matches a label when they share a distinctive word.

    Substring matching in either direction matched "insurance" against
    "ensurance"-free text by accident and matched short labels far too often.
    """
    stop = {"of", "on", "to", "the", "and", "for", "in", "a"}
    hw = {w.strip(".,:;") for w in head.split()} - stop
    kw = {w for w in key.split()} - stop
    if not kw:
        return False
    shared = hw & kw
    return bool(shared) and len(shared) >= min(2, len(kw))


@dataclass
class Finding:
    label: str
    clause_index: int
    number: str
    heading: str
    span: str
    passes: list[str] = field(default_factory=list)
    rule_evidence: str = ""
    neighbours: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "clause_index": self.clause_index,
            "number": self.number,
            "heading": self.heading,
            "span": self.span,
            "passes": self.passes,
            "agreement": len(self.passes),
            "rule_evidence": self.rule_evidence,
            "neighbours": self.neighbours,
            "confidence": round(self.confidence, 3),
        }


# How strong the BM25 vote has to be before the retrieval pass claims a label.
# Measured on 25 held-out CUAD contracts (see docs/BENCHMARK.md): at 8.0 the
# retrieval pass made 930 of 1,048 predictions at 0.257 span precision, against
# 0.827 when a rule agreed with it. The threshold is a product decision, not a
# constant to leave at its first value.
RETRIEVAL_VOTE_MIN = 8.0


def label_clause(
    clause: Clause, cuad: Cuad, *, knn: int = 20, retrieval_vote_min: float | None = None
) -> list[Finding]:
    """Three passes over one clause. Agreement is recorded, not averaged away."""
    found: dict[str, Finding] = {}

    # Pass 1: rules.
    for label, required, support in RULES:
        m = required.search(clause.text)
        if not m:
            continue
        if support and not support.search(clause.text):
            continue
        snippet = clause.text[max(0, m.start() - 40) : m.end() + 80].strip()
        found[label] = Finding(
            label=label,
            clause_index=clause.index,
            number=clause.number,
            heading=clause.heading,
            span=snippet,
            passes=["rule"],
            rule_evidence=m.group(0)[:120],
        )

    # Pass 2: retrieval over CUAD's expert spans (exemplar half only).
    if cuad.exemplars:
        # 20 neighbours, not 5: the confirmation step needs the rule's label to be
        # represented among them at all, and with 5 it almost never was - one
        # agreeing finding in a 61-clause contract.
        hits = rerank_slot(cuad.bm25.search(tokens(clause.text), limit=knn))
        votes: collections.Counter = collections.Counter()
        best: dict[str, dict] = {}
        for i, score in hits:
            ex = cuad.exemplars[i]
            votes[ex.label] += score
            if ex.label not in best or score > best[ex.label]["bm25"]:
                best[ex.label] = {"label": ex.label, "bm25": round(score, 2),
                                  "exemplar": ex.text[:180], "from_contract": ex.contract}
        threshold = RETRIEVAL_VOTE_MIN if retrieval_vote_min is None else retrieval_vote_min
        if votes:
            # Confirmation first: if any retrieved neighbour carries a label the rule
            # pass already found in this clause, that is agreement, and it is the
            # signal worth most (0.83 span precision against 0.26 for retrieval
            # alone). The first version only looked at the single top vote, so two
            # passes almost never agreed and the UI's "agreement only" view was empty.
            for label in list(found):
                if label in votes and votes[label] >= threshold / 2:
                    found[label].passes.append("retrieval")
                    found[label].neighbours = [best[label]]

            top_label, top_score = votes.most_common(1)[0]
            if (
                top_score >= threshold
                and top_label not in found
                and label_is_plausible(top_label, clause.text)
            ):
                found[top_label] = Finding(
                    label=top_label,
                    clause_index=clause.index,
                    number=clause.number,
                    heading=clause.heading,
                    span=clause.text[:240],
                    passes=["retrieval"],
                    neighbours=[best[top_label]],
                )

    # Pass 3: structure. A heading naming the label is independent evidence.
    # Contracts rarely put headings on their own line - they run them into the
    # clause ("8.2 Confidential Information. Each Party ..."), so the leading
    # title-case phrase counts as a heading too. Without this the structure pass
    # fired on almost nothing and every finding looked like a single-pass guess.
    head = (clause.heading or inline_heading(clause.text)).lower()
    for label, _, _ in RULES:
        key = label.split("/")[0].replace("-", " ").lower()
        if head and _heading_matches(head, key):
            f = found.get(label)
            if f:
                f.passes.append("structure")
            else:
                found[label] = Finding(
                    label=label,
                    clause_index=clause.index,
                    number=clause.number,
                    heading=clause.heading,
                    span=clause.text[:240],
                    passes=["structure"],
                )

    for f in found.values():
        # Agreement across independent passes is the confidence signal here.
        f.confidence = {1: 0.45, 2: 0.78, 3: 0.93}.get(len(set(f.passes)), 0.45)
    return list(found.values())


# --------------------------------------------------------------------------------------
# Obligations
# --------------------------------------------------------------------------------------

DEADLINE_RE = re.compile(
    # "forty-five (45) days" is written with a hyphen, and the first version's
    # \w+ stopped at it, so every hyphenated deadline came back empty.
    r"\bwithin\s+([\w-]+(?:\s+\(\d+\))?)\s+"
    r"(business\s+days|calendar\s+days|days|months|years|hours|weeks)\b",
    re.IGNORECASE,
)
TRIGGER_RE = re.compile(r"\b(?:upon|in\s+the\s+event\s+(?:of|that)|if|following|after|prior\s+to)\b[^.;]{5,120}", re.IGNORECASE)
REMEDY_RE = re.compile(r"\b(?:shall\s+be\s+entitled\s+to|may\s+terminate|liquidated\s+damages|specific\s+performance|injunctive\s+relief|reimburse)\b[^.;]{0,100}", re.IGNORECASE)
DUTY_RE = re.compile(r"\b(shall|must|agrees\s+to|is\s+obligated\s+to|will)\b\s+([^.;]{8,160})", re.IGNORECASE)


@dataclass
class Obligation:
    who: str
    what: str
    trigger: str
    deadline: str
    remedy: str
    clause_index: int
    number: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def obligations(clause: Clause) -> list[Obligation]:
    """Who owes what, on what trigger, by when, with what remedy.

    This is the structure CUAD does not carry, and the reason the product is more
    than a clause classifier: a reader wants the duty, not the label.
    """
    out: list[Obligation] = []
    for m in DUTY_RE.finditer(clause.text):
        window = clause.text[max(0, m.start() - 60) : m.end() + 60]
        if DEFINITION_RE.search(window):
            continue  # a definition, not a duty
        before = clause.text[max(0, m.start() - 90) : m.start()]
        party = PARTY_RE.findall(before)
        who = party[-1].title() if party else "(unnamed party)"
        what = " ".join(m.group(2).split())[:200]
        trig = TRIGGER_RE.search(clause.text)
        dead = DEADLINE_RE.search(clause.text)
        rem = REMEDY_RE.search(clause.text)
        out.append(
            Obligation(
                who=who,
                what=what,
                trigger=" ".join(trig.group(0).split())[:140] if trig else "",
                deadline=" ".join(dead.group(0).split()) if dead else "",
                remedy=" ".join(rem.group(0).split())[:140] if rem else "",
                clause_index=clause.index,
                number=clause.number,
            )
        )
        if len(out) >= 6:  # one clause rarely carries more than a handful
            break
    return out


# --------------------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------------------


def structure(contract: str, clauses: list[Clause]) -> dict:
    """The contract's own graph: parties, defined terms, cross-references."""
    defined = collections.Counter(DEFINED_TERM_RE.findall(contract))
    parties = collections.Counter(p.title() for p in PARTY_RE.findall(contract))
    xrefs: list[dict] = []
    by_number = {c.number.rstrip(".)"): c.index for c in clauses if c.number}
    for c in clauses:
        for ref in XREF_RE.findall(c.text):
            if ref in by_number and by_number[ref] != c.index:
                xrefs.append({"from": c.index, "to": by_number[ref], "number": ref})
    return {
        "parties": [{"name": k, "mentions": v} for k, v in parties.most_common(8)],
        "defined_terms": [{"term": k, "mentions": v} for k, v in defined.most_common(15)],
        "cross_references": xrefs[:60],
        "clauses": len(clauses),
    }


def review(contract: str, cuad: Cuad, *, contract_type: str = "") -> dict:
    clauses = segment(contract)
    findings: list[Finding] = []
    duties: list[Obligation] = []
    for c in clauses:
        findings.extend(label_clause(c, cuad))
        duties.extend(obligations(c))

    present = {f.label for f in findings}
    expected = EXPECTED_BY_TYPE.get(contract_type.lower(), EXPECTED_BY_TYPE["_default"])
    missing = [label for label in expected if label not in present]

    unusual = [
        f.to_dict()
        for f in findings
        if f.label in {"Uncapped Liability", "Most Favored Nation", "Non-Compete", "Change Of Control", "Price Restrictions"}
    ]

    findings.sort(key=lambda f: (-f.confidence, f.clause_index))
    return {
        "clauses": [c.to_dict() for c in clauses],
        "findings": [f.to_dict() for f in findings],
        "obligations": [o.to_dict() for o in duties[:80]],
        "missing_protections": missing,
        "unusual_terms": unusual,
        "structure": structure(contract, clauses),
        "contract_type": contract_type or "(not specified)",
        "counts": {
            "clauses": len(clauses),
            "findings": len(findings),
            "labels_found": len(present),
            "obligations": len(duties),
            "agreement_3": sum(1 for f in findings if len(set(f.passes)) == 3),
            "agreement_2": sum(1 for f in findings if len(set(f.passes)) == 2),
            "agreement_1": sum(1 for f in findings if len(set(f.passes)) == 1),
        },
        "method": "structure-first: contract numbering, rule cues, BM25 k-NN over CUAD exemplars; no model",
    }
