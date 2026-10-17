"""Tests for the review pipeline, written around the defects that actually happened."""

from __future__ import annotations

import pytest

from clause.corpus import cuad_label
from clause.pipeline import (
    DEFINITION_RE,
    EXPECTED_BY_TYPE,
    inline_heading,
    obligations,
    review,
    segment,
    structure,
)
from clause.report import markdown, pdf

CONTRACT = """
MASTER SERVICES AGREEMENT

This Agreement is entered into as of January 1, 2024 between Acme Corp (the "Company") and
Globex Ltd (the "Supplier").

1. SERVICES

1.1 Scope. The Supplier shall provide the services described in Exhibit A within thirty (30) days
of receipt of a purchase order from the Company.

1.2 Definitions. "Deliverable" means any work product delivered under this Agreement.

2. CONFIDENTIALITY

2.1 Each Party acknowledges that it may receive Confidential Information of the other Party and
shall not disclose such Confidential Information to any third party.

3. LIABILITY

3.1 In no event shall the aggregate liability of either Party exceed the fees paid in the twelve
months preceding the claim.

4. GOVERNING LAW

4.1 This Agreement shall be governed by and construed in accordance with the laws of the State of
New York. Any dispute shall be resolved by binding arbitration in New York.
"""


def test_segmentation_uses_the_contracts_own_numbering():
    clauses = segment(CONTRACT)
    numbers = [c.number for c in clauses if c.number]
    assert any(n.startswith("1.1") for n in numbers), numbers
    assert any(n.startswith("4.1") for n in numbers), numbers


def test_inline_heading_is_found():
    assert inline_heading("Scope. The Supplier shall provide the services") == "Scope"
    assert inline_heading("The Supplier shall provide the services") == ""


@pytest.mark.parametrize(
    "text,is_definition",
    [
        ('"Deliverable" means any work product.', True),
        ("Affiliate shall have the meaning set forth in Section 1.", True),
        ("The Supplier shall deliver the Products within thirty (30) days.", False),
    ],
)
def test_definitions_are_told_apart_from_duties(text, is_definition):
    """The pattern lost its word boundaries once and silently matched nothing."""
    assert bool(DEFINITION_RE.search(text)) is is_definition


def test_obligation_names_the_party_and_the_deadline():
    clauses = segment(CONTRACT)
    duties = [o for c in clauses for o in obligations(c)]
    assert duties, "no obligations extracted"
    supplier = [o for o in duties if o.who.lower().startswith("supplier")]
    assert supplier, [o.who for o in duties]
    assert any("thirty" in o.deadline for o in supplier), [o.deadline for o in supplier]


def test_definitions_do_not_become_obligations():
    clauses = segment(CONTRACT)
    duties = [o for c in clauses for o in obligations(c)]
    assert not [o for o in duties if "meaning" in o.what or o.what.startswith("mean")]


def test_each_party_is_recognised_as_a_party():
    """'Each Party shall ...' used to come back as (unnamed party)."""
    clauses = [c for c in segment(CONTRACT) if "Each Party" in c.text]
    assert clauses
    duties = [o for c in clauses for o in obligations(c)]
    assert duties
    assert any("part" in o.who.lower() for o in duties), [o.who for o in duties]


def test_review_without_cuad_still_works(tmp_path):
    """The rule and structure passes must not need the expert corpus."""
    from clause.corpus import Cuad

    empty = Cuad(tmp_path / "missing.json")
    out = review(CONTRACT, empty, contract_type="master services agreement")
    labels = {f["label"] for f in out["findings"]}
    assert "Governing Law" in labels, labels
    assert "Cap On Liability" in labels, labels
    assert "Confidentiality" in labels, labels


def test_confidentiality_is_found_when_the_duty_and_the_term_are_split(tmp_path):
    """The cue may sit in a different clause from the duty.

    The first version of this test asserted something the pipeline cannot do: it
    had the duty clause say "any such information", with the words "Confidential
    Information" only in a definition two clauses earlier. No rule keyed to a cue
    word can make that link, and pretending otherwise would have meant writing a
    rule that fires on the word "information". The real behaviour - the cue in one
    clause and the duty in the next - is what is tested here.
    """
    from clause.corpus import Cuad

    split = (
        '1. DEFINITIONS\n\n1.1 "Confidential Information" means any non-public information '
        "disclosed by either Party to the other Party under this Agreement.\n\n"
        "2. DUTIES\n\n2.1 The Supplier shall not disclose the Confidential Information of the "
        "Company to any third party without prior written consent.\n"
    )
    out = review(split, Cuad(tmp_path / "missing.json"), contract_type="confidentiality agreement")
    assert "Confidentiality" not in out["missing_protections"], out["missing_protections"]


def test_short_clauses_are_not_dropped():
    """A one-line governing-law clause is 11 words and used to be skipped entirely."""
    short = "7. GOVERNING LAW\n\n7.1 This Agreement is governed by the laws of Delaware.\n"
    clauses = segment(short)
    assert clauses, "the only clause in the contract was dropped"
    assert any("Delaware" in c.text for c in clauses), [c.text for c in clauses]


def test_missing_protections_names_what_is_absent(tmp_path):
    from clause.corpus import Cuad

    thin = "1. TERM\n\n1.1 This Agreement begins on the Effective Date and continues for one year.\n"
    out = review(thin, Cuad(tmp_path / "missing.json"), contract_type="master services agreement")
    assert "Governing Law" in out["missing_protections"]
    assert set(out["missing_protections"]) <= set(EXPECTED_BY_TYPE["master services agreement"])


def test_structure_finds_parties_defined_terms_and_crossrefs():
    clauses = segment(CONTRACT)
    s = structure(CONTRACT, clauses)
    assert any(p["name"].lower().startswith("compan") for p in s["parties"]), s["parties"]
    assert any(t["term"] == "Deliverable" for t in s["defined_terms"]), s["defined_terms"]


def test_cuad_label_takes_only_the_quoted_name():
    q = ('Highlight the parts (if any) of this contract related to "Ip Ownership Assignment" that '
         "should be reviewed by a lawyer. Details: Does intellectual property created ...")
    assert cuad_label(q) == "Ip Ownership Assignment"


def test_pdf_report_is_a_pdf(tmp_path):
    from clause.corpus import Cuad

    out = review(CONTRACT, Cuad(tmp_path / "missing.json"), contract_type="master services agreement")
    blob = pdf(out, name="Test", min_agreement=1)
    assert blob.startswith(b"%PDF-1.4")
    assert b"startxref" in blob and blob.rstrip().endswith(b"%%EOF")
    assert len(blob) > 800


def test_markdown_report_mentions_the_findings(tmp_path):
    from clause.corpus import Cuad

    out = review(CONTRACT, Cuad(tmp_path / "missing.json"), contract_type="master services agreement")
    md = markdown(out, name="Test", min_agreement=1)
    assert "Clause review" in md and "Obligations" in md
    assert "Governing Law" in md


# --- found by feeding the pipeline a contract written by hand ---------------


def test_hyphenated_deadline_is_read():
    r"""A hyphenated number: "within forty-five (45) days". \w+ stopped at the hyphen and the deadline
    came back empty, so the one column a reader checks first was blank."""
    clause = segment(
        "2. FEES\n\n2.1 The Company shall pay the Supplier the fees within forty-five (45) days "
        "of receipt of a valid invoice.\n"
    )
    duties = [o for c in clause for o in obligations(c)]
    assert duties
    assert any("forty-five" in o.deadline for o in duties), [o.deadline for o in duties]


@pytest.mark.parametrize(
    "deadline,text",
    [
        ("thirty (30) days", "The Supplier shall deliver within thirty (30) days of the order."),
        ("ten business days", "The Company shall respond within ten business days of notice."),
        ("twenty-one (21) days", "Either Party shall cure within twenty-one (21) days of notice."),
    ],
)
def test_deadline_shapes(deadline, text):
    duties = [o for c in segment("1. T\n\n1.1 " + text) for o in obligations(c)]
    assert any(deadline in o.deadline for o in duties), [o.deadline for o in duties]


def test_a_label_needs_a_word_it_is_named_for():
    """Retrieval alone reported an Insurance clause in a contract with no insurance."""
    from clause.pipeline import label_is_plausible

    assert label_is_plausible("Insurance", "The Supplier shall maintain insurance of $1m.")
    assert not label_is_plausible("Insurance", "The Supplier shall deliver the goods on Tuesday.")
    assert label_is_plausible("Governing Law", "governed by the laws of Delaware")
    assert not label_is_plausible("Audit Rights", "The parties agree to meet quarterly.")


def test_unknown_labels_are_not_silently_dropped():
    """The cue table is a filter on confident nonsense, not an allow-list."""
    from clause.pipeline import label_is_plausible

    assert label_is_plausible("Some Label With No Cue Listed", "any text at all")


def test_junk_is_not_reviewed_as_a_contract(tmp_path):
    from clause.corpus import Cuad

    junk = "milk eggs bread coffee " * 40
    out = review(junk, Cuad(tmp_path / "missing.json"))
    assert len(out["findings"]) <= 3, [f["label"] for f in out["findings"]]
