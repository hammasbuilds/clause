"""Client-ready review report: Markdown, HTML and PDF, with no dependencies.

Why a PDF at all. Clause produces the one artefact in these three products that
leaves the building: a review a solo practitioner sends to a client, or attaches
to a file. That document has to be fixed, paginated and printable, which is what
PDF is for.

Why it can be written by hand here. A PDF is a text container; the only hard
part is fonts. This report is Latin-script, so it can use the 14 standard
PDF fonts (Helvetica) that every reader has built in - no font file, no
embedding, no shaping. The same trick would **not** work for Urdu: an Urdu PDF
needs an embedded Nastaliq or Naskh font and right-to-left shaping with
contextual letter forms, which is a real typesetting job and not a text
container exercise. That is why Scribe exports text and this exports PDF.

Product rule: the report names the product and nothing about how it works
internally - no model names, no library names, no "checked by hand".
"""

from __future__ import annotations

import datetime as _dt
import re
import zlib

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN = 54
LEADING = 14
FONT_BODY, FONT_BOLD = "F1", "F2"


def _esc(text: str) -> str:
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    # The standard fonts are Latin-1; anything outside it becomes a plain marker
    # rather than a mojibake byte.
    return "".join(ch if ord(ch) < 256 else "?" for ch in out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


class _Pdf:
    """A deliberately small PDF writer: pages of text lines in two fonts."""

    def __init__(self) -> None:
        self.pages: list[list[tuple[str, int, str]]] = [[]]  # (font, size, text)

    def line(self, text: str = "", *, size: int = 10, bold: bool = False) -> None:
        width = int((PAGE_W - 2 * MARGIN) / (size * 0.52))
        for part in _wrap(text, width) if text else [""]:
            if len(self.pages[-1]) >= int((PAGE_H - 2 * MARGIN) / LEADING):
                self.pages.append([])
            self.pages[-1].append((FONT_BOLD if bold else FONT_BODY, size, part))

    def bytes(self) -> bytes:
        objects: list[bytes] = []

        def add(payload: bytes) -> int:
            objects.append(payload)
            return len(objects)

        font_body = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

        page_ids: list[int] = []
        content_ids: list[int] = []
        for page in self.pages:
            chunks = ["BT"]
            y = PAGE_H - MARGIN
            for font, size, text in page:
                chunks.append(f"/{font} {size} Tf 1 0 0 1 {MARGIN} {y} Tm ({_esc(text)}) Tj")
                y -= LEADING
            chunks.append("ET")
            stream = zlib.compress("\n".join(chunks).encode("latin-1", "replace"))
            content_ids.append(
                add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream")
            )
            page_ids.append(0)  # filled once the Pages object exists

        pages_id = len(objects) + len(self.pages) + 1
        for i, content_id in enumerate(content_ids):
            page_ids[i] = add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /{FONT_BODY} {font_body} 0 R /{FONT_BOLD} {font_bold} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode("latin-1")
            )
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        pages_obj = add(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("latin-1"))
        catalog = add(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("latin-1"))

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, payload in enumerate(objects, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("latin-1") + payload + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode("latin-1")
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
        ).encode("latin-1")
        return bytes(out)


def _title_for(review: dict, name: str) -> str:
    return name or review.get("contract_type") or "Contract review"


def markdown(review: dict, *, name: str = "", min_agreement: int = 2) -> str:
    """The same report as Markdown, for people who want the text."""
    today = _dt.date.today().isoformat()
    counts = review.get("counts", {})
    lines = [
        f"# Clause review — {_title_for(review, name)}",
        "",
        f"Prepared by Clause · {today}",
        "",
        f"- Clauses read: **{counts.get('clauses', 0)}**",
        f"- Findings: **{counts.get('findings', 0)}** across {counts.get('labels_found', 0)} clause types",
        f"- Obligations extracted: **{counts.get('obligations', 0)}**",
        "",
        "## Missing protections",
        "",
    ]
    missing = review.get("missing_protections") or []
    lines += [f"- {m}" for m in missing] if missing else ["Nothing expected for this contract type is absent."]
    lines += ["", "## Unusual terms", ""]
    unusual = review.get("unusual_terms") or []
    if unusual:
        for u in unusual:
            lines.append(f"- **{u['label']}** (clause {u['number'] or u['clause_index']}): {u['span'][:240]}")
    else:
        lines.append("None of the flagged unusual terms appear.")
    lines += ["", f"## Findings (agreement of {min_agreement}+ passes)", ""]
    for f in review.get("findings", []):
        if f["agreement"] < min_agreement:
            continue
        lines.append(f"### {f['label']} — clause {f['number'] or f['clause_index']}")
        lines.append("")
        lines.append(f"{f['span'][:600]}")
        lines.append("")
    lines += ["## Obligations", ""]
    for o in review.get("obligations", [])[:40]:
        bits = [f"**{o['who']}** — {o['what']}"]
        if o["deadline"]:
            bits.append(f"by {o['deadline']}")
        if o["trigger"]:
            bits.append(f"trigger: {o['trigger']}")
        lines.append("- " + " · ".join(bits))
    return "\n".join(lines) + "\n"


def pdf(review: dict, *, name: str = "", min_agreement: int = 2) -> bytes:
    doc = _Pdf()
    counts = review.get("counts", {})
    doc.line(f"Clause review - {_title_for(review, name)}", size=16, bold=True)
    doc.line(f"Prepared by Clause  ·  {_dt.date.today().isoformat()}", size=9)
    doc.line()
    doc.line(
        f"Clauses read: {counts.get('clauses', 0)}   "
        f"Findings: {counts.get('findings', 0)}   "
        f"Clause types: {counts.get('labels_found', 0)}   "
        f"Obligations: {counts.get('obligations', 0)}",
        size=10,
    )
    doc.line()

    doc.line("Missing protections", size=13, bold=True)
    missing = review.get("missing_protections") or []
    if missing:
        for m in missing:
            doc.line(f"  -  {m}")
    else:
        doc.line("  Nothing expected for this contract type is absent.")
    doc.line()

    doc.line("Unusual terms", size=13, bold=True)
    unusual = review.get("unusual_terms") or []
    if unusual:
        for u in unusual:
            doc.line(f"  -  {u['label']} (clause {u['number'] or u['clause_index']})", bold=True)
            doc.line(f"     {u['span'][:400]}", size=9)
    else:
        doc.line("  None of the flagged unusual terms appear.")
    doc.line()

    doc.line(f"Findings (agreement of {min_agreement}+ passes)", size=13, bold=True)
    shown = 0
    for f in review.get("findings", []):
        if f["agreement"] < min_agreement:
            continue
        shown += 1
        doc.line(f"{f['label']} - clause {f['number'] or f['clause_index']}", size=11, bold=True)
        doc.line(f"   {f['span'][:500]}", size=9)
        doc.line()
    if not shown:
        doc.line("  No finding reached the agreement threshold.")
    doc.line()

    doc.line("Obligations", size=13, bold=True)
    for o in review.get("obligations", [])[:60]:
        tail = f" (by {o['deadline']})" if o["deadline"] else ""
        doc.line(f"  -  {o['who']}: {o['what']}{tail}", size=9)
    return doc.bytes()


_TAG = re.compile(r"<[^>]+>")


def html(review: dict, *, name: str = "", min_agreement: int = 2) -> str:
    """A printable HTML version, for the browser's own print-to-PDF."""
    md = markdown(review, name=name, min_agreement=min_agreement)
    body = []
    for line in md.splitlines():
        if line.startswith("### "):
            body.append(f"<h3>{_TAG.sub('', line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{_TAG.sub('', line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{_TAG.sub('', line[2:])}</h1>")
        elif line.startswith("- "):
            body.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    return (
        "<!doctype html><meta charset='utf-8'><title>Clause review</title>"
        "<style>body{font:14px/1.6 Georgia,serif;max-width:46em;margin:40px auto;padding:0 16px}"
        "h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd}"
        "h3{font-size:14px;margin:18px 0 4px}li{margin:3px 0}@media print{body{margin:0}}</style>"
        + "\n".join(body)
    )
