#!/usr/bin/env python3
"""
Split data/synthetic_reports/by_omim/rest.txt into one file per OMIM id.

We treat any line that contains 'OMIM: <digits>' and is preceded by a blank
line as a new header, e.g.:
    'Joubert syndrome 10 (OMIM: 300804)'
"""

from __future__ import annotations
import os
import re
from pathlib import Path

BASE = Path("data/synthetic_reports/by_omim")
INP  = BASE / "rest.txt"
OUTD = BASE

OMIM_IN_LINE = re.compile(r"OMIM\s*:?\s*(\d{4,7})", re.IGNORECASE)

def maybe_header(line: str, prev_blank: bool) -> str | None:
    """Return OMIM id if this looks like a header line."""
    if not prev_blank:
        return None
    m = OMIM_IN_LINE.search(line)
    return m.group(1) if m else None

def write_block(outdir: Path, omim_id: str | None, buf: list[str]) -> int:
    """Write buffer to <omim_id>.txt; return 1 if written, else 0."""
    if not omim_id or not buf:
        return 0
    out = outdir / f"{omim_id}.txt"
    # avoid overwrite if a per-OMIM file already exists
    if out.exists():
        i = 2
        while (outdir / f"{omim_id}_{i}.txt").exists():
            i += 1
        out = outdir / f"{omim_id}_{i}.txt"
    out.write_text(("\n".join(buf)).strip() + "\n", encoding="utf-8")
    return 1

def main() -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    if not INP.exists():
        print(f"rest.txt not found at {INP}")
        return

    text  = INP.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    current_id: str | None = None
    buf: list[str] = []
    wrote = 0
    prev_blank = True  # treat BOF as blank

    for line in lines:
        omim = maybe_header(line, prev_blank)
        if omim:
            wrote += write_block(OUTD, current_id, buf)
            current_id = omim
            buf = []
            prev_blank = False
            continue

        buf.append(line)
        prev_blank = (line.strip() == "")

    wrote += write_block(OUTD, current_id, buf)
    print(f"Wrote {wrote} files from rest.txt")

if __name__ == "__main__":
    main()
