#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from collections import Counter
from pathlib import Path

# Match OMIM/MIM with flexible punctuation and exactly 6 digits (typical OMIM IDs)
OMIM_RE = re.compile(
    r"""(?ix)          # case-insensitive, verbose
    \b(?:OMIM|MIM)     # keyword
    [^0-9]{0,5}        # short non-digit separator(s) like ':', '#', ',', ')', ' '
    (?P<id>[1-9]\d{5}) # 6-digit ID (leading digit non-zero)
    \b
    """
)

def read_text(path: Path, encoding: str = "utf-8") -> str:
    with path.open("r", encoding=encoding, errors="replace") as f:
        return f.read()

def line_start(text: str, pos: int) -> int:
    """Return index of the start-of-line containing pos."""
    nl = text.rfind("\n", 0, pos)
    return 0 if nl == -1 else nl + 1

def split_chunks(text: str):
    """
    Return list of (id, chunk_text) cut from the line containing each match
    up to the line before the next match.
    """
    matches = list(OMIM_RE.finditer(text))
    if not matches:
        return []

    starts = [line_start(text, m.start()) for m in matches]
    ids = [m.group("id") for m in matches]

    chunks = []
    for i, (sid, sidx) in enumerate(zip(ids, starts)):
        eidx = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[sidx:eidx].rstrip()
        chunks.append((sid, chunk))
    return chunks

def write_chunks(chunks, outdir: Path, encoding: str, dry_run: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    seen_this_run = set()
    for omim_id, chunk in chunks:
        out_path = outdir / f"{omim_id}.txt"
        if not dry_run:
            # If first time touching this file this run, start fresh (avoid mixing runs)
            if omim_id not in seen_this_run and out_path.exists():
                out_path.unlink()
            with out_path.open("a", encoding=encoding) as f:
                if out_path.exists() and out_path.stat().st_size > 0:
                    f.write("\n\n" + "=" * 80 + "\n\n")
                f.write(chunk.strip() + "\n")
        seen_this_run.add(omim_id)

def summarize(chunks, text: str, outdir: Path):
    ids = [i for i, _ in chunks]
    counts = Counter(ids)
    total_tokens = len(ids)
    unique_ids = len(counts)
    multi = sum(1 for _, c in counts.items() if c > 1)
    print("\n=== Split summary ===")
    print(f"OMIM/MIM tokens found:   {total_tokens}")
    print(f"Unique OMIM IDs:         {unique_ids}")
    print(f"IDs with >1 chunk:       {multi}")
    if outdir.exists():
        existing = len(list(outdir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt")))
        print(f"Numeric files now in {outdir}: {existing}")

def main():
    p = argparse.ArgumentParser(description="Split rest.txt into per-OMIM files.")
    p.add_argument("--infile", required=True, help="Path to rest.txt")
    p.add_argument("--outdir", required=True, help="Directory to write per-OMIM files")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--dry_run", action="store_true", help="Parse and report only; write nothing.")
    args = p.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)

    text = read_text(infile, encoding=args.encoding)

    chunks = split_chunks(text)
    if not chunks:
        print("No OMIM/MIM markers found. Check encoding or pattern.")
        return

    write_chunks(chunks, outdir, args.encoding, args.dry_run)
    summarize(chunks, text, outdir)

if __name__ == "__main__":
    main()
