#!/usr/bin/env python3
from __future__ import annotations
import csv, re
from pathlib import Path

BASE = Path("data/synthetic_reports/by_omim")
OUT  = Path("data/synthetic_reports/index.csv")
OMIM_RE = re.compile(r"(\d{4,7})(?:_\d+)?\.txt$", re.I)

def main():
    rows = []
    for p in sorted(BASE.glob("*.txt")):
        m = OMIM_RE.search(p.name)
        if not m: 
            continue
        omim = m.group(1)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "omim_id": omim,
            "path": str(p),
            "n_chars": len(txt),
            "n_lines": txt.count("\n")+1,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omim_id","path","n_chars","n_lines"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
