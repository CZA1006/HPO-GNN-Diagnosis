#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_hpo_idf.py
Build HPO -> IDF (rarer terms get larger weights) from phenotype.hpoa.
压低“到处都有的常见表型”，抬高“跨疾病稀有、最能区分的表型”，让相似度和排序更有判别力

Saves:
  checkpoints/hpo_df.pt   # { "HP:xxxxxxx": int df }
  checkpoints/hpo_idf.pt  # { "HP:xxxxxxx": float idf }
"""

import argparse, csv, math, os
from collections import defaultdict
import torch

def parse_hpoa(path):
    # Returns: term -> set(diseases), and total number of unique diseases N
    term2diseases = defaultdict(set)
    diseases = set()
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        cols = {}
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                lower = [h.lower() for h in header]
                def find(col):
                    try:
                        return lower.index(col)
                    except ValueError:
                        return None
                cols["did"]  = find("database_id")
                cols["hpo"]  = find("hpo_id")
                cols["qual"] = find("qualifier")
                if cols["did"] is None or cols["hpo"] is None:
                    raise RuntimeError("Missing required columns database_id/hpo_id in HPOA header")
                continue
            did = row[cols["did"]].strip()
            hp  = row[cols["hpo"]].strip()
            if not did or not hp:
                continue
            if cols["qual"] is not None and cols["qual"] < len(row):
                if row[cols["qual"]].strip().upper() == "NOT":
                    continue  # ignore negative assertions
            diseases.add(did)
            term2diseases[hp].add(did)
    return term2diseases, len(diseases)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    term2d, N = parse_hpoa(args.hpoa)
    df = {hp: len(s) for hp, s in term2d.items()}

    # BM25-style IDF (non-negative)
    idf = {}
    for hp, d in df.items():
        val = math.log((N - d + 0.5) / (d + 0.5))
        idf[hp] = max(0.0, float(val))

    torch.save(df,  os.path.join(args.out_dir, "hpo_df.pt"))
    torch.save(idf, os.path.join(args.out_dir, "hpo_idf.pt"))
    print(f"Counted {N} unique diseases; built DF/IDF for {len(df)} HPO terms.")
    print(f"Saved: {args.out_dir}/hpo_df.pt, {args.out_dir}/hpo_idf.pt")

if __name__ == "__main__":
    main()
