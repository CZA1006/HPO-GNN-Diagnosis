import math
import torch
from collections import Counter

def compute_ic(hpoa_path="phenotype.hpoa"):
    """
    Parses the HPO annotation file to compute Information Content (IC)
    for each HPO term:
      IC(t) = -log( #diseases annotated with t / total #diseases )
    """
    term_counts = Counter()
    disease_set = set()

    with open(hpoa_path) as f:
        # skip comments, read header
        for line in f:
            if line.startswith("#"):
                continue
            header = line.strip().split("\t")
            break

        # map lowercase header names to indices
        idx = {h.lower(): i for i, h in enumerate(header)}
        disease_col = idx.get("database_id")
        term_col    = idx.get("hpo_id")
        if disease_col is None or term_col is None:
            raise RuntimeError(f"Header missing database_id or hpo_id: {header}")

        # process each record
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) <= max(disease_col, term_col):
                continue
            # split on commas
            dids = parts[disease_col].split(",")
            hpos = parts[term_col].split(",")
            for did in dids:
                disease_set.add(did)
                for h in hpos:
                    term_counts[h] += 1

    N = len(disease_set)
    ic = {t: -math.log(cnt / N) for t, cnt in term_counts.items()}
    return ic

if __name__ == "__main__":
    ic_map = compute_ic()
    torch.save(ic_map, "checkpoints/hpo_ic.pt")
    print(f"Computed IC for {len(ic_map)} terms; saved to checkpoints/hpo_ic.pt")
