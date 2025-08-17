# src/aggregate_disease_embeddings.py

import os
import re
import csv
import math
import torch
from collections import defaultdict

FREQ_MAP = {
    # HPO frequency terms -> approximate probabilities
    "HP:0040280": 1.00,  # Obligate (100%)
    "HP:0040281": 0.99,  # Very frequent (99%)
    "HP:0040282": 0.80,  # Frequent (80%)
    "HP:0040283": 0.30,  # Occasional (30%)
    "HP:0040284": 0.05,  # Rare (5%)
    "HP:0040285": 0.01,  # Very rare (<1%)
}

def parse_frequency(s: str) -> float:
    """Convert frequency field to a probability in [0,1]."""
    if not s or s == ".":
        return 1.0
    s = s.strip()
    # ratio like '1/2'
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            return max(0.0, min(1.0, a / b))
    # HPO frequency code
    if s in FREQ_MAP:
        return FREQ_MAP[s]
    # semicolon-separated list (take max)
    if ";" in s:
        return max(parse_frequency(tok) for tok in s.split(";"))
    return 1.0

def load_term_embeddings(node_list_path, term_embs_path):
    node_list = torch.load(node_list_path, weights_only=True)
    term_embs = torch.load(term_embs_path)
    return node_list, term_embs

def load_ic_map(path="checkpoints/hpo_ic.pt"):
    return torch.load(path)  # dict: term -> IC (float)

def load_disease_annotations(hpoa_path):
    """
    Returns mapping: disease_id -> list of (hpo_id, freq_weight)
    """
    mapping = defaultdict(list)
    with open(hpoa_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                # find columns
                try:
                    did_col  = header.index("database_id")
                    hpo_col  = header.index("hpo_id")
                    freq_col = header.index("frequency")
                except ValueError:
                    # fallback: best-effort search
                    did_col  = next(i for i,h in enumerate(header) if "database" in h and "id" in h)
                    hpo_col  = next(i for i,h in enumerate(header) if h.lower().startswith("hp"))
                    freq_col = next(i for i,h in enumerate(header) if "freq" in h.lower())
                continue

            disease = row[did_col].strip()
            hpo     = row[hpo_col].strip()
            freq    = parse_frequency(row[freq_col].strip() if freq_col < len(row) else "")
            if disease and hpo.startswith("HP:"):
                mapping[disease].append((hpo, freq))
    return mapping

def aggregate_weighted(disease2terms, node_list, term_embs, ic_map):
    term2idx = {t:i for i,t in enumerate(node_list)}
    disease_ids, disease_vecs = [], []
    for disease, pairs in disease2terms.items():
        idxs, wts = [], []
        for hp, f in pairs:
            if hp in term2idx:
                idxs.append(term2idx[hp])
                ic = float(ic_map.get(hp, 0.0))
                wts.append(max(1e-8, f * ic))  # IC × frequency
        if not idxs:
            continue
        idxs = torch.tensor(idxs, dtype=torch.long)
        w    = torch.tensor(wts, dtype=term_embs.dtype)
        embs = term_embs[idxs]  # [K, D]
        vec  = (embs * w.unsqueeze(1)).sum(dim=0) / w.sum()
        disease_ids.append(disease)
        disease_vecs.append(vec)
    if not disease_ids:
        raise RuntimeError("No disease embeddings computed; check your .hpoa parsing.")
    return disease_ids, torch.stack(disease_vecs, dim=0)

if __name__ == "__main__":
    node_list, term_embs = load_term_embeddings(
        "checkpoints/node_list.pt",
        "checkpoints/hpo_gcl_embeddings.pt",
    )
    ic_map = load_ic_map("checkpoints/hpo_ic.pt")
    disease2terms = load_disease_annotations("phenotype.hpoa")
    disease_ids, disease_embs = aggregate_weighted(disease2terms, node_list, term_embs, ic_map)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(disease_ids,  "checkpoints/disease_ids.pt")
    torch.save(disease_embs, "checkpoints/disease_embs.pt")
    print(f"Saved {len(disease_ids)} disease embeddings (IC × freq weighted).")
