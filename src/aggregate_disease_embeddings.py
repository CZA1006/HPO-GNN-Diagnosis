#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_disease_embeddings.py — Pool term embeddings into disease embeddings.

Weights each term vector by:
  w(term) = IC(term) × frequency_weight × (ancestor_decay^dist)  # dist=0 for original terms

New (optional):
  --ancestor_depth N   (default 0 → off)
  --ancestor_decay D   (default 0.7)
  --obo hp.obo         (required if --ancestor_depth > 0)

Outputs:
  checkpoints/disease_ids.pt   (list[str])
  checkpoints/disease_embs.pt  (torch.FloatTensor [num_diseases, dim])
"""
import os
import csv
import torch
import obonet
from collections import defaultdict, deque
from typing import Dict, List, Tuple

# HPO frequency term IDs → approximate probabilities
FREQ_MAP = {
    "HP:0040280": 1.00,  # Obligate
    "HP:0040281": 0.99,  # Very frequent
    "HP:0040282": 0.80,  # Frequent
    "HP:0040283": 0.30,  # Occasional
    "HP:0040284": 0.05,  # Rare
    "HP:0040285": 0.01,  # Very rare
}

def _parse_freq(value: str) -> float:
    if not value:
        return 1.0
    v = value.strip()
    if v in FREQ_MAP:
        return FREQ_MAP[v]
    if "/" in v:
        try:
            a, b = v.split("/", 1)
            a = float(a); b = float(b)
            return max(1e-8, min(1.0, a / b)) if b else 1.0
        except Exception:
            return 1.0
    if v.endswith("%"):
        try:
            return max(1e-8, min(1.0, float(v[:-1]) / 100.0))
        except Exception:
            return 1.0
    try:
        x = float(v)
        return max(1e-8, min(1.0, x if x <= 1.0 else x / 100.0))
    except Exception:
        return 1.0

def load_term_embeddings(node_list_path: str, term_embs_path: str):
    node_list = torch.load(node_list_path)
    term_embs = torch.load(term_embs_path)
    return node_list, term_embs

def load_ic_map(path: str = "checkpoints/hpo_ic.pt") -> Dict[str, float]:
    return torch.load(path)

def load_disease_annotations(hpoa_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns mapping: disease_id -> list of (hpo_id, freq_weight)
    """
    mapping: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open(hpoa_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        cols = {}
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                lower = [h.lower() for h in header]
                def find(colname, fallback=None):
                    try:
                        return lower.index(colname)
                    except ValueError:
                        return fallback
                cols["did"]  = find("database_id")
                cols["hpo"]  = find("hpo_id")
                cols["freq"] = find("frequency", None)
                if cols["did"] is None or cols["hpo"] is None:
                    raise RuntimeError(f"Header missing database_id or hpo_id: {header}")
                continue

            did  = row[cols["did"]].strip()
            hpo  = row[cols["hpo"]].strip()
            freq = _parse_freq(row[cols["freq"]].strip()) if cols["freq"] is not None and cols["freq"] < len(row) else 1.0
            if not did or not hpo:
                continue
            mapping[did].append((hpo, freq))
    return mapping

# -------- Ancestor utilities (shared logic with diagnose) --------
_parents = None
def _load_parents(obo_path: str):
    global _parents
    if _parents is not None:
        return
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"Cannot find obo file: {obo_path}")
    graph = obonet.read_obo(obo_path)
    parents = {n: set() for n in graph.nodes()}
    for n, d in graph.nodes(data=True):
        for p in d.get("is_a", []):
            p = p.split("!")[0].strip()
            if p.startswith("HP:"):
                parents[n].add(p)
        for rel in d.get("relationship", []):
            if "part_of" in rel and "HP:" in rel:
                pid = rel.split("HP:")[1][:7]
                parents[n].add("HP:" + pid)
    _parents = {k: sorted(list(v)) for k, v in parents.items() if v}

def _ancestors(term: str, max_depth: int = 2):
    if term not in _parents:
        return {}
    out = {}
    q = deque([(term, 0)])
    seen = {term}
    while q:
        t, d = q.popleft()
        if d == max_depth:
            continue
        for p in _parents.get(t, ()):
            if p in seen:
                continue
            seen.add(p)
            out[p] = d + 1
            q.append((p, d + 1))
    return out

def aggregate_weighted(
    disease2terms: Dict[str, List[Tuple[str, float]]],
    node_list: List[str],
    term_embs: torch.Tensor,
    ic_map: Dict[str, float],
    ancestor_depth: int = 0,
    ancestor_decay: float = 0.7,
):
    term2idx = {t: i for i, t in enumerate(node_list)}
    disease_ids, disease_vecs = [], []

    for disease, pairs in disease2terms.items():
        contrib = {}
        for hp, f in pairs:
            if hp in term2idx:
                w = max(1e-8, f * float(ic_map.get(hp, 0.0)))
                contrib[term2idx[hp]] = contrib.get(term2idx[hp], 0.0) + w

            # optional ancestor expansion
            if ancestor_depth > 0:
                for anc, dist in _ancestors(hp, max_depth=ancestor_depth).items():
                    if anc in term2idx:
                        w = max(1e-8, f * (ancestor_decay ** dist) * float(ic_map.get(anc, 0.0)))
                        idx = term2idx[anc]
                        contrib[idx] = contrib.get(idx, 0.0) + w

        if not contrib:
            continue

        idxs = torch.tensor(list(contrib.keys()), dtype=torch.long)
        wts  = torch.tensor([contrib[i] for i in idxs.tolist()], dtype=term_embs.dtype)
        embs = term_embs[idxs]
        vec  = (embs * wts.unsqueeze(1)).sum(dim=0) / wts.sum()
        disease_ids.append(disease)
        disease_vecs.append(vec)

    if not disease_ids:
        raise RuntimeError("No disease embeddings computed; check your .hpoa parsing.")
    return disease_ids, torch.stack(disease_vecs, dim=0)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_list", default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs", default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--hpoa",      default="phenotype.hpoa")
    ap.add_argument("--ic",        default="checkpoints/hpo_ic.pt")
    ap.add_argument("--out_dir",   default="checkpoints")
    ap.add_argument("--ancestor_depth", type=int, default=0, help="0 disables ancestor expansion")
    ap.add_argument("--ancestor_decay", type=float, default=0.7)
    ap.add_argument("--obo", default=None, help="Path to hp.obo (required if --ancestor_depth > 0)")
    args = ap.parse_args()

    if args.ancestor_depth > 0:
        if not args.obo:
            raise SystemExit("--obo is required when --ancestor_depth > 0")
        _load_parents(args.obo)

    node_list, term_embs = load_term_embeddings(args.node_list, args.term_embs)
    ic_map = load_ic_map(args.ic)
    disease2terms = load_disease_annotations(args.hpoa)
    disease_ids, disease_embs = aggregate_weighted(
        disease2terms, node_list, term_embs, ic_map,
        ancestor_depth=args.ancestor_depth, ancestor_decay=args.ancestor_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(disease_ids,  os.path.join(args.out_dir, "disease_ids.pt"))
    torch.save(disease_embs, os.path.join(args.out_dir, "disease_embs.pt"))
    print(f"Saved {len(disease_ids)} disease embeddings to {args.out_dir}/ (IC × freq × ancestor weighting).")
