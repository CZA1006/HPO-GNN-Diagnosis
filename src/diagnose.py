#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose.py — Embed a patient’s HPO set and rank candidate diseases.

- Lazily loads IC map and HPO parents on first use (or via ensure_loaded()).
- Safely clamps topk to the number of candidates.
- Supports ancestor expansion and optional subtraction of negated codes.
"""
import os
import torch
import obonet
from collections import deque
from typing import Dict, Iterable, List

# Globals (lazy-initialized)
_ic_map: Dict[str, float] = None
_parents: Dict[str, List[str]] = None
_default_ic: float = 0.0
_ic_path: str = "checkpoints/hpo_ic.pt"
_obo_path: str = "hp.obo"

# ---------- resource loaders ----------
def _load_ic(path: str = None):
    global _ic_map, _default_ic, _ic_path
    if path is not None:
        _ic_path = path
    if _ic_map is None:
        _ic_map = torch.load(_ic_path)
        _default_ic = min(_ic_map.values()) if _ic_map else 0.0

def _load_parents(obo_path: str = None):
    """Use obonet to get parent (is_a + part_of) relationships."""
    global _parents, _obo_path
    if obo_path is not None:
        _obo_path = obo_path
    if _parents is not None:
        return
    if not os.path.exists(_obo_path):
        raise FileNotFoundError(f"Cannot find obo file: {_obo_path}")

    graph = obonet.read_obo(_obo_path)
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

def ensure_loaded(obo_path: str = None, ic_path: str = None):
    """Public helper: call this once from other scripts after setting paths."""
    _load_ic(ic_path)
    _load_parents(obo_path)

# ---------- small utilities ----------
def _ancestors(term: str, max_depth: int = 2):
    """Return dict ancestor->distance (1..max_depth)."""
    if _parents is None:
        _load_parents()  # use last-known/default path
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

def load_embeddings(path_ids: str, path_embs: str):
    ids  = torch.load(path_ids)
    embs = torch.load(path_embs)
    return ids, embs

def _pooled_vector(codes: Iterable[str], term2idx: Dict[str, int], term_embs: torch.Tensor,
                   decay: float = 0.7, max_depth: int = 2) -> torch.Tensor:
    # Lazy init IC if needed
    if _ic_map is None:
        _load_ic()
    contrib: Dict[int, float] = {}
    for c in codes:
        if c in term2idx:
            w = float(_ic_map.get(c, _default_ic))
            contrib[term2idx[c]] = contrib.get(term2idx[c], 0.0) + w
        # ancestors
        for anc, dist in _ancestors(c, max_depth=max_depth).items():
            if anc in term2idx:
                w = (decay ** dist) * float(_ic_map.get(anc, _default_ic))
                idx = term2idx[anc]
                contrib[idx] = contrib.get(idx, 0.0) + w

    if not contrib:
        raise ValueError("No terms overlapped with embedding vocabulary.")

    idxs = torch.tensor(list(contrib.keys()), dtype=torch.long, device=term_embs.device)
    w    = torch.tensor([contrib[i] for i in idxs.tolist()], dtype=term_embs.dtype, device=term_embs.device)
    embs = term_embs[idxs]
    return (embs * w.unsqueeze(1)).sum(dim=0) / w.sum()

# ---------- public API ----------
def embed_patient(hpo_codes: List[str],
                  term_node_list: List[str],
                  term_embs: torch.Tensor,
                  neg_codes: List[str] = None,
                  decay: float = 0.7,
                  max_depth: int = 2,
                  neg_alpha: float = 0.5) -> torch.Tensor:
    # Ensure parents/IC are available (safe when imported)
    if _ic_map is None:
        _load_ic()
    if _parents is None:
        _load_parents()
    term2idx = {t: i for i, t in enumerate(term_node_list)}
    pos_vec = _pooled_vector(hpo_codes, term2idx, term_embs, decay=decay, max_depth=max_depth)

    if neg_codes:
        try:
            neg_vec = _pooled_vector(neg_codes, term2idx, term_embs, decay=decay, max_depth=max_depth)
            vec = pos_vec - neg_alpha * neg_vec
        except ValueError:
            vec = pos_vec
    else:
        vec = pos_vec
    return vec.unsqueeze(0)

def rank_diseases(patient_emb: torch.Tensor, disease_ids: List[str], disease_embs: torch.Tensor, topk: int = 10):
    """Return top-k (clamped) list of (disease_id, score)."""
    n = int(disease_embs.size(0)) if hasattr(disease_embs, "size") else len(disease_ids)
    if n == 0:
        return []
    k = max(1, min(topk, n))
    pe = patient_emb / (patient_emb.norm(dim=1, keepdim=True) + 1e-8)
    de = disease_embs / (disease_embs.norm(dim=1, keepdim=True) + 1e-8)
    sims = (pe @ de.t()).squeeze(0)
    vals, idxs = sims.topk(k, largest=True)
    return [(disease_ids[i], float(vals[j])) for j, i in enumerate(idxs)]

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo",              default="hp.obo")
    ap.add_argument("--term_node_list",   default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs",        default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--disease_ids",      default="checkpoints/disease_ids.pt")
    ap.add_argument("--disease_embs",     default="checkpoints/disease_embs.pt")
    ap.add_argument("--patient_hpos",     required=True, help="Comma-separated HP: codes")
    ap.add_argument("--patient_neg",      default="", help="Comma-separated negated HP: codes")
    ap.add_argument("--topk",             type=int, default=5)
    args = ap.parse_args()

    # explicit load with provided paths
    ensure_loaded(obo_path=args.obo, ic_path=_ic_path)

    term_nodes, term_embs   = load_embeddings(args.term_node_list, args.term_embs)
    disease_ids, disease_em = load_embeddings(args.disease_ids,   args.disease_embs)

    pos = [s.strip() for s in args.patient_hpos.split(",") if s.strip()]
    neg = [s.strip() for s in args.patient_neg.split(",") if s.strip()]
    emb = embed_patient(pos, term_nodes, term_embs, neg_codes=neg)

    top = rank_diseases(emb, disease_ids, disease_em, topk=args.topk)
    print("Top candidate diagnoses:")
    for did, score in top:
        print(f"{did}\t{score:.4f}")
