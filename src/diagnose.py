# src/diagnose.py

import os
import re
import torch
from collections import deque

_ic_map = None
_parents = None
_default_ic = 0.0

def _load_ic():
    global _ic_map, _default_ic
    if _ic_map is None:
        _ic_map = torch.load("checkpoints/hpo_ic.pt")
        _default_ic = min(_ic_map.values()) if _ic_map else 0.0

def _load_parents(obo_path="hp.obo"):
    """Parse hp.obo to build parent map: term -> set(parents) for is_a/part_of."""
    global _parents
    if _parents is not None:
        return
    parents = {}
    cur = None
    with open(obo_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = None
                continue
            if line.startswith("id: HP:"):
                cur = line.split("id: ")[1].strip()
                parents.setdefault(cur, set())
            elif cur and line.startswith("is_a: HP:"):
                pid = line.split("is_a: ")[1].split()[0]
                parents[cur].add(pid)
            elif cur and line.startswith("relationship: part_of HP:"):
                pid = line.split("part_of ")[1].split()[0]
                parents[cur].add(pid)
    _parents = parents

def _ancestors(term, max_depth=2):
    """Return dict ancestor->distance (1..max_depth)."""
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

def load_embeddings(path_ids, path_embs):
    ids  = torch.load(path_ids, weights_only=True)
    embs = torch.load(path_embs)
    return ids, embs

def _pooled_vector(codes, term2idx, term_embs, decay=0.7, max_depth=2):
    """IC-weighted pooling with ancestor expansion."""
    _load_ic()
    _load_parents()
    contrib = {}
    for c in codes:
        # self
        if c in term2idx:
            contrib[term2idx[c]] = contrib.get(term2idx[c], 0.0) + float(_ic_map.get(c, _default_ic))
        # ancestors
        for anc, dist in _ancestors(c, max_depth=max_depth).items():
            if anc in term2idx:
                w = (decay ** dist) * float(_ic_map.get(anc, _default_ic))
                idx = term2idx[anc]
                contrib[idx] = contrib.get(idx, 0.0) + w

    if not contrib:
        raise ValueError("No terms overlapped with embedding vocabulary.")

    idxs = torch.tensor(list(contrib.keys()), dtype=torch.long, device=term_embs.device)
    w    = torch.tensor([contrib[i] for i in idxs.tolist()],
                        dtype=term_embs.dtype, device=term_embs.device)
    embs = term_embs[idxs]
    return (embs * w.unsqueeze(1)).sum(dim=0) / w.sum()

def embed_patient(hpo_codes, term_node_list, term_embs,
                  neg_codes=None, decay=0.7, max_depth=2, neg_alpha=0.5):
    """
    Build patient embedding with IC+ancestor expansion and optional subtraction of negated codes.
    Returns [1, D] tensor.
    """
    term2idx = {t:i for i,t in enumerate(term_node_list)}
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

def rank_diseases(patient_emb, disease_ids, disease_embs, topk=10):
    pe = patient_emb / patient_emb.norm(dim=1, keepdim=True)
    de = disease_embs / disease_embs.norm(dim=1, keepdim=True)
    sims = (pe @ de.t()).squeeze(0)
    vals, idxs = sims.topk(topk, largest=True)
    return [(disease_ids[i], float(vals[j])) for j,i in enumerate(idxs)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--term_node_list", default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs",      default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--disease_ids",    default="checkpoints/disease_ids.pt")
    ap.add_argument("--disease_embs",   default="checkpoints/disease_embs.pt")
    ap.add_argument("--patient_hpos",   required=True)
    ap.add_argument("--patient_neg",    default="")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    term_nodes, term_embs   = load_embeddings(args.term_node_list, args.term_embs)
    disease_ids, disease_em = load_embeddings(args.disease_ids,   args.disease_embs)

    pos = [s.strip() for s in args.patient_hpos.split(",") if s.strip()]
    neg = [s.strip() for s in args.patient_neg.split(",") if s.strip()]
    emb = embed_patient(pos, term_nodes, term_embs, neg_codes=neg)

    top = rank_diseases(emb, disease_ids, disease_em, topk=args.topk)
    print("Top candidate diagnoses:")
    for did, score in top:
        print(f"{did}\t{score:.4f}")
