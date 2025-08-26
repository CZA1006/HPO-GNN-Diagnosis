#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose.py
Rank candidate diseases for a patient given HPO terms.

Patient vector = pooled sum of term embeddings with weights:
  w = IC(term) * (idf(term) ** idf_gamma) * (ancestor_decay ** distance)
Negated HPOs subtract their contribution.
"""

import argparse, os, math
import torch
import importlib.util

# ---------------- global caches ----------------
_ic_path = "checkpoints/hpo_ic.pt"
_ic_map = None
_idf_path = None
_idf_map = None
_idf_gamma = 1.0
_parents = None

def _import_obonet():
    import obonet
    return obonet

def _load_parents(obo_path):
    obonet = _import_obonet()
    g = obonet.read_obo(obo_path)
    parents = {}
    for n, d in g.nodes(data=True):
        if not n.startswith("HP:"): continue
        ps = set()
        for p in d.get("is_a", []):
            p = p.split("!")[0].strip()
            if p.startswith("HP:"):
                ps.add(p)
        for rel in d.get("relationship", []):
            if "part_of" in rel and "HP:" in rel:
                pid = rel.split("HP:")[1][:7]
                ps.add("HP:"+pid)
        if ps: parents[n] = sorted(ps)
    return parents

def ensure_loaded(obo_path="hp.obo", ic_path="checkpoints/hpo_ic.pt",
                  idf_path=None, idf_gamma=1.0):
    global _ic_path, _ic_map, _parents, _idf_path, _idf_map, _idf_gamma
    if _ic_map is None or _ic_path != ic_path:
        _ic_path = ic_path
        _ic_map = torch.load(ic_path)
    if _parents is None and os.path.exists(obo_path):
        _parents = _load_parents(obo_path)
    if idf_path and ( _idf_map is None or _idf_path != idf_path ):
        _idf_path  = idf_path
        _idf_map   = torch.load(idf_path)
        _idf_gamma = float(idf_gamma)

def _ancestors(term, max_depth):
    if _parents is None or max_depth <= 0: return {}
    from collections import deque
    out, q, seen = {}, deque([(term, 0)]), {term}
    while q:
        t,d = q.popleft()
        if d == max_depth: continue
        for p in _parents.get(t, ()):
            if p in seen: continue
            seen.add(p); out[p] = d+1; q.append((p, d+1))
    return out

def _pooled_vector(hpo_codes, term2idx, term_embs, decay=0.7, max_depth=2):
    # weighted sum with IC × (idf**gamma)
    acc = torch.zeros(term_embs.size(1), dtype=term_embs.dtype)
    for c in hpo_codes:
        w = float(_ic_map.get(c, 0.0))
        if _idf_map is not None:
            w *= float(_idf_map.get(c, 0.0)) ** float(_idf_gamma)
        if c in term2idx:
            acc += term_embs[term2idx[c]] * w
        for anc, dist in _ancestors(c, max_depth).items():
            if anc in term2idx:
                dec = (decay ** dist)
                w_anc = float(_ic_map.get(anc, 0.0))
                if _idf_map is not None:
                    w_anc *= float(_idf_map.get(anc, 0.0)) ** float(_idf_gamma)
                acc += term_embs[term2idx[anc]] * (w_anc * dec)
    # L2 normalize
    norm = acc.norm() + 1e-8
    return (acc / norm).unsqueeze(0)  # [1, dim]

def embed_patient(pos_codes, term_nodes, term_embs, neg_codes=None,
                  max_depth=2, decay=0.7):
    term2idx = {hp:i for i,hp in enumerate(term_nodes)}
    pos_emb = _pooled_vector(pos_codes, term2idx, term_embs, decay=decay, max_depth=max_depth)
    if neg_codes:
        neg_emb = _pooled_vector(neg_codes, term2idx, term_embs, decay=decay, max_depth=max_depth)
        emb = pos_emb - 0.5 * neg_emb
    else:
        emb = pos_emb
    return emb  # [1, dim]

def rank_diseases(patient_emb, disease_ids, disease_embs, topk=10):
    # cosine similarity
    pe = patient_emb / (patient_emb.norm(dim=1, keepdim=True) + 1e-8)
    de = disease_embs / (disease_embs.norm(dim=1, keepdim=True) + 1e-8)
    sims = (pe @ de.t()).squeeze(0)  # [num_diseases]
    k = min(topk, sims.numel())
    vals, idxs = sims.topk(k, largest=True)
    return [(disease_ids[i], float(vals[j])) for j,i in enumerate(idxs)]

def _import_evaluate():
    # allows `diagnose.py` to be used stand-alone from CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", default="hp.obo")
    ap.add_argument("--ic", default="checkpoints/hpo_ic.pt")
    ap.add_argument("--idf", default=None, help="Optional path to checkpoints/hpo_idf.pt")
    ap.add_argument("--idf_gamma", type=float, default=1.0)
    ap.add_argument("--term_node_list", default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs", default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--disease_ids", default="checkpoints/disease_ids.pt")
    ap.add_argument("--disease_embs", default="checkpoints/disease_embs.pt")
    ap.add_argument("--patient_hpos", required=True, help="Comma-separated HP:xxxxxxx list")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    ensure_loaded(obo_path=args.obo, ic_path=args.ic, idf_path=args.idf, idf_gamma=args.idf_gamma)
    term_nodes  = torch.load(args.term_node_list)
    term_embs   = torch.load(args.term_embs)
    disease_ids = torch.load(args.disease_ids)
    disease_embs= torch.load(args.disease_embs)

    pos = [t.strip() for t in args.patient_hpos.split(",") if t.strip()]
    emb = embed_patient(pos, term_nodes, term_embs, neg_codes=None, max_depth=2, decay=0.7)
    ranked = rank_diseases(emb, disease_ids, disease_embs, topk=args.topk)
    print("Top candidate diagnoses:")
    for did, sc in ranked:
        print(f"{did:12s} {sc:8.4f}")

if __name__ == "__main__":
    _import_evaluate()
