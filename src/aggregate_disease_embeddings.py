#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_disease_embeddings.py

Build disease embeddings by pooling HPO term embeddings from HPOA with weights:
  w = IC(term) * freq(term|disease) * (ancestor_decay ** distance) * (idf(term) ** idf_gamma)

Outputs:
  checkpoints/disease_ids.pt  # [str] disease CURIEs
  checkpoints/disease_embs.pt # torch.FloatTensor [num_diseases, dim]
"""

import argparse, csv, os, math
from collections import defaultdict
import torch

def load_pt(path):
    return torch.load(path)

def parse_hpoa(path):
    # disease_id -> list of (hp, freq_weight)
    ann = defaultdict(list)
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
                    try: return lower.index(col)
                    except ValueError: return None
                cols["did"]  = find("database_id")
                cols["hpo"]  = find("hpo_id")
                cols["qual"] = find("qualifier")
                cols["freq"] = find("frequency")
                assert cols["did"] is not None and cols["hpo"] is not None
                continue
            did = row[cols["did"]].strip()
            hp  = row[cols["hpo"]].strip()
            if not did or not hp:
                continue
            if cols["qual"] is not None and cols["qual"] < len(row):
                if row[cols["qual"]].strip().upper() == "NOT":
                    continue
            # very light frequency parsing; fallback to 1.0
            w = 1.0
            if cols["freq"] is not None and cols["freq"] < len(row):
                s = row[cols["freq"]].strip()
                if "/" in s:
                    try:
                        a,b = s.split("/")
                        w = float(a)/float(b)
                    except Exception:
                        pass
                else:
                    try:
                        v = float(s)
                        # treat 0..100 as percent
                        w = v/100.0 if v > 1.0 else max(0.0, min(1.0, v))
                    except Exception:
                        pass
            ann[did].append((hp, float(w)))
    return ann

def build_parents(obo_path):
    # minimal parent map (HP -> parents)
    import obonet
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

def ancestors(term, parents, max_depth):
    if max_depth <= 0: return {}
    from collections import deque
    out, q, seen = {}, deque([(term, 0)]), {term}
    while q:
        t,d = q.popleft()
        if d == max_depth: continue
        for p in parents.get(t, ()):
            if p in seen: continue
            seen.add(p); out[p] = d+1; q.append((p, d+1))
    return out

def aggregate_weighted(ann, node_list, term_embs, ic_map,
                       idf_map=None, idf_gamma=1.0,
                       parents=None, ancestor_depth=0, ancestor_decay=0.7):
    term2idx = {hp:i for i,hp in enumerate(node_list)}
    D, dim = len(ann), term_embs.size(1)
    disease_ids, disease_vecs = [], torch.zeros((D, dim), dtype=term_embs.dtype)

    def term_weight(hp, base_w):
        w = base_w
        if ic_map is not None:
            w *= float(ic_map.get(hp, 0.0))
        if idf_map is not None:
            w *= float(idf_map.get(hp, 0.0)) ** float(idf_gamma)
        return w

    for d_idx, (did, pairs) in enumerate(ann.items()):
        disease_ids.append(did)
        acc = torch.zeros(dim, dtype=term_embs.dtype)
        for hp, f in pairs:
            # self
            if hp in term2idx:
                acc += term_embs[term2idx[hp]] * term_weight(hp, f)
            # ancestors
            if parents and ancestor_depth > 0:
                for anc, dist in ancestors(hp, parents, ancestor_depth).items():
                    if anc in term2idx:
                        decay = (ancestor_decay ** dist)
                        acc += term_embs[term2idx[anc]] * term_weight(anc, f * decay)
        # L2 normalize to keep scale comparable
        norm = acc.norm() + 1e-8
        disease_vecs[d_idx] = acc / norm
    return disease_ids, disease_vecs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_list", default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs", default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--ic", default="checkpoints/hpo_ic.pt")
    ap.add_argument("--idf", default=None, help="Optional path to checkpoints/hpo_idf.pt")
    ap.add_argument("--idf_gamma", type=float, default=1.0, help="Exponent on IDF weight")
    ap.add_argument("--obo", default=None)
    ap.add_argument("--ancestor_depth", type=int, default=0)
    ap.add_argument("--ancestor_decay", type=float, default=0.7)
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    node_list = load_pt(args.node_list)
    term_embs  = load_pt(args.term_embs)
    ic_map     = load_pt(args.ic) if args.ic else None
    idf_map    = load_pt(args.idf) if args.idf else None

    ann = parse_hpoa(args.hpoa)
    parents = build_parents(args.obo) if args.obo and args.ancestor_depth > 0 else None

    disease_ids, disease_embs = aggregate_weighted(
        ann, node_list, term_embs, ic_map,
        idf_map=idf_map, idf_gamma=args.idf_gamma,
        parents=parents, ancestor_depth=args.ancestor_depth, ancestor_decay=args.ancestor_decay
    )

    path_ids  = os.path.join(args.out_dir, "disease_ids.pt")
    path_embs = os.path.join(args.out_dir, "disease_embs.pt")
    torch.save(disease_ids, path_ids)
    torch.save(disease_embs, path_embs)

    msg = "IC × freq"
    if args.obo and args.ancestor_depth > 0:
        msg += " × ancestor"
    if args.idf:
        msg += f" × idf^{args.idf_gamma:g}"
    print(f"Saved {len(disease_ids)} disease embeddings to {args.out_dir}/ ({msg}).")

if __name__ == "__main__":
    main()
