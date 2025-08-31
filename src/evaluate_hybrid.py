#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, random
from glob import glob
from collections import defaultdict, Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import diagnose  # uses embed_patient()

# ---------- helpers ----------

def _read_phenopacket(path, min_hpo=2):
    try:
        obj = json.load(open(path, "r"))
    except Exception:
        return None
    # gold id
    gold_id = None
    dx = obj.get("disease", []) or obj.get("diseases", [])
    if isinstance(dx, dict):
        dx = [dx]
    if dx:
        gold_id = (dx[0].get("term", {}) or dx[0].get("disease", {})).get("id")
        if gold_id:
            gold_id = gold_id.strip()
    # phenotypes
    pos, neg = set(), set()
    feats = obj.get("phenotypicFeatures", []) or obj.get("phenotypic_features", [])
    for f in feats:
        tid = (f.get("type") or {}).get("id") or (f.get("term") or {}).get("id") or f.get("id")
        if not tid:
            continue
        (neg if f.get("excluded", False) else pos).add(tid.strip())
    if len(pos) < min_hpo:
        return None
    return gold_id, pos, neg


def _load_phenopackets(folder, min_hpo=2):
    return [(p, *r) for p in sorted(glob(os.path.join(folder, "**", "*.json"), recursive=True))
            if (r := _read_phenopacket(p, min_hpo))]


def _load_hpoa(hpoa_path):
    d2t = defaultdict(set)
    with open(hpoa_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            did, hpo = cols[0].strip(), cols[3].strip()
            if did and hpo.startswith("HP:"):
                d2t[did].add(hpo)
    return d2t


def _load_obo_parents(obo_path):
    parents = defaultdict(set)
    cur = None
    with open(obo_path, "r") as f:
        for line in f:
            s = line.strip()
            if s == "[Term]":
                cur = None
                continue
            if s.startswith("id: "):
                cur = s[4:].strip()
            elif cur and s.startswith("is_a: "):
                pid = s.split("is_a: ", 1)[1].split(" ! ")[0].strip()
                parents[cur].add(pid)
    return parents


def _ancestors(term, parents, depth):
    out, frontier = {term}, [term]
    for _ in range(depth):
        nxt = []
        for t in frontier:
            for p in parents.get(t, ()):
                if p not in out:
                    out.add(p); nxt.append(p)
        frontier = nxt
        if not frontier:
            break
    return out


def _expand_with_ancestors(terms, parents, depth):
    if depth <= 0:
        return set(terms)
    ex = set()
    for t in terms:
        ex.update(_ancestors(t, parents, depth))
    return ex


def _hybrid_overlap_score(patient_hpos, disease_hpos, cos_sim, ic_map, idf_map, alpha, beta, idf_gamma):
    inter = patient_hpos & disease_hpos
    union = patient_hpos | disease_hpos
    jacc = (len(inter) / len(union)) if union else 0.0
    inter_ic  = sum(ic_map.get(h, 0.0)  for h in inter)
    inter_idf = sum(idf_map.get(h, 0.0) for h in inter)
    patient_ic = sum(ic_map.get(h, 0.0) for h in patient_hpos) + 1e-9
    overlap = (inter_ic + idf_gamma * inter_idf) / (1.0 + idf_gamma)
    overlap_norm = overlap / patient_ic
    rem = max(0.0, 1.0 - alpha - beta)
    return alpha * float(cos_sim) + beta * jacc + rem * overlap_norm


# ---------- args ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phenopackets_dir", type=str, default="phenopackets")
    p.add_argument("--node_list", type=str, default="checkpoints/node_list.pt")
    p.add_argument("--term_embs", type=str, default="checkpoints/hpo_gcl_embeddings.pt")
    p.add_argument("--disease_ids", type=str, default="checkpoints/disease_ids.pt")
    p.add_argument("--disease_embs", type=str, default="checkpoints/disease_embs.pt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min_hpo", type=int, default=2)

    p.add_argument("--hpoa", type=str, default="phenotype.hpoa")
    p.add_argument("--obo", type=str, default="hp.obo")
    p.add_argument("--mondo", type=str, default=None)

    p.add_argument("--ic", type=str, default="checkpoints/hpo_ic.pt")
    p.add_argument("--idf", type=str, default=None)
    p.add_argument("--idf_gamma", type=float, default=1.0)

    p.add_argument("--filter_by_overlap", action="store_true")
    p.add_argument("--filter_depth", type=int, default=1)
    p.add_argument("--filter_min_terms", type=int, default=2)
    p.add_argument("--filter_min_ic", type=float, default=0.0)
    p.add_argument("--filter_keep_top", type=int, default=500)

    p.add_argument("--patient_depth", type=int, default=0)
    p.add_argument("--patient_decay", type=float, default=0.7)

    p.add_argument("--hybrid_alpha", type=float, default=0.9)
    p.add_argument("--hybrid_beta",  type=float, default=0.1)

    p.add_argument("--report_top", type=int, nargs="*", default=[5, 10, 50, 100])

    p.add_argument("--roc_use", choices=["cosine", "hybrid"], default="cosine")
    p.add_argument("--roc_negatives", type=int, default=300)
    return p.parse_args()


# ---------- main ----------

def main():
    args = parse_args()

    # checkpoints
    term_nodes  = torch.load(args.node_list)
    term_embs   = torch.load(args.term_embs)
    disease_ids = torch.load(args.disease_ids)
    disease_embs = torch.load(args.disease_embs)

    ic_map = torch.load(args.ic)  if args.ic  and os.path.exists(args.ic)  else {}
    idf_map = torch.load(args.idf) if args.idf and os.path.exists(args.idf) else defaultdict(float)

    # CRITICAL: expose IC/IDF to diagnose (embed_patient() reads globals)
    diagnose._ic_map  = ic_map
    diagnose._idf_map = idf_map
    if not hasattr(diagnose, "_default_ic") or diagnose._default_ic is None:
        vals = list(ic_map.values())
        diagnose._default_ic = float(np.median(vals)) if vals else 0.0

    # normalize disease embs for cosine
    with torch.no_grad():
        dmat = disease_embs / (disease_embs.norm(dim=1, keepdim=True) + 1e-9)

    disease2terms = _load_hpoa(args.hpoa)
    parents = _load_obo_parents(args.obo)
    cases = _load_phenopackets(args.phenopackets_dir, min_hpo=args.min_hpo)

    # metrics
    top_hits = {k: 0 for k in args.report_top}
    top_hits_matched = {k: 0 for k in args.report_top}
    mrr = mrr_matched = 0.0
    n_all = n_matched = 0
    not_in_candidates = 0

    roc_y_true, roc_y_score = [], []

    gold_ns = Counter()
    cand_ns = Counter(s.split(":")[0] for s in disease_ids)
    for _, g, _, _ in cases:
        if g:
            gold_ns[g.split(":")[0]] += 1
    print("ID namespaces in gold (top 10):", gold_ns.most_common(10))
    print("ID namespaces in candidates (top 10):", cand_ns.most_common(10))

    id2idx = {did: i for i, did in enumerate(disease_ids)}

    for _, gold_id, pos_hpos, neg_hpos in cases:
        if len(pos_hpos) < args.min_hpo:
            continue

        pos_for_embed = _expand_with_ancestors(pos_hpos, parents, args.patient_depth)
        emb = diagnose.embed_patient(
            list(pos_for_embed),
            term_nodes,
            term_embs,
            neg_codes=list(neg_hpos),
            decay=args.patient_decay,
            max_depth=0,
        )
        # Ensure 1-D vector for mv()
        if hasattr(emb, 'dim') and emb.dim() > 1:
            emb = emb.reshape(-1)
        emb = emb.to(dmat.device).float()
        emb = emb / (emb.norm() + 1e-9)

        # filtering
        if args.filter_by_overlap:
            pat_for_filter = _expand_with_ancestors(pos_hpos, parents, args.filter_depth)
            scored = []
            for did, hset in disease2terms.items():
                inter = pat_for_filter & hset
                if len(inter) < args.filter_min_terms:
                    continue
                ic_sum = sum(ic_map.get(h, 0.0) for h in inter)
                if ic_sum < args.filter_min_ic:
                    continue
                idf_sum = sum(idf_map.get(h, 0.0) for h in inter)
                overlap = ic_sum + args.idf_gamma * idf_sum
                scored.append((overlap, did))
            scored.sort(reverse=True)
            kept = [d for _, d in scored[: args.filter_keep_top]]
        else:
            kept = disease_ids

        sub_ids, sub_rows = [], []
        for did in kept:
            j = id2idx.get(did)
            if j is not None:
                sub_ids.append(did); sub_rows.append(j)
        if not sub_rows:
            n_all += 1
            if gold_id not in id2idx:
                not_in_candidates += 1
            continue

        sub_embs = dmat[torch.tensor(sub_rows, dtype=torch.long, device=dmat.device)]
        sims = torch.mv(sub_embs, emb)

        pat_set_for_hybrid = _expand_with_ancestors(pos_hpos, parents, args.filter_depth)
        sub_scores = []
        for i, did in enumerate(sub_ids):
            hset = disease2terms.get(did, set())
            sub_scores.append(
                _hybrid_overlap_score(
                    pat_set_for_hybrid, hset, sims[i].item(),
                    ic_map, idf_map, args.hybrid_alpha, args.hybrid_beta, args.idf_gamma
                )
            )
        sub_scores = np.asarray(sub_scores, dtype=np.float32)
        order = np.argsort(-sub_scores)
        ranked_ids = [sub_ids[i] for i in order]

        n_all += 1
        if gold_id in id2idx:
            n_matched += 1

        if gold_id in ranked_ids:
            r = ranked_ids.index(gold_id) + 1
            mrr += 1.0 / r
            if gold_id in kept:
                mrr_matched += 1.0 / r
            for k in args.report_top:
                if r <= k:
                    top_hits[k] += 1
                    if gold_id in kept:
                        top_hits_matched[k] += 1
        else:
            if gold_id not in id2idx:
                not_in_candidates += 1

        # ROC accumulation
        if args.roc_use == "cosine":
            if gold_id in id2idx:
                gidx = id2idx[gold_id]
                roc_y_true.append(1)
                roc_y_score.append(float(torch.dot(dmat[gidx], emb)))
                total = len(disease_ids) - 1
                if total > 0:
                    take = min(args.roc_negatives, total)
                    pool = list(range(len(disease_ids))); pool.remove(gidx)
                    for j in random.sample(pool, take):
                        roc_y_true.append(0)
                        roc_y_score.append(float(torch.dot(dmat[j], emb)))
        else:
            if gold_id in sub_ids:
                g = sub_ids.index(gold_id)
                roc_y_true.append(1); roc_y_score.append(float(sub_scores[g]))
                total = len(sub_ids) - 1
                if total > 0:
                    take = min(args.roc_negatives, total)
                    pool = [i for i in range(len(sub_ids)) if i != g]
                    for j in random.sample(pool, take):
                        roc_y_true.append(0); roc_y_score.append(float(sub_scores[j]))

    print(f"Gold not present in candidates: {not_in_candidates} / {n_all}")
    print(f"Evaluated {n_all} cases (matched-in-candidates: {n_matched})")
    print("== Full candidate set ==")
    for k in args.report_top:
        overall_acc = top_hits[k] / max(1, n_all)
        matched_acc = top_hits_matched[k] / max(1, n_matched)
        print(f"Overall Top-{k}: {overall_acc:.4f} | MRR: {mrr / max(1, n_all):.4f}")
        print(f"Matched Top-{k}: {matched_acc:.4f} | MRR: {mrr_matched / max(1, n_matched):.4f}")

    if len(set(roc_y_true)) >= 2:
        fpr, tpr, _ = roc_curve(roc_y_true, roc_y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(loc="lower right")
        plt.tight_layout(); plt.show()
    else:
        print("[WARN] Not enough positives/negatives to plot ROC.")


if __name__ == "__main__":
    main()
