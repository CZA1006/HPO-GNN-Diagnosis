#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evaluate_hybrid.py : Batch evaluation on phenopackets with optional hybrid scoring on the filtered pool.
# Adds two flags:
#   --hybrid_alpha : weight on IC-overlap (0..1)
#   --hybrid_beta  : weight on overlap-count (0..1)
# Score on filtered pool = (1 - alpha - beta) * cosine + alpha * overlap_IC + beta * overlap_count

import os
import glob
import json
import csv
import random
import argparse
from collections import Counter, defaultdict, deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import importlib.util
import obonet  # for MONDO + HPO parent graph

# ---------- import diagnose ----------
def _import_diagnose():
    for candidate in ("diagnose.py", os.path.join("src", "diagnose.py")):
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("diagnose", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    import diagnose as mod
    return mod

diagnose = _import_diagnose()

# ---------- small utils ----------
def compute_metrics(results, k=5):
    topk_hits, mrr_total = 0, 0.0
    for true_id, ranked in results:
        preds = [did for did, _ in ranked]
        if true_id in preds[:k]:
            topk_hits += 1
            rank = preds.index(true_id) + 1
            mrr_total += 1.0 / rank
    n = len(results)
    return (topk_hits / n if n else 0.0, mrr_total / n if n else 0.0)

def extract_true_disease_id(data):
    # Common phenopacket patterns
    if "diseases" in data and isinstance(data["diseases"], list) and data["diseases"]:
        d0 = data["diseases"][0]
        if isinstance(d0, dict):
            term = d0.get("term") or d0.get("disease") or {}
            did = term.get("id") or d0.get("id")
            if isinstance(did, str):
                return did
    if isinstance(data.get("disease"), dict):
        did = data["disease"].get("id")
        if did:
            return did
    # last resort: first CURIE-like string
    for _, v in data.items():
        if isinstance(v, str) and ":" in v and v.split(":")[0].isupper():
            return v
    return None

def extract_hpo_sets(data):
    pos, neg = [], []
    feats = data.get("phenotypicFeatures") or data.get("phenotypes") or []
    for feat in feats:
        try:
            t = feat.get("type") or {}
            code = t.get("id") or feat.get("hpoId")
            negated = bool(feat.get("excluded") or feat.get("negated"))
            if code and code.startswith("HP:"):
                (neg if negated else pos).append(code)
        except Exception:
            continue
    return pos, neg

# ---------- HPO parents (for ancestor expansion during filtering) ----------
def _load_parents(obo_path: str):
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
    return {k: sorted(list(v)) for k, v in parents.items() if v}

def _ancestors(term: str, parents, max_depth: int):
    if term not in parents or max_depth <= 0:
        return {}
    out, q, seen = {}, deque([(term, 0)]), {term}
    while q:
        t, d = q.popleft()
        if d == max_depth:
            continue
        for p in parents.get(t, ()):
            if p in seen:
                continue
            seen.add(p)
            out[p] = d + 1
            q.append((p, d + 1))
    return out

# ---------- inverted index: HPO term -> set(disease indices) ----------
def _build_term_to_disease_index(hpoa_path: str, disease_ids: list):
    idx_by_id = {did: i for i, did in enumerate(disease_ids)}
    term2idxs = defaultdict(set)
    with open(hpoa_path, newline="") as f:
        reader = csv.reader(f, delimiter="	")
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
                    raise RuntimeError(f"Header missing database_id or hpo_id: {header}")
                continue
            did = row[cols["did"]].strip()
            hp  = row[cols["hpo"]].strip()
            if not did or not hp:
                continue
            if cols["qual"] is not None and cols["qual"] < len(row):
                if row[cols["qual"]].strip().upper() == "NOT":
                    continue
            if did in idx_by_id:
                term2idxs[hp].add(idx_by_id[did])
    return {k: sorted(v) for k, v in term2idxs.items()}

# ---------- MONDO canonicalization ----------
def build_mondo_index(mondo_path):
    g = obonet.read_obo(mondo_path)
    idx = {}
    for n, d in g.nodes(data=True):
        idx[n] = n  # MONDO maps to itself
        for alt in d.get("alt_id", []):
            idx[alt] = n
        for x in d.get("xref", []):  # e.g., "OMIM:123456"
            curie = x.split()[0]
            idx[curie] = n
    return idx

def canonize_results(results, idx):
    out = []
    for gold, ranked in results:
        g = idx.get(gold, gold)
        out.append((g, [(idx.get(did, did), sc) for did, sc in ranked]))
    return out

def _minmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenopackets_dir", default="phenopackets")
    ap.add_argument("--node_list", default="checkpoints/node_list.pt")
    ap.add_argument("--term_embs", default="checkpoints/hpo_gcl_embeddings.pt")
    ap.add_argument("--disease_ids", default="checkpoints/disease_ids.pt")
    ap.add_argument("--disease_embs", default="checkpoints/disease_embs.pt")
    ap.add_argument("--k", dest="topk", default=5, type=int)
    ap.add_argument("--min_hpo", default=2, type=int)
    ap.add_argument("--roc_negatives", default=1000, type=int)

    # overlap filtering
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--obo", default="hp.obo")
    ap.add_argument("--filter_by_overlap", action="store_true")
    ap.add_argument("--filter_depth", type=int, default=2)
    ap.add_argument("--report_top", nargs="*", type=int, default=[5, 10, 50, 100])

    # MONDO canonicalization
    ap.add_argument("--mondo", default=None, help="Path to mondo.obo for ID canonicalization")

    # IC-aware filtering knobs
    ap.add_argument("--ic", default="checkpoints/hpo_ic.pt")
    ap.add_argument("--filter_min_terms", type=int, default=2)
    ap.add_argument("--filter_min_ic", type=float, default=2.5)
    ap.add_argument("--filter_keep_top", type=int, default=500)

    # patient pooling knobs (optional)
    ap.add_argument("--patient_depth", type=int, default=2)
    ap.add_argument("--patient_decay", type=float, default=0.7)

    # IDF passthrough
    ap.add_argument("--idf", default=None)
    ap.add_argument("--idf_gamma", type=float, default=1.0)

    # NEW: hybrid scoring on filtered pool
    ap.add_argument("--hybrid_alpha", type=float, default=0.0,
                    help="weight on IC-overlap (0..1). 0 disables hybrid scoring.")
    ap.add_argument("--hybrid_beta", type=float, default=0.0,
                    help="additional weight on overlap-count (0..1).")

    args = ap.parse_args()

    # load resources
    term_nodes = torch.load(args.node_list)
    term_embs  = torch.load(args.term_embs)
    disease_ids = torch.load(args.disease_ids)
    disease_embs = torch.load(args.disease_embs)
    ic_map = torch.load(args.ic)

    diagnose.ensure_loaded(obo_path=args.obo, ic_path=args.ic, idf_path=args.idf, idf_gamma=args.idf_gamma)

    cand_set = set(disease_ids)

    # build overlap index if requested
    term2didxs = None
    parents = None
    if args.filter_by_overlap:
        term2didxs = _build_term_to_disease_index(args.hpoa, disease_ids)
        parents = _load_parents(args.obo)

    # MONDO canonicalization
    mondo_idx = None
    cand_set_canon = None
    if args.mondo:
        mondo_idx = build_mondo_index(args.mondo)
        cand_set_canon = {mondo_idx.get(d, d) for d in disease_ids}

    # collect phenopackets
    files = glob.glob(os.path.join(args.phenopackets_dir, "**", "*.json"), recursive=True)

    results = []
    results_filtered = []
    y_true, y_score = [], []
    skipped_parse = skipped_sparse = skipped_missing_did = 0
    candidate_sizes = []

    filtered_gold_kept = 0
    filtered_gold_total = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped_parse += 1
            continue

        pos, neg = extract_hpo_sets(data)
        if len(pos) < args.min_hpo:
            skipped_sparse += 1
            continue

        true_id = extract_true_disease_id(data)
        if not true_id:
            skipped_missing_did += 1
            continue

        # embed patient
        try:
            emb = diagnose.embed_patient(
                pos, term_nodes, term_embs,
                neg_codes=neg,
                max_depth=args.patient_depth,
                decay=args.patient_decay
            )
        except ValueError:
            skipped_sparse += 1
            continue

        # ---------- full ranking (baseline) ----------
        ranked_full = diagnose.rank_diseases(emb, disease_ids, disease_embs, topk=max(20, args.topk))
        results.append((true_id, ranked_full))

        # ---------- filtered ranking (IC-aware overlap + optional hybrid) ----------
        if args.filter_by_overlap:
            expanded = set(pos)
            for p in list(pos):
                expanded.update(_ancestors(p, parents, max_depth=args.filter_depth).keys())

            floor_ic = max(0.0, args.filter_min_ic / 3.0)
            expanded = {t for t in expanded if ic_map.get(t, 0.0) >= floor_ic}

            raw_cand_idx = set()
            for t in expanded:
                raw_cand_idx.update(term2didxs.get(t, ()))
            if not raw_cand_idx:
                raw_cand_idx = set(range(len(disease_ids)))

            overlap_cnt = {}
            overlap_ic  = {}
            for t in expanded:
                w = float(ic_map.get(t, 0.0))
                for i in term2didxs.get(t, ()):
                    if i not in raw_cand_idx:
                        continue
                    overlap_cnt[i] = overlap_cnt.get(i, 0) + 1
                    overlap_ic[i]  = overlap_ic.get(i, 0.0) + w

            cand_idx = [i for i in raw_cand_idx
                        if overlap_cnt.get(i, 0) >= args.filter_min_terms and
                           overlap_ic.get(i, 0.0) >= args.filter_min_ic]

            if len(cand_idx) > args.filter_keep_top:
                cand_idx = sorted(cand_idx, key=lambda i: overlap_ic.get(i, 0.0), reverse=True)[:args.filter_keep_top]

            if not cand_idx and raw_cand_idx:
                cand_idx = sorted(raw_cand_idx, key=lambda i: overlap_ic.get(i, 0.0), reverse=True)[:min(200, len(raw_cand_idx))]

            candidate_sizes.append(len(cand_idx))
            sub_ids  = [disease_ids[i] for i in cand_idx]
            sub_embs = disease_embs[cand_idx]

            # track filter recall
            if true_id in cand_set:
                filtered_gold_total += 1
                if true_id in sub_ids:
                    filtered_gold_kept += 1

            # (A) pure embedding ranking (baseline)
            ranked_sub = diagnose.rank_diseases(emb, sub_ids, sub_embs, topk=args.topk)

            # (B) optional HYBRID: combine cosine with overlap scores
            if (args.hybrid_alpha > 1e-9) or (args.hybrid_beta > 1e-9):
                pe = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                de = sub_embs / (sub_embs.norm(dim=1, keepdim=True) + 1e-8)
                sims = (pe @ de.t()).squeeze(0).cpu().numpy()

                ic_vec  = np.array([overlap_ic.get(i, 0.0) for i in cand_idx], dtype=float)
                cnt_vec = np.array([overlap_cnt.get(i, 0)   for i in cand_idx], dtype=float)

                s_sims = _minmax(sims)
                s_ic   = _minmax(ic_vec)
                s_cnt  = _minmax(cnt_vec)

                a = max(0.0, float(args.hybrid_alpha))
                b = max(0.0, float(args.hybrid_beta))
                base_w = max(0.0, 1.0 - a - b)
                combined = base_w * s_sims + a * s_ic + b * s_cnt

                order = np.argsort(-combined)
                ranked_sub = [(sub_ids[j], float(combined[j])) for j in order[:args.topk]]

            results_filtered.append((true_id, ranked_sub))

        # ---------- ROC bookkeeping (matched only; use full list for stability) ----------
        score_map = {did: sc for did, sc in ranked_full}
        if true_id not in cand_set:
            continue
        if true_id not in score_map:
            pe = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
            de = disease_embs / (disease_embs.norm(dim=1, keepdim=True) + 1e-8)
            sims_all = (pe @ de.t()).squeeze(0).cpu().numpy()
            idx = disease_ids.index(true_id)
            score_map[true_id] = float(sims_all[idx])

        y_true.append(1)
        y_score.append(score_map[true_id])

        if len(disease_ids) > 1:
            if "sims_all" not in locals():
                pe = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                de = disease_embs / (disease_embs.norm(dim=1, keepdim=True) + 1e-8)
                sims_all = (pe @ de.t()).squeeze(0).cpu().numpy()
            neg_pool = [i for i, did in enumerate(disease_ids) if did != true_id]
            sample_k = min(args.roc_negatives, len(neg_pool))
            for ni in random.sample(neg_pool, sample_k):
                y_true.append(0); y_score.append(float(sims_all[ni]))

    # ----- namespace report (pre-canonicalization) -----
    gold_prefix = Counter(y.split(":")[0] for y, _ in results if ":" in y)
    cand_prefix = Counter(d.split(":")[0] for d in disease_ids if ":" in d)
    matched_results_base = [(y, r) for (y, r) in results if y in cand_set]
    print("ID namespaces in gold (top 10):", gold_prefix.most_common(10))
    print("ID namespaces in candidates (top 10):", cand_prefix.most_common(10))
    print(f"Gold not present in candidates: {len(results) - len(matched_results_base)} / {len(results)}")

    # ----- MONDO canonicalization for metrics -----
    if mondo_idx is not None:
        results = canonize_results(results, mondo_idx)
        if results_filtered:
            results_filtered = canonize_results(results_filtered, mondo_idx)
        matched_results = [(y, r) for (y, r) in results if y in {mondo_idx.get(d, d) for d in cand_set}]
    else:
        matched_results = [(y, r) for (y, r) in results if y in cand_set]

    # ----- Metrics reports -----
    def report(label, res_list, tops):
        if not res_list:
            return
        for k in tops:
            acc, mrr = compute_metrics(res_list, k=k)
            print(f"{label} Top-{k}: {acc:.4f} | MRR: {mrr:.4f}")

    print(f"Evaluated {len(results)} cases (matched-in-candidates: {len(matched_results)})")
    print("== Full candidate set ==")
    report("Overall", results, args.report_top)
    report("Matched", matched_results, args.report_top)

    if results_filtered:
        if mondo_idx is not None:
            matched_results_f = [(y, r) for (y, r) in results_filtered if y in {mondo_idx.get(d, d) for d in cand_set}]
        else:
            matched_results_f = [(y, r) for (y, r) in results_filtered if y in cand_set]

        print("== Filtered by phenotype overlap ==")
        report("Overall (filtered)", results_filtered, args.report_top)
        report("Matched (filtered)", matched_results_f, args.report_top)
        if candidate_sizes:
            print(f"[INFO] Mean filtered candidate size: {np.mean(candidate_sizes):.1f} (median {np.median(candidate_sizes):.0f})")
        if filtered_gold_total:
            pct = 100.0 * filtered_gold_kept / filtered_gold_total
            print(f"[INFO] Filter recall (gold kept after filtering): {filtered_gold_kept}/{filtered_gold_total} ({pct:.1f}%)")

    print(f"[DEBUG] Skipped {skipped_sparse} sparse cases (<{args.min_hpo} HPOs)")
    print(f"[DEBUG] Skipped {skipped_missing_did} cases with missing disease ID")
    print(f"[DEBUG] Skipped {skipped_parse} files due to JSON parse errors")

    # ----- ROC (matched only; full list) -----
    if len(set(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    else:
        print("[WARN] Not enough positive/negative samples to plot ROC.")

if __name__ == "__main__":
    main()
