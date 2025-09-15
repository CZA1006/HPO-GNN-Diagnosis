#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hpo_space.py
A simple HPO-space baseline that scores diseases by weighted
overlap with a patient's HPO terms, with optional IC/IDF weighting,
patient ancestor expansion (depth/decay), candidate filtering, and
OMIM normalization via MONDO xrefs. Computes Top-K / MRR and an
approximate ROC/AUC with negative sampling, and can save a ROC plot.
"""

import os
import re
import json
import math
import glob
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

try:
    import torch
except Exception:
    torch = None

# ---------------------------
# OBO parsing (HP & MONDO)
# ---------------------------

def parse_hp_obo(obo_path: str) -> Dict[str, Set[str]]:
    """Parse hp.obo; return parents map: term -> set of parent terms (is_a)."""
    parents = defaultdict(set)
    if not obo_path or not os.path.exists(obo_path):
        return parents

    current_id = None
    in_term = False
    with open(obo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                in_term = True
                current_id = None
                continue
            if not line:
                in_term = False
                current_id = None
                continue
            if in_term:
                if line.startswith("id: "):
                    tid = line.split("id: ", 1)[1].strip()
                    if tid.startswith("HP:"):
                        current_id = tid
                elif line.startswith("is_a: ") and current_id and current_id.startswith("HP:"):
                    pid = line.split("is_a: ", 1)[1].split("!")[0].strip()
                    parents[current_id].add(pid)
    return parents


def parse_mondo_xrefs(mondo_path: str) -> Dict[str, Set[str]]:
    """Parse mondo.obo; return mapping of external IDs -> set of OMIM IDs.
       e.g., {'ORPHA:123': {'OMIM:600123'}}. Only records OMIM xrefs.
    """
    xmap = defaultdict(set)
    if not mondo_path or not os.path.exists(mondo_path):
        return xmap

    in_term = False
    current_id = None
    current_omims: Set[str] = set()
    current_xrefs: List[str] = []

    def flush():
        nonlocal current_id, current_omims, current_xrefs, xmap
        if current_omims:
            for xr in current_xrefs:
                # examples: "Orphanet:1234", "DECIPHER:12", "OMIM:123456"
                if xr.startswith("OMIM:"):
                    continue
                if ":" in xr:
                    xmap[xr].update(current_omims)

    with open(mondo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current_id is not None:
                    flush()
                in_term = True
                current_id = None
                current_omims = set()
                current_xrefs = []
                continue
            if not line:
                continue
            if in_term:
                if line.startswith("id: "):
                    current_id = line.split("id: ", 1)[1].strip()
                elif line.startswith("xref: "):
                    xr = line.split("xref: ", 1)[1].split()[0].strip()
                    # keep basic forms like OMIM:123456 Orphanet:1234 DECIPHER:12
                    if ":" in xr:
                        if xr.startswith("OMIM:"):
                            current_omims.add(xr)
                        else:
                            current_xrefs.append(xr)
    # flush last
    if current_id is not None:
        # flush the last term
        if current_omims:
            for xr in current_xrefs:
                if xr.startswith("OMIM:"):
                    continue
                if ":" in xr:
                    xmap[xr].update(current_omims)

    return xmap


def normalize_disease_id(did: str, mondo_map: Dict[str, Set[str]]) -> str:
    """Normalize disease ID to OMIM if a unique mapping exists in mondo_map."""
    if not did:
        return did
    did = did.strip()
    if did.startswith("OMIM:"):
        return did
    if did in mondo_map:
        omims = sorted(mondo_map[did])
        if len(omims) == 1:
            return omims[0]
    # try some common namespace aliases
    if did.startswith("ORPHA:") and did in mondo_map:
        omims = sorted(mondo_map[did])
        if len(omims) == 1:
            return omims[0]
    if did.startswith("DECIPHER:") and did in mondo_map:
        omims = sorted(mondo_map[did])
        if len(omims) == 1:
            return omims[0]
    return did


# ---------------------------
# HPOA parsing (disease -> HPOs)
# ---------------------------

def load_hpoa(hpoa_path: str,
              mondo_map: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Load phenotype.hpoa; return normalized_disease_id -> set(HPO)"""
    disease_to_hpos = defaultdict(set)
    if not hpoa_path or not os.path.exists(hpoa_path):
        return disease_to_hpos

    with open(hpoa_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            disease_id = parts[0].strip()       # e.g., OMIM:xxxx, ORPHA:xxxx
            hpo_id = parts[4].strip()           # HPO term
            if not hpo_id.startswith("HP:"):
                continue
            # qualifier column sometimes indicates NOT
            qualifier = parts[3].strip() if len(parts) > 3 else ""
            if qualifier == "NOT":
                continue
            norm_id = normalize_disease_id(disease_id, mondo_map)
            disease_to_hpos[norm_id].add(hpo_id)
    return disease_to_hpos


# ---------------------------
# Phenopacket parsing
# ---------------------------

def collect_json_files(root: str) -> List[str]:
    files = []
    for ext in ("*.json", "*.ndjson", "*.jsonl"):
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def extract_case(file_path: str) -> Tuple[List[str], List[str], str]:
    """Return (pos_hpos, neg_hpos, gold_disease_id) from a phenopacket JSON."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    # phenotypicFeatures
    pos, neg = [], []
    phenos = data.get("phenotypicFeatures") or get_nested(data, ["phenopacket", "phenotypicFeatures"]) or []
    for feat in phenos:
        if not isinstance(feat, dict):
            continue
        term = feat.get("type") or {}
        tid = term.get("id") or term.get("term") or ""
        if tid and tid.startswith("HP:"):
            if feat.get("excluded") is True:
                neg.append(tid)
            else:
                pos.append(tid)

    # disease
    gold = None
    diseases = data.get("diseases") or get_nested(data, ["phenopacket", "diseases"]) or []
    if isinstance(diseases, list) and diseases:
        di = diseases[0]
        term = di.get("term") or {}
        gold = term.get("id") or di.get("id") or di.get("disease") or None
    if not gold:
        # sometimes 'disease' singular
        di = data.get("disease") or get_nested(data, ["phenopacket", "disease"])
        if isinstance(di, dict):
            term = di.get("term") or {}
            gold = term.get("id") or di.get("id")

    return pos, neg, gold


# ---------------------------
# Ancestor expansion
# ---------------------------

def get_ancestors(term: str, parents: Dict[str, Set[str]], max_depth: int) -> Set[str]:
    """Return ancestors up to max_depth (>=1 means include parents; 0 -> none)."""
    if max_depth <= 0:
        return set()
    out = set()
    frontier = set([term])
    visited = set([term])
    for depth in range(max_depth):
        next_frontier = set()
        for t in frontier:
            for p in parents.get(t, []):
                if p not in visited:
                    visited.add(p)
                    next_frontier.add(p)
        out.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return out


def expand_patient_terms(pos: List[str],
                         parents: Dict[str, Set[str]],
                         depth: int,
                         decay: float) -> Dict[str, float]:
    """Return weighted patient term vector with ancestor propagation."""
    weights = defaultdict(float)
    for t in pos:
        weights[t] += 1.0  # direct
        if depth > 0:
            anc = get_ancestors(t, parents, depth)
            for a in anc:
                # depth-aware decays (distance at least 1)
                # we approximate each ancestor distance as 1 hop per BFS layer
                # For simplicity, use the same decay for all ancestors discovered
                weights[a] += decay
    return dict(weights)


# ---------------------------
# Weighting functions
# ---------------------------

def build_code_weight_func(weight_mode: str,
                           ic_map: Dict[str, float],
                           idf_map: Dict[str, float],
                           idf_gamma: float):
    """Return a function: w(code) -> float based on IC/IDF mode."""
    if weight_mode == "uniform":
        return lambda c: 1.0
    elif weight_mode == "ic":
        return lambda c: float(ic_map.get(c, 0.0))
    elif weight_mode == "idf":
        return lambda c: float(idf_map.get(c, 0.0)) ** float(idf_gamma)
    elif weight_mode == "icidf":
        return lambda c: float(ic_map.get(c, 0.0)) * (float(idf_map.get(c, 0.0)) ** float(idf_gamma))
    else:
        return lambda c: 1.0


# ---------------------------
# Scoring & Filtering
# ---------------------------

def score_disease(patient_vec: Dict[str, float],
                  disease_terms: Set[str],
                  code_w) -> float:
    """Weighted overlap score: sum_{c in intersection} (patient_weight(c) * w(c))."""
    s = 0.0
    for c, pw in patient_vec.items():
        if c in disease_terms:
            s += pw * float(code_w(c))
    return s


def filter_candidates(patient_vec: Dict[str, float],
                      disease_to_hpos: Dict[str, Set[str]],
                      code_w,
                      min_terms: int,
                      min_ic: float,
                      keep_top: int) -> List[Tuple[str, float, int, float]]:
    """Return a list of (disease_id, overlap_score, overlap_count, overlap_ic_sum)."""
    scored = []
    for d, terms in disease_to_hpos.items():
        overlap = [c for c in patient_vec.keys() if c in terms]
        if not overlap:
            continue
        count = len(overlap)
        ic_sum = sum(float(code_w(c)) for c in overlap)  # "ic-ish" sum per chosen weight function
        if count < min_terms or ic_sum < min_ic:
            continue
        s = sum(patient_vec[c] * float(code_w(c)) for c in overlap)
        scored.append((d, s, count, ic_sum))
    if not scored:
        return []
    scored.sort(key=lambda x: x[1], reverse=True)
    if keep_top and keep_top > 0:
        scored = scored[:keep_top]
    return scored


# ---------------------------
# Metrics
# ---------------------------

def compute_topk_and_mrr(ranked_ids: List[str],
                         gold_id: str,
                         topks: List[int]) -> Tuple[Dict[int, float], float]:
    """Return dict of hits@K and MRR for a single case (1 or 0 for hits)."""
    hits = {k: 0.0 for k in topks}
    mrr = 0.0
    if gold_id and gold_id in ranked_ids:
        rank = ranked_ids.index(gold_id) + 1
        for k in topks:
            if rank <= k:
                hits[k] = 1.0
        mrr = 1.0 / float(rank)
    return hits, mrr


def trapezoidal_auc(xs: List[float], ys: List[float]) -> float:
    """Compute AUC for a monotonic ROC curve given xs (FPR) and ys (TPR)."""
    if len(xs) < 2:
        return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += dx * (ys[i] + ys[i-1]) / 2.0
    return area


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenopackets_dir", type=str, required=True)
    ap.add_argument("--hpoa", type=str, required=True)
    ap.add_argument("--obo", type=str, required=True)
    ap.add_argument("--mondo", type=str, default="")

    ap.add_argument("--ic", type=str, default="")
    ap.add_argument("--idf", type=str, default="")
    ap.add_argument("--idf_gamma", type=float, default=1.0)
    ap.add_argument("--weight_mode", type=str, default="icidf",
                    choices=["uniform", "ic", "idf", "icidf"])

    ap.add_argument("--filter_by_overlap", action="store_true")
    ap.add_argument("--filter_depth", type=int, default=1)
    ap.add_argument("--filter_min_terms", type=int, default=2)
    ap.add_argument("--filter_min_ic", type=float, default=2.0)
    ap.add_argument("--filter_keep_top", type=int, default=500)

    ap.add_argument("--patient_depth", type=int, default=2)
    ap.add_argument("--patient_decay", type=float, default=0.7)

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--report_top", type=int, nargs="+", default=[5, 10, 50, 100])

    ap.add_argument("--roc_negatives", type=int, default=300)
    ap.add_argument("--roc_out", type=str, default="")

    args = ap.parse_args()

    # Load maps
    hp_parents = parse_hp_obo(args.obo)
    mondo_map = parse_mondo_xrefs(args.mondo) if args.mondo else defaultdict(set)

    ic_map = {}
    if args.ic and os.path.exists(args.ic) and torch is not None:
        try:
            ic_map = torch.load(args.ic)
        except Exception:
            ic_map = {}
    idf_map = {}
    if args.idf and os.path.exists(args.idf) and torch is not None:
        try:
            idf_map = torch.load(args.idf)
        except Exception:
            idf_map = {}

    code_w = build_code_weight_func(args.weight_mode, ic_map, idf_map, args.idf_gamma)

    # Build disease -> HPOs (normalized IDs)
    disease_to_hpos_raw = load_hpoa(args.hpoa, mondo_map)

    # Merge diseases by normalized OMIM if some are still non-OMIM but normalizeable
    disease_to_hpos: Dict[str, Set[str]] = defaultdict(set)
    for did, terms in disease_to_hpos_raw.items():
        norm = normalize_disease_id(did, mondo_map)
        disease_to_hpos[norm].update(terms)

    all_diseases = sorted(disease_to_hpos.keys())

    # Collect phenopackets
    files = collect_json_files(args.phenopackets_dir)
    n_cases = 0
    n_matched = 0
    n_gold_missing = 0

    # Aggregate metrics
    overall_hits = Counter()
    overall_mrr_sum = 0.0
    matched_hits = Counter()
    matched_mrr_sum = 0.0
    topks = sorted(set([args.k] + args.report_top))

    # ROC aggregation
    pos_scores = []
    neg_scores = []

    for fp in files:
        pos, neg, gold = extract_case(fp)
        if not pos and not neg:
            continue
        n_cases += 1

        gold_norm = normalize_disease_id(gold or "", mondo_map) if gold else None
        if not gold_norm or gold_norm not in disease_to_hpos:
            # We'll still compute a ranking, but mark as unmatched
            n_gold_missing += 1

        # Build patient vector with ancestors
        pvec = expand_patient_terms(pos, hp_parents, args.patient_depth, args.patient_decay)
        if not pvec:
            # If no positives, skip
            continue

        # Candidate filtering
        candidate_ids = list(all_diseases)
        if args.filter_by_overlap:
            filtered = filter_candidates(
                pvec, disease_to_hpos, code_w,
                min_terms=args.filter_min_terms,
                min_ic=args.filter_min_ic,
                keep_top=args.filter_keep_top
            )
            if filtered:
                candidate_ids = [d for d, _, _, _ in filtered]
            # Ensure the gold remains if present in the knowledge base
            if gold_norm and gold_norm in disease_to_hpos and gold_norm not in candidate_ids:
                candidate_ids.append(gold_norm)
            # If filtering wiped out everything, fall back
            if not candidate_ids:
                candidate_ids = list(all_diseases)

        # Score all candidates
        scores = []
        for did in candidate_ids:
            s = score_disease(pvec, disease_to_hpos[did], code_w)
            if s > 0.0:
                scores.append((did, s))
        if not scores:
            # If everything zero, keep a tiny score so there is a ranking
            scores = [(did, 0.0) for did in candidate_ids]

        scores.sort(key=lambda x: x[1], reverse=True)
        ranked = [d for d, _ in scores]

        # Metrics
        hits, mrr = compute_topk_and_mrr(ranked, gold_norm, topks)
        for k, v in hits.items():
            overall_hits[k] += v
        overall_mrr_sum += mrr

        if gold_norm and gold_norm in disease_to_hpos:
            n_matched += 1
            for k, v in hits.items():
                matched_hits[k] += v
            matched_mrr_sum += mrr

        # ROC sampling (collect positive + negatives per case)
        if gold_norm and gold_norm in ranked:
            gold_score = dict(scores)[gold_norm]
            pos_scores.append(gold_score)

            if args.roc_negatives > 0:
                # sample from candidates excluding gold with replacement if needed
                negatives = [d for d in ranked if d != gold_norm]
                if not negatives:
                    negatives = [d for d in all_diseases if d != gold_norm]
                if negatives:
                    # ensure we can index even if roc_negatives > len(negatives)
                    for _ in range(args.roc_negatives):
                        dn = random.choice(negatives)
                        neg_scores.append(dict(scores).get(dn, 0.0))

    # Report
    print(f"Evaluated {n_cases} cases")
    print(f"Gold not present in candidates: {n_gold_missing}")
    print("== HPO-space baseline ==")

    def summarize(hcounter: Counter, mrr_sum: float, denom: int, tag: str):
        if denom <= 0:
            print(f"{tag} Top-K/MRR: N/A (no matched cases)")
            return
        for k in args.report_top:
            v = hcounter.get(k, 0.0) / max(1, denom)
            print(f"{tag} Top-{k}: {v:.4f}")
        print(f"{tag} MRR: {mrr_sum / max(1, denom):.4f}")

    summarize(overall_hits, overall_mrr_sum, n_cases, "Overall")
    summarize(matched_hits, matched_mrr_sum, n_matched, "Matched")

    # ROC/AUC
    if pos_scores and neg_scores:
        # Build a global ROC by thresholding across all scores
        # Labels: 1 for each pos score, 0 for each neg score
        scores_all = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
        scores_all.sort(key=lambda x: x[0], reverse=True)
        P = float(len(pos_scores))
        N = float(len(neg_scores))

        tpr = []
        fpr = []
        tp = 0.0
        fp = 0.0
        prev_s = None
        for s, y in scores_all:
            if y == 1:
                tp += 1.0
            else:
                fp += 1.0
            # Record a point only when score changes to keep curve compact
            if s != prev_s:
                tpr.append(tp / P if P > 0 else 0.0)
                fpr.append(fp / N if N > 0 else 0.0)
                prev_s = s

        # Ensure curve starts at (0,0) and ends at (1,1)
        if not fpr or fpr[0] != 0.0:
            fpr = [0.0] + fpr
            tpr = [0.0] + tpr
        if fpr[-1] != 1.0 or tpr[-1] != 1.0:
            fpr.append(1.0)
            tpr.append(1.0)

        auc = trapezoidal_auc(fpr, tpr)
        print(f"[ROC] AUC: {auc:.4f} (Pos={int(P)}, Neg={int(N)})")

        # Plot if requested
        if args.roc_out:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # One chart, default colors, no custom style
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve (HPO-space baseline)")
                plt.legend(loc="lower right")

                os.makedirs(os.path.dirname(args.roc_out), exist_ok=True)
                plt.savefig(args.roc_out, bbox_inches="tight", dpi=160)
                plt.close()
                print(f"[ROC] Saved figure to {args.roc_out}")
            except Exception as e:
                print(f"[ROC] Failed to plot: {e}")
    else:
        print("[ROC] Not enough positives/negatives to compute ROC/AUC.")
    print("[Done]")


if __name__ == "__main__":
    main()
