
#!/usr/bin/env python3
"""
evaluate_hpo_space.py

Baseline evaluation in HPO-space (no GNN). Diseases are represented by their HPO
annotation sets (from phenotype.hpoa). A patient is represented by their HPOs
(from phenopackets). Optional ancestor propagation with depth/decay can be applied.
Similarity is computed as a weighted overlap between patient and disease terms.

Weights support:
  - uniform: 1.0 for each term
  - ic: IC(term)
  - idf: IDF(term)^gamma
  - icidf: IC(term) * IDF(term)^gamma

Filtering (optional):
  - Keep only diseases with at least FILTER_MIN_TERMS overlapping terms (after propagation),
    and total IC sum over the intersection >= FILTER_MIN_IC, then keep top FILTER_KEEP_TOP
    candidates by overlap count.

Evaluation reports:
  - Top-k accuracy and MRR for user-selected K values (e.g., 5 10 50 100)
  - ROC/AUC by pooling positives/negatives across cases (negatives are sampled among
    the scored candidates per case). A PNG can be saved with --roc_out.

This script is self-contained and does not require model checkpoints beyond
IC/IDF maps (torch-saved dicts) and the HPOA file.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------------------
# OBO parsing (HPO ancestor DAG)
# -----------------------------

def parse_obo_parents(obo_path: str) -> Dict[str, List[str]]:
    """
    Parse an OBO file and return a mapping term -> list of parents (is_a).
    Only parses [Term] stanzas with 'id:' and 'is_a:'.
    """
    parents: Dict[str, List[str]] = defaultdict(list)
    if not obo_path or not os.path.exists(obo_path):
        return parents

    with open(obo_path, 'r', encoding='utf-8') as f:
        term_id = None
        in_term = False
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                term_id = None
                continue
            if line == '' and in_term:
                in_term = False
                term_id = None
                continue
            if not in_term:
                continue

            if line.startswith('id: '):
                term_id = line.split('id: ')[1].strip()
            elif line.startswith('is_a: ') and term_id:
                # Example: is_a: HP:0000118 ! Phenotypic abnormality
                parent = line.split('is_a: ')[1].split('!')[0].strip()
                parents[term_id].append(parent)
    return parents


def ancestors_within_depth(code: str, parents: Dict[str, List[str]], depth: int) -> Set[str]:
    """
    Return the set of ancestors up to 'depth' steps (1 step = direct parent).
    Includes the code itself (depth >= 0).
    """
    if depth <= 0:
        return {code}
    result = {code}
    frontier = {code}
    for _ in range(depth):
        next_frontier = set()
        for t in frontier:
            for p in parents.get(t, []):
                if p not in result:
                    result.add(p)
                    next_frontier.add(p)
        if not next_frontier:
            break
        frontier = next_frontier
    return result


# -----------------------------
# Phenopacket & HPOA utilities
# -----------------------------

def load_phenopacket(path: str) -> Tuple[Set[str], str]:
    """
    Parse a GA4GH Phenopacket JSON. Returns (positive_hpo_codes, gold_disease_id_or_None).
    Tries several common locations for disease IDs.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect positive HPOs (skip negated)
    hpos = set()
    feats = data.get('phenotypicFeatures') or data.get('phenotypic_features') or []
    for feat in feats:
        if feat.get('negated'):
            continue
        t = feat.get('type') or feat.get('term') or {}
        cid = t.get('id') or t.get('code')
        if cid and cid.startswith('HP:'):
            hpos.add(cid)

    # Try to find a gold disease ID
    gold = None

    # 1) diseases[]
    diseases = data.get('diseases') or []
    for d in diseases:
        term = d.get('term') or {}
        did = term.get('id')
        if not did:
            did = d.get('disease') or d.get('id')
        if isinstance(did, str) and (did.startswith('OMIM:') or did.startswith('ORPHA:') or did.startswith('DECIPHER:') or did.upper().startswith('MONDO:')):
            gold = did
            break

    # 2) interpretations[]
    if gold is None:
        interps = data.get('interpretations') or []
        for it in interps:
            dx = it.get('diagnosis') or {}
            dis = dx.get('disease') or {}
            term = dis.get('term') or {}
            did = term.get('id') or dis.get('id')
            if isinstance(did, str) and (did.startswith('OMIM:') or did.startswith('ORPHA:') or did.startswith('DECIPHER:') or did.upper().startswith('MONDO:')):
                gold = did
                break

    return hpos, gold


def iter_phenopackets(pp_dir: str) -> List[Tuple[str, Set[str], str]]:
    """
    Walk a directory and yield (filename, hpo_set, gold_disease_id_or_None) for .json files.
    """
    out = []
    for root, _, files in os.walk(pp_dir):
        for fn in files:
            if fn.lower().endswith('.json'):
                path = os.path.join(root, fn)
                try:
                    hpos, gold = load_phenopacket(path)
                    out.append((path, hpos, gold))
                except Exception as e:
                    print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
    return out


def load_hpoa(hpoa_path: str) -> Dict[str, Set[str]]:
    """
    Load phenotype.hpoa (tab-separated). Returns mapping disease_id -> set(HPO).
    Skips NOT qualifiers and comment lines.
    Columns (HPOA v2.3 typical): database_id, disease_name, qualifier, hpo_id, ...
    """
    mapping: Dict[str, Set[str]] = defaultdict(set)
    if not hpoa_path or not os.path.exists(hpoa_path):
        return mapping
    with open(hpoa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            disease_id = parts[0].strip()
            qualifier = parts[2].strip()
            hpo_id = parts[4].strip()
            if qualifier.upper() == 'NOT':
                continue
            if hpo_id.startswith('HP:'):
                mapping[disease_id].add(hpo_id)
    return mapping


# -----------------------------
# Weights and scoring
# -----------------------------

def build_weight_map(codes: Set[str],
                     ic_map: Dict[str, float],
                     idf_map: Dict[str, float],
                     idf_gamma: float,
                     weight_mode: str) -> Dict[str, float]:
    """
    Assign a weight per code according to weight_mode.
    """
    w = {}
    for c in codes:
        ic = float(ic_map.get(c, 0.0))
        idf = float(idf_map.get(c, 0.0))
        if weight_mode == 'uniform':
            val = 1.0
        elif weight_mode == 'ic':
            val = ic
        elif weight_mode == 'idf':
            # If idf is 0.0, keep it at 0.0 (rare terms have larger idf in most conventions).
            val = (idf ** idf_gamma) if idf_gamma != 0 else 1.0
        elif weight_mode == 'icidf':
            base = ic
            mult = (idf ** idf_gamma) if idf_gamma != 0 else 1.0
            val = base * mult
        else:
            val = 1.0
        if val != 0.0:
            w[c] = val
    return w


def propagate_codes_with_decay(codes: Set[str],
                               parents: Dict[str, List[str]],
                               depth: int,
                               decay: float) -> Dict[str, float]:
    """
    Expand term weights to ancestors up to 'depth', with multiplicative decay per level.
    Returns a dict code -> accumulated weight (start with 1.0 for self).
    """
    weights: Dict[str, float] = defaultdict(float)
    for c in codes:
        weights[c] += 1.0  # self
        if depth <= 0:
            continue
        frontier = {c}
        current_weight = 1.0
        for _ in range(depth):
            current_weight *= decay
            next_frontier = set()
            for t in frontier:
                for p in parents.get(t, []):
                    weights[p] += current_weight
                    next_frontier.add(p)
            if not next_frontier:
                break
            frontier = next_frontier
    return dict(weights)


def weighted_overlap_score(patient_w: Dict[str, float],
                           disease_w: Dict[str, float],
                           term_weight_map: Dict[str, float]) -> float:
    """
    Score = sum_{t in intersection} (patient_weight(t) * disease_weight(t) * term_weight(t))
    where term_weight(t) is from IC/IDF scheme; patient_weight/disease_weight are from propagation.
    """
    s = 0.0
    if len(patient_w) < len(disease_w):
        it = (t for t in patient_w.keys() if t in disease_w and t in term_weight_map)
    else:
        it = (t for t in disease_w.keys() if t in patient_w and t in term_weight_map)
    for t in it:
        s += patient_w[t] * disease_w[t] * term_weight_map[t]
    return s


# -----------------------------
# Metrics
# -----------------------------

def ranks_and_hits(scores: List[Tuple[str, float]], gold_id: str) -> Tuple[int, float]:
    """
    Returns (rank, score_of_gold) where rank starts at 1 for best.
    If gold not present, returns (-1, 0.0).
    """
    score_by_id = dict(scores)
    if gold_id not in score_by_id:
        return -1, 0.0
    gold_score = score_by_id[gold_id]
    # Rank: number of strictly higher scores + 1 (ties give worst-rank within tie)
    better = sum(1 for _, s in scores if s > gold_score)
    rank = better + 1
    return rank, gold_score


def topk_and_mrr(all_ranks: List[int], ks: List[int]) -> Tuple[Dict[int, float], float]:
    """
    From a list of per-case ranks (-1 if missing), compute top-k accuracies and MRR.
    'overall' style: include missing (count as failure). Caller can compute 'matched' separately.
    """
    n = len(all_ranks)
    top = {}
    for k in ks:
        hits = sum(1 for r in all_ranks if r != -1 and r <= k)
        top[k] = hits / max(1, n)
    # MRR
    rr = [1.0 / r for r in all_ranks if r != -1]
    mrr = (sum(rr) / max(1, n)) if rr else 0.0
    return top, mrr


def mann_whitney_auc(pos: List[float], neg: List[float]) -> float:
    """
    AUC via Mann-Whitney statistic: proportion that a random positive > random negative.
    """
    if not pos or not neg:
        return 0.0
    # Rank all together
    y = np.concatenate([np.array(pos), np.array(neg)])
    order = np.argsort(y, kind='mergesort')  # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y) + 1, dtype=float)
    n_pos = len(pos)
    rank_sum_pos = ranks[:n_pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * len(neg))
    return float(auc)


def make_roc_curve(pos: List[float], neg: List[float], num_thresh: int = 256) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Approximate ROC by sweeping 'num_thresh' quantile thresholds.
    """
    if not pos or not neg:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0

    all_scores = np.concatenate([np.array(pos), np.array(neg)])
    # Use quantile thresholds for efficiency
    qs = np.linspace(0.0, 1.0, num_thresh)
    thresh = np.quantile(all_scores, qs)

    pos_arr = np.array(pos)
    neg_arr = np.array(neg)

    tpr = []
    fpr = []
    for t in thresh[::-1]:  # high to low
        tp = (pos_arr >= t).sum()
        fp = (neg_arr >= t).sum()
        tpr.append(tp / max(1, len(pos_arr)))
        fpr.append(fp / max(1, len(neg_arr)))

    # Ensure (0,0) and (1,1) endpoints exist
    tpr = np.array([0.0] + tpr + [1.0])
    fpr = np.array([0.0] + fpr + [1.0])

    # AUC via trapezoid
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phenopackets_dir', type=str, default='phenopackets')
    p.add_argument('--hpoa', type=str, required=True, help='phenotype.hpoa')
    p.add_argument('--obo', type=str, required=True, help='hp.obo (HPO)')
    p.add_argument('--mondo', type=str, default=None, help='mondo.obo (unused here, accepted for parity)')

    p.add_argument('--ic', type=str, default=None, help='torch-saved dict: {HP: IC}')
    p.add_argument('--idf', type=str, default=None, help='torch-saved dict: {HP: IDF}')
    p.add_argument('--idf_gamma', type=float, default=1.0)

    p.add_argument('--weight_mode', type=str, default='icidf', choices=['uniform', 'ic', 'idf', 'icidf'])

    # Filtering options
    p.add_argument('--filter_by_overlap', action='store_true')
    p.add_argument('--filter_depth', type=int, default=1)
    p.add_argument('--filter_min_terms', type=int, default=1)
    p.add_argument('--filter_min_ic', type=float, default=0.0)
    p.add_argument('--filter_keep_top', type=int, default=1000)

    # Patient expansion
    p.add_argument('--patient_depth', type=int, default=1)
    p.add_argument('--patient_decay', type=float, default=0.7)

    # Evaluation
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--report_top', type=int, nargs='+', default=[5, 10, 50, 100])
    p.add_argument('--min_hpo', type=int, default=2)

    # ROC
    p.add_argument('--roc_negatives', type=int, default=300)
    p.add_argument('--roc_out', type=str, default=None)

    args = p.parse_args()

    # Load resources
    ic_map = torch.load(args.ic) if args.ic and os.path.exists(args.ic) else {}
    idf_map = torch.load(args.idf) if args.idf and os.path.exists(args.idf) else {}

    parents = parse_obo_parents(args.obo)
    disease2hpo = load_hpoa(args.hpoa)
    packets = iter_phenopackets(args.phenopackets_dir)

    if not packets:
        print("[ERROR] No phenopackets found.")
        sys.exit(1)

    # Precompute disease expanded weights (for filtering and scoring)
    # We propagate each disease's HPO set using the same depth as patient_depth,
    # but you can change this to args.filter_depth if you want to use a different
    # expansion for the filter stage.
    disease_expanded_cache: Dict[str, Dict[str, float]] = {}
    for did, hpos in disease2hpo.items():
        disease_expanded_cache[did] = propagate_codes_with_decay(hpos, parents, args.patient_depth, args.patient_decay)

    ks = sorted(set(args.report_top + [args.k]))

    overall_ranks: List[int] = []
    matched_ranks: List[int] = []

    pos_scores_pool: List[float] = []
    neg_scores_pool: List[float] = []

    sparse_skips = 0
    gold_missing = 0

    # Precompute term weights universe to avoid recomputing per case? No,
    # term weights depend on the set of terms considered (patient expansion).
    # We'll compute per patient for accuracy.
    eval_count = 0

    for path, patient_hpos, gold in packets:
        if len(patient_hpos) < args.min_hpo:
            sparse_skips += 1
            continue

        # Patient expanded with decay
        p_expanded = propagate_codes_with_decay(patient_hpos, parents, args.patient_depth, args.patient_decay)
        # Build term-weight (IC/IDF) map ONLY for terms seen in patient or diseases (intersection later)
        # But simpler: build for all observed HPO codes in patient expansion to speed dictionary checks.
        term_weight_map = build_weight_map(set(p_expanded.keys()), ic_map, idf_map, args.idf_gamma, args.weight_mode)

        # Candidate diseases
        candidates = list(disease2hpo.keys())

        # Optional phenotype-overlap filtering
        if args.filter_by_overlap:
            filt_scores = []
            for did in candidates:
                d_expanded = disease_expanded_cache[did]

                # raw overlap set (unweighted) for gatekeeping
                inter = set(p_expanded.keys()).intersection(d_expanded.keys())
                if not inter:
                    continue

                # Count and IC sum for filtering thresholds
                overlap_cnt = len(inter)
                ic_sum = sum(float(ic_map.get(t, 0.0)) for t in inter)

                if overlap_cnt >= args.filter_min_terms and ic_sum >= args.filter_min_ic:
                    filt_scores.append((did, overlap_cnt))

            # Keep top-N by overlap count
            filt_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [d for d, _ in filt_scores[: args.filter_keep_top]]
            if not candidates:
                # No overlap; fall back to all diseases to allow a score (very rare)
                candidates = list(disease2hpo.keys())

        # Score all candidates
        scored: List[Tuple[str, float]] = []
        p_terms = set(p_expanded.keys())
        for did in candidates:
            d_expanded = disease_expanded_cache[did]
            # Limit per-term weight map to terms that could intersect (optional)
            # Compute final weighted overlap
            s = weighted_overlap_score(p_expanded, d_expanded, term_weight_map)
            if s != 0.0:
                scored.append((did, float(s)))

        # If everything is zero, keep zeros too for ranking determinism
        if not scored:
            scored = [(did, 0.0) for did in candidates]

        # Rank (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        eval_count += 1

        # Rank of gold
        r, gold_score = ranks_and_hits(scored, gold) if gold else (-1, 0.0)
        overall_ranks.append(r)
        if r != -1:
            matched_ranks.append(r)
        else:
            gold_missing += 1

        # Negatives for ROC
        if r != -1 and args.roc_negatives > 0:
            # Choose negatives from the scored IDs (so lookups can't fail)
            scored_ids = [d for d, _ in scored if d != gold]
            n_neg = min(args.roc_negatives, len(scored_ids))
            if n_neg > 0:
                neg_samps = random.sample(scored_ids, n_neg)
                score_by_id = dict(scored)
                neg_vals = [score_by_id[d] for d in neg_samps]
                pos_scores_pool.append(gold_score)
                neg_scores_pool.extend(neg_vals)

    # Reporting
    def print_report(group_name: str, ranks: List[int]):
        if not ranks:
            print(f"== {group_name} ==\nNo cases.\n")
            return
        top, mrr = topk_and_mrr(ranks, ks)
        print(f"== {group_name} ==")
        for k in ks:
            print(f"Overall Top-{k}: {top[k]:.4f}")
        print(f"Overall MRR: {mrr:.4f}")
        print()

    print(f"Evaluated {eval_count} cases")
    print(f"Gold not present in candidates: {gold_missing}")

    print("== HPO-space baseline ==")
    # Overall (includes missing as failures)
    top_overall, mrr_overall = topk_and_mrr(overall_ranks, ks)
    for k in ks:
        print(f"Overall Top-{k}: {top_overall[k]:.4f}")
    print(f"Overall MRR: {mrr_overall:.4f}")

    # Matched only
    matched_only = [r for r in overall_ranks if r != -1]
    if matched_only:
        top_matched, mrr_matched = topk_and_mrr(matched_only, ks)
        for k in ks:
            print(f"Matched Top-{k}: {top_matched[k]:.4f}")
        print(f"Matched MRR: {mrr_matched:.4f}")
    else:
        print("Matched Top-K/MRR: N/A (no matched cases)")

    # ROC/AUC
    if pos_scores_pool and neg_scores_pool:
        fpr, tpr, auc_trap = make_roc_curve(pos_scores_pool, neg_scores_pool, num_thresh=256)
        auc_mw = mann_whitney_auc(pos_scores_pool, neg_scores_pool)
        print(f"[ROC] Pooled AUC (trapz): {auc_trap:.4f} | (Mann-Whitney): {auc_mw:.4f}")
        if args.roc_out:
            os.makedirs(os.path.dirname(args.roc_out), exist_ok=True)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc_trap:.3f}")
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.roc_out, dpi=180)
            plt.close()
            print(f"[Saved ROC] {args.roc_out}")
    else:
        print("[ROC] Not enough positives/negatives to compute ROC/AUC.")

    print(f"[Done]")


if __name__ == '__main__':
    main()
