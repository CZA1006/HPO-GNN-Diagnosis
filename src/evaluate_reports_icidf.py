#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate synthetic free-text case reports in 'HPO space' with IC/IDF weights.

Usage (example):
  python src/evaluate_reports_icidf.py \
    --reports_dir data/synthetic_reports/by_omim \
    --hpoa phenotype.hpoa --obo hp.obo \
    --ic checkpoints/hpo_ic.pt \
    --idf checkpoints/hpo_idf.pt --idf_gamma 1.0 \
    --weight_mode icidf \
    --patient_depth 2 --patient_decay 0.7 \
    --filter_by_overlap --filter_depth 2 --filter_min_terms 2 --filter_min_ic 2.6 \
    --roc_negatives 300 --roc_out results/Figure_reports_icidf.png \
    --report_top 5 10 50 100
"""
from __future__ import annotations
import argparse, os, re, math, random, json, collections, time, sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reports_dir", required=True)
    p.add_argument("--hpoa", required=True)
    p.add_argument("--obo", required=True)
    p.add_argument("--ic", required=True)
    p.add_argument("--idf", required=True)
    p.add_argument("--idf_gamma", type=float, default=1.0)
    p.add_argument("--weight_mode", choices=["binary","ic","idf","icidf"], default="icidf")
    p.add_argument("--patient_depth", type=int, default=0)
    p.add_argument("--patient_decay", type=float, default=1.0)
    # simple candidate filter (like your other scripts)
    p.add_argument("--filter_by_overlap", action="store_true")
    p.add_argument("--filter_depth", type=int, default=1)
    p.add_argument("--filter_min_terms", type=int, default=0)
    p.add_argument("--filter_min_ic", type=float, default=0.0)
    p.add_argument("--filter_keep_top", type=int, default=1000000)
    # metrics / outputs
    p.add_argument("--report_top", nargs="+", type=int, default=[5,10,50,100])
    p.add_argument("--roc_negatives", type=int, default=0)
    p.add_argument("--roc_out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# ---------- OBO + HPOA parsing ----------

def load_hp_obo(path: str):
    """Return: parents: Dict[term, Set[parent]], namesyn: Dict[phrase_lower -> term]"""
    parents: Dict[str, Set[str]] = collections.defaultdict(set)
    namesyn: Dict[str, str] = {}
    cur = {}
    def flush(cur):
        if not cur: return
        if cur.get("id","").startswith("HP:"):
            tid = cur["id"]
            if "name" in cur:
                namesyn[cur["name"].lower()] = tid
            for s in cur.get("synonym", []):
                # s like: "foo" EXACT [PMID:xxx]
                m = re.match(r'"(.+?)"\s+\w+', s)
                if m:
                    phrase = m.group(1).strip().lower()
                    if len(phrase) >= 3:
                        namesyn.setdefault(phrase, tid)
            for is_a in cur.get("is_a", []):
                pid = is_a.split("!", 1)[0].strip()
                parents[tid].add(pid)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush(cur); cur = {}
            elif line == "":
                continue
            else:
                if ":" not in line: continue
                k, v = line.split(":", 1); v = v.strip()
                if k in ("id","name"):
                    cur[k] = v
                elif k in ("is_a","synonym"):
                    cur.setdefault(k, []).append(v)
        flush(cur)
    return parents, namesyn

def ancestors(term: str, parents: Dict[str, Set[str]], depth:int) -> List[Tuple[str,int]]:
    out = []
    cur = [(term,0)]
    seen = {term}
    while cur:
        t,d = cur.pop()
        if d == depth: continue
        for p in parents.get(t, ()):
            if p not in seen:
                seen.add(p)
                out.append((p,d+1))
                cur.append((p,d+1))
    return out

def load_hpoa(path: str) -> Dict[str, Set[str]]:
    """Return disease_id (e.g., OMIM:300088) -> set(HPO)"""
    dis2hpos: Dict[str, Set[str]] = collections.defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5: continue
            disease_id = parts[0]        # e.g., OMIM:123456
            qualifier  = parts[2]        # 'NOT' if negated
            hpo_id     = parts[3]        # e.g., HP:0001250
            if qualifier.strip().upper() == "NOT": 
                continue
            if not disease_id.startswith(("OMIM:","ORPHA:","DECIPHER:")):
                continue
            if hpo_id.startswith("HP:"):
                dis2hpos[disease_id].add(hpo_id)
    return dis2hpos

# ---------- text -> HPO extraction (very simple exact-phrase matcher) ----------

def build_phrase_index(namesyn: Dict[str,str]):
    """Map first token -> list[(tokens, term)]. Speeds up scanning."""
    first2phr = collections.defaultdict(list)
    for phrase, tid in namesyn.items():
        toks = phrase.split()
        if not toks: continue
        first2phr[toks[0]].append((toks, tid))
    # sort by length desc so longer phrases win first
    for k in first2phr:
        first2phr[k].sort(key=lambda x: -len(x[0]))
    return first2phr

def extract_hpos(text: str, first2phr) -> Set[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    words = text.split()
    found = set()
    i = 0
    while i < len(words):
        candidates = first2phr.get(words[i], [])
        matched = False
        for toks, tid in candidates:
            L = len(toks)
            if i+L <= len(words) and words[i:i+L] == toks:
                found.add(tid)
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return found

# ---------- vector utils ----------

def weight_of(term: str, ic_map, idf_map, mode: str, gamma: float) -> float:
    ic  = float(ic_map.get(term, 0.0))
    idf = float(idf_map.get(term, 0.0))
    if mode == "binary": return 1.0
    if mode == "ic":     return ic
    if mode == "idf":    return (idf ** gamma)
    # icidf
    return ic * (idf ** gamma)

def make_vector(base_terms: Set[str], parents, depth:int, decay:float, ic_map, idf_map, mode:str, gamma:float) -> Dict[str,float]:
    vec: Dict[str,float] = collections.defaultdict(float)
    for t in base_terms:
        vec[t] += weight_of(t, ic_map, idf_map, mode, gamma)
        if depth > 0 and decay > 0:
            for anc, d in ancestors(t, parents, depth):
                vec[anc] += (decay ** d) * weight_of(anc, ic_map, idf_map, mode, gamma)
    return dict(vec)

def cosine_sparse(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None: dot += va * vb
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

# ---------- metrics ----------

def topk_metrics(ranks: List[int], ks: List[int]) -> Dict[str,float]:
    out = {}
    for k in ks:
        out[f"top{k}"] = sum(1 for r in ranks if r is not None and r <= k) / max(1,len(ranks))
    mrr = 0.0
    cnt = 0
    for r in ranks:
        if r is not None:
            mrr += 1.0 / r
            cnt += 1
    out["mrr"] = (mrr / max(1,cnt)) if cnt else 0.0
    return out

def roc_auc(pos: List[float], neg: List[float]) -> Tuple[List[float],List[float],float]:
    if not pos or not neg:
        return [0,1],[0,1],float("nan")
    all_scores = [(s,1) for s in pos] + [(s,0) for s in neg]
    all_scores.sort(key=lambda x: x[0], reverse=True)
    P = len(pos); N = len(neg)
    tp = fp = 0
    tpr=[]; fpr=[]
    last = None
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    for s,y in all_scores:
        if last is None or s != last:
            tpr.append(tp / P); fpr.append(fp / N)
            # trapezoid
            auc += (fpr[-1]-prev_fpr) * (tpr[-1]+prev_tpr) / 2.0
            prev_fpr, prev_tpr = fpr[-1], tpr[-1]
            last = s
        if y==1: tp += 1
        else:    fp += 1
    tpr.append(1.0); fpr.append(1.0)
    auc += (1.0-prev_fpr) * (1.0+prev_tpr) / 2.0
    return fpr, tpr, auc

# ---------- main ----------

def main():
    args = parse_args()
    random.seed(args.seed)

    parents, namesyn = load_hp_obo(args.obo)
    first2phr = build_phrase_index(namesyn)
    dis2hpos = load_hpoa(args.hpoa)

    ic_map  = torch.load(args.ic)
    idf_map = torch.load(args.idf)

    # Precompute disease vectors (IC/IDF + optional ancestor expansion for filtering only)
    disease_vec: Dict[str, Dict[str,float]] = {}
    disease_base: Dict[str, Set[str]] = {}
    for did, hpos in dis2hpos.items():
        if not hpos: continue
        disease_base[did] = set(hpos)
        disease_vec[did]  = make_vector(
            base_terms=set(hpos),
            parents=parents,
            depth=args.filter_depth if args.filter_by_overlap else 0,
            decay=1.0,
            ic_map=ic_map, idf_map=idf_map,
            mode=args.weight_mode, gamma=args.idf_gamma
        )

    # list reports
    rep_paths = sorted([p for p in Path(args.reports_dir).glob("*.txt") if re.match(r"\d{4,7}(?:_\d+)?\.txt$", p.name)])
    print(f"Found {len(rep_paths)} reports.")

    ranks = []
    pos_scores=[]; neg_scores=[]
    matched = 0; gold_not_found = 0

    t0 = time.time()
    for p in rep_paths:
        omim = re.match(r"(\d{4,7})", p.stem).group(1)
        gold_id = f"OMIM:{omim}"
        text = p.read_text(encoding="utf-8", errors="ignore")

        patient_terms = extract_hpos(text, first2phr)

        # Optional coarse filtering
        candidates = list(disease_vec.keys())
        if args.filter_by_overlap:
            # basic overlap on base terms expanded to ancestors up to filter_depth
            exp = set(patient_terms)
            for t in list(patient_terms):
                for anc,_d in ancestors(t, parents, args.filter_depth):
                    exp.add(anc)
            # score overlap quickly using base disease terms
            scored = []
            for did in candidates:
                overlap = len(exp & disease_vec[did].keys())
                # apply gate(s)
                if overlap < args.filter_min_terms: 
                    continue
                if args.filter_min_ic > 0:
                    ic_sum = sum(float(ic_map.get(t,0.0)) for t in (set([t for t in dis2hpos.get(did,())]) & patient_terms))
                    if ic_sum < args.filter_min_ic: 
                        continue
                scored.append((overlap, did))
            scored.sort(reverse=True)
            candidates = [d for _o,d in scored[:args.filter_keep_top]]

        # build patient vector (with ancestor expansion for embedding)
        patient_vec = make_vector(
            base_terms=patient_terms,
            parents=parents,
            depth=args.patient_depth,
            decay=args.patient_decay,
            ic_map=ic_map, idf_map=idf_map,
            mode=args.weight_mode, gamma=args.idf_gamma
        )

        # score
        score_by = {}
        for did in candidates:
            score_by[did] = cosine_sparse(patient_vec, disease_vec[did])

        if gold_id not in score_by:
            gold_not_found += 1
            ranks.append(None)
            continue

        matched += 1
        # rank
        s_gold = score_by[gold_id]
        rank = 1 + sum(1 for s in score_by.values() if s > s_gold)
        ranks.append(rank)

        # ROC sampling
        if args.roc_negatives > 0:
            all_neg = [d for d in score_by.keys() if d != gold_id]
            if all_neg:
                k = min(args.roc_negatives, len(all_neg))
                negs = random.sample(all_neg, k)
                pos_scores.append(s_gold)
                neg_scores.extend(score_by[d] for d in negs)

    # metrics
    ks = list(sorted(set(args.report_top)))
    overall = topk_metrics(ranks, ks)
    print(f"Evaluated {len(rep_paths)} reports (matched in candidates: {matched})")
    if gold_not_found:
        print(f"Gold not present in candidates: {gold_not_found}")
    print("== HPO-space on reports ==")
    for k in ks:
        print(f"Overall Top-{k}: {overall[f'top{k}']:.4f}")
    print(f"Overall MRR: {overall['mrr']:.4f}")

    # ROC / AUC
    if args.roc_negatives > 0 and pos_scores and neg_scores:
        fpr, tpr, auc = roc_auc(pos_scores, neg_scores)
        print(f"[ROC] AUC = {auc:.3f} (pos={len(pos_scores)}, neg={len(neg_scores)})")
        if args.roc_out:
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Reports, HPO-space)")
            plt.legend(loc="lower right")
            Path(args.roc_out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.roc_out, dpi=160, bbox_inches="tight")
            print(f"[ROC] Saved to {args.roc_out}")
    else:
        print("[ROC] Not enough positives/negatives to compute ROC/AUC.")
    print(f"[Done] {time.time()-t0:.1f}s")
    
if __name__ == "__main__":
    main()
