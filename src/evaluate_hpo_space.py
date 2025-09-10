#!/usr/bin/env python3
"""
HPO-space baseline (Task 1b)
- Represents diseases and patients as sparse HPO vectors (optionally incl. ancestors)
- Supports weighting: binary | ic | idf | icidf  (with --idf_gamma)
- Same overlap filtering knobs you already use (depth/min_terms/min_ic/keep_top)
- Reports Overall/Matched Top-K & MRR and plots a ROC curve.

Usage (example):
python src/evaluate_hpo_space.py \
  --phenopackets_dir phenopackets \
  --hpoa phenotype.hpoa --obo hp.obo \
  --ic checkpoints/hpo_ic.pt \
  --idf checkpoints/hpo_idf.pt --idf_gamma 1.0 \
  --weight_mode icidf \
  --filter_by_overlap --filter_depth 2 \
  --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500 \
  --patient_depth 2 --patient_decay 0.7 \
  --roc_negatives 300 \
  --report_top 5 10 50 100 \
  --roc_out results/Figure_hpo_space.png
"""
import argparse, json, os, random, math, re, collections, time
from typing import Dict, Set, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- small OBO parser (id -> name, parents) ----------
def load_obo(path:str):
    name = {}
    parents = collections.defaultdict(set)
    cur = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.rstrip()
            if line == "[Term]":
                cur = {}
            elif not line and cur:
                if 'id' in cur:
                    tid = cur['id']
                    if 'name' in cur: name[tid]=cur['name']
                    for p in cur.get('is_a', []):
                        parents[tid].add(p)
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur['id'] = line[4:].strip()
                elif line.startswith("name: "):
                    cur['name'] = line[6:].strip()
                elif line.startswith("is_a: "):
                    pid = line[6:].split(' ! ')[0].strip()
                    cur.setdefault('is_a', []).append(pid)
        # flush last
        if cur and 'id' in cur:
            tid = cur['id']
            if 'name' in cur: name[tid]=cur['name']
            for p in cur.get('is_a', []):
                parents[tid].add(p)
    return name, parents

def ancestors(term:str, parents:Dict[str,Set[str]], max_depth:int) -> Set[str]:
    out=set()
    frontier=[(term,0)]
    seen=set([term])
    while frontier:
        t,d=frontier.pop(0)
        if d>=max_depth: 
            continue
        for p in parents.get(t, ()):
            if p not in seen:
                seen.add(p); out.add(p)
                frontier.append((p,d+1))
    return out

# ---------- read HPOA: disease -> set of HPOs (no NOT) ----------
def load_hpoa(path:str) -> Dict[str, Set[str]]:
    mapping=collections.defaultdict(set)
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): continue
            # HPOA is tab-separated; columns documented by Monarch
            cols=line.rstrip('\n').split('\t')
            if len(cols) < 5: continue
            disease_id = cols[0].strip()  # database_id (e.g., OMIM:123)
            qual = cols[2].strip()
            hpo_id = cols[3].strip()
            if qual == 'NOT': 
                continue
            if hpo_id:
                mapping[disease_id].add(hpo_id)
    return dict(mapping)

# ---------- phenopacket utilities ----------
def read_patient(file_path:str) -> Tuple[str, List[str]]:
    """returns (gold_disease_id or None, list of positive HPO ids)"""
    with open(file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    # HPOs
    hpos=[]
    for feat in data.get('phenotypicFeatures', []):
        if feat.get('excluded', False): 
            continue
        t = feat.get('type') or {}
        hid = t.get('id')
        if hid: hpos.append(hid)
    # disease gold
    gold=None
    for dz in data.get('diseases', []):
        term = dz.get('term',{}) or dz.get('diseaseCode',{})
        cand = term.get('id') or term.get('code')
        if cand:
            gold=cand; break
    return gold, hpos

# ---------- vectorization ----------
def expand_with_ancestors(ids:List[str], parents, depth:int) -> Set[str]:
    out=set(ids)
    if depth>0:
        for h in ids:
            out |= ancestors(h, parents, depth)
    return out

def build_weight_map(id_set:Set[str], ic_map, idf_map, gamma:float, mode:str,
                     depth:int, parents, decay:float) -> Dict[str,float]:
    """Return weighted sparse vector w[id]=weight; ancestors added with geometric decay^dist."""
    # BFS with depth & decay
    dist = {hid:0 for hid in id_set}
    frontier=list(id_set)
    seen=set(id_set)
    while frontier:
        cur=frontier.pop(0)
        d=dist[cur]
        if d>=depth: 
            continue
        for p in parents.get(cur,()):
            if p not in seen:
                seen.add(p)
                dist[p]=d+1
                frontier.append(p)
    # assign weights
    w={}
    for hid, d in dist.items():
        base = 1.0
        if mode in ('ic','icidf'):
            base *= float(ic_map.get(hid, 0.0))
        if mode in ('idf','icidf'):
            base *= (1.0 + gamma*float(idf_map.get(hid, 0.0)))
        # if purely binary requested
        if mode == 'binary':
            base = 1.0
        if d>0:
            base *= (decay ** d)
        if base>0.0:
            w[hid]=w.get(hid,0.0)+base
    return w

def cosine_sparse(a:Dict[str,float], b:Dict[str,float]) -> float:
    if not a or not b: return 0.0
    # dot
    dot=0.0
    # iterate smaller
    (small, large) = (a,b) if len(a)<len(b) else (b,a)
    for k,v in small.items():
        if k in large:
            dot += v*large[k]
    if dot==0.0: return 0.0
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)

# ---------- metrics ----------
def topk_and_mrr(ranks:List[int], ks:List[int]) -> Tuple[Dict[int,float], float]:
    # ranks: 1-based rank of gold if found, else None/inf
    mrr_vals=[]
    topk = {k:0 for k in ks}
    for r in ranks:
        if r is None: 
            continue
        mrr_vals.append(1.0/r)
        for k in ks:
            if r<=k: topk[k]+=1
    N=len(ranks)
    topk={k:(topk[k]/N if N else 0.0) for k in ks}
    mrr=(sum(mrr_vals)/N) if N else 0.0
    return topk, mrr

def roc_curve_and_auc(scores:List[float], labels:List[int]):
    # sort by score descending
    order = np.argsort(-np.array(scores))
    y = np.array(labels)[order]
    tp = np.cumsum(y==1)
    fp = np.cumsum(y==0)
    P = tp[-1] if len(tp)>0 else 0
    N = fp[-1] if len(fp)>0 else 0
    tpr = (tp / P) if P>0 else np.zeros_like(tp, dtype=float)
    fpr = (fp / N) if N>0 else np.zeros_like(fp, dtype=float)
    # AUC via trapezoid
    auc = np.trapz(tpr, fpr) if len(tpr)>1 else 0.0
    return fpr, tpr, float(auc)

# ---------- main ----------
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phenopackets_dir', default='phenopackets')
    p.add_argument('--hpoa', required=True)
    p.add_argument('--obo', required=True)
    p.add_argument('--ic', default=None)
    p.add_argument('--idf', default=None)
    p.add_argument('--idf_gamma', type=float, default=1.0)
    p.add_argument('--weight_mode', choices=['binary','ic','idf','icidf'], default='icidf')

    # filtering
    p.add_argument('--filter_by_overlap', action='store_true')
    p.add_argument('--filter_depth', type=int, default=2)
    p.add_argument('--filter_min_terms', type=int, default=2)
    p.add_argument('--filter_min_ic', type=float, default=0.0)
    p.add_argument('--filter_keep_top', type=int, default=1000)

    # patient expansion
    p.add_argument('--patient_depth', type=int, default=2)
    p.add_argument('--patient_decay', type=float, default=0.7)

    # evaluation
    p.add_argument('--report_top', nargs='+', type=int, default=[5,10,50,100])
    p.add_argument('--roc_negatives', type=int, default=300)
    p.add_argument('--roc_out', default='Figure_hpo_space.png')
    args=p.parse_args()

    t0=time.time()
    name_map, parents = load_obo(args.obo)
    disease_hpos = load_hpoa(args.hpoa)
    ic_map = torch.load(args.ic) if args.ic and os.path.exists(args.ic) else {}
    idf_map = torch.load(args.idf) if args.idf and os.path.exists(args.idf) else {}

    # precompute disease vectors (patient-style depth/decay to keep symmetry)
    dz_vectors={}
    dz_sets_for_filter={}
    for dz, hset in disease_hpos.items():
        # for filtering we only need presence (depth=filter_depth)
        fset = expand_with_ancestors(list(hset), parents, args.filter_depth)
        dz_sets_for_filter[dz]=fset|set(hset)
        # for similarity, patient_depth/decay to mirror patient settings
        dz_vectors[dz]=build_weight_map(set(hset), ic_map, idf_map, args.idf_gamma, 
                                        args.weight_mode, args.patient_depth, parents, args.patient_decay)

    # load patients
    files=[os.path.join(args.phenopackets_dir,f) for f in os.listdir(args.phenopackets_dir) if f.endswith('.json')]
    files.sort()

    overall_ranks=[]
    matched_ranks=[]
    roc_scores=[]
    roc_labels=[]
    gold_not_present=0

    for fp in files:
        gold, patient_hpos = read_patient(fp)
        if not patient_hpos: 
            continue

        # patient vector
        p_vec=build_weight_map(set(patient_hpos), ic_map, idf_map, args.idf_gamma,
                               args.weight_mode, args.patient_depth, parents, args.patient_decay)

        # candidate filtering
        candidates=list(disease_hpos.keys())
        if args.filter_by_overlap:
            p_filter_set = expand_with_ancestors(patient_hpos, parents, args.filter_depth) | set(patient_hpos)
            # IC overlap score to prioritize
            scored=[]
            for dz in candidates:
                overlap = p_filter_set & dz_sets_for_filter[dz]
                if len(overlap) < args.filter_min_terms:
                    continue
                ic_sum = sum(float(ic_map.get(h,0.0)) for h in overlap)
                if ic_sum < args.filter_min_ic:
                    continue
                scored.append((ic_sum, dz))
            if not scored:
                # if nothing passes, fall back to all
                pass
            else:
                scored.sort(reverse=True)
                keepN = min(args.filter_keep_top, len(scored))
                candidates = [dz for _,dz in scored[:keepN]]

        # similarities
        sims=[]
        for dz in candidates:
            s = cosine_sparse(p_vec, dz_vectors[dz])
            sims.append((s,dz))
        sims.sort(reverse=True, key=lambda x:x[0])

        # rank
        rank=None
        if gold not in disease_hpos:
            gold_not_present+=1
        else:
            for i,(_,dz) in enumerate(sims, start=1):
                if dz==gold:
                    rank=i; break

        overall_ranks.append(rank)
        if gold in candidates:
            matched_ranks.append(rank)
        else:
            matched_ranks.append(None)

        # ROC: positive + sampled negatives
        if gold in disease_hpos:
            pos = cosine_sparse(p_vec, dz_vectors[gold])
            roc_scores.append(pos); roc_labels.append(1)
            # negatives from all diseases (exclude gold)
            pool=[dz for dz in disease_hpos.keys() if dz!=gold]
            random.shuffle(pool)
            for dz in pool[:args.roc_negatives]:
                roc_scores.append(cosine_sparse(p_vec, dz_vectors[dz]))
                roc_labels.append(0)

    ks = args.report_top
    overall_topk, overall_mrr = topk_and_mrr(overall_ranks, ks)
    matched_topk, matched_mrr = topk_and_mrr(matched_ranks, ks)

    print(f"Evaluated {len(overall_ranks)} cases")
    print(f"Gold not present in candidates: {gold_not_present}")
    print("== HPO-space baseline ==")
    for k in ks:
        print(f"Overall Top-{k}: {overall_topk[k]:.4f}")
    print(f"Overall MRR: {overall_mrr:.4f}")
    for k in ks:
        print(f"Matched Top-{k}: {matched_topk[k]:.4f}")
    print(f"Matched MRR: {matched_mrr:.4f}")

    # ROC
    if roc_scores:
        fpr,tpr,auc = roc_curve_and_auc(roc_scores, roc_labels)
        plt.figure(figsize=(5,5))
        plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        out=args.roc_out or 'Figure_hpo_space.png'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        plt.savefig(out,bbox_inches='tight',dpi=140)
        print(f"[ROC] Saved to {out}")
    print(f"[Done] {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
