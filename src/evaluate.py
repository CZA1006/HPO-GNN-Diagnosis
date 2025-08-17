# src/evaluate.py

import os
import glob
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import importlib.util

# --- import from src/diagnose.py dynamically ---
spec = importlib.util.spec_from_file_location("diagnose", os.path.join("src", "diagnose.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)

embed_patient   = diagnose.embed_patient
load_embeddings = diagnose.load_embeddings

# ----------------- helpers -----------------
def compute_metrics(results, k=5):
    """results: list of (true_id, [(pred_id, score), ...])"""
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
    """Try multiple phenopacket fields to find a disease ID."""
    # 1) top-level "disease": { id }
    did = (data.get("disease") or {}).get("id")
    if did:
        return did
    # 2) "diseases": [ { term: { id } } ]
    for d in data.get("diseases", []):
        term = d.get("term") or {}
        if term.get("id"):
            return term["id"]
    # 3) "interpretations": [ { diagnosis: { disease: { id } } } ]
    for it in data.get("interpretations", []):
        diag = (it.get("diagnosis") or {}).get("disease") or {}
        if diag.get("id"):
            return diag["id"]
    return None

# ----------------- main -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate HPO-GNN pipeline with IC/ancestors + ROC")
    p.add_argument("--phenopacket_dir", required=True, help="Folder containing phenopacket JSONs")
    p.add_argument("--topk", type=int, default=5)
    # checkpoints
    p.add_argument("--term_node_list", default="checkpoints/node_list.pt")
    p.add_argument("--term_embs",      default="checkpoints/hpo_gcl_embeddings.pt")
    p.add_argument("--disease_ids",    default="checkpoints/disease_ids.pt")
    p.add_argument("--disease_embs",   default="checkpoints/disease_embs.pt")
    # patient pooling knobs
    p.add_argument("--min_hpo",    type=int, default=2, help="Min positive HPOs to evaluate a case")
    p.add_argument("--decay",      type=float, default=0.7, help="Ancestor weight decay per hop")
    p.add_argument("--max_depth",  type=int, default=2, help="Ancestor expansion depth")
    p.add_argument("--neg_alpha",  type=float, default=0.5, help="Weight to subtract negated terms")
    # ROC sampling
    p.add_argument("--roc_negatives", type=int, default=50, help="# negative diseases to sample per case")
    args = p.parse_args()

    # 1) Load embeddings
    term_nodes, term_embs     = load_embeddings(args.term_node_list, args.term_embs)
    disease_ids, disease_embs = load_embeddings(args.disease_ids,   args.disease_embs)
    print(f"[DEBUG] Loaded {len(disease_ids)} diseases")

    # normalize diseases once
    disease_norm = disease_embs / disease_embs.norm(dim=1, keepdim=True)

    # 2) Discover phenopacket files
    pattern = os.path.join(args.phenopacket_dir, "**", "*.json")
    files   = glob.glob(pattern, recursive=True)
    print(f"[DEBUG] Found {len(files)} JSONs")

    results = []
    y_true, y_score = [], []

    # counters for analysis
    skipped_sparse = 0
    skipped_missing_did = 0
    skipped_parse = 0

    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            skipped_parse += 1
            continue

        # extract pos/neg HPOs
        pos_hpos, neg_hpos = [], []
        for feat in data.get("phenotypicFeatures", []):
            code = ((feat.get("type") or {}).get("id")) or None
            if not code:
                continue
            excluded = bool(feat.get("excluded", False))
            observed = (feat.get("observed") or "").upper()
            if excluded or observed == "NEGATIVE":
                neg_hpos.append(code)
            else:
                pos_hpos.append(code)

        if len(pos_hpos) < args.min_hpo:
            skipped_sparse += 1
            continue

        # extract true disease id
        true_id = extract_true_disease_id(data)
        if not true_id or true_id not in disease_ids:
            skipped_missing_did += 1
            continue

        # 3) Patient embedding (IC + ancestor expansion + negatives)
        emb = embed_patient(
            pos_hpos, term_nodes, term_embs,
            neg_codes=neg_hpos, decay=args.decay, max_depth=args.max_depth, neg_alpha=args.neg_alpha
        )
        emb_norm = emb / emb.norm()

        # 4) Rank + collect Top-K
        sims_all = (emb_norm @ disease_norm.T).squeeze(0).cpu().numpy()
        raw = np.argsort(-sims_all)[: args.topk]
        topk_idxs = [int(i) for i in raw]  # ensure python ints for indexing
        ranked = [(disease_ids[i], float(sims_all[i])) for i in topk_idxs]
        results.append((true_id, ranked))

        # 5) ROC points: 1 positive + sampled negatives
        pos_idx = disease_ids.index(true_id)
        y_true.append(1); y_score.append(sims_all[pos_idx])
        # guard if label space is tiny
        if len(disease_ids) > 1:
            neg_pool = [i for i in range(len(disease_ids)) if i != pos_idx]
            sample_k = min(args.roc_negatives, len(neg_pool))
            for ni in random.sample(neg_pool, sample_k):
                y_true.append(0); y_score.append(sims_all[ni])

    # 6) Metrics
    topk_acc, mrr = compute_metrics(results, k=args.topk)
    print(f"Evaluated {len(results)} cases")
    print(f"Top-{args.topk} Accuracy: {topk_acc:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print(f"[DEBUG] Skipped {skipped_sparse} sparse cases (<{args.min_hpo} HPOs)")
    print(f"[DEBUG] Skipped {skipped_missing_did} cases with missing/unmatched disease ID")
    print(f"[DEBUG] Skipped {skipped_parse} files due to JSON parse errors")

    # 7) ROC curve
    if len(set(y_true)) >= 2:  # need both classes
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
