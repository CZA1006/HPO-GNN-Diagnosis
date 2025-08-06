# src/evaluate.py

import torch
import json
import glob
from collections import defaultdict
from diagnose import embed_patient, rank_diseases, load_embeddings

def compute_metrics(results, k=5):
    """
    results: list of tuples (true_disease, ranked_list_of_diseases)
    returns: top-k accuracy for k, and MRR
    """
    topk_hits = 0
    mrr_total = 0.0
    n = len(results)
    for true_disease, ranked in results:
        ids = [did for did, _ in ranked]
        if true_disease in ids[:k]:
            topk_hits += 1
            rank = ids.index(true_disease) + 1
            mrr_total += 1.0 / rank
        else:
            mrr_total += 0.0
    topk_acc = topk_hits / n if n else 0.0
    mrr = mrr_total / n if n else 0.0
    return topk_acc, mrr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluate on phenopacket JSONs")
    parser.add_argument('--phenopacket_dir', type=str, required=True,
                        help='Directory containing phenopacket JSON files')
    parser.add_argument('--topk', type=int, default=5,
                        help='Compute Top-k accuracy')
    parser.add_argument('--term_node_list', default='checkpoints/node_list.pt')
    parser.add_argument('--term_embs',      default='checkpoints/hpo_gcl_embeddings.pt')
    parser.add_argument('--disease_ids',    default='checkpoints/disease_ids.pt')
    parser.add_argument('--disease_embs',   default='checkpoints/disease_embs.pt')
    args = parser.parse_args()

    # Load embeddings once
    term_node_list, term_embs    = load_embeddings(args.term_node_list, args.term_embs)
    disease_ids, disease_embs    = load_embeddings(args.disease_ids, args.disease_embs)

    # Collect results
    results = []
    files = glob.glob(f"{args.phenopacket_dir}/*.json")
    for filepath in files:
        data = json.load(open(filepath))
        # Adjust these keys based on your phenopacket schema
        true_disease = data.get('disease', {}).get('id') or \
                       data.get('meta', {}).get('diseaseId')
        patient_hpos = [feat['type']['id'] for feat in data.get('phenotypicFeatures', [])]
        if not true_disease or not patient_hpos:
            continue
        patient_emb = embed_patient(patient_hpos, term_node_list, term_embs)
        ranked = rank_diseases(patient_emb, disease_ids, disease_embs, topk=args.topk)
        results.append((true_disease, ranked))

    # Compute metrics
    topk_acc, mrr = compute_metrics(results, k=args.topk)
    print(f"Evaluated {len(results)} cases")
    print(f"Top-{args.topk} Accuracy: {topk_acc:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
