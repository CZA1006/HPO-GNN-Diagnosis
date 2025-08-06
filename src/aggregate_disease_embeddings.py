import os
import torch
import csv
from collections import defaultdict

def load_term_embeddings(node_list_path, term_embs_path):
    node_list = torch.load(node_list_path, weights_only=True)
    term_embs = torch.load(term_embs_path)
    return node_list, term_embs

def load_disease_annotations(hpoa_path):
    """
    Reads the .hpoa file, skips comment lines, auto-detects
    the 'disease' and 'hpo' column indices, and returns:
      dict[disease_id] = set(hpo_ids)
    """
    mapping = defaultdict(set)
    with open(hpoa_path) as f:
        # find header line
        for line in f:
            if line.startswith('#'):
                continue
            header = line.strip().split('\t')
            break
        # auto-detect columns
        disease_col = next((i for i,h in enumerate(header)
                            if 'disease' in h.lower()), None)
        hpo_col     = next((i for i,h in enumerate(header)
                            if 'hpo' in h.lower()), None)
        if disease_col is None or hpo_col is None:
            raise ValueError(f"Could not find disease/HPO cols in header: {header}")

        # read each data line
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            cols    = line.strip().split('\t')
            disease = cols[disease_col]
            hpo     = cols[hpo_col]
            mapping[disease].add(hpo)
    return mapping

def aggregate(disease2terms, node_list, term_embs):
    """
    Mean-pool term_embs for each disease’s HPO set.
    Returns (disease_ids, tensor [N_diseases, D])
    """
    term2idx = {t:i for i,t in enumerate(node_list)}
    disease_ids = []
    disease_vecs = []
    for disease, terms in disease2terms.items():
        idxs = [term2idx[t] for t in terms if t in term2idx]
        if not idxs:
            continue
        vec = term_embs[idxs].mean(dim=0)
        disease_ids.append(disease)
        disease_vecs.append(vec)
    if not disease_ids:
        raise RuntimeError("No disease embeddings computed; check your .hpoa parsing.")
    return disease_ids, torch.stack(disease_vecs, dim=0)

if __name__ == '__main__':
    # 1) load term embeddings
    node_list, term_embs = load_term_embeddings(
        'checkpoints/node_list.pt',
        'checkpoints/hpo_gcl_embeddings.pt'
    )

    # 2) load disease→HPO mapping
    disease2terms = load_disease_annotations('phenotype.hpoa')

    # 3) aggregate
    disease_ids, disease_embs = aggregate(disease2terms, node_list, term_embs)

    # 4) save results
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(disease_ids,  'checkpoints/disease_ids.pt')
    torch.save(disease_embs, 'checkpoints/disease_embs.pt')
    print(f"Saved {len(disease_ids)} disease embeddings")
