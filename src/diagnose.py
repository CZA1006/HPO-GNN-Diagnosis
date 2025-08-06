import torch
from collections import defaultdict

def load_embeddings(path_ids, path_embs):
    ids  = torch.load(path_ids, weights_only=True)
    embs = torch.load(path_embs)
    return ids, embs

def embed_patient(hpo_codes, term_node_list, term_embs):
    term2idx = {t:i for i,t in enumerate(term_node_list)}
    idxs = [term2idx[c] for c in hpo_codes if c in term2idx]
    if not idxs:
        raise ValueError("None of the provided HPO codes are in the embedding vocab")
    return term_embs[idxs].mean(dim=0, keepdim=True)

def rank_diseases(patient_emb, disease_ids, disease_embs, topk=10):
    pe = patient_emb / patient_emb.norm(dim=1, keepdim=True)
    de = disease_embs / disease_embs.norm(dim=1, keepdim=True)
    sims = (pe @ de.t()).squeeze(0)  # [n_diseases]
    vals, idxs = sims.topk(topk, largest=True)
    return [(disease_ids[i], float(vals[j])) for j,i in enumerate(idxs)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--term_node_list", default="checkpoints/node_list.pt")
    parser.add_argument("--term_embs",      default="checkpoints/hpo_gcl_embeddings.pt")
    parser.add_argument("--disease_ids",    default="checkpoints/disease_ids.pt")
    parser.add_argument("--disease_embs",   default="checkpoints/disease_embs.pt")
    parser.add_argument("--patient_hpos",   required=True,
                        help="Comma-separated HPO codes, e.g. HP:0001250,HP:0004321")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    # load
    term_node_list, term_embs     = load_embeddings(args.term_node_list, args.term_embs)
    disease_ids, disease_embs     = load_embeddings(args.disease_ids, args.disease_embs)
    patient_hpos = args.patient_hpos.split(",")

    # embed & rank
    patient_emb = embed_patient(patient_hpos, term_node_list, term_embs)
    top         = rank_diseases(patient_emb, disease_ids, disease_embs, topk=args.topk)

    print("Top candidate diagnoses:")
    for did, score in top:
        print(f"{did}\t{score:.4f}")
