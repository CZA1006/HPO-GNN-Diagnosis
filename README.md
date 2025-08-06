# HPO-GNN-Diagnosis

This repository implements a GT2Vec-inspired pipeline on the Human Phenotype Ontology (HPO) to generate joint text+graph embeddings for symptom-based disease diagnosis.

---

## Repository Structure

```
HPO-GNN-Diagnosis/
├── src/
│   ├── hpo_tsdae.py       # TSDAE fine-tuning on HPO term texts
│   └── hpo_gcl.py         # Graph Contrastive Learning on HPO DAG
├── hp.obo                 # HPO ontology file (not versioned if in .gitignore)
├── phenotype.hpoa         # Disease↔HPO annotation (not versioned if in .gitignore)
└── .gitignore
```

---

## Prerequisites

- Python 3.8+  
- macOS/Linux/Windows with CPU or GPU  

**Install dependencies**  
```bash
python -m venv .venv      # create virtual env
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```  

> requirements.txt should include:
> ```
> torch
> transformers
> obonet
> networkx
> torch-geometric
> matplotlib
> ```

---

## Data Download

1. **HPO Ontology**  
   ```bash
   wget -O hp.obo \
     https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo
   ```
2. **HPO Annotation**  
   ```bash
   wget -O phenotype.hpoa \
     https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-05-06/phenotype.hpoa
   ```

Place both files in the project root.

---

## Step 1: TSDAE Textual Encoder

Fine-tune BioBERT on HPO term definitions.

```bash
python src/hpo_tsdae.py \
  --hpo_obo hp.obo \
  --model_name dmis-lab/biobert-v1.1 \
  --batch_size 16 \
  --lr 5e-5 \
  --epochs 5 \
  --device cpu 
```  
**Output:**
- `checkpoints/hpo_tsdae_encoder/` containing the fine-tuned BERT model.

**Notes:**
- Use fewer epochs or a smaller subset for quick tests: add `--epochs 1` or slice dataset.

---

## Step 2: Graph Contrastive Learning (GCL)

Generate joint text+graph embeddings via contrastive GIN.

```bash
python src/hpo_gcl.py \
  --obo hp.obo \
  --tsdae_ckpt checkpoints/hpo_tsdae_encoder \
  --epochs 100 \
  --device cpu
```  
**Output:**
- `checkpoints/hpo_gcl_embeddings.pt` tensor of shape `[num_terms, hidden_dim]`.

---

## Next Steps

1. **Aggregate term embeddings**
   - Mean or attention pooling over each disease’s HPO term set (via `phenotype.hpoa`).
2. **Generate patient narratives**
   - Use LLM to convert phenopacket JSON to free-text.
3. **Extract HPO terms from text**
   - Rule-based or LLM/NLP-based mapping back to HPO codes.
4. **Embed patients & diagnose**
   - Compute cosine similarity between patient and disease vectors.

---

## License

Specify your license here.

---

_For questions or issues, please open an issue on GitHub._

