# HPO-GNN-Diagnosis

An end-to-end pipeline that combines text and graph neural embeddings of the Human Phenotype Ontology (HPO) to automatically diagnose diseases from patient phenotypic features.

---

## 📂 Repository Structure

```
HPO-GNN-Diagnosis/
├── checkpoints/                     # Model outputs and embeddings
│   ├── hpo_tsdae_encoder/           # Fine-tuned BioBERT TSDAE weights
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── node_list.pt                 # Ordered list of HPO term IDs
│   ├── hpo_gcl_embeddings.pt        # Joint text+graph term embeddings
│   ├── disease_ids.pt               # List of disease names/IDs
│   └── disease_embs.pt              # Pooled disease embeddings
├── phenopackets/                    # Sample patient phenopackets
│   └── test.json                    # Example phenopacket (JSON)
├── src/                              
│   ├── hpo_tsdae.py                  # TSDAE fine-tuning on term texts
│   ├── hpo_gcl.py                    # Graph contrastive GNN training
│   ├── aggregate_disease_embeddings.py  # Pool term → disease vectors
│   ├── diagnose.py                   # Rank diseases for one patient
│   ├── evaluate.py                   # Batch-evaluate on JSON folder
│   └── requirements.txt              # Python dependencies
├── hp.obo                            # HPO ontology (OBO format)
├── phenotype.hpoa                    # Disease ↔ HPO annotations
├── README.md                         # This file
└── .gitignore                        # Ignore checkpoints, envs, data
```

---

## 🛠️ Setup

1. **Clone & create venv**

   ```bash
   git clone https://github.com/CZA1006/HPO-GNN-Diagnosis.git
   cd HPO-GNN-Diagnosis
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   ```

2. **Install dependencies**

   ```bash
   pip install -r src/requirements.txt
   ```

3. **Download data files**

   ```bash
   wget -O hp.obo \
     https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo

   wget -O phenotype.hpoa \
     https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-05-06/phenotype.hpoa
   ```

---

## 🚀 Usage

### 1. Fine-tune TSDAE on HPO Terms

```bash
python src/hpo_tsdae.py \
  --hpo_obo hp.obo \
  --model_name dmis-lab/biobert-v1.1 \
  --batch_size 16 \
  --lr 5e-5 \
  --epochs 5 \
  --device cuda    # or cpu
```

Outputs: `checkpoints/hpo_tsdae_encoder/`

---

### 2. Graph Contrastive Learning (GCL)

```bash
python src/hpo_gcl.py \
  --obo hp.obo \
  --tsdae_ckpt checkpoints/hpo_tsdae_encoder \
  --model_name dmis-lab/biobert-v1.1 \
  --epochs 100 \
  --device cuda    # or cpu
```

Creates:

- `checkpoints/node_list.pt`
- `checkpoints/hpo_gcl_embeddings.pt`

---

### 3. Aggregate Disease Embeddings

```bash
python src/aggregate_disease_embeddings.py
```

Produces:

- `checkpoints/disease_ids.pt`
- `checkpoints/disease_embs.pt`

---

### 4. Diagnose a Single Patient

```bash
python src/diagnose.py \
  --term_node_list checkpoints/node_list.pt \
  --term_embs      checkpoints/hpo_gcl_embeddings.pt \
  --disease_ids    checkpoints/disease_ids.pt \
  --disease_embs   checkpoints/disease_embs.pt \
  --patient_hpos   HP:0001250,HP:0004321 \
  --topk           10
```

---

### 5. Batch Evaluation

```bash
python src/evaluate.py \
  --phenopacket_dir phenopackets \
  --topk 5
```

Outputs Top-5 accuracy and Mean Reciprocal Rank over all JSON cases.

---

## 📈 Next Steps

- **Expand test set** with real or synthetic phenopackets.
- **Improve aggregation** (weighted pooling or attention).
- **Extract HPO codes** automatically from clinical narratives.
- **Deploy** as an API (Flask / FastAPI).

---

## ⚠️ Notes

- **GPU recommended** for reasonable training speed.
- Paths and hyperparameters are configurable via CLI flags.
- Feel free to open issues or PRs for enhancements!

