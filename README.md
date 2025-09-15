# HPO-GNN Diagnosis — Working README

This repo contains scripts to (a) build disease embeddings from HPO, (b) score candidate diagnoses for phenopackets or free-text reports, and (c) evaluate Top‑K and ROC/AUC.

> **Status (Sept 2025)**  
> We added IC+IDF weighting, a phenotype-overlap filter, a hybrid scorer, experiment logging, and two new baselines:
> 1) an **HPO-space (IC×IDF) baseline** and  
> 2) a **free‑text synthetic reports** evaluation.  
> See **Results** below for the latest numbers and figure paths.

---

## 1) Setup

```bash
conda activate hpo-gnn-diagnosis   # your environment
pip install -r requirements.txt    # if not already done
```

Place ontology and annotation files at the repo root (or pass paths explicitly):
- `hp.obo` (Human Phenotype Ontology)
- `phenotype.hpoa` (HPOA)
- `mondo.obo` (MONDO; optional but recommended)

If you need MONDO:
```bash
curl -L -o mondo.obo https://purl.obolibrary.org/obo/mondo.obo
```

---

## 2) Precompute term IC, disease embeddings

```bash
# 2.1 Information content (IC) for HPO terms
python src/compute_ic.py   --hpoa phenotype.hpoa   --obo hp.obo   --out checkpoints/hpo_ic.pt

# 2.2 Aggregate disease embeddings (uses GNN term embeddings + HPOA)
python src/aggregate_disease_embeddings.py   --node_list checkpoints/node_list.pt   --term_embs checkpoints/hpo_gcl_embeddings.pt   --hpoa phenotype.hpoa   --ic checkpoints/hpo_ic.pt   --obo hp.obo   --ancestor_depth 2 --ancestor_decay 0.7   --out_ids  checkpoints/disease_ids.pt   --out_embs checkpoints/disease_embs.pt
```

> Notes
> - `checkpoints/hpo_gcl_embeddings.pt` must exist (your trained/loaded term embeddings).
> - `checkpoints/node_list.pt` maps HPO:#### terms to embedding indices.

---

## 3) Hybrid evaluator (phenopackets)

The hybrid score combines:
- **Vector sim** between patient embedding and disease embedding
- **IC/IDF overlap** (term‑level)
- Optional **phenotype‑overlap filter** to shrink the candidate set

### 3.1 Example run

```bash
python src/evaluate_hybrid.py   --phenopackets_dir phenopackets   --k 5   --hpoa phenotype.hpoa --obo hp.obo --mondo mondo.obo   --ic  checkpoints/hpo_ic.pt   --idf checkpoints/hpo_idf.pt --idf_gamma 1.0   --filter_by_overlap --filter_depth 2   --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500   --patient_depth 2 --patient_decay 0.7   --hybrid_alpha 0.90 --hybrid_beta 0.10   --roc_use hybrid --roc_negatives 300   --report_top 5 10 50 100
```

- `--hybrid_alpha` weights vector sim, `--hybrid_beta` weights IC×IDF overlap.  
- `--patient_depth / --patient_decay` expand patient terms to shallow ancestors.

---

## 4) Baseline in HPO space (no GNN)

A pure IC/IDF overlap scorer that ignores disease embeddings:

```bash
python src/evaluate_hpo_space.py   --phenopackets_dir phenopackets   --hpoa phenotype.hpoa --obo hp.obo --mondo mondo.obo   --ic  checkpoints/hpo_ic.pt   --idf checkpoints/hpo_idf.pt --idf_gamma 1.0   --weight_mode icidf   --filter_by_overlap --filter_depth 2   --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500   --patient_depth 2 --patient_decay 0.7   --roc_negatives 300   --report_top 5 10 50 100   --roc_out results/Figure_hpo_space.png
```

---

## 5) Free‑text synthetic reports (IC×IDF baseline)

### 5.1 Prepare reports

Put raw TARs under `data/synthetic_reports/raw/`:
- `reports.tar.gz`
- `reports_70.tar.gz`
- (optional) `reports.tar.gz` from Monica’s email
- `reports/rest.txt` (a big TXT that we split per‑OMIM)

Then:

```bash
# Split rest.txt into one file per OMIM ID
python src/split_rest_txt.py

# Expand the tarballs to per‑OMIM text files
mkdir -p data/synthetic_reports/by_omim
tar -xzf data/synthetic_reports/raw/reports_70.tar.gz -C data/synthetic_reports/by_omim
tar -xzf data/synthetic_reports/raw/reports.tar.gz    -C data/synthetic_reports/by_omim

# Build an index CSV (omim_id, path, n_chars, n_lines)
python src/index_synthetic_reports.py
```

### 5.2 Evaluate (IC×IDF)

```bash
python src/evaluate_reports_icidf.py   --reports_dir data/synthetic_reports/by_omim   --hpoa phenotype.hpoa --obo hp.obo   --ic  checkpoints/hpo_ic.pt   --idf checkpoints/hpo_idf.pt --idf_gamma 1.0   --weight_mode icidf   --patient_depth 2 --patient_decay 0.7   --filter_by_overlap --filter_depth 2   --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500   --roc_negatives 300   --report_top 5 10 50 100   --roc_out results/Figure_reports_icidf.png
```

To run a *looser* variant (no overlap filter, more gold retained, different trade‑off):
```bash
python src/evaluate_reports_icidf.py   --reports_dir data/synthetic_reports/by_omim   --hpoa phenotype.hpoa --obo hp.obo   --ic  checkpoints/hpo_ic.pt   --idf checkpoints/hpo_idf.pt --idf_gamma 1.0   --weight_mode icidf   --patient_depth 2 --patient_decay 0.7   --roc_negatives 300   --report_top 5 10 50 100   --roc_out results/Figure_reports_icidf_loose.png
```

---

## 6) Track experiments

Two helper scripts live in **`src/`**:

- `log_run.py` — run any command, capture stdout/stderr, and append a row to CSV.
- `summarize_results.py` — read the CSV and print a small leaderboard.

Example:

```bash
# single run
python src/log_run.py --out results/experiments.csv --   python src/evaluate_hybrid.py     --phenopackets_dir phenopackets     --k 5     --hpoa phenotype.hpoa --obo hp.obo --mondo mondo.obo     --ic checkpoints/hpo_ic.pt --idf checkpoints/hpo_idf.pt --idf_gamma 1.0     --filter_by_overlap --filter_depth 2     --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500     --patient_depth 2 --patient_decay 0.7     --hybrid_alpha 0.9 --hybrid_beta 0.1     --roc_use hybrid --roc_negatives 300     --report_top 5 10 50 100

# summarize
python src/summarize_results.py --csv results/experiments.csv
```

Logs are saved under `logs/`, figures under `results/`.

---

## 7) Results (so far)

### 7.1 Phenopackets — Hybrid scorer
Representative configs and outcomes from your runs:

- **Post‑filter ranking (overlap filter applied, candidate cap=500)**  
  `--filter_by_overlap --filter_depth 2 --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500`  
  `--patient_depth 2 --patient_decay 0.7 --hybrid_alpha 0.90 --hybrid_beta 0.10`  
  **Top‑5 ≈ 0.63**, **MRR ≈ 0.54** (Matched numbers are essentially identical)  
  Figure: `results/Figure_10.png` / `results/Figure_11.png`

- **Full candidate set (no post‑filter ranking)**  
  Same IC/IDF and patient expansion, **Top‑5 ≈ 0.077**, **Top‑10 ≈ 0.123**.

> Interpretation: the overlap filter prunes ~90% of the catalog while *increasing* precision considerably.  
> We continue tuning patient ancestor depth/decay and α/β.

### 7.2 Free‑text synthetic reports — IC×IDF baseline

- **Strict overlap filter**  
  **Top‑5 = 0.1075**, **Top‑10 = 0.1729**, **Top‑50 = 0.4112**, **Top‑100 = 0.5047**, **MRR = 0.1210**  
  **AUC = 0.846** (pos=133, neg=37,717)  
  Figure: `results/Figure_reports_icidf.png`

- **Looser variant (no overlap filter)**  
  **Top‑5 = 0.0561**, **Top‑10 = 0.0888**, **Top‑50 = 0.3131**, **Top‑100 = 0.3925**, **MRR = 0.0412**  
  **AUC = 0.888** (pos=212, neg=63,600)  
  Figure: `results/Figure_reports_icidf_loose.png`

> Trade‑off: removing the overlap filter keeps more positives and increases AUC, but Top‑K precision drops.

---

## 8) Next steps

- Train the GNN longer (≥10–50 epochs) and **fine‑tune** on phenopackets; re‑export `hpo_gcl_embeddings.pt`.
- Try `hybrid_alpha`/`beta` sweeps with `log_run.py` and compare in `results/experiments.csv`.
- Explore deeper/shallower patient ancestor pooling (`--patient_depth`, `--patient_decay`).

---

## 9) Troubleshooting

- If you see warnings like `torch.load ... weights_only=False`, they’re safe; PyTorch is deprecating the default.  
- If MONDO lookup errors occur, download a fresh `mondo.obo` (see §1) and pass `--mondo mondo.obo`.
- If *“Gold not present in candidates”* is high, reduce filtering or increase `--filter_keep_top`.

---

## 10) Figures to check in this repo

- `Figure_10.png` and/or `Figure_11.png` — Hybrid ROC/diagnostic curves
- `results/Figure_reports_icidf.png` and `_loose.png` — Reports baseline ROC

---

*Last updated:* 2025-09-15 09:53:23
