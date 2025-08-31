#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

# Search grid
ALPHAS=(1.00 0.95 0.90 0.85)
BETAS=(0.00 0.05 0.10)
DEPTHS=(1 2)
DECAYS=(0.70 0.85)

# Common args (edit here as needed)
COMMON_ARGS=(
  --phenopackets_dir phenopackets
  --k 5
  --hpoa phenotype.hpoa --obo hp.obo --mondo mondo.obo
  --ic checkpoints/hpo_ic.pt
  --idf checkpoints/hpo_idf.pt --idf_gamma 1.0
  --filter_by_overlap --filter_depth 2
  --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500
  --roc_use cosine --roc_negatives 300
  --report_top 5 10 50 100
)

# Replace '.' with 'p' for filenames
slug() { echo "$1" | sed 's/\./p/g'; }

for a in "${ALPHAS[@]}"; do
  for b in "${BETAS[@]}"; do
    for pd in "${DEPTHS[@]}"; do
      for pc in "${DECAYS[@]}"; do
        aS=$(slug "$a"); bS=$(slug "$b"); pcS=$(slug "$pc")
        tag="a${aS}_b${bS}_pd${pd}_pc${pcS}"
        echo "=== ${tag} ==="

        # Skip if we already have a log for this combo
        if [[ -f "logs/${tag}.log" ]]; then
          echo "[skip] logs/${tag}.log exists"
          continue
        fi

        # Run + log: log_run.py appends a one-line summary to results/experiments.csv
        python src/log_run.py --out results/experiments.csv -- \
          python src/evaluate_hybrid.py \
            "${COMMON_ARGS[@]}" \
            --patient_depth "${pd}" --patient_decay "${pc}" \
            --hybrid_alpha "${a}" --hybrid_beta "${b}" \
          2>&1 | tee "logs/${tag}.log"
      done
    done
  done
done

# Optional: build a quick leaderboard after the sweep
python src/summarize_results.py results/experiments.csv \
  --by f_overall_top5 --top 20 --to_md results/leaderboard.md
echo "Done. See results/experiments.csv and results/leaderboard.md"
