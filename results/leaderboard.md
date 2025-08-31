# Leaderboard (sorted by `f_overall_top5`)

| timestamp | cmd | returncode | duration_s | k | filter_depth | filter_min_terms | filter_min_ic | filter_keep_top | patient_depth | patient_decay | idf_gamma | hybrid_alpha | hybrid_beta | roc_use | roc_negatives | overall_top5 | overall_mrr5 | log_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20250829-015429 | python src/evaluate_hybrid.py --phenopackets_dir phenopackets --k 5 --hpoa phenotype.hpoa --obo hp.obo --mondo mondo.obo --ic checkpoints/hpo_ic.pt --idf checkpoints/hpo_idf.pt --idf_gamma 1.0 --filter_by_overlap --filter_depth 2 --filter_min_terms 2 --filter_min_ic 2.6 --filter_keep_top 500 --patient_depth 2 --patient_decay 0.7 --hybrid_alpha 0.9 --hybrid_beta 0.1 --roc_use hybrid --roc_negatives 300 --report_top 5 10 50 100 | 0 | 197.222 | 5 | 2 | 2 | 2.6 | 500 | 2 | 0.7 | 1.0 | 0.9 | 0.1 | hybrid | 300 | 0.0767 | 0.0619 | logs/run_20250829-015429.log |
