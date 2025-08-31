#!/usr/bin/env python3
import argparse, csv, os, sys

def read_rows(path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return r.fieldnames, rows

def to_float(x):
    try:
        return float(x)
    except:
        return float("-inf")

def main():
    p = argparse.ArgumentParser(description="Summarize experiment CSV.")
    p.add_argument("csv_path", help="results/experiments.csv")
    p.add_argument("--by", default="f_overall_top5", help="Metric column to sort by (desc). Default: f_overall_top5")
    p.add_argument("--top", type=int, default=20, help="How many rows to show")
    p.add_argument("--to_md", default="", help="Optional path to save a Markdown leaderboard")
    args = p.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"CSV not found: {args.csv_path}", file=sys.stderr)
        sys.exit(2)

    header, rows = read_rows(args.csv_path)
    rows_sorted = sorted(rows, key=lambda r: to_float(r.get(args.by, "")), reverse=True)

    # Print a compact table
    keep_cols = [c for c in [
        "timestamp","cmd","returncode","duration_s",
        "k","filter_depth","filter_min_terms","filter_min_ic","filter_keep_top",
        "patient_depth","patient_decay","idf_gamma","hybrid_alpha","hybrid_beta",
        "roc_use","roc_negatives",
        "overall_top5","overall_mrr5",
        "f_overall_top5","f_overall_mrr5",
        "f_recall_kept","f_recall_total","f_mean_cand_size","f_median_cand_size",
        "roc_auc","pr_auc",
        "log_path"
    ] if c in header]

    def format_row(r):
        return " | ".join(f"{c}={r.get(c,'')}" for c in keep_cols)

    print(f"== Top {args.top} by '{args.by}' ==")
    for r in rows_sorted[:args.top]:
        print(format_row(r))

    if args.to_md:
        with open(args.to_md, "w") as f:
            f.write(f"# Leaderboard (sorted by `{args.by}`)\n\n")
            f.write("| " + " | ".join(keep_cols) + " |\n")
            f.write("|" + "|".join(["---"]*len(keep_cols)) + "|\n")
            for r in rows_sorted[:args.top]:
                f.write("| " + " | ".join(r.get(c,"") for c in keep_cols) + " |\n")
        print(f"[summarize] Wrote markdown to {args.to_md}")

if __name__ == "__main__":
    main()
