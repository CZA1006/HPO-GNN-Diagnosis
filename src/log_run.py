#!/usr/bin/env python3
import argparse, os, subprocess, time, re, csv, sys, shlex
from datetime import datetime
from collections import defaultdict

def _strip_extra_dashes(cmd):
    # If user accidentally puts an extra "--", drop all leading "--" tokens
    while cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd

def _join_command(cmd):
    return " ".join(shlex.quote(x) for x in cmd)

def parse_metrics(text):
    m = {}
    def grab(pattern, key, flags=0):
        mo = re.search(pattern, text, flags)
        if mo:
            m[key] = mo.group(1)
    def grab2(pattern, key1, key2, flags=0):
        mo = re.search(pattern, text, flags)
        if mo:
            m[key1] = mo.group(1)
            m[key2] = mo.group(2)

    # Overall/Matched Top-K (unfiltered)
    for k in [5,10,50,100]:
        grab2(rf"Overall Top-{k}:\s*([0-9.]+)\s*\|\s*MRR:\s*([0-9.]+)", f"overall_top{k}", f"overall_mrr{k}")
        grab2(rf"Matched Top-{k}:\s*([0-9.]+)\s*\|\s*MRR:\s*([0-9.]+)", f"matched_top{k}", f"matched_mrr{k}")

    # Filtered Top-K
    for k in [5,10,50,100]:
        grab2(rf"Overall \(filtered\) Top-{k}:\s*([0-9.]+)\s*\|\s*MRR:\s*([0-9.]+)", f"f_overall_top{k}", f"f_overall_mrr{k}")
        grab2(rf"Matched \(filtered\) Top-{k}:\s*([0-9.]+)\s*\|\s*MRR:\s*([0-9.]+)", f"f_matched_top{k}", f"f_matched_mrr{k}")

    # Filter recall & candidate size
    grab2(r"Filter recall \(gold kept after filtering\):\s*(\d+)/(\d+)", "f_recall_kept", "f_recall_total")
    grab2(r"Mean filtered candidate size:\s*([0-9.]+)\s*\(median\s*([0-9.]+)\)", "f_mean_cand_size", "f_median_cand_size")

    # Optional AUCs
    grab(r"(?:ROC[- ]?AUC|AUC ROC)\s*[:=]\s*([0-9.]+)", "roc_auc")
    grab(r"(?:PR[- ]?AUC|AUC PR)\s*[:=]\s*([0-9.]+)", "pr_auc")

    return m

def extract_flags(cmd):
    # Pull useful flags & values into columns
    want = {
        "--k":"k",
        "--filter_depth":"filter_depth",
        "--filter_min_terms":"filter_min_terms",
        "--filter_min_ic":"filter_min_ic",
        "--filter_keep_top":"filter_keep_top",
        "--patient_depth":"patient_depth",
        "--patient_decay":"patient_decay",
        "--idf_gamma":"idf_gamma",
        "--hybrid_alpha":"hybrid_alpha",
        "--hybrid_beta":"hybrid_beta",
        "--roc_use":"roc_use",
        "--roc_negatives":"roc_negatives",
    }
    out = {}
    i = 0
    while i < len(cmd):
        t = cmd[i]
        if t in want:
            key = want[t]
            # boolean flags have no value; others do
            if i+1 < len(cmd) and not cmd[i+1].startswith("--"):
                out[key] = cmd[i+1]
                i += 2
                continue
            else:
                out[key] = "1"
        i += 1
    return out

def write_row(csv_path, row):
    # Append with header auto-management
    existing_header = []
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            for r in reader:
                rows.append(r)

    # Merge headers
    all_keys = set(existing_header or [])
    all_keys.update(row.keys())
    header = sorted(all_keys)

    # Rewrite file (keeps existing rows)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
        w.writerow({k: row.get(k, "") for k in header})

def main():
    p = argparse.ArgumentParser(description="Run an eval command, capture stdout, parse metrics, append to CSV.")
    p.add_argument("--out", required=True, help="Path to CSV results file, e.g., results/experiments.csv")
    p.add_argument("--logdir", default="logs", help="Where to store raw logs (default: logs)")
    p.add_argument("--workdir", default=".", help="Working directory for the command")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after '--'")
    args = p.parse_args()

    cmd = _strip_extra_dashes(args.command)
    if not cmd:
        print("ERROR: No command provided. Usage:\n  python src/log_run.py --out results/experiments.csv -- python src/evaluate_hybrid.py ...")
        sys.exit(2)

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, f"run_{ts}.log")

    print(f"[log_run] Running: {_join_command(cmd)}")
    print(f"[log_run] workdir: {os.path.abspath(args.workdir)}")
    print(f"[log_run] logging to: {log_path}")
    t0 = time.time()
    run = subprocess.run(cmd, cwd=args.workdir, text=True, capture_output=True)
    dt = time.time() - t0

    # Tee to terminal
    if run.stdout:
        print(run.stdout, end="")
    if run.stderr:
        print(run.stderr, end="", file=sys.stderr)

    # Save combined log
    with open(log_path, "w") as f:
        if run.stdout: f.write(run.stdout)
        if run.stderr: f.write("\n--- STDERR ---\n" + run.stderr)

    # Parse metrics
    metrics = parse_metrics((run.stdout or "") + "\n" + (run.stderr or ""))
    flags   = extract_flags(cmd)

    row = {
        "timestamp": ts,
        "duration_s": f"{dt:.3f}",
        "returncode": str(run.returncode),
        "cmd": _join_command(cmd),
        "workdir": os.path.abspath(args.workdir),
        "log_path": log_path,
    }
    row.update(flags)
    row.update(metrics)

    write_row(args.out, row)
    print(f"[log_run] Appended results to {args.out}")
    if run.returncode != 0:
        sys.exit(run.returncode)

if __name__ == "__main__":
    main()
