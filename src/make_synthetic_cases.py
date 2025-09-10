#!/usr/bin/env python3
"""
Generate simple synthetic narratives from existing phenopackets (Task 2b).
Outputs:
  --out_txt  : TSV (case_id \t text)
  --out_csv  : CSV with {case_id, prompt, baseline_text} to use with an LLM.

Example:
python src/make_synthetic_cases.py \
  --phenopackets_dir phenopackets \
  --obo hp.obo \
  --out_txt data/synthetic_texts.tsv \
  --out_csv data/synthetic_prompts.csv
"""
import argparse, os, json, csv

def load_obo_labels(obo_path):
    labels={}
    cur=None
    with open(obo_path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.rstrip()
            if line=="[Term]": cur={}
            elif not line and cur:
                if 'id' in cur and 'name' in cur:
                    labels[cur['id']]=cur['name']
                cur=None
            elif cur is not None:
                if line.startswith("id: "): cur['id']=line[4:].strip()
                elif line.startswith("name: "): cur['name']=line[6:].strip()
    return labels

def narrative(case_id, hpo_ids, labels):
    terms=[labels.get(h, h) for h in hpo_ids]
    if not terms:
        return f"Case {case_id}: No specific phenotypic features were recorded."
    lead = f"Case {case_id}: The patient presents with {terms[0]}"
    if len(terms)>1:
        lead += ", " + ", ".join(terms[1:-1])
        if len(terms)>2:
            lead += f", and {terms[-1]}"
    lead += "."
    return lead

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phenopackets_dir', required=True)
    ap.add_argument('--obo', required=True)
    ap.add_argument('--out_txt', required=True)
    ap.add_argument('--out_csv', required=True)
    args=ap.parse_args()

    labels=load_obo_labels(args.obo)
    files=[f for f in os.listdir(args.phenopackets_dir) if f.endswith('.json')]
    files.sort()

    os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    with open(args.out_txt,'w',encoding='utf-8') as ft, \
         open(args.out_csv,'w',encoding='utf-8',newline='') as fc:
        w=csv.DictWriter(fc, fieldnames=['case_id','prompt','baseline_text'])
        w.writeheader()
        for fn in files:
            path=os.path.join(args.phenopackets_dir, fn)
            with open(path,'r',encoding='utf-8') as f:
                data=json.load(f)
            cid=data.get('id', os.path.splitext(fn)[0])
            hpos=[pf.get('type',{}).get('id') for pf in data.get('phenotypicFeatures',[]) if not pf.get('excluded',False)]
            hpos=[h for h in hpos if h]
            text=narrative(cid, hpos, labels)
            ft.write(f"{cid}\t{text}\n")
            prompt=( "Create a diagnosis-blinded clinical narrative from the following HPO features. "
                     "Do NOT mention the diagnosis or gene. Use concise clinical language.\n"
                     f"HPO IDs: {', '.join(hpos)}" )
            w.writerow({'case_id':cid,'prompt':prompt,'baseline_text':text})
    print(f"Wrote {args.out_txt} and {args.out_csv}")

if __name__=='__main__':
    main()
