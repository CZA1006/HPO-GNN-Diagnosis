#!/usr/bin/env python3
"""
Convert a CSV corpus of case reports to phenopacket JSONs by tagging HPOs.

Input CSV columns:
  case_id,text,disease_id

Example:
python src/prepare_text_corpus_to_phenopackets.py \
  --csv data/rolando_cases.csv \
  --obo hp.obo \
  --out_dir phenopackets_text
"""
import argparse, csv, json, os
from text_to_hpo import build_phrase_index, extract_hpos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--obo', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--min_terms', type=int, default=2)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index=build_phrase_index(args.obo)
    kept=0; total=0
    with open(args.csv,'r',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        for row in rd:
            total+=1
            cid=row['case_id'].strip()
            text=row['text']
            dz=row.get('disease_id','').strip() or None
            hpos=extract_hpos(text, index)
            if len(hpos) < args.min_terms:
                continue
            pp={
                "id": cid,
                "phenotypicFeatures": [{"type":{"id":hid}} for hid in sorted(set(hpos))],
            }
            if dz:
                pp["diseases"]=[{"term":{"id":dz}}]
            out=os.path.join(args.out_dir, f"{cid}.json")
            with open(out,'w',encoding='utf-8') as fo:
                json.dump(pp, fo, ensure_ascii=False, indent=2)
            kept+=1
    print(f"Wrote {kept}/{total} phenopackets to {args.out_dir}")

if __name__=='__main__':
    main()
