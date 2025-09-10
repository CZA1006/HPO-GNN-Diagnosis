#!/usr/bin/env python3
"""
Very simple text → HPO matcher using names & synonyms from hp.obo.
Conservative exact/phrase matching with word boundaries (case-insensitive).

Examples:
  echo "short stature and brachydactyly" | python src/text_to_hpo.py --obo hp.obo
  python src/text_to_hpo.py --obo hp.obo --infile case.txt
"""
import argparse, re
from typing import Dict, List, Tuple

def build_phrase_index(obo_path:str) -> List[Tuple[re.Pattern,str,str]]:
    items=[]
    cur=None
    with open(obo_path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.rstrip()
            if line=="[Term]":
                cur={'syn':[]}
            elif not line and cur:
                if 'id' in cur and 'name' in cur:
                    hid=cur['id']; names=[cur['name']]+cur['syn']
                    for nm in names:
                        s=re.escape(nm.lower())
                        pat=re.compile(rf'(?<!\w){s}(?!\w)')  # whole phrase
                        items.append((pat,hid,nm))
                cur=None
            elif cur is not None:
                if line.startswith("id: "): cur['id']=line[4:].strip()
                elif line.startswith("name: "): cur['name']=line[6:].strip()
                elif line.startswith("synonym: "):
                    # synonym: "text" EXACT/RELATED ...
                    m=re.match(r'synonym:\s*"(.*)"', line)
                    if m: cur['syn'].append(m.group(1))
    return items

def extract_hpos(text:str, index) -> List[str]:
    t=text.lower()
    found=[]
    seen=set()
    for pat,hid,_ in index:
        if pat.search(t):
            if hid not in seen:
                seen.add(hid); found.append(hid)
    return found

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--obo', required=True)
    ap.add_argument('--text', default=None)
    ap.add_argument('--infile', default=None)
    args=ap.parse_args()
    if args.text is None and args.infile is None:
        print("Provide --text or --infile"); return
    text = args.text
    if args.infile:
        with open(args.infile,'r',encoding='utf-8') as f:
            text = f.read()
    index=build_phrase_index(args.obo)
    ids=extract_hpos(text, index)
    print('\n'.join(ids))

if __name__=='__main__':
    main()
