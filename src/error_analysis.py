# src/error_analysis.py

import os
import glob
import json
from collections import Counter

# 1) Gather all phenopacket JSON paths
phenos = glob.glob(os.path.join("phenopackets", "**", "*.json"), recursive=True)
print(f"Found {len(phenos)} phenopacket files.\n")

# 2) Count HPO terms per file
counter = Counter()
for fp in phenos:
    try:
        data = json.load(open(fp))
        hpos = [ feat.get("type", {}).get("id")
                 for feat in data.get("phenotypicFeatures", []) ]
        # also include any under legacy key        
        hpos += [ feat.get("hpoId")
                  for feat in data.get("phenotypes", []) if feat.get("hpoId") ]
        count = len([x for x in hpos if x])
        counter[count] += 1
    except Exception as e:
        print("  ⚠️ Failed to parse", fp, ":", e)

# 3) Print distribution summary
print("HPO‐term counts per phenopacket:")
for num_terms in sorted(counter):
    print(f"  {num_terms:2d} terms  → {counter[num_terms]:5d} files")

total = sum(counter.values())
mean = sum(k * v for k, v in counter.items()) / total
print(f"\nMean HPO terms per file: {mean:.2f}")
print(f"Files with ≥5 terms: {sum(v for k,v in counter.items() if k>=5)} / {total}")
print(f"Files with  0 terms: {counter.get(0,0)}")

# 4) Identify a few example “very sparse” failures
print("\nExample files with ≤1 HPO term (likely to fail diagnosis):")
examples = [fp for fp in phenos
            if len(json.load(open(fp)).get("phenotypicFeatures", [])) <= 1]
for fp in examples[:5]:
    print(" ", fp)
