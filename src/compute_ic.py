#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ic.py — Compute Information Content (IC) for HPO terms using ancestor propagation.

For each disease in phenotype.hpoa, we take its annotated HPO terms and add all
ancestors (is_a + part_of). Each disease contributes at most 1 count per term.
IC(t) = -log( count(t) / N ), where N is the #distinct diseases.

Outputs:
  checkpoints/hpo_ic.pt  (dict[str, float] like {"HP:0000001": 3.14, ...})
"""

import os
import csv
import math
import torch
import obonet
from collections import defaultdict, deque
from typing import Dict, Set, List

def _load_parents(obo_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"Cannot find obo file: {obo_path}")
    graph = obonet.read_obo(obo_path)
    parents = {n: set() for n in graph.nodes()}
    for n, d in graph.nodes(data=True):
        for p in d.get("is_a", []):
            p = p.split("!")[0].strip()
            if p.startswith("HP:"):
                parents[n].add(p)
        for rel in d.get("relationship", []):
            # strings like "part_of HP:0000118 ! Phenotypic abnormality"
            if "part_of" in rel and "HP:" in rel:
                pid = rel.split("HP:")[1][:7]
                parents[n].add("HP:" + pid)
    return {k: sorted(list(v)) for k, v in parents.items() if v}

def _ancestors(term: str, parents: Dict[str, List[str]]) -> Set[str]:
    """Full ancestor set (excluding the term itself)."""
    if term not in parents:
        return set()
    out, q, seen = set(), deque([term]), {term}
    while q:
        t = q.popleft()
        for p in parents.get(t, ()):
            if p in seen:
                continue
            seen.add(p)
            out.add(p)
            q.append(p)
    return out

def _read_hpoa(hpoa_path: str) -> Dict[str, Set[str]]:
    """Return disease -> set(HPO IDs). Ignores NOT/negated qualifiers if present."""
    if not os.path.exists(hpoa_path):
        raise FileNotFoundError(f"Cannot find HPOA file: {hpoa_path}")
    mapping: Dict[str, Set[str]] = defaultdict(set)
    with open(hpoa_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        cols = {}
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                lower = [h.lower() for h in header]
                def find(col):
                    try:
                        return lower.index(col)
                    except ValueError:
                        return None
                cols["did"]  = find("database_id")
                cols["hpo"]  = find("hpo_id")
                cols["qual"] = find("qualifier")
                if cols["did"] is None or cols["hpo"] is None:
                    raise RuntimeError(f"Header missing database_id or hpo_id: {header}")
                continue
            did = row[cols["did"]].strip()
            hp  = row[cols["hpo"]].strip()
            if not did or not hp:
                continue
            # Skip explicit NOT if column exists
            if cols["qual"] is not None and cols["qual"] < len(row):
                if row[cols["qual"]].strip().upper() == "NOT":
                    continue
            mapping[did].add(hp)
    return mapping

def compute_ic(obo_path: str = "hp.obo", hpoa_path: str = "phenotype.hpoa") -> Dict[str, float]:
    parents = _load_parents(obo_path)
    disease2terms = _read_hpoa(hpoa_path)
    term_counts = defaultdict(int)
    diseases = list(disease2terms.keys())

    for did in diseases:
        terms = disease2terms[did]
        # propagate to ancestors; count each disease at most once per term
        closure = set(terms)
        for t in list(terms):
            closure.update(_ancestors(t, parents))
        for t in closure:
            term_counts[t] += 1

    N = len(diseases)
    if N == 0:
        raise RuntimeError("No diseases parsed from phenotype.hpoa.")
    ic = {t: -math.log(max(1e-12, cnt / N)) for t, cnt in term_counts.items()}
    return ic

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    ic_map = compute_ic()
    torch.save(ic_map, "checkpoints/hpo_ic.pt")
    print(f"Computed IC for {len(ic_map)} terms with ancestor propagation; saved to checkpoints/hpo_ic.pt")
