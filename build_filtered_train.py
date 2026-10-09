#!/usr/bin/env python3
"""
Build filtered training data (Person 3, post-processing)
========================================================
Takes verifier scores from the LLM judge and produces:

    data/filtered_negatives.jsonl   : flagged false-negative pairs only
    data/train_pairs_filtered.jsonl : cleaned train data (FNs removed from negatives)

The downstream training script (Person 1, scripts/train.py) consumes
train_pairs_filtered.jsonl as the contrastive training file.

Usage
-----
    python build_filtered_train.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

TRAIN_PATH         = Path("data/train.json")
SCORES_PATH        = Path("data/verifier_scores.jsonl")
FILTERED_NEG_PATH  = Path("data/filtered_negatives.jsonl")
OUTPUT_PATH        = Path("data/train_pairs_filtered.jsonl")

def main():
    # ---- Load full train data ----
    with TRAIN_PATH.open() as f:
        train = json.load(f)
    print(f"[load] train.json: {len(train)} examples")

    # ---- Load verifier scores ----
    # Map query text -> set of candidate texts flagged as false negatives.
    fn_by_query: dict[str, set[str]] = defaultdict(set)
    flagged_records: list[dict] = []
    n_scores = n_fn = 0
    with SCORES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_scores += 1
            if rec.get("is_false_negative"):
                fn_by_query[rec["query"]].add(rec["candidate_text"])
                flagged_records.append(rec)
                n_fn += 1

    print(f"[load] verifier_scores.jsonl: {n_scores} judgments, "
          f"{n_fn} flagged false negatives")
    print(f"[load] queries with >=1 FN: {len(fn_by_query)}")

    # ---- Save flagged-only file ----
    FILTERED_NEG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FILTERED_NEG_PATH.open("w") as f:
        for rec in flagged_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {FILTERED_NEG_PATH} ({len(flagged_records)} rows)")

    # ---- Build filtered training data ----
    removed_total = 0
    examples_changed = 0
    with OUTPUT_PATH.open("w") as f:
        for ex in train:
            query = ex["query"]
            fn_set = fn_by_query.get(query, set())
            kept = [n for n in ex["negatives"] if n not in fn_set]
            removed = len(ex["negatives"]) - len(kept)
            if removed:
                examples_changed += 1
                removed_total += removed
            f.write(json.dumps({
                "query":      query,
                "positive":   ex["positive"],
                "negatives":  kept,
                "num_removed": removed,
            }, ensure_ascii=False) + "\n")

    print(f"[write] {OUTPUT_PATH}")
    print(f"  examples touched : {examples_changed} / {len(train)}")
    print(f"  negatives removed: {removed_total}")
    print(f"  avg removed per touched ex: "
          f"{(removed_total / examples_changed) if examples_changed else 0:.2f}")

if __name__ == "__main__":
    main()
