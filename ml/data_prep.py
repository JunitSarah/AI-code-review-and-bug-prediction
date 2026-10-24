"""Build train/val/test JSONL from Defects4J and BugsInPy bug-fix patches.

Expected local layout (clone the datasets yourself, see README):
  BugsInPy:   <bugsinpy>/projects/<project>/bugs/<id>/bug_patch.txt
  Defects4J:  <defects4j>/framework/projects/<Project>/patches/<id>.src.patch

Usage:
  python data_prep.py --bugsinpy ./data/BugsInPy --defects4j ./data/defects4j --out ./data/processed

Split is GROUPED BY PROJECT-BUG so the buggy/fixed halves of one hunk never straddle
train and test (that leak inflates F1 badly). For a stricter test, hold out whole projects.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from diff_utils import parse_unified_diff

MIN_CHARS, MAX_CHARS = 30, 4000


def iter_patches(bugsinpy: Path | None, defects4j: Path | None):
    if bugsinpy:
        for p in bugsinpy.glob("projects/*/bugs/*/bug_patch.txt"):
            yield f"bugsinpy/{p.parents[2].name}-{p.parent.name}", p
    if defects4j:
        for p in defects4j.glob("framework/projects/*/patches/*.src.patch"):
            yield f"defects4j/{p.parents[1].name}-{p.name.split('.')[0]}", p


def build_samples(bugsinpy, defects4j):
    samples = []
    for group, path in iter_patches(bugsinpy, defects4j):
        text = path.read_text(errors="ignore")
        for hi, h in enumerate(parse_unified_diff(text)):
            for code, label in ((h.before, 1), (h.after, 0)):
                if MIN_CHARS <= len(code) <= MAX_CHARS:
                    # pair_id links the buggy and fixed twins of one hunk (used by evaluate.py)
                    samples.append({"code": code, "label": label, "group": group, "file": h.file,
                                    "pair_id": f"{group}#{hi}"})
    return samples


def grouped_split(samples, seed=42, val=0.1, test=0.1):
    groups = sorted({s["group"] for s in samples})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    test_g = set(groups[: int(n * test)])
    val_g = set(groups[int(n * test): int(n * (test + val))])
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        key = "test" if s["group"] in test_g else "val" if s["group"] in val_g else "train"
        out[key].append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugsinpy", type=Path)
    ap.add_argument("--defects4j", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    if not (args.bugsinpy or args.defects4j):
        ap.error("pass at least one of --bugsinpy / --defects4j")

    samples = build_samples(args.bugsinpy, args.defects4j)
    splits = grouped_split(samples)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(args.out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in rows)
        print(f"{name:5s} n={len(rows):6d} buggy={pos:6d} ({pos / max(len(rows), 1):.0%})")


if __name__ == "__main__":
    main()
