"""Evaluate the fine-tuned model honestly: baselines, pair accuracy, shortcut checks, CIs, threshold.

  # full run (needs the trained model folder):
  python evaluate.py --data data/processed --model ../artifacts/codebert-defect --out ../artifacts/codebert-defect

  # baselines only (no model / no GPU needed):
  python evaluate.py --data data/processed --skip_model --out results

Writes into --out:
  results.json    all numbers
  results.md      table to paste into the paper
  threshold.json  operating threshold chosen on the VALIDATION set (the backend picks this up)
  curves.png      ROC and precision-recall curves (test set)

Metrics
  accuracy / precision / recall / F1 / AUC   standard, at the chosen threshold (AUC is threshold-free)
  pair accuracy   for each hunk, does the BUGGY twin score above its FIXED twin?  (0.5 = chance)
                  This is the question the tool is really asked, and it ignores calibration.
  flag rate       share of test hunks flagged "defect-prone" (0.5 on a balanced set is the neutral value)
  95% CIs         bootstrap, resampling whole bugs so twins stay together
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

ALWAYS = "Always 'buggy'"
BERT = "CodeBERT (fine-tuned)"


def load(path: Path):
    rows = [json.loads(l) for l in open(path)]
    return ([r["code"] for r in rows], np.array([r["label"] for r in rows]),
            [r["group"] for r in rows], [r.get("pair_id") for r in rows])


def pick_threshold(y, p) -> float:
    """Maximise balanced accuracy (Youden's J) on validation data; robust to class balance."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def make_pairs(y, pair_ids):
    """(buggy_index, fixed_index) for every pair_id that has both twins."""
    seen: dict = {}
    for i, (pid, lab) in enumerate(zip(pair_ids, y)):
        if pid is not None:
            seen.setdefault(pid, {})[int(lab)] = i
    return [(d[1], d[0]) for d in seen.values() if 1 in d and 0 in d]


def pair_accuracy(p, pairs) -> float | None:
    if not pairs:
        return None
    b = np.array([p[i] for i, _ in pairs])
    f = np.array([p[j] for _, j in pairs])
    return float(np.mean((b > f) + 0.5 * (b == f)))


def metrics_at(y, p, t) -> dict:
    pred = (p >= t).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(set(p)) > 1 else 0.5,
        "avg_precision": average_precision_score(y, p),
        "flag_rate": float(pred.mean()),
    }


def bootstrap_ci(y, p, groups, pairs, t, n=1000, seed=0):
    """95% CI resampling whole bugs (groups), so paired hunks stay together."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    idx_by_g = {g: [i for i, x in enumerate(groups) if x == g] for g in uniq}
    pairs_by_g: dict = {}
    for b, f in pairs:
        pairs_by_g.setdefault(groups[b], []).append((b, f))
    keys = ["accuracy", "f1", "auc", "pair_accuracy"]
    samples = {k: [] for k in keys}
    for _ in range(n):
        picked = [uniq[i] for i in rng.choice(len(uniq), len(uniq), replace=True)]
        idx = np.concatenate([idx_by_g[g] for g in picked])
        if len(set(y[idx])) < 2:
            continue
        m = metrics_at(y[idx], p[idx], t)
        for k in ("accuracy", "f1", "auc"):
            samples[k].append(m[k])
        pp = [pr for g in picked for pr in pairs_by_g.get(g, [])]
        pa = pair_accuracy(p, pp)
        if pa is not None:
            samples["pair_accuracy"].append(pa)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items() if v}


def shortcut_check(texts, y) -> dict:
    """AUC of trivial features. Values near 0.5 mean the model can't be cheating on size/comments."""
    feats = {
        "chars": [len(t) for t in texts],
        "lines": [t.count("\n") + 1 for t in texts],
        "comment_lines": [len(re.findall(r"^\s*(#|//|/\*|\*)", t, re.M)) for t in texts],
    }
    return {k: float(roc_auc_score(y, v)) for k, v in feats.items() if len(set(v)) > 1}


def score_codebert(model_dir: str, texts: list[str], batch=32, max_len=512) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist())
    return np.array(out)


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else (f"{v:.3f}" if v is not None else "n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip_model", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, _, _ = load(args.data / "train.jsonl")
    Xva, yva, _, _ = load(args.data / "val.jsonl")
    Xte, yte, gte, pte = load(args.data / "test.jsonl")
    pairs = make_pairs(yte, pte)
    if not pairs:
        print("note: no pair_id in the data; re-run data_prep.py to get pair accuracy")

    scores_val, scores_test = {}, {}

    # Baseline 1: flag everything (shows why F1 alone misleads on balanced data)
    scores_val[ALWAYS], scores_test[ALWAYS] = np.ones(len(yva)), np.ones(len(yte))

    # Baseline 2: TF-IDF (code tokens) + logistic regression
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", ngram_range=(1, 2),
                          min_df=2, max_features=200_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(vec.fit_transform(Xtr), ytr)
    scores_val["TF-IDF + LogReg"] = clf.predict_proba(vec.transform(Xva))[:, 1]
    scores_test["TF-IDF + LogReg"] = clf.predict_proba(vec.transform(Xte))[:, 1]

    if not args.skip_model:
        model_dir = args.model or str(args.out)
        scores_val[BERT] = score_codebert(model_dir, Xva)
        scores_test[BERT] = score_codebert(model_dir, Xte)

    results, chosen = {}, {}
    for name in scores_val:
        t = 0.5 if name == ALWAYS else pick_threshold(yva, scores_val[name])
        chosen[name] = t
        m = metrics_at(yte, scores_test[name], t)
        m["threshold"] = t
        m["pair_accuracy"] = pair_accuracy(scores_test[name], pairs)
        m["n_pairs"] = len(pairs)
        m["ci95"] = {} if name == ALWAYS else bootstrap_ci(yte, scores_test[name], gte, pairs, t)
        results[name] = m

    results["_shortcut_check_auc_test"] = shortcut_check(Xte, yte)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC (95% CI) | Pair acc. (95% CI) | Flag rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        if name.startswith("_"):
            continue
        ci = m["ci95"]
        lines.append(f"| {name} | {m['threshold']:.2f} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {fmt_ci(m['auc'], ci.get('auc'))} | "
                     f"{fmt_ci(m['pair_accuracy'], ci.get('pair_accuracy'))} | {m['flag_rate']:.2f} |")
    sc = results["_shortcut_check_auc_test"]
    table = "\n".join(lines) + "\n\nShortcut check (AUC of trivial features on test; 0.5 = no signal): " + \
        ", ".join(f"{k} {v:.2f}" for k, v in sc.items())
    (args.out / "results.md").write_text(table + "\n")
    print(table)

    if BERT in chosen:
        (args.out / "threshold.json").write_text(json.dumps({"threshold": chosen[BERT],
                                                              "chosen_on": "validation set (Youden's J)"}))
        print(f"\nthreshold.json written (t={chosen[BERT]:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for name, s in scores_test.items():
            if name == ALWAYS:
                continue
            fpr, tpr, _ = roc_curve(yte, s)
            pr, rc, _ = precision_recall_curve(yte, s)
            ax[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['auc']:.2f})")
            ax[1].plot(rc, pr, label=name)
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
        ax[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(args.out / "curves.png", dpi=150)
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
