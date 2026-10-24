"""Fine-tune CodeBERT for binary defect prediction.

  python train.py --data ./data/processed --out ./artifacts/codebert-defect --epochs 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, EarlyStoppingCallback)

BASE = "microsoft/codebert-base"


def compute_metrics(pred):
    logits, labels = pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    preds = probs.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    out = {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}
    if len(set(labels)) > 1:
        out["auc"] = roc_auc_score(labels, probs[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/codebert-defect"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("json", data_files={s: str(args.data / f"{s}.jsonl") for s in ("train", "val", "test")})
    tok = AutoTokenizer.from_pretrained(BASE)

    def encode(batch):
        return tok(batch["code"], truncation=True, max_length=args.max_len)

    ds = ds.map(encode, batched=True, remove_columns=["code", "group", "file"]).rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      tokenizer=tok, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
