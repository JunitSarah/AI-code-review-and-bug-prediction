"""Loads the fine-tuned CodeBERT once and scores code snippets."""
from __future__ import annotations

import threading


class DefectPredictor:
    def __init__(self, model_path: str, quantize: bool = True):
        # heavy imports live here so the API (and its tests) can be imported without torch installed
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        if quantize:
            from torch.ao.quantization import quantize_dynamic
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self._lock = threading.Lock()

    def score(self, snippets: list[str], batch_size: int = 8) -> list[float]:
        """Return P(defect-prone) for each snippet."""
        torch = self._torch
        out: list[float] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(snippets), batch_size):
                batch = snippets[i: i + batch_size]
                enc = self.tok(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
                probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1]
                out.extend(probs.tolist())
        return out
