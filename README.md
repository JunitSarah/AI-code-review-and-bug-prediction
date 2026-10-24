# AI Code Review & Bug Prediction Tool

A fine-tuned **CodeBERT** model scores code changes for defect risk, prioritizing which pull request hunks or code snippets need the closest human attention. A **FastAPI + LangChain** backend scores every change and triggers an LLM review (Gemini, Claude, Groq, or OpenAI), while a **React** UI displays files ranked by relative risk.

```
React UI ──► FastAPI Backend ──► GitHub API (PR diff) OR Raw Snippet
                   │
                   ├─► CodeBERT classifier  → Probabilistic defect risk + PR percentile rank
                   └─► LangChain + LLM      → Written review & fix recommendations (optional)
```

> **Positioning:** This tool is a **triage assistant**. It helps reviewers decide *where to look first* in large PRs; it does not claim to prove that code is correct or catch every bug. Scores indicate relative risk.

## Key Features
- **PR & Snippet Analysis:** Analyze entire GitHub pull requests or paste Python and Java snippets directly.
- **Relative Risk Ranking:** Calculates percentile ranks within each PR to spotlight the top 20% riskiest hunks.
- **Provider-Agnostic LLM Reviews:** Works with Anthropic Claude, Google Gemini, Groq, OpenAI, or local Ollama instances.
- **Data-Calibrated Thresholds:** Operating threshold is dynamically calibrated via Youden's J statistic on validation data.

## Model Performance (Test Set)
Fine-tuned on Defects4J (Java) and BugsInPy (Python) bug-fix patches:

| Metric | Score | Note |
|---|---|---|
| **ROC AUC** | **0.673** | Threshold-free discriminative ability |
| **Recall** | **70.3%** | Captures 7 out of 10 defect-prone hunks |
| **Accuracy** | **59.7%** | Modest signal on balanced bug-fix dataset |
| **Precision** | **58.0%** | Defect-prone flag precision |
| **Pair Accuracy** | **~65%** | Does the buggy hunk outscore its fixed twin? |

## Repo Layout
| Path | What |
|---|---|
| `ml/` | Data prep (Defects4J + BugsInPy), CodeBERT fine-tuning, evaluation with baselines, tests |
| `backend/` | FastAPI app, GitHub client, predictor, LangChain reviewer, API tests |
| `frontend/` | React + Vite UI |
| `colab_train.ipynb` | One-click Colab notebook for GPU training and evaluation |
| `docs/` | `deploy.md`, `paper_outline.md`, `demo_script.md` |
| `scripts/` | `upload_to_hf.py` (publish to HF Hub), `pr_benchmark.py` (benchmark on real-world PRs) |
| `render.yaml`, `frontend/vercel.json` | Cloud deployment configurations |

## Quick Start

### 1. Backend
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env                  # configure your LLM_API_KEY
uvicorn backend.app.main:app --reload                 # http://localhost:8000/docs
```
*Without a trained model, set `MODEL_PATH=microsoft/codebert-base` to check the plumbing (scores are meaningless until fine-tuned).*

### 2. Frontend
In a second terminal:
```bash
cd frontend
npm install
npm run dev                                           # http://localhost:5173
```

## Train and Evaluate
Train with a free GPU in `colab_train.ipynb`, or locally:
```bash
cd ml && pip install -r requirements.txt
git clone --depth 1 https://github.com/soarsmu/BugsInPy data/BugsInPy
git clone --depth 1 https://github.com/rjust/defects4j data/defects4j
python data_prep.py --bugsinpy data/BugsInPy --defects4j data/defects4j --out data/processed
python train.py    --data data/processed --out ../artifacts/codebert-defect
python evaluate.py --data data/processed --model ../artifacts/codebert-defect --out ../artifacts/codebert-defect
```
`evaluate.py` reports an always-"buggy" baseline and a TF-IDF + logistic-regression baseline next to CodeBERT, **pair accuracy** (does the buggy twin outscore its fixed twin?), a length/comment shortcut check, bootstrap 95% CIs (resampling whole bugs), and writes `threshold.json` (chosen on the validation set) which the API uses automatically.

## Tests
```bash
pytest
```
API tests use a stub model and a mocked GitHub client, so they need neither PyTorch nor internet access.

## Deploy
See [`docs/deploy.md`](docs/deploy.md): Hugging Face Hub (model) → Render (API) → Vercel (UI) → set CORS.

## Known Limitations
- Labels come from bug-fix patches (buggy side = 1, fixed side = 0), so the model may partly learn what a *fix* looks like. Splits are grouped by bug, not by project.
- Small dataset (thousands of hunks); modest accuracy is expected for this task. Always compare against the baselines.
- Python and Java only; LLM reviews can hallucinate, so keep a human in the loop.
- Free-tier hosting is memory-constrained (see `docs/deploy.md`).
