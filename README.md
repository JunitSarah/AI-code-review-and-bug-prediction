# AI Code Review & Bug Prediction Tool

Fine-tuned **CodeBERT** predicts defect-prone code; a **FastAPI + LangChain** backend pulls PR diffs from the **GitHub API**, scores every hunk, and has an LLM write a review; a **React** UI shows the results.

```
React (Vercel) ──► FastAPI (Render) ──► GitHub API (PR diff)
                        │
                        ├─► CodeBERT classifier  → risk per hunk
                        └─► LangChain + LLM      → written review
```

## Repo layout
| Path | What |
|---|---|
| `ml/` | data prep from Defects4J / BugsInPy, CodeBERT fine-tuning, tests |
| `backend/` | FastAPI app, GitHub client, predictor, LangChain reviewer |
| `frontend/` | React + Vite UI |
| `render.yaml`, `frontend/vercel.json` | deploy configs |

## 1. Train the model
```bash
cd ml && pip install -r requirements.txt
git clone https://github.com/soarsmu/BugsInPy data/BugsInPy
git clone https://github.com/rjust/defects4j data/defects4j
python data_prep.py --bugsinpy data/BugsInPy --defects4j data/defects4j --out data/processed
python train.py --data data/processed --out ../artifacts/codebert-defect   # needs a GPU (Colab/Kaggle is fine)
cat ../artifacts/codebert-defect/test_metrics.json                          # <-- your resume numbers
```
Push the result to the Hugging Face Hub so Render can load it: `huggingface-cli upload yourname/codebert-defect artifacts/codebert-defect`.

## 2. Run the backend
```bash
cd backend && pip install -r requirements.txt && cp .env.example .env
cd .. && uvicorn backend.app.main:app --reload       # http://localhost:8000/docs
```

## 3. Run the frontend
```bash
cd frontend && npm install && cp .env.example .env && npm run dev
```

## 4. Deploy
- **Render**: connect the repo (uses `render.yaml`), set `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, `MODEL_PATH`, `CORS_ORIGINS`.
- **Vercel**: root directory `frontend`, env var `VITE_API_URL` = your Render URL.

## Honest limitations (put these in your paper)
- Defects4J/BugsInPy are bug-fix datasets, not labelled snippets. Labels come from patch hunks (buggy side = 1, fixed side = 0), so the model partly learns "what a fix looks like". Report this as a threat to validity.
- Splits are grouped by bug; for a stronger claim, also hold out whole projects.
- Render's free tier has 512 MB RAM: the backend uses int8 quantisation and a CPU-only torch wheel. If it still OOMs, use a paid instance.
