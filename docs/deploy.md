# Deployment guide

Order matters: model → Hugging Face Hub → Render (API) → Vercel (UI) → fix CORS.

## 1. Publish the model to the Hugging Face Hub
Render can't read your laptop, so the model must live online.

1. Create a free account at huggingface.co → Settings → Access Tokens → new token with **write** access.
2. In Colab (or locally):
   ```bash
   pip install -q huggingface_hub
   huggingface-cli login            # paste the token
   python scripts/upload_to_hf.py artifacts/codebert-defect YOUR_USERNAME/codebert-defect
   ```
   Make sure `threshold.json` (from `evaluate.py`) is inside that folder before uploading.
3. Your `MODEL_PATH` is now `YOUR_USERNAME/codebert-defect`. Test locally first:
   ```powershell
   $env:MODEL_PATH="YOUR_USERNAME/codebert-defect"
   uvicorn backend.app.main:app
   ```
   If the repo is private, also set `HF_TOKEN` on the server.

## 2. Push the project to GitHub
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main && git remote add origin https://github.com/YOUR_USERNAME/code-review-ai.git
git push -u origin main
```
`.gitignore` already excludes `artifacts/`, `data/`, `node_modules/` and `.env`. **Never commit API keys.**

## 3. Backend on Render
1. render.com → New → **Blueprint** → pick your repo (it reads `render.yaml`), or New → Web Service and copy the commands.
2. Environment variables:

   | Key | Value |
   |---|---|
   | `MODEL_PATH` | `YOUR_USERNAME/codebert-defect` |
   | `GITHUB_TOKEN` | a GitHub token with no scopes (public-repo read) |
   | `ANTHROPIC_API_KEY` | optional; enables the written review |
   | `CORS_ORIGINS` | fill in after step 4 |
   | `QUANTIZE` | `1` |
   | `PYTHON_VERSION` | `3.11.9` |

3. Wait for the build, then open `https://YOUR-SERVICE.onrender.com/health`.

**Free-tier caveats (be honest about these in your demo):**
- 512 MB RAM. CodeBERT is ~500 MB in fp32; int8 quantisation helps but it may still run out of memory. Symptom: the service restarts while loading. Fix: a paid instance, or run the model on a Hugging Face Space and call it from the API.
- The free service sleeps after inactivity, so the first request can take 30–60 s. Open `/health` before a live demo.

## 4. Frontend on Vercel
1. vercel.com → Add New Project → pick the repo → **Root Directory: `frontend`**.
2. Environment variable: `VITE_API_URL` = `https://YOUR-SERVICE.onrender.com` (no trailing slash).
3. Deploy, copy the Vercel URL.

## 5. Close the loop (CORS)
Back on Render set `CORS_ORIGINS=https://YOUR-APP.vercel.app` and redeploy. Without this, the browser blocks every call.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| UI shows "Failed to fetch" | `VITE_API_URL` wrong, or CORS_ORIGINS missing the Vercel URL. |
| Render build fails on torch | Keep the CPU wheel index in `render.yaml`; check the Python version. |
| `/health` OK but PR analysis 429 | GitHub rate limit; set `GITHUB_TOKEN`. |
| Service restarts on startup | Out of memory (see caveats above). |
