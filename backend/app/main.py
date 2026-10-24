from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# share the diff helpers with the training code
sys.path.append(str(Path(__file__).resolve().parents[2] / "ml"))
from diff_utils import split_patch_into_hunks  # noqa: E402

from .config import get_settings, load_threshold
from .schemas import FileResult, HunkResult, PRRequest, ReviewResponse, SnippetRequest
from .services.github_client import GitHubError, fetch_pr_files
from .services.predictor import DefectPredictor
from .services.reviewer import build_chain, summarize

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    state["predictor"] = DefectPredictor(s.model_path, quantize=s.quantize)
    state["threshold"] = load_threshold(s.model_path, s.risk_threshold_env)
    state["chain"] = None
    if s.llm_api_key:
        if not s.llm_model:
            raise RuntimeError("LLM_API_KEY is set but LLM_MODEL is empty; set LLM_MODEL (and LLM_BASE_URL "
                               "for LLM_PROVIDER=openai_compatible)")
        state["chain"] = build_chain(s.llm_provider, s.llm_api_key, s.llm_model, s.llm_base_url)
    yield
    state.clear()


app = FastAPI(title="AI Code Review & Bug Prediction", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


def _label(risk: float) -> str:
    return "defect-prone" if risk >= state["threshold"] else "likely clean"


async def _review(hunks: list[dict]) -> tuple[str | None, str | None]:
    """LLM review is optional: a failure (rate limit, bad key, timeout) must never cost the user the scores."""
    if not state["chain"]:
        return None, None
    try:
        return await summarize(state["chain"], hunks), None
    except Exception as e:  # noqa: BLE001 - any provider error should degrade gracefully
        return None, f"Written review unavailable ({type(e).__name__}); the risk scores above are still valid."


def _rank_pct(risks: list[float]) -> list[float]:
    """Percentile rank of each hunk within this request (1.0 = riskiest)."""
    order = sorted(range(len(risks)), key=lambda i: risks[i])
    pct = [0.0] * len(risks)
    for rank, i in enumerate(order):
        pct[i] = (rank + 1) / len(risks)
    return pct


@app.get("/health")
def health():
    return {"status": "ok", "model": state["predictor"].model_path,
            "threshold": state["threshold"], "llm": state["chain"] is not None}


@app.post("/api/analyze/snippet", response_model=ReviewResponse)
async def analyze_snippet(req: SnippetRequest):
    risk = (await run_in_threadpool(state["predictor"].score, [req.code]))[0]
    hunk = HunkResult(filename=f"snippet.{req.language}", hunk_index=0, risk=risk, rank_percentile=1.0,
                      label=_label(risk), snippet=req.code)
    summary, summary_error = await _review([{"filename": hunk.filename, "risk": risk, "snippet": req.code}])
    return ReviewResponse(overall_risk=risk, files=[FileResult(filename=hunk.filename, max_risk=risk, hunks=[hunk])],
                          summary=summary, summary_error=summary_error,
                          model_version=state["predictor"].model_path, threshold=state["threshold"])


@app.post("/api/analyze/pr", response_model=ReviewResponse)
async def analyze_pr(req: PRRequest):
    s = get_settings()
    try:
        files = await fetch_pr_files(req.pr_url, s.github_token, s.max_files)
    except GitHubError as e:
        raise HTTPException(e.status, str(e))
    if not files:
        raise HTTPException(422, "No supported code files with diffs in this PR")

    flat: list[tuple[str, int, str]] = []
    for f in files:
        for i, chunk in enumerate(split_patch_into_hunks(f["patch"])):
            flat.append((f["filename"], i, chunk))
    if not flat:
        raise HTTPException(422, "No reviewable hunks found in this PR")
    risks = await run_in_threadpool(state["predictor"].score, [c for _, _, c in flat])
    pcts = _rank_pct(risks)

    by_file: dict[str, list[HunkResult]] = {}
    for (fname, idx, chunk), risk, pct in zip(flat, risks, pcts):
        by_file.setdefault(fname, []).append(
            HunkResult(filename=fname, hunk_index=idx, risk=risk, rank_percentile=pct,
                       label=_label(risk), snippet=chunk[:2000]))
    file_results = [FileResult(filename=n, max_risk=max(h.risk for h in hs), hunks=hs) for n, hs in by_file.items()]
    file_results.sort(key=lambda f: f.max_risk, reverse=True)

    summary, summary_error = await _review([h.model_dump() for f in file_results for h in f.hunks])
    return ReviewResponse(overall_risk=max(risks), files=file_results, summary=summary,
                          summary_error=summary_error, model_version=state["predictor"].model_path,
                          threshold=state["threshold"])