from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# share the diff helpers with the training code
sys.path.append(str(Path(__file__).resolve().parents[2] / "ml"))
from diff_utils import split_patch_into_hunks  # noqa: E402

from .config import get_settings
from .schemas import FileResult, HunkResult, PRRequest, ReviewResponse, SnippetRequest
from .services.github_client import GitHubError, fetch_pr_files
from .services.predictor import DefectPredictor
from .services.reviewer import build_chain, summarize

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    state["predictor"] = DefectPredictor(s.model_path, quantize=s.quantize)
    state["chain"] = build_chain(s.anthropic_api_key, s.llm_model) if s.anthropic_api_key else None
    yield
    state.clear()


app = FastAPI(title="AI Code Review & Bug Prediction", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


def _label(risk: float) -> str:
    return "defect-prone" if risk >= get_settings().risk_threshold else "likely clean"


@app.get("/health")
def health():
    return {"status": "ok", "model": state["predictor"].model_path, "llm": state["chain"] is not None}


@app.post("/api/analyze/snippet", response_model=ReviewResponse)
async def analyze_snippet(req: SnippetRequest):
    risk = state["predictor"].score([req.code])[0]
    hunk = HunkResult(filename=f"snippet.{req.language}", hunk_index=0, risk=risk,
                      label=_label(risk), snippet=req.code)
    summary = None
    if state["chain"]:
        summary = await summarize(state["chain"], [{"filename": hunk.filename, "risk": risk, "snippet": req.code}])
    return ReviewResponse(overall_risk=risk, files=[FileResult(filename=hunk.filename, max_risk=risk, hunks=[hunk])],
                          summary=summary, model_version=state["predictor"].model_path)


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
    risks = state["predictor"].score([c for _, _, c in flat])

    by_file: dict[str, list[HunkResult]] = {}
    for (fname, idx, chunk), risk in zip(flat, risks):
        by_file.setdefault(fname, []).append(
            HunkResult(filename=fname, hunk_index=idx, risk=risk, label=_label(risk), snippet=chunk[:2000]))
    file_results = [FileResult(filename=n, max_risk=max(h.risk for h in hs), hunks=hs) for n, hs in by_file.items()]
    file_results.sort(key=lambda f: f.max_risk, reverse=True)

    summary = None
    if state["chain"]:
        summary = await summarize(state["chain"], [h.model_dump() for f in file_results for h in f.hunks])
    return ReviewResponse(overall_risk=max(risks), files=file_results, summary=summary,
                          model_version=state["predictor"].model_path)
