# Demo video script (target 3–4 minutes)

Record the **local** app (backend + frontend running) — free Render instances sleep and cold-start slowly, so don't
depend on them for the recording. Rehearse once; keep GitHub and API keys out of frame.

| Time | Show | Say |
|---|---|---|
| 0:00 | Title slide / README top | One sentence: the problem (reviewers have limited time) and what the tool does. |
| 0:20 | Architecture diagram | React → FastAPI → GitHub API → CodeBERT scorer → LLM reviewer. |
| 0:50 | Training: Colab results + `results.md` | Data = Defects4J + BugsInPy patches; show the table **including the baselines**; state plainly it is a modest, above-baseline signal. |
| 1:30 | UI, PR mode: paste a prepared public PR | Files sorted by risk, expand the top hunk, read the LLM review. |
| 2:30 | UI, snippet mode: off-by-one example | Show the score, then explain it is a *relative* risk. |
| 3:00 | `pytest` passing + `/health` + deployed URLs | Engineering quality: tests, quantised model, Render + Vercel. |
| 3:30 | Limitations slide | Threats to validity, in your own words. Then future work. |

Prepare in advance: 2–3 PRs that give interesting output, a snippet pair, and a fallback screen recording in case the
network fails during the take.
