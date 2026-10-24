"""Measure real-world behaviour of the running API on a list of public PRs.

Produces the numbers a project report needs: how many PRs/repos were tested, how many hunks,
response-time median / p95, how often the API failed, and the risk distribution.

  # urls.txt: one GitHub PR URL per line (use 20-30 PRs from different repos and languages)
  python scripts/pr_benchmark.py urls.txt --api http://127.0.0.1:8000 --out results/pr_benchmark.csv

Set GITHUB_TOKEN on the server first, or you will hit GitHub's 60 requests/hour limit.
"""
import argparse
import csv
import statistics as st
import time
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("urls", type=Path)
ap.add_argument("--api", default="http://127.0.0.1:8000")
ap.add_argument("--out", type=Path, default=Path("results/pr_benchmark.csv"))
ap.add_argument("--timeout", type=float, default=120)
args = ap.parse_args()

urls = [u.strip() for u in args.urls.read_text().splitlines() if u.strip() and not u.startswith("#")]
rows = []
with httpx.Client(timeout=args.timeout) as client:
    for u in urls:
        t0 = time.perf_counter()
        try:
            r = client.post(f"{args.api}/api/analyze/pr", json={"pr_url": u})
            dt = time.perf_counter() - t0
            if r.status_code == 200:
                b = r.json()
                hunks = [h for f in b["files"] for h in f["hunks"]]
                rows.append({"pr_url": u, "status": 200, "seconds": round(dt, 2), "files": len(b["files"]),
                             "hunks": len(hunks), "max_risk": round(b["overall_risk"], 3),
                             "flagged_hunks": sum(h["label"] == "defect-prone" for h in hunks)})
            else:
                rows.append({"pr_url": u, "status": r.status_code, "seconds": round(dt, 2), "error": r.text[:120]})
        except httpx.HTTPError as e:
            rows.append({"pr_url": u, "status": "error", "seconds": round(time.perf_counter() - t0, 2),
                         "error": type(e).__name__})
        print(rows[-1])

args.out.parent.mkdir(parents=True, exist_ok=True)
fields = ["pr_url", "status", "seconds", "files", "hunks", "max_risk", "flagged_hunks", "error"]
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

ok = [r for r in rows if r["status"] == 200]
repos = {"/".join(r["pr_url"].split("/")[3:5]) for r in rows}
print(f"\nPRs tried: {len(rows)} across {len(repos)} repos | succeeded: {len(ok)} | failed: {len(rows) - len(ok)}")
if ok:
    secs = sorted(r["seconds"] for r in ok)
    p95 = secs[min(len(secs) - 1, int(0.95 * len(secs)))]
    print(f"response time: median {st.median(secs):.1f}s, p95 {p95:.1f}s, max {secs[-1]:.1f}s")
    print(f"hunks scored: {sum(r['hunks'] for r in ok)}, flagged: {sum(r['flagged_hunks'] for r in ok)}")
