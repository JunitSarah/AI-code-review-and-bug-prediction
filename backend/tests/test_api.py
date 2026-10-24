"""API tests with a stub model and a mocked GitHub client (no torch, no network needed).

Run from the repo root:  python -m pytest backend/tests -q
"""
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services.github_client import GitHubError, parse_pr_url


class StubPredictor:
    model_path = "stub"

    def __init__(self, *a, **k):
        pass

    def score(self, snippets, batch_size=8):
        return [0.9 if "len(xs) + 1" in s else 0.1 for s in snippets]


BUGGY = "def f(xs):\n    for i in range(len(xs) + 1):\n        print(xs[i])"
CLEAN = "def f(xs):\n    return xs[-3:]"

PATCH = """@@ -1,3 +1,3 @@
 def f(xs):
-    for i in range(len(xs)):
+    for i in range(len(xs) + 1):
         print(xs[i])
@@ -20,2 +20,2 @@
 x = 1
-y = 2
+y = 3
"""


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "DefectPredictor", StubPredictor)
    monkeypatch.setattr(main.get_settings(), "llm_api_key", None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model"] == "stub" and r["llm"] is False


def test_snippet_ranks_buggy_above_clean(client):
    bug = client.post("/api/analyze/snippet", json={"code": BUGGY}).json()
    ok = client.post("/api/analyze/snippet", json={"code": CLEAN}).json()
    assert bug["overall_risk"] > ok["overall_risk"]
    assert bug["files"][0]["hunks"][0]["label"] == "defect-prone"
    assert ok["files"][0]["hunks"][0]["label"] == "likely clean"
    assert bug["summary"] is None  # no LLM key configured


def test_snippet_validation(client):
    assert client.post("/api/analyze/snippet", json={"code": ""}).status_code == 422
    assert client.post("/api/analyze/snippet", data="not json").status_code == 422


def test_pr_flow_and_ranking(client, monkeypatch):
    async def fake_fetch(url, token, max_files):
        return [{"filename": "a.py", "patch": PATCH}]

    monkeypatch.setattr(main, "fetch_pr_files", fake_fetch)
    r = client.post("/api/analyze/pr", json={"pr_url": "https://github.com/o/r/pull/1"})
    assert r.status_code == 200
    body = r.json()
    hunks = body["files"][0]["hunks"]
    assert len(hunks) == 2
    assert body["overall_risk"] == pytest.approx(0.9)
    top = max(hunks, key=lambda h: h["risk"])
    assert top["rank_percentile"] == 1.0 and top["hunk_index"] == 0


def test_pr_github_errors_are_mapped(client, monkeypatch):
    async def boom(url, token, max_files):
        raise GitHubError("PR not found", 404)

    monkeypatch.setattr(main, "fetch_pr_files", boom)
    r = client.post("/api/analyze/pr", json={"pr_url": "https://github.com/o/r/pull/1"})
    assert r.status_code == 404 and "not found" in r.json()["detail"].lower()


def test_pr_without_code_files(client, monkeypatch):
    async def empty(url, token, max_files):
        return []

    monkeypatch.setattr(main, "fetch_pr_files", empty)
    r = client.post("/api/analyze/pr", json={"pr_url": "https://github.com/o/r/pull/1"})
    assert r.status_code == 422


def test_parse_pr_url():
    assert parse_pr_url("https://github.com/octo/repo/pull/42") == ("octo", "repo", 42)
    with pytest.raises(GitHubError):
        parse_pr_url("https://example.com/not-a-pr")
