import re

import httpx

PR_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")
CODE_EXTS = (".py", ".java", ".js", ".ts", ".go", ".c", ".cpp")


class GitHubError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_RE.search(url)
    if not m:
        raise GitHubError("Not a valid GitHub pull request URL", 400)
    return m.group(1), m.group(2), int(m.group(3))


async def fetch_pr_files(url: str, token: str | None, max_files: int) -> list[dict]:
    owner, repo, number = parse_pr_url(url)
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(api, headers=headers, params={"per_page": 100})
    if r.status_code == 404:
        raise GitHubError("PR not found (private repo? set GITHUB_TOKEN)", 404)
    if r.status_code in (403, 429):
        raise GitHubError("GitHub rate limit hit; set GITHUB_TOKEN", 429)
    if r.status_code != 200:
        raise GitHubError(f"GitHub API error {r.status_code}")
    files = [f for f in r.json() if f["filename"].endswith(CODE_EXTS) and f.get("patch")]
    return files[:max_files]
