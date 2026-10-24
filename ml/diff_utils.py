"""Turn unified-diff patches (Defects4J / BugsInPy bug-fix patches) into labelled training pairs.

Idea: a bug-fix patch tells us what the code looked like BEFORE (buggy) and AFTER (fixed).
For every hunk we emit two samples:
    buggy side  (context + removed lines) -> label 1
    fixed side  (context + added lines)   -> label 0
This is a standard, if imperfect, way to get snippet-level labels from bug-fix datasets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


@dataclass
class Hunk:
    file: str
    before: str  # buggy version of the hunk
    after: str   # fixed version of the hunk


def parse_unified_diff(patch_text: str, exts: tuple[str, ...] = (".py", ".java")) -> list[Hunk]:
    hunks: list[Hunk] = []
    cur_file = None
    before: list[str] = []
    after: list[str] = []
    in_hunk = False

    def flush():
        nonlocal before, after, in_hunk
        if in_hunk and cur_file and (before or after):
            hunks.append(Hunk(cur_file, "\n".join(before), "\n".join(after)))
        before, after, in_hunk = [], [], False

    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            flush()
            cur_file = None
        elif line.startswith("+++ ") and not in_hunk:
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            cur_file = path if path.endswith(exts) else None
        elif line.startswith("--- ") and not in_hunk:
            continue
        elif HUNK_RE.match(line):
            flush()
            in_hunk = True
        elif in_hunk:
            if line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            elif line.startswith(" "):
                before.append(line[1:])
                after.append(line[1:])
            # "\ No newline at end of file" and others are ignored
    flush()
    return hunks


def added_code_from_patch(patch: str) -> str:
    """Rebuild the 'after' side of a single-file GitHub PR patch (context + added lines)."""
    out = []
    for line in patch.splitlines():
        if line.startswith("@@") or line.startswith("\\"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
        elif line.startswith(" "):
            out.append(line[1:])
    return "\n".join(out)


def split_patch_into_hunks(patch: str) -> list[str]:
    """Split a GitHub PR file patch into per-hunk 'after' code strings."""
    chunks, cur = [], []
    for line in patch.splitlines():
        if line.startswith("@@"):
            if cur:
                chunks.append("\n".join(cur))
            cur = []
        elif line.startswith("+") or line.startswith(" "):
            cur.append(line[1:])
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if c.strip()]
