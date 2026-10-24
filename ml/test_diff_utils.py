from diff_utils import parse_unified_diff, split_patch_into_hunks

PATCH = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,4 +1,4 @@
 def f(xs):
-    for i in range(len(xs) + 1):
+    for i in range(len(xs)):
         print(xs[i])
@@ -10,2 +10,2 @@
 x = 1
-y = None.foo
+y = 2
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""


def test_parse_pairs():
    hunks = parse_unified_diff(PATCH)
    assert len(hunks) == 2  # README.md filtered out
    assert "len(xs) + 1" in hunks[0].before and "len(xs) + 1" not in hunks[0].after
    assert "y = 2" in hunks[1].after


def test_split_patch():
    body = PATCH.split("+++ b/src/a.py\n")[1].split("diff --git")[0]
    chunks = split_patch_into_hunks(body)
    assert len(chunks) == 2 and "range(len(xs))" in chunks[0]
