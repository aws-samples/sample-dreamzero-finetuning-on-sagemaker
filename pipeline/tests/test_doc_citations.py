"""Every `file.py:NNN` citation in the docs must land on real code.

This exists because the failure it catches happened three times in one day. The
docs cite specific line numbers as evidence — SECURITY.md opens by promising each
row is reproducible — and any edit to a cited file silently shifts them. A stale
citation is worse than a missing one: it looks precise and points at something
unrelated, so a reader checking your work concludes the claim is wrong rather than
the pointer.

Deliberately a weak check. It asserts the cited line exists and is not blank; it
cannot know whether the code there still supports the claim. It catches drift,
not wrongness. Ranges are checked at their start line only.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# file.py:12 / file.sh:12-30 — the shapes the docs actually use.
CITATION = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|sh|json|yaml|yml)):(\d+)(?:-(\d+))?")

# Docs cite by bare filename as often as by path, so resolve against the
# directories that hold citable code.
SEARCH_DIRS = ("", "pipeline", "cdk", "docker", "evaluation",
               "cdk/dreamzero_pipeline", "pipeline/tests", "inference")


def _tracked_markdown():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md"],
                         capture_output=True, text=True)
    if out.returncode != 0:          # not a git checkout — nothing to assert
        return []
    return [REPO / p for p in out.stdout.split()]


def _resolve(cited):
    for d in SEARCH_DIRS:
        p = REPO / d / Path(cited).name if d else REPO / cited
        if p.is_file():
            return p
    return None


def test_doc_line_citations_resolve():
    md_files = _tracked_markdown()
    if not md_files:
        print("  skipped: not a git checkout")
        return

    checked, bad = 0, []
    for md in md_files:
        text = md.read_text(errors="replace")
        for m in CITATION.finditer(text):
            cited, start = m.group(1), int(m.group(2))
            target = _resolve(cited)
            if target is None:
                # A citation naming a file that does not exist at all is its own
                # bug, but keep this test focused on line drift: a doc may
                # legitimately reference an upstream file we do not vendor.
                continue
            lines = target.read_text(errors="replace").split("\n")
            checked += 1
            rel = md.relative_to(REPO)
            if start > len(lines):
                bad.append(f"{rel} cites {m.group(0)} but "
                           f"{target.relative_to(REPO)} has {len(lines)} lines")
            elif not lines[start - 1].strip():
                bad.append(f"{rel} cites {m.group(0)} which is a BLANK line in "
                           f"{target.relative_to(REPO)}")

    assert checked, "found no citations to check — has the citation format changed?"
    assert not bad, (
        f"{len(bad)} stale line citation(s) — the cited file was edited without "
        f"updating the doc:\n  " + "\n  ".join(bad))
    print(f"  {checked} line citations across {len(md_files)} markdown files")
