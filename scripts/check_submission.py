"""Validate submission assets: summary length, screenshots, and artifact safety."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
FORBIDDEN = ("private_key", "access_token", "vertax_creds", "/Users/", "/home/")
SCREENSHOTS = (
    "01-overview.png",
    "02-research-journey.png",
    "03-experiment-lineage.png",
    "04-monitoring.png",
)


def main() -> int:
    failures: list[str] = []

    summary = (ROOT / "SUBMISSION.md").read_text(encoding="utf-8")
    body = summary.split("\n", 1)[1]
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-/]*", body))
    if words != 100:
        failures.append(f"submission summary must be exactly 100 words, found {words}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in SCREENSHOTS:
        image = ROOT / "docs" / "images" / name
        if not image.is_file():
            failures.append(f"missing screenshot {name}")
        elif image.stat().st_size < 50_000:
            failures.append(f"screenshot likely unrendered: {name}")
        if name not in readme:
            failures.append(f"README does not reference {name}")

    for link in re.findall(r"\]\((?!http)([^)#]+)\)", readme):
        if not (ROOT / link).exists():
            failures.append(f"README links to missing path: {link}")

    bundle = ROOT / "app" / "data" / "demo_bundle.json"
    if not bundle.is_file():
        failures.append("presentation bundle is missing")
    else:
        text = bundle.read_text(encoding="utf-8")
        if bundle.stat().st_size > 2_000_000:
            failures.append("presentation bundle is too large for the hosted tier")
        for marker in FORBIDDEN:
            if marker.lower() in text.lower():
                failures.append(f"bundle contains forbidden marker: {marker}")

    for message in failures:
        print(f"FAIL {message}")
    if failures:
        return 1
    print(f"submission assets valid ({words}-word summary, {len(SCREENSHOTS)} screenshots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
