"""Print FinMind's public documentation index (CI only; no token, read-only GET).

The development environment cannot reach finmind.github.io or finmindtrade.com, so a GitHub
runner reads llms.txt and prints it, plus the lines of linked pages that mention ETFs,
holdings or weights, so the ETF datasets can be chosen from the official description.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request

USER_AGENT = "stock-research/0.1 (personal research tool; docs probe)"
INDEXES = (
    "https://finmind.github.io/llms.txt",
    "https://finmind.github.io/llms-full.txt",
    "https://finmindtrade.com/llms.txt",
)
KEYWORDS = re.compile(r"ETF|Holding|Weight|成分|成份|持股|權重|主動", re.IGNORECASE)
MAX_LINES = 1500
SECTIONS = (
    "TaiwanStockInfo",
    "TaiwanStockActiveETFInfo",
    "TaiwanStockActiveETFHolding",
    "TaiwanStockActiveETFHoldingChange",
    "TaiwanStockMarketValueWeight",
    "TaiwanStockMarketValue",
    "TaiwanStockPrice",
    "TaiwanStockDividend",
)


def fetch(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # report and continue
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    pages: list[str] = []
    for url in INDEXES:
        status, text = fetch(url)
        lines = text.splitlines()
        print(f"=== {url} -> {status}, {len(text)} chars, {len(lines)} lines")
        if status != 200:
            print(text[:300])
            continue
        if url.endswith("llms-full.txt"):
            # Long: the full sections of the datasets that may hold ETF data, plus the
            # headings of every section so nothing ETF-related is missed.
            for line in lines:
                if line.startswith("#"):
                    print("HEADING " + line)
            for i, line in enumerate(lines):
                if line.startswith("### ") and any(f"### {n} " in line + " " for n in SECTIONS):
                    after = range(i + 1, len(lines))
                    end = next((j for j in after if lines[j].startswith("#")), i + 40)
                    print("\n".join(lines[i:end]))
            continue
        if url != INDEXES[0]:
            continue
        for line in lines[:MAX_LINES]:
            if KEYWORDS.search(line) or "Free" in line or "Tier" in line:
                print(line)
        pages += [u for u in re.findall(r"\((https?://[^)\s]+)\)", text) if KEYWORDS.search(u)]
    for url in dict.fromkeys(pages):
        status, text = fetch(url)
        print(f"=== page {url} -> {status}, {len(text)} chars")
        for line in text.splitlines():
            if KEYWORDS.search(line):
                print("  " + line.strip()[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
