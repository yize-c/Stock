"""Print the structure of official data files the project may read later (CI only).

Run by .github/workflows/probe-sources.yml after probe_sources.py. Reads:
* the central bank's monthly history of the five largest banks' deposit and lending rates
  (a13rate.xls; openpyxl is installed in CI only, it is not a project dependency);
* DGBAS's monthly CPI XML (pr0101a1m.xml).
Read-only; prints sheet layouts and the first rows so parsers can be written from them.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

USER_AGENT = "stock-research/0.1 (personal research tool; source probe)"
RATES = "https://www.cbc.gov.tw/public/data/a13rate.xls"
CPI = "https://ws.dgbas.gov.tw/001/Upload/461/relfile/11525/230555/pr0101a1m.xml"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def show_rates(out: Path) -> None:
    import xlrd  # CI only

    body = fetch(RATES)
    (out / "a13rate.xls").write_bytes(body)
    book = xlrd.open_workbook(file_contents=body)
    for sheet in book.sheets():
        print(f"=== sheet {sheet.name!r}: {sheet.nrows} rows x {sheet.ncols} cols")
        for r in list(range(min(14, sheet.nrows))) + list(
            range(max(14, sheet.nrows - 6), sheet.nrows)
        ):
            print(r, [sheet.cell_value(r, c) for c in range(sheet.ncols)])


def show_cpi(out: Path) -> None:
    body = fetch(CPI)
    (out / "pr0101a1m.xml").write_bytes(body)
    text = body.decode("utf-8", errors="replace")
    print(f"=== CPI XML: {len(body)} bytes")
    print(text[:2500])
    print("...")
    print(text[-800:])


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "probe")
    out.mkdir(parents=True, exist_ok=True)
    for name, show in (("rates", show_rates), ("cpi", show_cpi)):
        try:
            show(out)
        except Exception as exc:  # report and continue with the next source
            print(f"=== {name} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
