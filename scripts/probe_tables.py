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
    import io

    import openpyxl  # CI only; the file is xlsx despite its .xls name

    body = fetch(RATES)
    (out / "a13rate.xlsx").write_bytes(body)
    book = openpyxl.load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    print("=== sheets:", [s.title for s in book.worksheets])
    for sheet in book.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        title = str(rows[0][0]) if rows and rows[0] else ""
        if "平均" not in sheet.title and "平均" not in title:
            continue
        print(f"=== sheet {sheet.title!r}: {len(rows)} rows; title {title!r}")
        for i in range(min(5, len(rows))):
            print(i, "|".join("" if v is None else str(v) for v in rows[i]))
        # Year-month (ROC) and the one-year time deposit, fixed and floating (columns 13, 14),
        # from ROC year 92 (2003) on.
        for row in rows[5:]:
            month = "" if row[0] is None else str(row[0]).strip()
            if month[:3].isdigit() and int(month[:3]) >= 92:
                print(month, row[13], row[14], sep="|")


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
    for name, show in (("rates", show_rates),):
        try:
            show(out)
        except Exception as exc:  # report and continue with the next source
            print(f"=== {name} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
