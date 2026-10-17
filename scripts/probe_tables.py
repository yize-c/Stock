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


# The central bank's five largest banks: 彰銀 until October 2008 (ROC 97/10), 土銀 from
# November 2008. One sheet each; column 13 is the one-year time deposit, fixed rate (header
# rows 2-4 of every sheet).
FIVE_BANKS_BEFORE = ("台銀", "合庫", "一銀", "華銀", "彰銀")
FIVE_BANKS = ("台銀", "土銀", "合庫", "一銀", "華銀")
SWITCH_MONTH = "09711"
ONE_YEAR_FIXED = 13


def show_rates(out: Path) -> None:
    import calendar
    import io

    import openpyxl  # CI only; the file is xlsx despite its .xls name

    body = fetch(RATES)
    (out / "a13rate.xlsx").write_bytes(body)
    book = openpyxl.load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    print("=== sheets:", [s.title for s in book.worksheets])
    by_month: dict[str, list[float]] = {}
    for title in dict.fromkeys(FIVE_BANKS_BEFORE + FIVE_BANKS):
        rows = list(book[title].iter_rows(values_only=True))
        months = []
        for row in rows[5:]:
            month = "" if row[0] is None else str(row[0]).strip()
            value = row[ONE_YEAR_FIXED] if len(row) > ONE_YEAR_FIXED else None
            if not (month[:3].isdigit() and len(month) == 5 and value):
                continue
            months.append(month)
            members = FIVE_BANKS if month >= SWITCH_MONTH else FIVE_BANKS_BEFORE
            if int(month[:3]) >= 92 and title in members:
                by_month.setdefault(month, []).append(float(value))
        print(f"=== {title}: {len(months)} months, {min(months)} to {max(months)}")
    # Five-bank average per month, merged into periods of equal rate (percent, 4 decimals).
    periods: list[list] = []
    for month in sorted(by_month):
        values = by_month[month]
        average = round(sum(values) / len(values), 4)
        label = (int(month[:3]) + 1911, int(month[3:]))
        if periods and periods[-1][2] == average and periods[-1][3] == len(values):
            periods[-1][1] = label
        else:
            periods.append([label, label, average, len(values)])
    print(f"=== one-year fixed, five-bank average: {len(by_month)} months, {len(periods)} periods")
    for (y0, m0), (y1, m1), average, banks in periods:
        last = calendar.monthrange(y1, m1)[1]
        print(f"RATE {y0}-{m0:02d}-01 {y1}-{m1:02d}-{last:02d} {average} banks={banks}")


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
