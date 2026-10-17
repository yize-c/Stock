"""Probe official Taiwanese data sources that the development environment cannot reach.

Run by .github/workflows/probe-sources.yml (GitHub's runners are not behind that network
policy). For each URL it prints the HTTP status, content type, size and the start of the body,
and saves the body under probe/ so fetchers can be written from real responses. Read-only:
plain GET requests, no credentials. Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "stock-research/0.1 (personal research tool; source probe)"
TIMEOUT_SECONDS = 60
PREVIEW_CHARS = 600

# name -> URL. Kept small: one request each, spaced by the runner's own latency.
PROBES = {
    # TWSE OpenAPI: listed-company basic data (includes the listing date).
    "twse_company_info": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
    # TWSE OpenAPI: all listed stocks' latest daily quotes.
    "twse_stock_day_all": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    # TWSE: one stock's daily quotes for a month (2412 around its 2008 capital reduction).
    "twse_stock_day_2412_200801": (
        "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
        "?date=20080101&stockNo=2412&response=json"
    ),
    # TWSE: capital-reduction resumption reference prices (pre-2011 coverage check).
    "twse_capital_reduction_2007_2010": (
        "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU"
        "?startDate=20070101&endDate=20101231&response=json"
    ),
    # TWSE: TAIEX total-return index for a month.
    "twse_total_return_index_202401": (
        "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U?date=20240101&response=json"
    ),
    # MOPS (old and new hosts).
    "mops_index": "https://mops.twse.com.tw/mops/web/index",
    "mopsov_index": "https://mopsov.twse.com.tw/mops/web/index",
    # Central bank: home page and the interest-rate statistics page.
    "cbc_home": "https://www.cbc.gov.tw/tw/mp-1.html",
    "cbc_rates": "https://www.cbc.gov.tw/tw/lp-370-1.html",
    # DGBAS (CPI) and the government open-data portal.
    "dgbas_home": "https://www.dgbas.gov.tw/",
    "dgbas_statdb": "https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?k=dgmain",
    "stat_gov_tw_home": "https://www.stat.gov.tw/",
    # DGBAS long-run CPI workbook (location as published on the DGBAS price-statistics page).
    "dgbas_cpi_workbook": "https://ws.dgbas.gov.tw/001/Upload/463/relfile/10315/2/cpispl.xls",
    "data_gov_tw_cpi_page": "https://data.gov.tw/dataset/6019",
}

# Pages whose links are listed when the link text or address contains one of the keywords,
# to find the data files behind them.
LINK_KEYWORDS = {
    "cbc_home": ("利率", "存款", "牌告"),
    "cbc_rates": ("利率", "存款", "牌告", ".xls", ".csv", ".ods"),
    "dgbas_statdb": ("物價", "CPI", "消費者"),
    "stat_gov_tw_home": ("物價", "CPI", "消費者"),
    "data_gov_tw_cpi_page": (".csv", ".xml", ".json", "下載"),
}


def probe(name: str, url: str, out: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body, status, content_type = exc.read(), exc.code, exc.headers.get("Content-Type", "")
    except Exception as exc:  # network errors: record them, keep probing the rest
        return {"name": name, "url": url, "error": f"{type(exc).__name__}: {exc}"}
    (out / f"{name}.body").write_bytes(body)
    text = body.decode("utf-8", errors="replace")
    return {
        "name": name,
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": len(body),
        "preview": " ".join(text[:PREVIEW_CHARS].split()),
    }


ANCHOR = re.compile(r"<a\s[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
TAG = re.compile(r"<[^>]+>")


def links(out: Path, name: str, keywords: tuple[str, ...]) -> None:
    path = out / f"{name}.body"
    if not path.exists():
        return
    text = path.read_bytes().decode("utf-8", errors="replace")
    print(f"=== links in {name}")
    seen = set()
    for href, label in ANCHOR.findall(text):
        label = " ".join(TAG.sub("", label).split())
        if any(k in href or k in label for k in keywords) and (href, label) not in seen:
            seen.add((href, label))
            print(f"{label[:60]} -> {href}")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "probe")
    out.mkdir(parents=True, exist_ok=True)
    results = [probe(name, url, out) for name, url in PROBES.items()]
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for r in results:
        print(f"=== {r['name']}")
        for key in ("url", "status", "content_type", "bytes", "error", "preview"):
            if key in r:
                print(f"{key}: {r[key]}")
    for name, keywords in LINK_KEYWORDS.items():
        links(out, name, keywords)
    # Listing dates for the stocks whose dates are configured by hand.
    info = out / "twse_company_info.body"
    if info.exists():
        try:
            rows = json.loads(info.read_text(encoding="utf-8"))
        except ValueError:
            rows = []
        wanted = {"2885", "2886", "2887", "2890", "2891", "2892", "3008", "3017", "3034",
                  "3037", "3045", "4904", "6505", "3231"}  # fmt: skip
        print("=== listing dates (TWSE t187ap03_L)")
        for row in rows:
            code = str(row.get("公司代號", "")).strip()
            if code in wanted:
                print(code, row.get("公司簡稱"), row.get("上市日期"), sep=" | ")
        if rows:
            print("columns:", list(rows[0].keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
