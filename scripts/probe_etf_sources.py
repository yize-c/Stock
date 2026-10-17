"""Print what free public sources say about ETF holdings and a few official closes (CI only).

Read-only GET requests without credentials, from a GitHub runner (the development environment
cannot reach these hosts). Two purposes:
* official closes (TWSE, TPEx) for the days where FinMind and Yahoo disagree;
* where issuers and exchanges publish ETF holdings or creation/redemption baskets (PCF) for
  0050 (index) and 00981A (active), and what their terms say about reuse.
Only public pages are read; nothing behind a login or paywall.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

USER_AGENT = "Mozilla/5.0 (stock-research personal research tool; source probe)"
KEYWORDS = re.compile(
    r"持股|權重|比重|成分|申購買回|PCF|pcf|holding|Holding|weight|Weight|"
    r"轉載|重製|著作權|使用條款|授權|robots|Disallow|api/|API",
)


def fetch(url: str) -> tuple[int, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            kind = response.headers.get("Content-Type", "")
            return response.status, kind, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, "", ""
    except Exception as exc:  # report and continue
        return -1, "", f"{type(exc).__name__}: {exc}"


def show(url: str, *, max_lines: int = 60, raw: int = 0) -> str:
    status, kind, text = fetch(url)
    print(f"=== {url} -> {status} {kind} {len(text)} chars")
    if status != 200:
        print("  ", text[:300])
        return ""
    if raw:
        print(text[:raw])
        return text
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    if title:
        print("  title:", " ".join(title.group(1).split())[:200])
    for src in sorted(set(re.findall(r"""(?:src|href)=["']([^"']+\.js[^"']*)["']""", text)))[:20]:
        print("  script:", src)
    for api in sorted(set(re.findall(r"""["'](/?api/[^"'\s]{2,120})["']""", text)))[:40]:
        print("  api:", api)
    shown = 0
    for line in re.sub(r"<[^>]+>", "\n", text).splitlines():
        line = " ".join(line.split())
        if line and KEYWORDS.search(line) and shown < max_lines:
            print("  |", line[:300])
            shown += 1
    return text


def closes() -> None:
    print("##### official closes, March 2024")
    for code in ("2412", "1216", "2603", "2382"):
        status, _, text = fetch(
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
            f"?date=20240301&stockNo={code}&response=json"
        )
        print(f"=== TWSE STOCK_DAY {code} 2024-03 -> {status}")
        try:
            body = json.loads(text)
            print("  fields:", body.get("fields"))
            for row in body.get("data", [])[-8:]:
                print("  ", row)
        except ValueError:
            print("  ", text[:200])
    for url in (
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
        "?code=6488&date=2024%2F03%2F01&response=json",
        "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php"
        "?l=zh-tw&d=113/03&stkno=6488",
    ):
        show(url, raw=3000)


def etf_sources() -> None:
    print("##### ETF holdings sources")
    show("https://openapi.twse.com.tw/v1/swagger.json", raw=0, max_lines=0)
    status, _, text = fetch("https://openapi.twse.com.tw/v1/swagger.json")
    if status == 200:
        spec = json.loads(text)
        for path, ops in spec.get("paths", {}).items():
            summary = " ".join(str(op.get("summary", "")) for op in ops.values())
            if re.search(r"ETF|基金|受益|持股|成分", summary + path):
                print("  openapi:", path, summary[:120])
    for url in (
        "https://www.twse.com.tw/zh/ETFortune/etfInfo/0050",
        "https://www.twse.com.tw/zh/ETFortune/etfInfo/00981A",
        "https://www.yuantaetfs.com/robots.txt",
        "https://www.yuantaetfs.com/product/detail/0050/ratio",
        "https://www.yuantaetfs.com/product/detail/0050/pcf",
        "https://www.ezmoney.com.tw/robots.txt",
        "https://www.ezmoney.com.tw/ETF/Fund",
    ):
        show(url)
    # Find 00981A's fund page on the issuer site from the ETF list.
    _, _, page = fetch("https://www.ezmoney.com.tw/ETF/Fund")
    pattern = r"""href=["']([^"']*00981A[^"']*|[^"']*fundCode=[^"']+)"""
    for match in sorted(set(re.findall(pattern, page)))[:20]:
        print("  ezmoney link:", match)


def main() -> int:
    for part in (closes, etf_sources):
        try:
            part()
        except Exception as exc:  # report and continue with the next part
            print(f"##### {part.__name__} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
