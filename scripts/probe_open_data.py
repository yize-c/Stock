"""Official open data for the portfolio page and uniform active-ETF holdings (CI only).

Read-only GET requests without credentials, from a GitHub runner (the development environment
cannot reach these hosts). Prints:
* TWSE OpenAPI: the all-stocks daily close (STOCK_DAY_ALL) and every endpoint about ex-rights,
  ex-dividends or dividends, with a few rows each, and the open-data licence text;
* TPEx OpenAPI: the licence text (its daily quotes and ex-rights endpoints are already used);
* TWSE's website terms on reusing its data;
* MOPS and SITCA (the investment trust association): whether they publish active-ETF holdings
  for every issuer in one place.
Only public pages are read; nothing behind a login or paywall.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (stock-research personal research tool; source probe)"
LICENCE = re.compile(r"授權|開放資料|政府資料|轉載|引用|著作權|使用規範|條款|License|license")
HOLDINGS = re.compile(r"主動式|ETF|持股|投資組合|成分|申購買回|PCF")


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


def lines(text: str, pattern: re.Pattern[str], limit: int = 25) -> None:
    shown = 0
    for line in re.sub(r"<[^>]+>", "\n", text).splitlines():
        line = " ".join(line.split())
        if line and pattern.search(line) and len(line) < 220 and shown < limit:
            print("  |", line)
            shown += 1


def rows(url: str, codes: tuple[str, ...] = ()) -> None:
    status, _, text = fetch(url)
    print(f"=== {url} -> {status} {len(text)} chars")
    if status != 200:
        print("  ", text[:200])
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("  not JSON:", text[:200])
        return
    print(f"  rows: {len(data)}")
    for row in data[:2]:
        print("  sample:", json.dumps(row, ensure_ascii=False)[:400])
    for code in codes:
        hit = next((r for r in data if code in json.dumps(r, ensure_ascii=False)[:40]), None)
        print(f"  {code}:", json.dumps(hit, ensure_ascii=False)[:300] if hit else "not found")


def twse() -> None:
    rows(
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        ("2330", "0050", "00981A", "006208"),
    )
    status, _, text = fetch("https://openapi.twse.com.tw/v1/swagger.json")
    if status == 200:
        words = re.compile(r"除權|除息|股利|配息|權息")
        for path, methods in json.loads(text).get("paths", {}).items():
            for method in methods.values():
                label = f"{method.get('summary', '')} {method.get('description', '')}"
                if words.search(label):
                    print(f"  ex-rights endpoint: {path} | {' '.join(label.split())[:120]}")
                    rows(f"https://openapi.twse.com.tw/v1{path}")
    for url in ("https://openapi.twse.com.tw/", "https://www.twse.com.tw/zh/page/terms/use.html"):
        status, _, text = fetch(url)
        print(f"=== licence/terms {url} -> {status} {len(text)} chars")
        lines(text, LICENCE)


def tpex() -> None:
    status, _, text = fetch("https://www.tpex.org.tw/openapi/")
    print(f"=== licence https://www.tpex.org.tw/openapi/ -> {status} {len(text)} chars")
    lines(text, LICENCE)


def uniform_holdings() -> None:
    for url in (
        "https://mops.twse.com.tw/mops/web/t78sb39_new",
        "https://mopsov.twse.com.tw/mops/web/t78sb39_new",
        "https://www.sitca.org.tw/ROC/Industry/IN2629.aspx",
        "https://www.sitca.org.tw/ROC/Industry/IN2106.aspx",
        "https://www.sitca.org.tw/",
    ):
        status, kind, text = fetch(url)
        print(f"=== {url} -> {status} {kind} {len(text)} chars")
        title = re.search(r"<title>(.*?)</title>", text, re.S)
        if title:
            print("  title:", " ".join(title.group(1).split())[:160])
        lines(text, HOLDINGS, 20)


def terms_text() -> None:
    """Full text of TWSE's terms (reuse clauses) and the open-data platform's licence."""
    status, _, text = fetch("https://www.twse.com.tw/zh/page/terms/use.html")
    print(f"=== TWSE terms full text -> {status}")
    body = " ".join(
        re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<(script|style).*?</\1>", "", text)).split()
    )
    for sentence in re.split(r"(?<=[。；])", body):
        if re.search(r"轉載|重製|引用|散布|商業|授權|開放資料|資料", sentence):
            print("  |", sentence.strip()[:300])
    for query in ("每日收盤行情", "除權除息預告"):
        url = "https://data.gov.tw/api/v2/rest/dataset?" + urllib.parse.urlencode({"q": query})
        status, _, text = fetch(url)
        print(f"=== data.gov.tw search {query} -> {status} {len(text)} chars")
        print("  ", text[:1500])


def mops_active_etf() -> None:
    """Links and forms behind MOPS's ETF menu entries for domestic active ETFs."""
    status, _, text = fetch("https://mopsov.twse.com.tw/mops/web/t78sb39_new")
    print(f"=== MOPS ETF menu -> {status}")
    for match in re.finditer(r"主動式|持股", text):
        start = max(0, match.start() - 300)
        print("  ...", " ".join(text[start : match.end() + 300].split())[:600])


def main() -> None:
    terms_text()
    mops_active_etf()


if __name__ == "__main__":
    main()
