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
    print("##### official closes, March 2024 (second batch)")
    status, _, text = fetch(
        "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20240301&stockNo=9921&response=json"
    )
    print(f"=== TWSE STOCK_DAY 9921 2024-03 -> {status}")
    try:
        for row in json.loads(text).get("data", [])[-12:]:
            print("  ", row[0], "close", row[6])
    except ValueError:
        print("  ", text[:200])
    status, _, text = fetch(
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
        "?code=5347&date=2024%2F03%2F01&response=json"
    )
    print(f"=== TPEx tradingStock 5347 2024-03 -> {status}")
    try:
        for row in json.loads(text)["tables"][0]["data"][-12:]:
            print("  ", row[0], "close", row[6])
    except (ValueError, KeyError, IndexError):
        print("  ", text[:200])


def opener() -> urllib.request.OpenerDirector:
    """Follows redirects and keeps cookies (some issuer sites redirect through a cookie)."""
    import http.cookiejar

    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )


def context(text: str, needle: str, width: int = 400, limit: int = 3) -> None:
    start = 0
    for _ in range(limit):
        at = text.find(needle, start)
        if at < 0:
            return
        print(f"  ...{needle} @{at}:", text[max(0, at - width // 2) : at + width // 2])
        start = at + 1


def etf_sources() -> None:
    print("##### Yuanta 0050")
    _, _, page = fetch("https://www.yuantaetfs.com/product/detail/0050/ratio")
    context(page, "台積電")
    for link in sorted(set(re.findall(r"""href=["']([^"']+)["']""", page))):
        if re.search(r"term|law|privacy|statement|declare|copyright|聲明|條款", link, re.I):
            print("  footer link:", link)
    scripts = sorted(set(re.findall(r"""src=["'](/_nuxt/[^"']+\.js)["']""", page)))
    apis: set[str] = set()
    for src in scripts:
        _, _, js = fetch("https://www.yuantaetfs.com" + src)
        apis |= set(re.findall(r"""["'`](/?api/[A-Za-z0-9_/.-]{2,80})""", js))
        apis |= set(
            re.findall(r"""["'`](https?://[a-z0-9.-]*yuanta[a-z0-9.-]*/[^"'`\s]{0,80})""", js)
        )
    for api in sorted(apis):
        print("  api:", api)

    print("##### TWSE ETF e-tianfu 00981A")
    _, _, js = fetch("https://www.twse.com.tw/rsrc/sites/etfortune/js/etf-product.js")
    for api in sorted(set(re.findall(r"""["'`](/zh/ETFortune/[A-Za-z]+)""", js))):
        print("  ajax:", api)

    print("##### Uni-President (ezmoney) 00981A")
    browser = opener()
    for url in (
        "https://www.ezmoney.com.tw/",
        "https://www.ezmoney.com.tw/ETF/Fund",
        "https://www.ezmoney.com.tw/robots.txt",
    ):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with browser.open(request, timeout=60) as response:
                body = response.read().decode("utf-8", errors="replace")
                print(f"=== {url} -> {response.status} final={response.geturl()} {len(body)} chars")
                title = re.search(r"<title>(.*?)</title>", body, re.S)
                print("  title:", " ".join(title.group(1).split())[:200] if title else None)
                found = set(
                    re.findall(r"""href=["']([^"']*(?:00981A|fundCode|FundCode)[^"']*)""", body)
                )
                for link in sorted(found)[:20]:
                    print("  link:", link)
                for link in sorted(set(re.findall(r"""href=["']([^"']+)["']""", body))):
                    if re.search(r"term|privacy|statement|copyright|聲明|條款", link, re.I):
                        print("  footer link:", link)
                if url.endswith("robots.txt"):
                    print(body[:500])
        except urllib.error.HTTPError as exc:
            print(f"=== {url} -> HTTP {exc.code} location={exc.headers.get('Location')}")
        except Exception as exc:
            print(f"=== {url} -> {type(exc).__name__}: {exc}")


def main() -> int:
    for part in (closes, etf_sources):
        try:
            part()
        except Exception as exc:  # report and continue with the next part
            print(f"##### {part.__name__} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
