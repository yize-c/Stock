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
    print("##### official closes, 5347 September 2024")
    status, _, text = fetch(
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
        "?code=5347&date=2024%2F09%2F01&response=json"
    )
    print(f"=== TPEx tradingStock 5347 2024-09 -> {status}")
    try:
        for row in json.loads(text)["tables"][0]["data"]:
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


def browse(browser: urllib.request.OpenerDirector, url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with browser.open(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"=== {url} -> {response.status} {len(body)} chars")
            return body
    except urllib.error.HTTPError as exc:
        print(f"=== {url} -> HTTP {exc.code}")
    except Exception as exc:
        print(f"=== {url} -> {type(exc).__name__}: {exc}")
    return ""


def terms(body: str, base: str, browser: urllib.request.OpenerDirector) -> None:
    """Follow footer links that look like terms of use and print their reuse clauses."""
    links = sorted(set(re.findall(r"""href=["']([^"'#]+)["'][^>]*>([^<]{0,30})<""", body)))
    for link, label in links:
        if re.search(r"條款|聲明|著作|版權|使用規範|term|copyright|disclaimer", link + label, re.I):
            url = link if link.startswith("http") else base + "/" + link.lstrip("/")
            print(f"  terms link: {label.strip()} -> {url}")
            page = browse(browser, url)
            text = " ".join(re.sub(r"<[^>]+>", " ", page).split())
            for clause in re.findall(
                r"[^。]{0,120}(?:轉載|重製|著作權|智慧財產|引用|散布)[^。]{0,160}。", text
            )[:6]:
                print("   |", clause)


def etf_sources() -> None:
    browser = opener()
    print("##### Yuanta 0050 holdings table and terms")
    page = browse(browser, "https://www.yuantaetfs.com/product/detail/0050/ratio")
    cells = re.findall(r"<span data-v-818b5120>([^<]{1,40})</span>", page)
    print("  first table cells:", cells[:24])
    home = browse(browser, "https://www.yuantaetfs.com/")
    terms(home + page, "https://www.yuantaetfs.com", browser)

    print("##### Uni-President (ezmoney) fund pages")
    found = None
    for code in (
        "36YTW",
        "37YTW",
        "44YTW",
        "46YTW",
        "47YTW",
        "49YTW",
        "50YTW",
        "61YTW",
        "63YTW",
        "64YTW",
        "65YTW",
    ):
        body = browse(browser, f"https://www.ezmoney.com.tw/ETF/Fund/Info?fundCode={code}")
        title = re.search(r"<title>(.*?)</title>", body, re.S)
        has = "00981A" in body
        name = " ".join(title.group(1).split())[:80] if title else None
        print(f"  {code}: 00981A={has} title={name}")
        if has and found is None:
            found = (code, body)
    if found:
        code, body = found
        print(f"##### 00981A page {code}")
        for api in sorted(
            set(
                re.findall(
                    r"""["'](/[A-Za-z]+/[A-Za-z/]*(?:Asset|Hold|Stock|Portfolio|PCF|Pcf|Weight)[A-Za-z/]*)["']""",
                    body,
                )
            )
        ):
            print("  api:", api)
        text = " ".join(re.sub(r"<[^>]+>", " ", body).split())
        for kw in ("持股", "投資組合", "申購買回", "PCF", "權重"):
            at = text.find(kw)
            if at >= 0:
                print(f"  {kw}:", text[max(0, at - 80) : at + 200])
        terms(body, "https://www.ezmoney.com.tw", browser)


def main() -> int:
    for part in (closes, etf_sources):
        try:
            part()
        except Exception as exc:  # report and continue with the next part
            print(f"##### {part.__name__} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
