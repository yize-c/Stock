"""Find free public sources of daily holdings for the domestic active ETFs (CI only).

Read-only GET requests without credentials, from a GitHub runner (the development environment
cannot reach most of these hosts). Prints:
* TWSE OpenAPI endpoints whose path or summary mentions ETFs, funds, holdings or baskets
  (open data, reusable under the government open-data licence);
* for each issuer's site: title, links that look like ETF holdings / PCF pages or APIs, and
  links to terms of use, so their reuse terms can be read before anything is built.
Only public pages are read; nothing behind a login or paywall.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (stock-research personal research tool; source probe)"
LINK = re.compile(r"""(?:href|src|data-url|url)\s*[=:]\s*["']([^"'#\s]{3,200})["']""")
INTEREST = re.compile(r"ETF|etf|PCF|pcf|持股|成分|申購買回|投資組合|holding|Holding|portfolio|api/")
TERMS = re.compile(
    r"使用條款|使用規範|網站聲明|著作權|版權|免責|條款|terms|Terms|copyright|Copyright"
)

ISSUERS = {
    "野村 00980A 00985A": "https://www.nomurafunds.com.tw/",
    "統一 00981A": "https://www.ezmoney.com.tw/",
    "群益 00982A 00992A": "https://www.capitalfund.com.tw/",
    "台新 00987A": "https://www.tsit.com.tw/",
    "復華 00991A": "https://www.fhtrust.com.tw/",
    "安聯 00993A": "https://etf.allianzgi.com.tw/",
    "第一金 00994A": "https://www.fsitc.com.tw/",
    "中信 00995A": "https://www.ctbcinvestments.com/",
    "兆豐 00996A": "https://www.megafunds.com.tw/",
}


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


def twse_openapi() -> None:
    status, _, text = fetch("https://openapi.twse.com.tw/v1/swagger.json")
    print(f"=== TWSE OpenAPI swagger -> {status} {len(text)} chars")
    if status != 200:
        print("  ", text[:300])
        return
    spec = json.loads(text)
    words = re.compile(r"ETF|ETN|基金|受益|持股|申購|買回|PCF|指數股票型", re.I)
    for path, methods in spec.get("paths", {}).items():
        for method in methods.values():
            label = f"{method.get('summary', '')} {method.get('description', '')}"
            if words.search(path) or words.search(label):
                print(f"  {path}  | {' '.join(label.split())[:160]}")


def issuer(name: str, url: str) -> None:
    status, kind, text = fetch(url)
    print(f"\n=== {name}: {url} -> {status} {kind} {len(text)} chars")
    if status != 200:
        print("  ", text[:300])
        return
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    if title:
        print("  title:", " ".join(title.group(1).split())[:160])
    links = sorted({urllib.parse.urljoin(url, m) for m in LINK.findall(text)})
    interesting = [x for x in links if INTEREST.search(x)]
    for link in interesting[:40]:
        print("  link:", link)
    plain = re.sub(r"<[^>]+>", "\n", text)
    for line in plain.splitlines():
        line = " ".join(line.split())
        if line and TERMS.search(line) and len(line) < 120:
            print("  terms text:", line)
    anchors = re.findall(r"""<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>""", text, re.S)
    for href, label in anchors:
        label = " ".join(re.sub(r"<[^>]+>", "", label).split())
        if TERMS.search(label):
            print(f"  terms link: {label[:40]} -> {urllib.parse.urljoin(url, href)}")


def main() -> None:
    twse_openapi()
    for name, url in ISSUERS.items():
        issuer(name, url)


if __name__ == "__main__":
    main()
