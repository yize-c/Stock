"""ETF sources for the user's answers 1b, 2b, 3a (CI only).

Read-only requests without credentials, from a GitHub runner. Prints:
* TWSE open data "基金基本資料彙總表" (t187ap47_L): its columns and the rows of the active ETFs
  and the ETFs on the watchlist (issuer, index, fees if listed), answer 3a;
* for every issuer of those ETFs and for 臺灣指數公司: the terms-of-use pages linked from the
  home page and their sentences on automated access and reuse, plus links that look like
  holdings or basket (PCF) pages, answers 1b and 2b.
Only public pages; nothing behind a login or paywall; at most a few pages per site.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (stock-research personal research tool; source probe)"
WATCHED = ("0050", "006208", "00952", "00962", "00762", "00851", "00737")
REUSE = re.compile(
    r"轉載|重製|爬蟲|自動化|自動程式|擷取|機器人|下載|引用|散布|商業|授權|著作權|禁止"
)
TERMS = re.compile(
    r"使用條款|使用規範|網站聲明|使用聲明|著作權|版權|免責|條款|terms|copyright", re.I
)
INTEREST = re.compile(r"PCF|pcf|持股|成分|申購買回|投資組合|holding|portfolio|basket", re.I)
LINK = re.compile(r"""<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>""", re.S)
# Issuer name fragments -> home page (the open-data rows name the issuer).
HOMES = {
    "元大": "https://www.yuantaetfs.com/",
    "富邦": "https://websys.fsit.com.tw/FubonETF/",
    "國泰": "https://www.cathaysite.com.tw/",
    "凱基": "https://www.kgifund.com.tw/",
    "台新": "https://www.tsit.com.tw/",
    "野村": "https://www.nomurafunds.com.tw/",
    "統一": "https://www.ezmoney.com.tw/",
    "群益": "https://www.capitalfund.com.tw/",
    "復華": "https://www.fhtrust.com.tw/",
    "安聯": "https://etf.allianzgi.com.tw/",
    "第一金": "https://www.fsitc.com.tw/",
    "中國信託": "https://www.ctbcinvestments.com/",
    "中信": "https://www.ctbcinvestments.com/",
    "兆豐": "https://www.megafunds.com.tw/",
    "永豐": "https://www.sinopacfunds.com.tw/",
    "新光": "https://www.skit.com.tw/",
    "大華銀": "https://www.uobam.com.tw/",
    "聯博": "https://www.abfunds.com.tw/",
    "摩根": "https://am.jpmorgan.com/tw/zh/asset-management/adv/",
    "街口": "https://www.jkoam.com/",
    "貝萊德": "https://www.blackrock.com/tw",
    "保德信": "https://www.pgim.com.tw/",
    "華南永昌": "https://www.hnitc.com.tw/",
    "日盛": "https://www.jsfunds.com.tw/",
    "台灣指數公司": "https://taiwanindex.com.tw/",
}


def fetch(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return response.status, raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # report and continue
        return -1, f"{type(exc).__name__}: {exc}"


def text_of(html: str) -> str:
    html = re.sub(r"(?s)<(script|style).*?</\1>", " ", html)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def funds() -> set[str]:
    """The open-data fund table; returns the issuers named in the rows printed."""
    status, text = fetch("https://openapi.twse.com.tw/v1/opendata/t187ap47_L")
    print(f"=== t187ap47_L -> {status} {len(text)} chars")
    if status != 200:
        print("  ", text[:300])
        return set()
    rows = json.loads(text)
    print(f"  rows: {len(rows)}; columns: {list(rows[0]) if rows else []}")
    issuers = set()
    for row in rows:
        values = " ".join(str(v) for v in row.values())
        code = next((v for v in row.values() if re.fullmatch(r"00\d{2,4}[A-Z]?", str(v))), "")
        if code.endswith("A") or code in WATCHED:
            print("  row:", json.dumps(row, ensure_ascii=False)[:600])
            issuers |= {name for name in HOMES if name in values}
    return issuers


def site(name: str, home: str) -> None:
    status, html = fetch(home)
    print(f"\n=== {name}: {home} -> {status} {len(html)} chars")
    if status != 200:
        print("  ", html[:200])
        return
    links = [
        (urllib.parse.urljoin(home, href), " ".join(re.sub(r"<[^>]+>", "", label).split()))
        for href, label in LINK.findall(html)
    ]
    for url, label in links:
        if INTEREST.search(label) or INTEREST.search(url):
            print(f"  holdings-like link: {label[:30]} -> {url[:160]}")
    terms = []
    for url, label in links:
        if (TERMS.search(label) or TERMS.search(url)) and url not in terms:
            terms.append(url)
    print(f"  terms links: {terms[:6]}")
    for url in terms[:3]:
        status, page = fetch(url)
        print(f"  --- {url} -> {status} {len(page)} chars")
        shown = 0
        for sentence in re.split(r"(?<=[。；！？])", text_of(page)):
            if REUSE.search(sentence) and shown < 12:
                print("    |", sentence.strip()[:260])
                shown += 1


def main() -> None:
    issuers = funds()
    print(f"\nissuers found in the rows: {sorted(issuers)}")
    for name in sorted(issuers | {"元大", "台灣指數公司"}):
        site(name, HOMES[name])


if __name__ == "__main__":
    main()
