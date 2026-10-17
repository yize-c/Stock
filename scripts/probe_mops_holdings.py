"""Can domestic active ETFs' quarterly holdings come from MOPS? (CI only, decision 8 = a.)

Read-only requests without credentials, from a GitHub runner (the development environment
cannot reach MOPS). Prints:
* the terms that govern MOPS and TWSE pages (automated access, reuse, open data);
* whether the holdings are published as open data instead (TWSE OpenAPI, data.gov.tw);
* the form behind MOPS's "基金每季持股明細" page for active ETFs (t78sb39_q4) and one
  answer for one fund, to learn the format.
Only public pages; nothing behind a login or paywall. One query per page, no loops.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (stock-research personal research tool; source probe)"
REUSE = re.compile(
    r"爬蟲|自動化|程式|機器人|大量|下載|轉載|重製|引用|散布|商業|授權|開放資料|著作權|禁止"
)
TERMS_LINK = re.compile(r"條款|聲明|著作權|使用規範|使用說明|隱私|免責")
HOLDINGS = re.compile(r"ETF|基金|持股|主動|交易所交易")
MOPS = "https://mopsov.twse.com.tw/mops/web/"


def fetch(url: str, data: dict[str, str] | None = None) -> tuple[int, str]:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
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


def sentences(text: str, pattern: re.Pattern[str], limit: int = 40) -> None:
    shown = 0
    for sentence in re.split(r"(?<=[。；！？])", text):
        if pattern.search(sentence) and shown < limit:
            print("  |", sentence.strip()[:300])
            shown += 1


def terms() -> None:
    """MOPS's own terms links (if any) and TWSE's terms, sentence by sentence."""
    for url in (MOPS + "index", "https://mops.twse.com.tw/mops/"):
        status, html = fetch(url)
        print(f"=== {url} -> {status} {len(html)} chars")
        links = re.findall(r'href="([^"]+)"[^>]*>([^<]{0,30})<', html)
        for href, label in links:
            if TERMS_LINK.search(label) or TERMS_LINK.search(href):
                print(f"  terms link: {label.strip()} -> {href}")
        sentences(text_of(html), REUSE, 15)
    for url in (
        "https://www.twse.com.tw/zh/page/terms/use.html",
        "https://mopsov.twse.com.tw/mops/web/t146sb10",  # MOPS site guide, if it exists
    ):
        status, html = fetch(url)
        print(f"=== terms {url} -> {status} {len(html)} chars")
        sentences(text_of(html), REUSE)


def open_data() -> None:
    """ETF holdings published as open data (allowed to reuse with the source stated)."""
    status, text = fetch("https://openapi.twse.com.tw/v1/swagger.json")
    print(f"=== TWSE OpenAPI swagger -> {status}")
    if status == 200:
        for path, methods in json.loads(text).get("paths", {}).items():
            for method in methods.values():
                label = " ".join(
                    f"{method.get('summary', '')} {method.get('description', '')}".split()
                )
                if HOLDINGS.search(label):
                    print(f"  endpoint: {path} | {label[:120]}")
    for query in ("主動式ETF", "ETF持股", "基金每季持股", "交易所交易基金 持股"):
        url = "https://data.gov.tw/api/v2/rest/dataset?" + urllib.parse.urlencode({"q": query})
        status, text = fetch(url)
        print(f"=== data.gov.tw {query} -> {status} {len(text)} chars")
        try:
            records = json.loads(text).get("result", [])
        except (json.JSONDecodeError, AttributeError):
            print("  ", text[:300])
            continue
        for record in records[:10] if isinstance(records, list) else []:
            title = record.get("title") or record.get("name")
            print(
                f"  dataset: {title} | {record.get('publisher', '')} | {record.get('license', '')}"
            )


def form_fields(html: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for tag in re.findall(r"<input[^>]*>", html, re.I):
        name = re.search(r'name="([^"]+)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            fields.setdefault(name.group(1), []).append(value.group(1) if value else "")
    for name, body in re.findall(
        r'<select[^>]*name="([^"]+)"[^>]*>(.*?)</select>', html, re.I | re.S
    ):
        fields[name] = re.findall(r'<option[^>]*value="([^"]*)"', body)[:40]
    return fields


def quarterly_holdings() -> None:
    """The quarterly holdings page for active ETFs and one answer."""
    status, html = fetch(MOPS + "t78sb39_q4")
    print(f"=== {MOPS}t78sb39_q4 -> {status} {len(html)} chars")
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    print("  title:", " ".join(title.group(1).split()) if title else None)
    print("  ajax targets:", sorted(set(re.findall(r"ajax_[a-z0-9_]+", html)))[:10])
    fields = form_fields(html)
    for name, values in fields.items():
        print(f"  field {name}: {values[:12]}")
    sentences(text_of(html), HOLDINGS, 20)
    # One query: the form's defaults plus the first fund listed, if any.
    data = {name: (values[0] if values else "") for name, values in fields.items()}
    data.update({"encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1"})
    status, answer = fetch(MOPS + "ajax_t78sb39_q4", data)
    sent = json.dumps(data, ensure_ascii=False)[:300]
    print(f"=== ajax_t78sb39_q4 {sent} -> {status} {len(answer)}")
    print("  ", text_of(answer)[:2500])
    headers = re.findall(r"<th[^>]*>(.*?)</th>", answer, re.S)
    print("  headers:", [text_of(h) for h in headers][:30])


def main() -> None:
    terms()
    open_data()
    quarterly_holdings()


if __name__ == "__main__":
    main()
