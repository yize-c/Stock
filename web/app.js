// 台股穩定性 — static viewer for the daily export (data/index.json, data/stocks/<code>.json).
// Every number shown here was computed and formatted by the Python pipeline; this file only
// lays it out. No prediction, target price or buy/sell advice is produced or shown.
"use strict";

const VERDICTS = {
  UNSTABLE: "⚠️ 不穩定",
  REGIME_SHIFT: "🔄 結構轉變",
  STABLE: "✅ 穩定",
  INSUFFICIENT: "❔ 資料不足",
};
const FILTERS = [
  ["all", "全部"],
  ["fav", "★ 自選"],
  ["calm", "三面向都沒有⚠️"],
  ["unstable", "有面向⚠️不穩定"],
  ["problem", "資料檢查未過"],
];
const WINDOWS = ["V1", "V3", "V2"];
const WINDOW_LABEL = { V1: "V1 2020後", V3: "V3 2020前", V2: "V2 全期" };
const LEGEND = [
  "比較 2020 年以後（V1）與以前（V3）；V2 全期只當參考。",
  "✅ 穩定：前後一致。🔄 結構轉變：2020 後換了樣子，但新樣子本身穩定。",
  "⚠️ 不穩定：一直在變，歷史數字參考價值低。❔ 資料不足：無法判斷。",
  "整體判定取所有項目（含框架條件）中最差的一個，所以多半是不穩定；列表改看營收、獲利、風險三個面向。",
  "綜合分數只描述符合投資人原則的程度，不是買賣建議。",
];

const state = { index: null, filter: "all", query: "", favs: loadFavs() };
const view = document.getElementById("view");
const statusBox = document.getElementById("status");
const title = document.getElementById("title");
const back = document.getElementById("back");

function loadFavs() {
  try {
    return new Set(JSON.parse(localStorage.getItem("favs") || "[]"));
  } catch {
    return new Set();
  }
}
function saveFavs() {
  try {
    localStorage.setItem("favs", JSON.stringify([...state.favs]));
  } catch {
    /* private mode: favourites last for this visit only */
  }
}

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
function chip(code, label) {
  return `<span class="chip v-${esc(code)}">${esc(label || VERDICTS[code] || code)}</span>`;
}
function pct(value) {
  return value === null || value === undefined ? "N/A" : `${Math.round(value * 100)}%`;
}
function table(headers, rows) {
  const head = headers.map((h) => `<th>${esc(h)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((c, i) => `<td${i ? ' class="num"' : ""}>${c}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

async function getJSON(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function showStatus(index) {
  const generated = new Date(index.generated_at);
  const hours = (Date.now() - generated.getTime()) / 36e5;
  const when = generated.toLocaleString("zh-TW", { hour12: false });
  const stale = hours > index.stale_after_hours;
  statusBox.className = stale ? "status stale" : "status";
  statusBox.textContent = stale
    ? `⚠️ 資料可能過期：資料日期 ${index.as_of}，產生於 ${when}（已超過 ${index.stale_after_hours} 小時未更新）`
    : `資料日期 ${index.as_of}・更新於 ${when}`;
}

// --- list ----------------------------------------------------------------------------------

function matches(stock) {
  const q = state.query.trim().toLowerCase();
  if (q && !`${stock.code} ${stock.name} ${stock.industry || ""}`.toLowerCase().includes(q)) {
    return false;
  }
  switch (state.filter) {
    case "all": return true;
    case "fav": return state.favs.has(stock.code);
    case "problem": return stock.status !== "ok";
    case "calm":
      return stock.status === "ok" && aspectCodes(stock).every((c) => c !== "UNSTABLE");
    case "unstable":
      return stock.status === "ok" && aspectCodes(stock).includes("UNSTABLE");
    default: return true;
  }
}

// The overall verdict is the worst of dozens of items and is UNSTABLE for nearly every stock,
// so the list shows the three aspects (revenue, profitability, risk) instead.
const ASPECT_SHORT = { revenue: "營收", profitability: "獲利", risk: "風險" };
const ICON = { STABLE: "✅", REGIME_SHIFT: "🔄", UNSTABLE: "⚠️", INSUFFICIENT: "❔" };
function aspectCodes(stock) {
  return Object.values(stock.aspects).map((a) => a.code);
}
function aspectMini(stock) {
  return Object.entries(ASPECT_SHORT)
    .map(([key, label]) => {
      const code = stock.aspects[key].code;
      return `<span class="mini v-${esc(code)}" title="${esc(label)}：${esc(VERDICTS[code])}">${label}${ICON[code]}</span>`;
    })
    .join("");
}

function rowHTML(stock) {
  const fav = state.favs.has(stock.code);
  const star = `<button class="star" data-code="${esc(stock.code)}" aria-pressed="${fav}" aria-label="自選">${fav ? "★" : "☆"}</button>`;
  if (stock.status !== "ok") {
    return `<li><a class="row" href="#/s/${esc(stock.code)}">
      <div class="main"><div class="name">${esc(stock.code)} ${esc(stock.name)}</div>
      <div class="sub">${esc(stock.problem)}</div></div>
      <div class="right">${chip(stock.status, stock.status === "blocked" ? "檢查未過" : "錯誤")}</div></a>${star}</li>`;
  }
  return `<li><a class="row" href="#/s/${esc(stock.code)}">
    <div class="main"><div class="name">${esc(stock.code)} ${esc(stock.name)}</div>
    <div class="sub">${esc(stock.industry)}</div>
    <div class="minis">${aspectMini(stock)}</div></div>
    <div class="right"><div class="small muted">V1 分數</div><div class="big-score">${pct(stock.composite_v1)}</div>${stock.reference_label ? `<div class="small muted">回測：${esc(stock.reference_label)}</div>` : ""}</div>
    </a>${star}</li>`;
}

function renderList() {
  title.textContent = "台股穩定性";
  back.hidden = true;
  const stocks = state.index.stocks.filter(matches);
  const chips = FILTERS.map(
    ([key, label]) => `<button data-filter="${key}" aria-pressed="${state.filter === key}">${esc(label)}</button>`
  ).join("");
  view.innerHTML = `
    <div class="controls">
      <input type="search" id="q" placeholder="搜尋代號、名稱或產業" value="${esc(state.query)}" autocomplete="off">
      <div class="chips" role="group" aria-label="篩選">${chips}</div>
    </div>
    <ul class="list" id="list">${stocks.map(rowHTML).join("") || '<li class="muted">沒有符合的股票</li>'}</ul>
    <p class="disclaimer">${state.index.disclaimer.map(esc).join("<br>")}</p>`;
  const input = document.getElementById("q");
  input.addEventListener("input", () => {
    state.query = input.value;
    document.getElementById("list").innerHTML =
      state.index.stocks.filter(matches).map(rowHTML).join("") || '<li class="muted">沒有符合的股票</li>';
  });
}

view.addEventListener("click", (event) => {
  const filter = event.target.closest("[data-filter]");
  if (filter) {
    state.filter = filter.dataset.filter;
    renderList();
    return;
  }
  const star = event.target.closest(".star");
  if (star) {
    event.preventDefault();
    const code = star.dataset.code;
    if (state.favs.has(code)) state.favs.delete(code);
    else state.favs.add(code);
    saveFavs();
    star.setAttribute("aria-pressed", state.favs.has(code));
    star.textContent = state.favs.has(code) ? "★" : "☆";
  }
});

// --- detail --------------------------------------------------------------------------------

function section(titleText, body, open = false) {
  return `<details class="card"${open ? " open" : ""}><summary>${esc(titleText)}</summary>${body}</details>`;
}

function healthHTML(health) {
  const rows = health.checks.map((c) => [
    esc(c.name), c.passed ? "通過" : `<strong>失敗</strong>`, c.passed ? "" : esc(c.detail),
  ]);
  const limits = health.limitations.map((l) => `<li>${esc(l)}</li>`).join("");
  return table(["檢查", "結果", "說明"], rows) + (limits ? `<p class="small muted">資料來源限制：</p><ul class="small muted">${limits}</ul>` : "");
}

function renderProblem(stock) {
  view.innerHTML = `
    <div class="error"><strong>${stock.status === "blocked" ? "資料檢查未通過，未產生分析" : "分析失敗"}</strong><br>${esc(stock.problem)}</div>
    ${stock.health ? section("資料檢查明細", healthHTML(stock.health), true) : ""}
    <p class="disclaimer">${(stock.disclaimer || []).map(esc).join("<br>")}</p>`;
}

// Historical backtest reference (CLAUDE.md core principle 4): labels come from the program's
// fixed rule and are always shown with their statistics and caveats.
const LABEL_CLASS = { "買進參考": "STABLE", "持有參考": "INSUFFICIENT", "賣出參考": "UNSTABLE" };
function signedPct(value) {
  if (value === null || value === undefined) return "—";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}
function backtestHTML(ref) {
  if (!ref || !ref.available) {
    return `<p class="small muted">${esc(ref ? ref.reason : "沒有歷史回測資料")}</p>`;
  }
  const rows = Object.entries(ref.horizons).map(([months, st]) => [
    `${esc(months)} 個月`,
    signedPct(st.mean),
    st.n ? `${signedPct(st.ci_low)} ~ ${signedPct(st.ci_high)}` : "—",
    st.hit_rate === null ? "—" : `${Math.round(st.hit_rate * 100)}%`,
    String(st.n),
    signedPct(ref.overall[months] ? ref.overall[months].mean : null),
  ]);
  const label = ref.label
    ? `<span class="chip v-${LABEL_CLASS[ref.label]}">${esc(ref.label)}</span>`
    : '<span class="chip v-INSUFFICIENT">不標示</span>';
  return `<p>V1 綜合分數 ${pct(ref.score)}，屬於 ${pct(ref.bucket[0])}–${pct(ref.bucket[1])} 區間：${label}</p>
    <p class="small muted">${esc(ref.label_reason)}。${esc(ref.rule)}</p>
    ${table(["期間", "平均超額報酬", "信賴區間", "勝率", "樣本", "全部樣本平均"], rows)}
    <p class="small muted">回測自 ${esc(ref.start)}，資料至 ${esc(ref.data_as_of)}，產生於 ${esc(ref.generated_at)}。超額報酬 = 相對加權報酬指數。</p>
    <p class="small"><strong>${esc(ref.disclaimer)}</strong></p>`;
}

function renderDetail(s) {
  const aspects = Object.values(s.aspects)
    .map((a) => `<div>${esc(a.name)}<br>${chip(a.code)}</div>`)
    .join("");
  const drivers = s.drivers.length ? `<p class="small">主要變動項目：${s.drivers.slice(0, 6).map(esc).join("、")}</p>` : "";
  const composite = table(
    ["", ...WINDOWS.map((w) => WINDOW_LABEL[w])],
    [["綜合分數", ...WINDOWS.map((w) => pct(s.composite[w]))]]
  );
  const metrics = table(
    ["指標", "判定", ...WINDOWS],
    s.metrics.map((m) => [esc(m.name), chip(m.verdict.code), ...WINDOWS.map((w) => esc(m.values[w]))])
  );
  const fundamentals = table(
    ["項目", ...WINDOWS],
    s.fundamentals.map((f) => [esc(f.name), ...WINDOWS.map((w) => esc(f.values[w]))])
  );
  const frameworks = s.frameworks
    .map((f) => {
      const scores = WINDOWS.map((w) => `${w} ${pct(f.scores[w])}`).join("・");
      const items = table(
        ["條件", ...WINDOWS],
        f.items.map((i) => [esc(i.name), ...WINDOWS.map((w) => esc(i.outcomes[w]))])
      );
      const na = f.not_applicable ? `<p class="small muted">不適用：${esc(f.not_applicable)}</p>` : "";
      return `<div class="fw"><strong>${esc(f.name)}</strong> ${f.stability ? chip(f.stability.code) : ""}
        <div class="small muted">${scores}</div>${na}${items}</div>`;
    })
    .join("");
  const risk = WINDOWS.map((w) => {
    const r = s.risk[w];
    if (!r.available) return `<p><strong>${WINDOW_LABEL[w]}</strong>：資料不足（${esc(r.reason)}）</p>`;
    const note = r.sharpe_note ? `<p class="small muted">Sharpe：${esc(r.sharpe_note)}</p>` : "";
    return `<p><strong>${WINDOW_LABEL[w]}</strong><br><span class="small muted">${esc(r.label)}</span></p>${table(["指標", "數值"], r.rows.map(([k, v]) => [esc(k), esc(v)]))}${note}`;
  }).join("");
  const technicals = `<p class="small muted">${esc(s.technicals.note)}</p><ul>${s.technicals.descriptions.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`;
  view.innerHTML = `
    <section class="card">
      <div class="small muted">${esc(s.market)}・${esc(s.industry)}・資料日期 ${esc(s.as_of)}</div>
      <div class="big">${esc(s.code)} ${esc(s.name)}</div>
      <p>整體判定 ${chip(s.overall.code)}</p>
      <div class="aspects">${aspects}</div>
      ${drivers}
    </section>
    <section class="card">${composite}${s.composite_notes.length ? `<p class="small muted">${s.composite_notes.map(esc).join("<br>")}</p>` : ""}</section>
    ${section("歷史回測參考（非投資建議）", backtestHTML(s.backtest_reference), true)}
    ${section("這些判定是什麼意思？", `<ul class="legend">${LEGEND.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`)}
    ${section("穩定性指標（V1 vs V3）", metrics, true)}
    ${section("基本面與月營收（未加權）", fundamentals)}
    ${section("投資框架評分卡", frameworks)}
    ${section("風險指標（歷史統計值）", risk)}
    ${section("技術指標（僅描述）", technicals)}
    ${section("資料檢查", healthHTML(s.health))}
    ${section("完整報告", `<div class="md">${markdown(s.report_markdown)}</div>`)}
    <p class="disclaimer">${s.disclaimer.map(esc).join("<br>")}</p>`;
}

async function showStock(code) {
  back.hidden = false;
  title.textContent = code;
  view.innerHTML = '<p class="muted pad">載入中…</p>';
  try {
    const stock = await getJSON(`data/stocks/${encodeURIComponent(code)}.json`);
    title.textContent = `${stock.code} ${stock.name || ""}`;
    if (stock.status === "ok") renderDetail(stock);
    else renderProblem(stock);
    window.scrollTo(0, 0);
  } catch (error) {
    view.innerHTML = `<div class="error">無法載入 ${esc(code)}：${esc(error.message)}</div>`;
  }
}

// --- minimal Markdown (headings, tables, lists, bold) ---------------------------------------

function inline(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
}
function markdown(source) {
  const lines = source.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      const cells = (row) => row.replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const [head, , ...body] = block;
      out.push(table([], []).replace("<tr></tr>", `<tr>${cells(head).map((c) => `<th>${c}</th>`).join("")}</tr>`)
        .replace("<tbody></tbody>", `<tbody>${body.map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`));
      continue;
    }
    if (line.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(`<li>${inline(lines[i++].slice(2))}</li>`);
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    const heading = line.match(/^(#{1,4}) (.*)$/);
    if (heading) out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
    else if (line.trim() === "---") out.push("<hr>");
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    i += 1;
  }
  return out.join("");
}

// --- routing -------------------------------------------------------------------------------

async function route() {
  if (!state.index) {
    try {
      state.index = await getJSON("data/index.json");
      showStatus(state.index);
    } catch (error) {
      statusBox.className = "status stale";
      statusBox.textContent = "";
      view.innerHTML = `<div class="error">無法載入資料：${esc(error.message)}<br>請確認網路，或稍後再試（每日開市前更新）。</div>`;
      return;
    }
  }
  const match = location.hash.match(/^#\/s\/(\d{4,6})$/);
  if (match) showStock(match[1]);
  else renderList();
}

window.addEventListener("hashchange", route);
route();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {
    /* the app works without offline support */
  });
}
