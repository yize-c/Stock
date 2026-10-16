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
  ["UNSTABLE", VERDICTS.UNSTABLE],
  ["REGIME_SHIFT", VERDICTS.REGIME_SHIFT],
  ["STABLE", VERDICTS.STABLE],
  ["problem", "資料檢查未過"],
];
const WINDOWS = ["V1", "V3", "V2"];
const WINDOW_LABEL = { V1: "V1 2020後", V3: "V3 2020前", V2: "V2 全期" };
const LEGEND = [
  "比較 2020 年以後（V1）與以前（V3）；V2 全期只當參考。",
  "✅ 穩定：前後一致。🔄 結構轉變：2020 後換了樣子，但新樣子本身穩定。",
  "⚠️ 不穩定：一直在變，歷史數字參考價值低。❔ 資料不足：無法判斷。",
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
function price(close) {
  if (!close || close.value === null) return "";
  return close.value.toLocaleString("zh-TW", { maximumFractionDigits: 2 });
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
    default: return stock.status === "ok" && stock.overall.code === state.filter;
  }
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
    <div class="sub">${esc(stock.industry)}・收盤 ${price(stock.close)}</div></div>
    <div class="right">${chip(stock.overall.code)}<div class="small muted">V1 分數 ${pct(stock.composite_v1)}</div></div>
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
      <div>收盤 ${price(s.close)} <span class="small muted">（${esc(s.close.date)}，${esc(s.close.note)}）</span></div>
      <p>整體判定 ${chip(s.overall.code)}</p>
      <div class="aspects">${aspects}</div>
      ${drivers}
    </section>
    <section class="card">${composite}${s.composite_notes.length ? `<p class="small muted">${s.composite_notes.map(esc).join("<br>")}</p>` : ""}</section>
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
