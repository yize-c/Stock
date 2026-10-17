// 台股穩定性 — static viewer for the daily export (data/index.json, data/stocks/<code>.json).
// Every number shown here was computed and formatted by the Python pipeline (including each
// capital tier's score); this file only lays it out and switches between precomputed tiers.
// No prediction, target price or buy/sell advice is produced or shown.
"use strict";

const VERDICTS = { STABLE: "穩定", REGIME_SHIFT: "結構轉變", UNSTABLE: "不穩定", INSUFFICIENT: "資料不足" };
const MAIN_FILTERS = [["all", "全部"], ["fav", "自選"], ["calm", "無不穩定"]];
const MORE_FILTERS = [["unstable", "有不穩定"], ["problem", "檢查未過"]];
const WINDOWS = ["V1", "V3", "V2"];
const WINDOW_LABEL = { V1: "V1 2020後", V3: "V3 2020前", V2: "V2 全期" };
const ASPECTS = { revenue: "營收", profitability: "獲利", risk: "風險" };
const LEGEND = [
  "比較 2020 年以後（V1）與以前（V3）；V2 全期只當參考。",
  "穩定：前後一致。結構轉變：2020 後換了樣子，但新樣子本身穩定。",
  "不穩定：一直在變，歷史數字參考價值低。資料不足：無法判斷。",
  "整體判定取所有項目（含框架條件）中最差的一個，所以多半是不穩定；列表改看營收、獲利、風險三個面向。",
  "分數只描述符合條件的程度與歷史回測平均，不是買賣建議。",
];
const CHEVRON = '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const state = { index: null, filter: "all", query: "", more: false, favs: loadSet("favs"), capital: null, listScroll: 0, current: null, detail: null };
const listView = document.getElementById("listView");
const detailPane = document.getElementById("detailPane");
const staleBox = document.getElementById("stale");
const dateBox = document.getElementById("date");
const wide = window.matchMedia("(min-width: 1024px)");

// --- storage (per-viewer conveniences only; the page works without it) ----------------------

function loadSet(key) {
  try {
    return new Set(JSON.parse(localStorage.getItem(key) || "[]"));
  } catch {
    return new Set();
  }
}
function store(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode: remembered for this visit only */
  }
}
function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

// --- helpers -------------------------------------------------------------------------------

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
function verdict(code, label) {
  return `<span class="verdict t-${esc(code)}"><i class="dot v-${esc(code)}"></i>${esc(label || VERDICTS[code] || code)}</span>`;
}
function pct(value) {
  return value === null || value === undefined ? "N/A" : `${Math.round(value * 100)}%`;
}
function signedPct(value) {
  if (value === null || value === undefined) return "—";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}
// headers become data-labels, so phones can show each row as a stacked card.
function table(headers, rows) {
  const head = headers.length ? `<thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>` : "";
  const body = rows
    .map((r) => `<tr>${r.map((c, i) => `<td data-label="${esc(headers[i] || "")}">${c === "" ? "" : `<span>${c}</span>`}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="tablewrap"><table class="t">${head}<tbody>${body}</tbody></table></div>`;
}
function section(titleText, body) {
  return `<details class="sec"><summary><span>${esc(titleText)}</span>${CHEVRON}</summary><div class="body">${body}</div></details>`;
}
function disclaimer(lines) {
  return `<footer class="disclaimer">${(lines || []).map((l) => `<p>${esc(l)}</p>`).join("")}</footer>`;
}
async function getJSON(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

// --- status: date next to the title, a warning bar only when the data is stale --------------

function showStatus(index) {
  const generated = new Date(index.generated_at);
  const hours = (Date.now() - generated.getTime()) / 36e5;
  const when = generated.toLocaleString("zh-TW", { hour12: false });
  if (hours > index.stale_after_hours) {
    staleBox.hidden = false;
    staleBox.textContent = `資料可能過期：資料日期 ${index.as_of}，產生於 ${when}（已超過 ${index.stale_after_hours} 小時未更新）`;
    dateBox.textContent = "";
  } else {
    staleBox.hidden = true;
    dateBox.textContent = `資料 ${index.as_of}`;
    dateBox.title = `更新於 ${when}`;
  }
}

// --- list ----------------------------------------------------------------------------------

function aspectCodes(stock) {
  return Object.values(stock.aspects).map((a) => a.code);
}
function matches(stock) {
  const q = state.query.trim().toLowerCase();
  if (q && !`${stock.code} ${stock.name} ${stock.industry || ""}`.toLowerCase().includes(q)) return false;
  switch (state.filter) {
    case "fav": return state.favs.has(stock.code);
    case "problem": return stock.status !== "ok";
    case "calm": return stock.status === "ok" && !aspectCodes(stock).includes("UNSTABLE");
    case "unstable": return stock.status === "ok" && aspectCodes(stock).includes("UNSTABLE");
    default: return true;
  }
}
function tierLabel(capital) {
  const tier = state.index.capital_tiers.find((t) => t.capital === capital);
  return tier ? tier.label : "";
}

function scoreCell(stock) {
  if (stock.status !== "ok") {
    return `<span class="bad">${stock.status === "blocked" ? "檢查未過" : "錯誤"}</span>`;
  }
  const score = stock.score;
  const tier = score && score.available ? score.tiers[String(state.capital)] : null;
  if (!tier) return `<span class="off" title="${esc(score ? score.reason : "")}">未上線</span>`;
  return `<b>${Math.round(tier.value)}</b><small>${esc(tier.label)}</small>`;
}
function dots(stock) {
  if (stock.status !== "ok") return "";
  const parts = Object.entries(ASPECTS).map(([key, name]) => [name, stock.aspects[key].code]);
  const text = parts.map(([name, code]) => `${name} ${VERDICTS[code]}`).join("・");
  return `<span class="dots" data-tip="${esc(text)}" role="img" aria-label="${esc(text)}">${parts
    .map(([, code]) => `<i class="dot v-${esc(code)}"></i>`).join("")}</span>`;
}
function rowHTML(stock) {
  const fav = state.favs.has(stock.code);
  const current = state.current === stock.code ? ' aria-current="page"' : "";
  const sub = stock.status === "ok" ? esc(stock.industry || "") : esc(stock.problem);
  return `<li class="item"><a class="row" href="#/s/${esc(stock.code)}"${current}>
      <span class="who"><span><span class="code">${esc(stock.code)}</span><span class="name">${esc(stock.name)}</span></span><span class="ind">${sub}</span></span>
      ${dots(stock)}<span class="score">${scoreCell(stock)}</span></a>
    <button class="star" type="button" data-code="${esc(stock.code)}" aria-pressed="${fav}" aria-label="自選 ${esc(stock.code)}">${fav ? "★" : "☆"}</button></li>`;
}
function listHTML() {
  return state.index.stocks.filter(matches).map(rowHTML).join("") || '<li class="empty">沒有符合的股票</li>';
}
function filterButtons() {
  const extra = state.more || MORE_FILTERS.some(([key]) => key === state.filter);
  const buttons = [...MAIN_FILTERS, ...(extra ? MORE_FILTERS : [])].map(
    ([key, label]) => `<button type="button" data-filter="${key}" aria-pressed="${state.filter === key}">${esc(label)}</button>`
  );
  if (!extra) buttons.push('<button type="button" class="more" data-more aria-expanded="false">更多</button>');
  return buttons.join("");
}

function renderList() {
  const tiers = state.index.capital_tiers
    .map((t) => `<option value="${t.capital}"${t.capital === state.capital ? " selected" : ""}>${esc(t.label)}</option>`)
    .join("");
  const [first, ...rest] = state.index.disclaimer;
  listView.innerHTML = `
    <div class="controls">
      <label class="capital"><span>本金</span><select id="capital">${tiers}</select></label>
      <input type="search" id="q" placeholder="搜尋代號、名稱或產業" aria-label="搜尋代號、名稱或產業" value="${esc(state.query)}" autocomplete="off">
      <div class="filters" id="filters" role="group" aria-label="篩選">${filterButtons()}</div>
    </div>
    <ul class="list" id="list">${listHTML()}</ul>
    <footer class="foot"><details><summary>${esc(first)}${CHEVRON}</summary>${rest.map((l) => `<p>${esc(l)}</p>`).join("")}</details></footer>`;
  document.getElementById("q").addEventListener("input", (event) => {
    state.query = event.target.value;
    document.getElementById("list").innerHTML = listHTML();
  });
  document.getElementById("capital").addEventListener("change", (event) => {
    state.capital = Number(event.target.value);
    store("capital", String(state.capital));
    document.getElementById("list").innerHTML = listHTML();
    if (state.detail && state.detail.status === "ok") renderDetail(state.detail);
  });
}

listView.addEventListener("click", (event) => {
  const filter = event.target.closest("[data-filter]");
  if (filter) {
    state.filter = filter.dataset.filter;
    document.getElementById("filters").innerHTML = filterButtons();
    document.getElementById("list").innerHTML = listHTML();
    document.querySelector(`[data-filter="${state.filter}"]`).focus(); // keep keyboard focus
    return;
  }
  if (event.target.closest("[data-more]")) {
    state.more = true;
    document.getElementById("filters").innerHTML = filterButtons();
    document.querySelector('[data-filter="unstable"]').focus();
    return;
  }
  const star = event.target.closest(".star");
  if (star) {
    const code = star.dataset.code;
    if (state.favs.has(code)) state.favs.delete(code);
    else state.favs.add(code);
    store("favs", JSON.stringify([...state.favs]));
    star.setAttribute("aria-pressed", state.favs.has(code));
    star.textContent = state.favs.has(code) ? "★" : "☆";
  }
});

// Long-press on the aspect dots shows their names (hover does this on desktop).
let pressTimer = null;
let suppressClick = false;
listView.addEventListener("touchstart", (event) => {
  const target = event.target.closest(".dots");
  if (!target) return;
  pressTimer = setTimeout(() => {
    target.classList.add("tip");
    suppressClick = true;
    setTimeout(() => target.classList.remove("tip"), 1800);
  }, 450);
}, { passive: true });
listView.addEventListener("touchend", () => clearTimeout(pressTimer));
listView.addEventListener("touchmove", () => clearTimeout(pressTimer), { passive: true });
listView.addEventListener("click", (event) => {
  if (suppressClick && event.target.closest(".row")) {
    event.preventDefault();
    suppressClick = false;
  }
}, true);

// --- detail --------------------------------------------------------------------------------

function header(s) {
  const meta = [
    s.market ? esc(s.market) : "",
    s.industry ? `<span class="ind">${esc(s.industry)}</span>` : "",
    `資料日期 ${esc(s.as_of)}`,
  ].filter(Boolean);
  return `<header class="dhead"><a class="back" href="#/">‹ 列表</a>
    <h2>${esc(s.code)} ${esc(s.name || "")}</h2>
    <p class="meta">${meta.join('<span class="sep">・</span>')}</p></header>`;
}
function currentTier(research) {
  return research.tiers.find((t) => t.capital === state.capital)
    || research.tiers.find((t) => t.capital === research.default_capital);
}

function heroHTML(s) {
  const r = s.research_score;
  const composite = `<p class="small muted">投資大師原則 V1 ${pct(s.composite.V1)}・V3 ${pct(s.composite.V3)}</p>`;
  if (!r || !r.available) {
    return `<section class="hero"><div class="scoreline"><span class="big muted">—</span><span class="label muted">分數未上線</span></div>
      <p class="small muted">${esc(r ? r.reason : "沒有分數資料")}</p>${composite}</section>`;
  }
  const tier = currentTier(r);
  const warnings = (r.notes || []).map((n) => `<p class="warn"><i class="dot"></i><span>${esc(n)}</span></p>`).join("");
  return `<section class="hero"><div class="scoreline"><span class="big">${Math.round(tier.score)}</span><span class="label">${esc(tier.label)}</span></div>
    <p class="small muted">本金 ${esc(tier.capital_label)}・歷史回測參考，非投資建議</p>${composite}${warnings}</section>`;
}

function scoreBody(s) {
  const r = s.research_score;
  const composite = table(
    ["", ...WINDOWS.map((w) => WINDOW_LABEL[w])],
    [["投資大師原則（綜合分數）", ...WINDOWS.map((w) => pct(s.composite[w]))]]
  ) + (s.composite_notes.length ? `<p class="small muted">${s.composite_notes.map(esc).join("<br>")}</p>` : "");
  if (!r || !r.available) return `<p class="muted">${esc(r ? r.reason : "沒有分數資料")}</p>${composite}`;
  const cal = r.calibration;
  const rows = r.tiers.map((t) => [
    esc(t.capital_label),
    `<strong>${Math.round(t.score)}</strong> ${esc(t.label)}`,
    `${t.factor.toFixed(1)}${t.odd_lot ? "（零股）" : ""}`,
    t.cost_ratio === null ? "—" : `${(t.cost_ratio * 100).toFixed(2)}%`,
    t.cap === null ? "—" : `${t.cap}：${esc(t.cap_reason)}`,
    `${signedPct(t.history.mean)}（${signedPct(t.history.ci_low)} ~ ${signedPct(t.history.ci_high)}，n=${t.history.n}）`,
    `${signedPct(t.history.etf_mean)} / ${signedPct(t.history.rf_mean)}`,
  ]);
  const parts = r.parts.map((p) => [esc(p.name), p.points === null ? "—" : `${p.points.toFixed(1)} / ${p.max}`, esc(p.detail)]);
  const notes = r.tiers.filter((t) => t.label_note).map((t) => `${esc(t.capital_label)}：${esc(t.label_note)}`);
  return `<p>基本分 ${r.base.toFixed(1)}，屬於第 ${r.group} 組（共 ${r.groups} 組，1 為最低）。各本金級距的分數 = 基本分 × 本金係數，再套用歷史校準上限。</p>
    ${table(["本金級距", "分數", "本金係數", "來回成本占比", "上限", `第 ${r.group} 組歷史回測平均（${cal.horizon_months} 個月，扣成本）`, "同期 0050 / 定存"], rows)}
    ${notes.length ? `<p class="small muted">${notes.join("<br>")}</p>` : ""}
    ${table(["項目", "得分", "說明"], parts)}
    <p class="small muted">樣本外基準日 ${esc(cal.out_of_sample[0])}～${esc(cal.out_of_sample[1])}（${cal.months} 個月），資料至 ${esc(cal.data_as_of)}。歷史回測平均，不是預期報酬。</p>
    ${composite}
    <p class="small"><strong>${esc(r.disclaimer)}</strong></p>`;
}

function metricsBody(s) {
  return table(
    ["指標", "判定", ...WINDOWS.map((w) => WINDOW_LABEL[w]), "walk-forward"],
    s.metrics.map((m) => [esc(m.name), verdict(m.verdict.code), ...WINDOWS.map((w) => esc(m.values[w])), esc(m.walk_forward || "—")])
  ) + (s.walk_forward_note ? `<p class="small muted">${esc(s.walk_forward_note)}</p>` : "");
}

function sensitivityHTML(h) {
  if (!h) return "";
  const rows = [["綜合分數", ...h.composite.values, h.composite.note || "差距在容忍度內"]].concat(
    h.rows.map((r) => [r.name, ...r.values, r.note])
  );
  const same = h.rows.length ? "" : '<p class="small muted">所有框架項目的通過／未通過在各半衰期下都相同。</p>';
  return `<h4>半衰期敏感度（V1）</h4>
    <p class="small muted">所有半衰期乘上不同倍數重算 V1 框架（目前設定為 1 倍）。</p>
    ${table(["項目", ...h.multipliers, "說明"], rows.map((r) => r.map(esc)))}${same}`;
}

function frameworksBody(s) {
  const items = s.frameworks.map((f) => {
    const stability = f.stability ? verdict(f.stability.code) : "";
    const na = f.not_applicable ? `<p class="small muted">不適用：${esc(f.not_applicable)}</p>` : "";
    const scores = `<p class="small muted">${WINDOWS.map((w) => `${WINDOW_LABEL[w]} ${pct(f.scores[w])}`).join("・")}</p>`;
    const rows = table(["條件", ...WINDOWS.map((w) => WINDOW_LABEL[w])], f.items.map((i) => [esc(i.name), ...WINDOWS.map((w) => esc(i.outcomes[w]))]));
    return `<details class="fw"><summary><span class="grow">${esc(f.name)} <span class="val">${pct(f.scores.V1)}</span></span>${stability}${CHEVRON}</summary>
      <div class="body" style="padding:0 0 12px">${scores}${na}${rows}</div></details>`;
  });
  return `<div>${items.join("")}</div>${sensitivityHTML(s.half_life_sensitivity)}`;
}

function riskBody(s) {
  return WINDOWS.map((w) => {
    const r = s.risk[w];
    if (!r.available) return `<p><strong>${WINDOW_LABEL[w]}</strong>：資料不足（${esc(r.reason)}）</p>`;
    const note = r.sharpe_note ? `<p class="small muted">Sharpe：${esc(r.sharpe_note)}</p>` : "";
    return `<p><strong>${WINDOW_LABEL[w]}</strong><br><span class="small muted">${esc(r.label)}</span></p>${table(["指標", "數值"], r.rows.map(([k, v]) => [esc(k), esc(v)]))}${note}`;
  }).join("");
}

function healthBody(health) {
  const rows = health.checks.map((c) => [esc(c.name), c.passed ? "通過" : '<strong class="t-error">失敗</strong>', c.passed ? "" : esc(c.detail)]);
  const limits = health.limitations.map((l) => `<li>${esc(l)}</li>`).join("");
  return table(["檢查", "結果", "說明"], rows) + (limits ? `<p class="small muted">資料來源限制：</p><ul class="small muted">${limits}</ul>` : "");
}

function renderDetail(s) {
  const aspects = Object.entries(ASPECTS).map(([key, name]) => `<span><span class="key">${name}</span>${verdict(s.aspects[key].code)}</span>`).join("");
  const drivers = s.drivers.length ? `<p class="drivers">主要變動：${s.drivers.slice(0, 3).map(esc).join("、")}</p>` : "";
  const legend = `<details class="info"><summary aria-label="這些判定是什麼意思？" title="這些判定是什麼意思？">ⓘ</summary><ul class="legend">${LEGEND.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>`;
  const technicals = `<p class="small muted">${esc(s.technicals.note)}</p><ul>${s.technicals.descriptions.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`;
  const fundamentals = table(["項目", ...WINDOWS.map((w) => WINDOW_LABEL[w])], s.fundamentals.map((f) => [esc(f.name), ...WINDOWS.map((w) => esc(f.values[w]))]));
  detailPane.innerHTML = `<article class="detail">${header(s)}${heroHTML(s)}
    <div class="line">${aspects}</div>
    <div class="line"><span><span class="key">整體</span>${verdict(s.overall.code)}</span>${legend}</div>
    ${drivers}
    <div class="sections">
      ${section("分數明細", scoreBody(s))}
      ${section("穩定性指標（V1 vs V3）", metricsBody(s))}
      ${section("投資框架", frameworksBody(s))}
      ${section("風險指標（歷史統計值）", riskBody(s))}
      ${section("基本面與月營收（未加權）", fundamentals)}
      ${section("技術指標（僅描述）", technicals)}
      ${section("資料檢查", healthBody(s.health))}
      ${section("完整報告", `<div class="md">${markdown(s.report_markdown)}</div>`)}
    </div>${disclaimer(s.disclaimer)}</article>`;
}

function renderProblem(s) {
  detailPane.innerHTML = `<article class="detail">${header(s)}
    <div class="error"><strong>${s.status === "blocked" ? "資料檢查未通過，未產生分析" : "分析失敗"}</strong><br>${esc(s.problem)}</div>
    ${s.health ? `<div class="sections"><details class="sec" open><summary><span>資料檢查明細</span>${CHEVRON}</summary><div class="body">${healthBody(s.health)}</div></details></div>` : ""}
    ${disclaimer(s.disclaimer)}</article>`;
}

function renderHint() {
  state.detail = null;
  detailPane.innerHTML = `<div class="hint"><p>選擇左側的股票查看詳細資料。</p>
    ${state.index ? state.index.disclaimer.map((l) => `<p class="small">${esc(l)}</p>`).join("") : ""}</div>`;
}

async function showStock(code) {
  detailPane.innerHTML = `<article class="detail"><header class="dhead"><a class="back" href="#/">‹ 列表</a><h2>${esc(code)}</h2></header><p class="muted">載入中…</p></article>`;
  try {
    const stock = await getJSON(`data/stocks/${encodeURIComponent(code)}.json`);
    if (state.current !== code) return; // the user moved on while this loaded
    state.detail = stock;
    if (stock.status === "ok") renderDetail(stock);
    else renderProblem(stock);
    document.title = `${stock.code} ${stock.name || ""}・台股穩定性`;
  } catch (error) {
    detailPane.innerHTML = `<article class="detail"><header class="dhead"><a class="back" href="#/">‹ 列表</a></header><div class="error">無法載入 ${esc(code)}：${esc(error.message)}</div></article>`;
  }
  if (wide.matches) detailPane.scrollTop = 0;
  else window.scrollTo(0, 0);
}

// --- minimal Markdown (headings, tables, lists, bold); status emoji become dots ------------

const EMOJI = { "✅": "STABLE", "🔄": "REGIME_SHIFT", "⚠️": "UNSTABLE", "⚠": "UNSTABLE", "❔": "INSUFFICIENT" };
function inline(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/✅|🔄|⚠️|⚠|❔/gu, (e) => `<i class="dot v-${EMOJI[e]}"></i>`);
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
      const cells = (row) => row.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const [head, , ...body] = block;
      const headers = cells(head).map((h) => h.replace(/\*\*/g, ""));
      out.push(table(headers, body.map((r) => cells(r).map(inline))));
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
    else if (line.startsWith("> ")) out.push(`<p>${inline(line.slice(2))}</p>`);
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
      const saved = Number(stored("capital"));
      const known = state.index.capital_tiers.some((t) => t.capital === saved);
      state.capital = known ? saved : state.index.default_capital;
      renderList();
    } catch (error) {
      listView.innerHTML = `<div class="error">無法載入資料：${esc(error.message)}<br>請確認網路，或稍後再試（每日開市前更新）。</div>`;
      detailPane.innerHTML = "";
      return;
    }
  }
  const match = location.hash.match(/^#\/s\/(\d{4,6})$/);
  const leaving = document.body.dataset.view === "list";
  if (match) {
    if (leaving && !wide.matches) state.listScroll = window.scrollY;
    state.current = match[1];
    document.body.dataset.view = "detail";
    document.getElementById("list").innerHTML = listHTML();
    showStock(match[1]);
  } else {
    state.current = null;
    document.body.dataset.view = "list";
    document.title = "台股穩定性";
    document.getElementById("list").innerHTML = listHTML();
    renderHint();
    if (!wide.matches) window.scrollTo(0, state.listScroll);
  }
}

window.addEventListener("hashchange", route);
route();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {
    /* the app works without offline support */
  });
}
