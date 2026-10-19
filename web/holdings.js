// 我的持股 and 本金試算 (user decisions 2026-10-17). Lays out what web/portfolio.js computes
// from the viewer's own records; computes nothing itself. The records live only in this
// browser (localStorage) and leave it only as a backup file the viewer saves.
// Uses the page helpers of app.js (esc, table, section, CHEVRON, getJSON, state).
"use strict";

const Holdings = (() => {
  const P = Portfolio;
  const KEY = "portfolio.v1";
  const LEVEL = { high: "高", mid: "中", low: "低" };
  const PLAN_NOTE = "建議只依程式事先固定的規則與條件符合度（追求比 0050 抗跌）產生，不代表未來，不構成投資建議。";
  const pf = {
    state: null,
    loadError: null,
    saveError: false,
    market: null,
    history: {},
    lastPlan: null,
    principles: {},
  };
  const listView = document.getElementById("listView");
  const listPane = document.getElementById("listPane");
  const detailPane = document.getElementById("detailPane");
  const sheet = document.getElementById("sheet");

  // --- storage (this device only) --------------------------------------------------------------

  function load() {
    let raw = null;
    try {
      raw = localStorage.getItem(KEY);
    } catch {
      pf.saveError = true;
    }
    if (!raw) return P.emptyState();
    try {
      return P.importJSON(raw);
    } catch (error) {
      // Never overwrite records that could not be read; the viewer can import a backup.
      pf.loadError = `讀不到這台裝置上的紀錄（${error.message}），請匯入備份檔。`;
      return P.emptyState();
    }
  }
  function save() {
    if (pf.loadError) return false;
    try {
      localStorage.setItem(KEY, P.exportJSON(pf.state));
      pf.saveError = false;
    } catch {
      pf.saveError = true;
      return false;
    }
    try {
      if (navigator.storage && navigator.storage.persist) navigator.storage.persist();
    } catch {
      /* best effort */
    }
    return true;
  }
  function records() {
    if (!pf.state) pf.state = load();
    return pf.state;
  }

  // --- data from the daily export ----------------------------------------------------------------

  function today() {
    return new Date().toLocaleDateString("sv-SE"); // YYYY-MM-DD in local time
  }
  async function ensureMarket() {
    if (pf.market) return pf.market;
    if (!state.index.archive) throw new Error("網站還沒有收盤價資料，下一次每日更新後才會有。");
    const [closes, events, params] = await Promise.all([
      getJSON("data/market/closes.json"),
      getJSON("data/market/events.json"),
      getJSON("data/params.json"),
    ]);
    pf.market = { closes: closes.closes, date: closes.date, source: closes.source, events: events.events, params };
    return pf.market;
  }
  async function ensureHistory(codes) {
    const missing = [...new Set(codes)].filter((c) => !(c in pf.history));
    await Promise.all(
      missing.map(async (code) => {
        const response = await fetch(`data/market/history/${encodeURIComponent(code)}.json`, { cache: "no-cache" });
        if (response.status === 404) pf.history[code] = { closes: [], scores: [] };
        else if (!response.ok) throw new Error(`${code} 的歷史資料：HTTP ${response.status}`);
        else pf.history[code] = await response.json();
      })
    );
  }
  function ctx() {
    const m = pf.market;
    return { params: m.params, events: m.events, closes: m.closes, dataDate: m.date, today: today() };
  }
  async function reviewNow() {
    const s = records();
    const m = await ensureMarket();
    await ensureHistory([...s.trades.map((t) => t.code), m.params.compare_etf]);
    const pick = (field) => Object.fromEntries(Object.entries(pf.history).map(([c, h]) => [c, h[field]]));
    return P.review(s, {
      ...ctx(),
      scoresAsOf: state.index.as_of,
      history: pick("closes"),
      scores: pick("scores"),
    });
  }
  function nameOf(code) {
    const quote = pf.market && pf.market.closes[code];
    if (quote && quote[0]) return quote[0];
    const row = state.index.stocks.find((r) => r.code === code);
    return row ? row.name : "";
  }

  // --- formatting only -------------------------------------------------------------------------

  function money(x) {
    if (x === null || x === undefined) return "—";
    const sign = x < 0 ? "−" : "";
    return `${sign}NT$ ${Math.round(Math.abs(x)).toLocaleString("zh-TW")}`;
  }
  function signedMoney(x) {
    if (x === null || x === undefined) return "—";
    return x > 0 ? `+${money(x)}` : money(x);
  }
  function percent(x) {
    if (x === null || x === undefined) return "—";
    return `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(1)}%`;
  }
  function tone(x) {
    return x > 0 ? "gain" : x < 0 ? "loss" : "";
  }
  function count(x) {
    return Number(x).toLocaleString("zh-TW");
  }
  function price(x, digits = 4) {
    return x === null || x === undefined ? "—" : Number(x).toLocaleString("zh-TW", { maximumFractionDigits: digits });
  }
  function lotsText(item) {
    if (item.oddLotOnly) return `${count(item.shares)} 股（全部零股）`;
    return item.odd ? `${count(item.lots)} 張 + ${count(item.odd)} 股零股` : `${count(item.lots)} 張`;
  }
  function worst(alerts) {
    return alerts.find((a) => a.level === "high") || alerts.find((a) => a.level === "mid") || alerts[0];
  }
  function alertDot(alerts) {
    if (!alerts.length) return '<span class="alerts"></span>';
    const top = worst(alerts);
    const text = `${top.action}：${alerts.map((a) => a.reason).join("；")}`;
    return `<span class="alerts" role="img" aria-label="${esc(text)}" title="${esc(text)}"><i class="dot lv-${top.level}"></i></span>`;
  }
  function banner(text, extra = "", kind = "warn") {
    return `<p class="banner ${kind}"><i class="dot"></i><span>${text}</span>${extra}</p>`;
  }
  function footer(extra = []) {
    const m = pf.market;
    const lines = [
      state.index.disclaimer[0],
      "持股與入出金紀錄只存在這台裝置的瀏覽器，不會上傳；換手機或清除網站資料前請先匯出備份。",
      ...(m ? [m.params.note, m.source] : []),
      ...extra,
    ];
    return disclaimer(lines);
  }
  function applyColors() {
    document.body.dataset.up = records().settings.up_color === "green" ? "green" : "red";
  }

  // --- 我的持股 ----------------------------------------------------------------------------------

  async function showHoldings(code) {
    applyColors();
    if (!pf.market) listView.innerHTML = '<p class="muted pad">載入中…</p>';
    let r;
    try {
      r = await reviewNow();
    } catch (error) {
      listView.innerHTML = `<div class="error">無法載入收盤價資料：${esc(error.message)}</div>${footer()}`;
      detailPane.innerHTML = "";
      return;
    }
    logAlerts(r);
    const open = openSections();
    listView.innerHTML = homeHTML(r);
    if (code) detailPane.innerHTML = holdingHTML(r, code);
    else detailPane.innerHTML = `<div class="hint"><p>選擇一檔持股查看交易紀錄、成本與提醒。</p></div>`;
    reopen(open);
  }

  // A re-render after saving keeps the sections the viewer had open.
  function sectionKey(details) {
    return `${details.closest("#detailPane") ? "d" : "l"}|${details.querySelector("summary span").textContent.split("（")[0]}`;
  }
  function openSections() {
    return new Set([...document.querySelectorAll("details.sec[open]")].map(sectionKey));
  }
  function reopen(keys) {
    for (const details of document.querySelectorAll("details.sec")) {
      if (keys.has(sectionKey(details))) details.open = true;
    }
  }

  function logAlerts(r) {
    const s = records();
    const added = r.held.flatMap((row) => P.newAlertLog(s.alert_log, row.code, row.alerts, pf.market.date));
    if (added.length) {
      s.alert_log = [...s.alert_log, ...added];
      save();
    }
  }

  function homeHTML(r) {
    const s = records();
    const top = [];
    if (pf.loadError) top.push(banner(esc(pf.loadError), "", "bad"));
    if (pf.saveError) top.push(banner("無法儲存到這台裝置（可能是私密瀏覽或空間不足），請立即匯出備份。", "", "bad"));
    if (!s.trades.length && !s.cash_flows.length) return top.join("") + emptyHTML();
    if (r.book.negativeCashDates.length) top.push(banner(`現金為負，請補記入金（${esc(r.book.negativeCashDates[0])} 起）`));
    if (r.newCash !== null) {
      top.push(banner(`有新的可投入現金 ${money(r.newCash)}`, ' <a href="#/c">前往本金試算 ›</a>', "info"));
    }
    if (r.backupDue) {
      const last = s.last_backup ? `上次 ${esc(s.last_backup)}` : "還沒有備份過";
      top.push(banner(`超過 ${pf.market.params.backup_reminder_days} 天沒有備份（${last}）`, ' <button type="button" class="link" data-action="export-json">匯出備份</button>'));
    }
    if (r.unquoted.length) top.push(banner(`沒有收盤價：${r.unquoted.map(esc).join("、")}（不計入市值與損益）`, "", "info"));
    for (const p of r.book.pendingStockDividends) {
      top.push(banner(`${esc(p.code)} ${esc(p.date)} 除權：${esc(p.note || "配股資料待確認")}`, "", "info"));
    }
    return `${top.join("")}${summaryHTML(r)}
      <div class="lhead"><h3>持股</h3><span class="small muted">${r.held.length} 檔</span></div>
      <ul class="list">${r.held.map(rowHTML).join("") || '<li class="empty">目前沒有持股</li>'}</ul>
      <div class="sections pad">
        ${r.closed.length ? section("已出清", closedHTML(r.closed)) : ""}
        ${section("入出金紀錄", flowsHTML(s))}
        ${section("報酬明細", detailsHTML(r))}
        ${section(`提醒紀錄（${s.alert_log.length}）`, logHTML(s))}
        ${section("設定", settingsHTML(s))}
        ${section("備份", backupHTML(s))}
      </div>
      <div class="pad"><p class="small muted backup-line">上次備份：${esc(s.last_backup || "尚未備份")}</p>${footer()}</div>`;
  }

  function emptyHTML() {
    return `<section class="hero empty-state">
        <p>還沒有任何紀錄。</p>
        <p class="small muted">記錄每一筆買賣與入出金，這裡會算出持股、損益和提醒。紀錄只存在這台裝置的瀏覽器。</p>
        <div class="actions"><button type="button" class="primary" data-action="trade">新增第一筆交易</button>
        <button type="button" data-action="deposit">入金</button>
        <button type="button" data-action="import-json">匯入備份</button></div>
      </section>${homeScreenNote()}<div class="pad">${footer()}</div>`;
  }

  function homeScreenNote() {
    const standalone = navigator.standalone || window.matchMedia("(display-mode: standalone)").matches;
    if (standalone) return "";
    return `<p class="small muted pad">建議用 Safari「分享 › 加入主畫面」：沒有加到主畫面的網站，Safari 可能會清除長期沒使用的資料。</p>`;
  }

  function summaryHTML(r) {
    const etf = r.etf;
    let compare = "";
    if (etf && etf.missing) {
      compare = `${esc(etf.code)} 同期：${esc(etf.missing)} 還沒有收盤價紀錄${etf.since ? `（網站從 ${esc(etf.since)} 起保存）` : ""}，無法比較`;
    } else if (etf && etf.shortfall) {
      compare = `${esc(etf.code)} 同期：${esc(etf.shortfall)} 的出金大於 ${esc(etf.code)} 部位，無法比較`;
    } else if (etf) {
      compare = `同樣入出金改買 ${esc(etf.code)}：總損益 <span class="${tone(etf.gain)}">${signedMoney(etf.gain)}</span>・時間加權 <span class="${tone(etf.twr)}">${percent(etf.twr)}</span>`;
    }
    const estimated = r.twrEstimated ? '<span class="small muted">（部分日期沒有收盤價，以成交價估算）</span>' : "";
    return `<section class="hero">
        <p class="small muted">總資產（持股市值＋現金）</p>
        <div class="big">${money(r.totalAssets)}</div>
        <p class="gl">總損益 <span class="${tone(r.totalGain)}">${signedMoney(r.totalGain)}</span>・時間加權 <span class="${tone(r.twr)}">${percent(r.twr)}</span>${estimated}</p>
        ${compare ? `<p class="small muted">${compare}</p>` : ""}
        <p class="small muted">收盤價（資料日期 ${esc(pf.market.date)}），不是即時報價</p>
      </section>
      <div class="cashline pad"><span>現金 ${money(r.cash)}</span>
        <button type="button" data-action="deposit">入金</button><button type="button" data-action="withdraw">出金</button></div>`;
  }

  function rowHTML(row) {
    const name = nameOf(row.code);
    const result = row.returnPct === null
      ? '<span class="off">無報價</span>'
      : `<b class="${tone(row.returnPct)}">${percent(row.returnPct)}</b><small class="${tone(row.unrealized)}">${signedMoney(row.unrealized)}</small>`;
    return `<li class="item solo"><a class="row" href="#/p/${esc(row.code)}"${state.current === row.code ? ' aria-current="page"' : ""}>
        <span class="who"><span><span class="code">${esc(row.code)}</span><span class="name">${esc(name)}</span></span><span class="sub">${count(row.shares)} 股</span></span>
        ${alertDot(row.alerts)}<span class="score">${result}</span></a></li>`;
  }

  function closedHTML(rows) {
    return table(
      ["代號", "名稱", "已實現損益", "現金股利"],
      rows.map((r) => [esc(r.code), esc(nameOf(r.code)), `<span class="${tone(r.realized)}">${signedMoney(r.realized)}</span>`, money(r.dividends)])
    );
  }

  function flowsHTML(s) {
    const flows = [...s.cash_flows].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
    const rows = flows.map((f) => [
      esc(f.date),
      f.type === "deposit" ? "入金" : "出金",
      money(f.amount),
      esc(f.note || ""),
      `<button type="button" class="link" data-action="edit-flow" data-id="${esc(f.id)}">修改</button> <button type="button" class="link" data-action="delete-flow" data-id="${esc(f.id)}">刪除</button>`,
    ]);
    return `${rows.length ? table(["日期", "類型", "金額", "備註", ""], rows) : '<p class="muted">還沒有入出金紀錄。</p>'}
      <p class="small muted">修改或刪除過去的入出金後，現金與報酬率會全部重新計算；會讓之後的出金超過當時現金的修改不會儲存。</p>`;
  }

  function detailsHTML(r) {
    const b = r.book;
    const rows = [
      ["累計入金", money(r.deposits)],
      ["累計出金", money(r.withdrawals)],
      ["持股市值", money(r.marketValue)],
      ["現金", money(r.cash)],
      ["總損益", `<span class="${tone(r.totalGain)}">${signedMoney(r.totalGain)}</span>`],
      ["時間加權報酬率", `<span class="${tone(r.twr)}">${percent(r.twr)}</span>`],
      ["資金加權報酬率（XIRR，年化）", r.xirr === null ? "—（期間太短或無法求解）" : `<span class="${tone(r.xirr)}">${percent(r.xirr)}</span>`],
      ["已實現損益", `<span class="${tone(b.realized)}">${signedMoney(b.realized)}</span>`],
      ["現金股利", money(b.dividends)],
      ["手續費", money(b.fees)],
      ["證交稅", money(b.taxes)],
    ];
    return `${table(["項目", "金額"], rows)}
      <p class="small muted">時間加權報酬率每次入出金切一段再連乘，反映選股本身，和什麼時候加減本金無關；XIRR 反映實際的賺賠，含加減碼時機。${b.estimatedCosts ? `有 ${b.estimatedCosts} 筆交易的手續費或稅是估算的。` : ""}</p>
      <p class="small muted">${esc(pf.market.params.note)}</p>`;
  }

  function logHTML(s) {
    if (!s.alert_log.length) return '<p class="muted">還沒有提醒。</p>';
    const rows = [...s.alert_log].reverse().slice(0, 200).map((e) => [
      esc(e.date),
      `${esc(e.code)} ${esc(nameOf(e.code))}`,
      `<span class="lv"><i class="dot lv-${esc(e.level)}"></i>${esc(LEVEL[e.level] || e.level)}</span>`,
      esc(e.action),
      esc(e.reason),
    ]);
    return `${table(["資料日期", "股票", "等級", "提醒", "原因"], rows)}
      <p class="small muted">每次出現的提醒都記在這裡（同一天同一原因只記一次），之後可以回頭看提醒準不準。</p>`;
  }

  function settingsHTML(s) {
    const v = P.settingsForInput(s.settings);
    const params = pf.market.params;
    return `<form class="form" data-form="settings">
        <label>手續費折扣<input name="fee_discount" type="number" step="any" min="0" max="1" inputmode="decimal" value="${esc(v.fee_discount)}" placeholder="${esc(params.costs.default_fee_discount)}"></label>
        <p class="small muted">例：0.6 為六折；留空用預設值。只影響沒有填實際手續費的交易。</p>
        <label>停損（%）<input name="stop_loss" type="number" step="any" max="0" inputmode="decimal" value="${esc(v.stop_loss)}" placeholder="例：-15"></label>
        <label>停利（%）<input name="take_profit" type="number" step="any" min="0" inputmode="decimal" value="${esc(v.take_profit)}" placeholder="例：30"></label>
        <label>漲跌顏色<select name="up_color"><option value="red"${v.up_color === "red" ? " selected" : ""}>紅漲綠跌（台股習慣）</option><option value="green"${v.up_color === "green" ? " selected" : ""}>綠漲紅跌</option></select></label>
        <p class="form-errors" role="alert"></p>
        <div class="actions"><button type="submit" class="primary">儲存設定</button></div>
      </form>`;
  }

  function backupHTML(s) {
    return `<p>上次備份：${esc(s.last_backup || "尚未備份")}</p>
      <div class="actions"><button type="button" data-action="export-json">匯出備份（JSON）</button>
      <button type="button" data-action="import-json">匯入備份</button>
      <button type="button" data-action="export-csv">匯出 CSV（Excel）</button></div>
      <p class="small muted">匯入會取代這台裝置上的所有紀錄。CSV 只給試算表看，不能匯入。</p>${homeScreenNote()}`;
  }

  // --- one holding -------------------------------------------------------------------------------

  function holdingHTML(r, code) {
    const row = r.held.find((x) => x.code === code) || r.closed.find((x) => x.code === code);
    const back = '<a class="back" href="#/p">‹ 我的持股</a>';
    if (!row) return `<article class="detail"><header class="dhead">${back}<h2>${esc(code)}</h2></header><p class="muted">沒有這檔股票的紀錄。</p></article>`;
    const name = nameOf(code);
    const inIndex = state.index.stocks.some((s) => s.code === code);
    const quote = row.close === null ? "沒有收盤價" : `收盤價 ${price(row.close)}（資料日期 ${esc(pf.market.date)}），不是即時報價`;
    const alerts = row.alerts || [];
    const hero = row.shares > 0
      ? `<section class="hero"><div class="scoreline"><span class="big ${tone(row.returnPct)}">${percent(row.returnPct)}</span><span class="label ${tone(row.unrealized)}">${signedMoney(row.unrealized)}</span></div>
          <p class="small muted">${count(row.shares)} 股・平均成本 ${price(row.average, 2)}・未實現損益已扣估算的賣出手續費與證交稅</p></section>`
      : `<section class="hero"><p>已出清・已實現損益 <span class="${tone(row.realized)}">${signedMoney(row.realized)}</span></p></section>`;
    const alertList = alerts.length
      ? `<ul class="plain bare">${alerts.map((a) => `<li><span class="lv"><i class="dot lv-${a.level}"></i>${esc(a.action)}</span>：${esc(a.reason)}</li>`).join("")}</ul>
         <p class="small muted">依你的停損停利設定與程式的固定規則產生，股價漲跌本身不是理由；不是投資建議。</p>`
      : '<p class="muted">目前沒有提醒。</p>';
    const pnl = table(["項目", "金額"], [
      ["持股市值", money(row.value)],
      ["持有成本（含買進手續費）", money(row.cost)],
      ["估算賣出成本（手續費＋證交稅）", money(row.sellCost)],
      ["未實現損益", `<span class="${tone(row.unrealized)}">${signedMoney(row.unrealized)}</span>`],
      ["已實現損益", `<span class="${tone(row.realized)}">${signedMoney(row.realized)}</span>`],
      ["現金股利", money(row.dividends)],
    ]);
    const trades = table(
      ["日期", "買賣", "價格", "股數", "手續費", "稅", ""],
      [...row.trades].reverse().map((t) => [
        esc(t.date),
        t.side === "buy" ? "買進" : "賣出",
        price(t.price),
        count(t.shares),
        t.fee === null || t.fee === undefined || t.fee === "" ? "估算" : money(t.fee),
        t.side === "buy" ? "" : t.tax === null || t.tax === undefined || t.tax === "" ? "估算" : money(t.tax),
        `<button type="button" class="link" data-action="edit-trade" data-id="${esc(t.id)}">修改</button> <button type="button" class="link" data-action="delete-trade" data-id="${esc(t.id)}">刪除</button>`,
      ])
    );
    return `<article class="detail"><header class="dhead">${back}<h2>${esc(code)} ${esc(name)}</h2><p class="meta">${quote}</p></header>
      ${hero}
      <div class="sections">
        ${row.shares > 0 ? section(`提醒（${alerts.length}）`, alertList) : ""}
        ${section("損益明細", pnl)}
        ${section(`交易紀錄（${row.trades.length}）`, trades)}
        ${row.shares > 0 ? section("買進時紀錄", snapshotHTML(row)) : ""}
      </div>
      ${inIndex ? `<p class="pad-y"><a href="#/s/${esc(code)}">查看 ${esc(code)} 的分析 ›</a></p>` : ""}
      ${footer()}</article>`;
  }

  function snapshotHTML(row) {
    const b = row.bought;
    if (!b) return `<p class="muted">無買進時紀錄（買進日 ${esc(row.opened ? row.opened.date : "")} 早於網站保存分數的日期，或不在觀察清單）。</p>`;
    if (b.status !== "ok") return `<p>${esc(b.as_of)} 的資料檢查未通過，沒有條件符合度。</p>`;
    const aspects = Object.entries(ASPECTS)
      .map(([k, n]) => `<span><span class="key">${n}</span>${b.aspects && b.aspects[k] ? verdict(b.aspects[k]) : "—"}</span>`)
      .join("");
    const tiers = Object.entries(b.tiers || {}).map(([c, v]) => `${esc(capitalLabel(Number(c)))} ${Math.round(v)}`).join("・");
    return `<p>買進時參考 ${esc(b.as_of)} 的匯出：</p><div class="line">${aspects}</div>
      <p class="small muted">條件符合度：${tiers || "—"}</p>`;
  }

  function capitalLabel(capital) {
    const tier = state.index.capital_tiers.find((t) => t.capital === capital);
    return tier ? tier.label : String(capital);
  }

  // --- forms (bottom sheet) ------------------------------------------------------------------------

  function openSheet(html) {
    sheet.innerHTML = html;
    if (typeof sheet.showModal === "function") sheet.showModal();
    else sheet.setAttribute("open", "");
    const first = sheet.querySelector("input:not([type=radio]), select");
    if (first) first.focus();
  }
  function closeSheet() {
    if (typeof sheet.close === "function") sheet.close();
    else sheet.removeAttribute("open");
    sheet.innerHTML = "";
  }

  function tradeForm(trade) {
    const t = trade || { side: "buy", date: pf.market.date, code: "", price: "", shares: "", fee: "", tax: "" };
    const options = Object.entries(pf.market.closes).map(([c, [n]]) => `<option value="${esc(c)}">${esc(n)}</option>`).join("");
    const value = (x) => (x === null || x === undefined ? "" : esc(x));
    return `<form class="form sheet-form" data-form="trade" data-id="${value(t.id)}">
        <h3>${trade ? "修改交易" : "新增交易"}</h3>
        <div class="seg" role="radiogroup" aria-label="買賣">
          <label><input type="radio" name="side" value="buy"${t.side === "buy" ? " checked" : ""}><span>買進</span></label>
          <label><input type="radio" name="side" value="sell"${t.side === "sell" ? " checked" : ""}><span>賣出</span></label>
        </div>
        <label>代號<input name="code" list="codes" autocomplete="off" autocapitalize="characters" required value="${value(t.code)}" placeholder="例：2330"></label>
        <p class="small muted" data-name>${esc(t.code ? nameOf(t.code) : "")}</p>
        <datalist id="codes">${options}</datalist>
        <label>成交日期<input type="date" name="date" required max="${esc(pf.market.date)}" value="${value(t.date)}"></label>
        <p class="small muted">最晚到資料日期 ${esc(pf.market.date)}。</p>
        <label>成交價<input type="number" name="price" step="any" min="0" inputmode="decimal" required value="${value(t.price)}"></label>
        <label>股數（零股也用股數）<input type="number" name="shares" step="1" min="1" inputmode="numeric" required value="${value(t.shares)}"></label>
        <label>手續費（可留空，自動估算）<input type="number" name="fee" step="any" min="0" inputmode="decimal" value="${value(t.fee)}"></label>
        <label data-sell${t.side === "sell" ? "" : " hidden"}>證交稅（可留空，自動估算）<input type="number" name="tax" step="any" min="0" inputmode="decimal" value="${value(t.tax)}"></label>
        <p class="form-errors" role="alert"></p>
        <div class="actions"><button type="button" data-action="close">取消</button><button type="submit" class="primary">儲存</button></div>
      </form>`;
  }

  function flowForm(type, flow) {
    const f = flow || { type, date: today(), amount: "", note: "" };
    return `<form class="form sheet-form" data-form="flow" data-id="${esc(f.id || "")}">
        <h3>${flow ? "修改入出金" : f.type === "deposit" ? "入金" : "出金"}</h3>
        <div class="seg" role="radiogroup" aria-label="入金或出金">
          <label><input type="radio" name="type" value="deposit"${f.type === "deposit" ? " checked" : ""}><span>入金</span></label>
          <label><input type="radio" name="type" value="withdraw"${f.type === "withdraw" ? " checked" : ""}><span>出金</span></label>
        </div>
        <label>日期<input type="date" name="date" required max="${esc(today())}" value="${esc(f.date)}"></label>
        <label>金額（元）<input type="number" name="amount" step="any" min="0" inputmode="decimal" required value="${esc(f.amount)}"></label>
        <label>備註<input name="note" maxlength="60" value="${esc(f.note || "")}" placeholder="例：初始本金"></label>
        <p class="form-errors" role="alert"></p>
        <div class="actions"><button type="button" data-action="close">取消</button><button type="submit" class="primary">儲存</button></div>
      </form>`;
  }

  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }
  function field(form, name) {
    const value = form.elements[name] ? form.elements[name].value.trim() : "";
    return value === "" ? null : value;
  }
  function showErrors(form, errors, warnings = []) {
    const box = form.querySelector(".form-errors");
    box.innerHTML = [...errors.map((e) => `<span class="bad">${esc(e)}</span>`), ...warnings.map((w) => `<span>${esc(w)}</span>`)].join("<br>");
  }

  async function submitTrade(form) {
    const s = records();
    const id = form.dataset.id || newId();
    const old = s.trades.find((t) => t.id === id);
    const trade = {
      id,
      code: (field(form, "code") || "").toUpperCase(),
      side: form.elements.side.value,
      date: field(form, "date"),
      price: Number(field(form, "price")),
      shares: Number(field(form, "shares")),
      fee: field(form, "fee") === null ? null : Number(field(form, "fee")),
      tax: form.elements.side.value === "sell" && field(form, "tax") !== null ? Number(field(form, "tax")) : null,
    };
    const errors = P.validateTrade(trade, s, ctx());
    if (errors.length) return showErrors(form, errors);
    const warnings = P.tradeWarnings(trade, s, ctx());
    if (trade.side === "buy") {
      await ensureHistory([trade.code]);
      const keep = old && old.side === "buy" && old.date === trade.date && old.code === trade.code;
      trade.snapshot = keep ? old.snapshot : P.snapshotFor(pf.history[trade.code].scores, trade.date);
    }
    const trades = old ? s.trades.map((t) => (t.id === id ? trade : t)) : [...s.trades, trade];
    const next = { ...s, trades };
    const broken = P.ledgerErrors(next, ctx()).filter((e) => !e.includes("出金"));
    if (broken.length) return showErrors(form, broken);
    pf.state = next;
    save();
    if (warnings.length && !form.dataset.warned) {
      form.dataset.warned = "1"; // saved; say why the cash looks wrong, close on the next tap
      showErrors(form, [], [...warnings, "已儲存。"]);
      form.querySelector("[type=submit]").textContent = "完成";
      form.dataset.done = "1";
      rerender();
      return;
    }
    closeSheet();
    rerender();
  }

  function submitFlow(form) {
    const s = records();
    const id = form.dataset.id || newId();
    const flow = {
      id,
      date: field(form, "date"),
      type: form.elements.type.value,
      amount: Number(field(form, "amount")),
      note: field(form, "note") || "",
    };
    const flows = s.cash_flows.some((f) => f.id === id)
      ? s.cash_flows.map((f) => (f.id === id ? flow : f))
      : [...s.cash_flows, flow];
    const errors = P.validateCashFlows(flows, s, ctx());
    if (errors.length) return showErrors(form, errors);
    pf.state = { ...s, cash_flows: flows };
    save();
    closeSheet();
    rerender();
  }

  function submitSettings(form) {
    const s = records();
    const input = Object.fromEntries(["fee_discount", "stop_loss", "take_profit", "up_color"].map((n) => [n, field(form, n)]));
    const { settings, errors } = P.settingsFromInput(input, s.settings);
    if (errors.length) return showErrors(form, errors);
    pf.state = { ...s, settings };
    save();
    rerender();
  }

  function remove(kind, id) {
    const s = records();
    const next = kind === "trade"
      ? { ...s, trades: s.trades.filter((t) => t.id !== id) }
      : { ...s, cash_flows: s.cash_flows.filter((f) => f.id !== id) };
    const errors = P.ledgerErrors(next, ctx());
    if (errors.length) {
      window.alert(`不能刪除：${errors.join("；")}`);
      return;
    }
    if (!window.confirm(kind === "trade" ? "刪除這筆交易？" : "刪除這筆入出金？")) return;
    pf.state = next;
    save();
    rerender();
  }

  // --- backup ------------------------------------------------------------------------------------

  async function download(text, name, type) {
    const file = typeof File === "function" ? new File([text], name, { type }) : null;
    if (navigator.standalone && file && navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file] });
        return true;
      } catch {
        return false; // cancelled
      }
    }
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return true;
  }
  async function exportJSON() {
    const backup = { ...records(), last_backup: today() };
    if (!(await download(P.exportJSON(backup), `stock-portfolio-${today()}.json`, "application/json"))) return;
    pf.state = backup;
    save();
    rerender();
  }
  function exportCSV() {
    download("﻿" + P.toCSV(records()), `stock-portfolio-${today()}.csv`, "text/csv"); // BOM for Excel
  }
  function importJSON() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.addEventListener("change", async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      try {
        const imported = P.importJSON(await file.text());
        const n = imported.trades.length + imported.cash_flows.length;
        if (!window.confirm(`匯入 ${n} 筆紀錄，並取代這台裝置上的所有紀錄？`)) return;
        pf.state = imported;
        pf.loadError = null;
        save();
        rerender();
      } catch (error) {
        window.alert(`無法匯入：${error.message}`);
      }
    });
    input.click();
  }

  // --- 本金試算 (version A) ------------------------------------------------------------------------

  async function showCapital() {
    applyColors();
    if (!pf.market) listView.innerHTML = '<p class="muted pad">載入中…</p>';
    let r;
    try {
      r = await reviewNow();
    } catch (error) {
      listView.innerHTML = `<div class="error">無法載入收盤價資料：${esc(error.message)}</div>${footer()}`;
      detailPane.innerHTML = "";
      return;
    }
    const rule = pf.market.params.allocation;
    const suggested = r.newCash !== null ? r.newCash : r.cash > 0 ? r.cash : null;
    const amount = pf.lastPlan ? pf.lastPlan.amount : suggested;
    const withHoldings = pf.lastPlan ? pf.lastPlan.withHoldings : false;
    listView.innerHTML = `<section class="hero"><form class="form" data-form="capital">
        <label>可投入金額（元）<input type="number" name="amount" min="1" step="1" inputmode="numeric" required value="${amount === null ? "" : Math.floor(amount)}"></label>
        <p class="small muted">${suggested === null ? "沒有記錄現金時請自行輸入。" : `預設帶入目前的現金 ${money(suggested)}。`}</p>
        <label class="check"><input type="checkbox" name="with_holdings"${withHoldings ? " checked" : ""}${r.held.length ? "" : " disabled"}><span>把現有持股一起算進去（以總資產重新分配，只買差額，不建議賣出）</span></label>
        <p class="form-errors" role="alert"></p>
        <div class="actions"><button type="submit" class="primary">試算</button></div>
      </form></section>
      <div class="pad">${section("建議規則", `<ul class="plain small">
        <li>依金額決定檔數：${state.index.capital_tiers.map((t) => esc(t.label)).join("、")}（從匯出參數讀取）。</li>
        <li>候選：該級距條件符合度 ≥ ${rule.min_score}，且沒有被標示「${esc(rule.excluded_cap_reason)}」。</li>
        <li>等權重，單一個股最多 ${Math.round(rule.max_weight * 100)}%，保留 ${Math.round(rule.cash_reserve * 100)}% 現金；不足的檔數或上限以外的部分放 ${esc(rule.fallback_etf)}。</li>
        <li>分配金額夠買一張就以整張為單位，剩下用零股補；整張與零股分兩筆下單、各收手續費。</li>
        <li>來回成本超過 ${(rule.cost_ratio_warning * 100).toFixed(1)}% 時提醒手續費占比偏高。</li></ul>`)}</div>`;
    pf.review = r;
    if (pf.lastPlan) await renderPlan();
    else detailPane.innerHTML = `<article class="detail"><p class="muted pad-y">輸入金額後按「試算」。</p>${footer([PLAN_NOTE])}</article>`;
  }

  async function submitCapital(form) {
    const amount = Number(field(form, "amount"));
    if (!(amount > 0)) return showErrors(form, ["請輸入大於 0 的金額"]);
    showErrors(form, []);
    pf.lastPlan = { amount, withHoldings: form.elements.with_holdings.checked };
    await renderPlan();
    if (!wide.matches) detailPane.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function renderPlan() {
    const r = pf.review;
    const m = pf.market;
    const current = pf.lastPlan.withHoldings ? r.held.map((h) => ({ code: h.code, name: nameOf(h.code), value: h.value })) : [];
    const candidates = P.candidatesFrom(state.index.stocks, m.closes);
    const plan = P.allocate(pf.lastPlan.amount, candidates, { ...ctx(), state: records() }, current);
    const date = esc(m.date);
    const cards = plan.items.map((item) => cardHTML(item, plan, date)).join("");
    const fallback = plan.fallback ? cardHTML(plan.fallback, plan, date, true) : "";
    const reductions = plan.reductions.length
      ? section("需要減碼的部位（只列出，不建議賣出）", table(["股票", "超出目標", "原因"], plan.reductions.map((x) => [`${esc(x.code)} ${esc(x.name || nameOf(x.code))}`, money(x.over), esc(x.reason)])))
      : "";
    const scaled = plan.scale < 1 ? `<p class="small muted">現金不夠補足每一檔的目標，各檔依同樣比例縮小。</p>` : "";
    detailPane.innerHTML = `<article class="detail"><header class="dhead"><h2>試算結果</h2>
        <p class="meta">${esc(capitalLabel(plan.tier.capital))}（以${pf.lastPlan.withHoldings ? "總資產" : "金額"} ${money(plan.total)} 判斷）・收盤價資料日期 ${date}，不是即時報價</p></header>
      ${plan.items.length || plan.fallback ? "" : '<p class="muted">這個金額買不到任何一股，或沒有符合條件的股票。</p>'}
      <div class="cards">${cards}${fallback}</div>
      ${scaled}
      <section class="hero"><p>共需 ${money(plan.spent)}（含手續費）</p>
        <p class="small muted">剩餘現金 ${money(plan.cashLeft)}，其中保留 ${money(plan.reserve)}</p></section>
      ${reductions ? `<div class="sections">${reductions}</div>` : ""}
      ${footer([PLAN_NOTE])}</article>`;
    loadReasons(plan.items.map((x) => x.code));
  }

  function cardHTML(item, plan, date, isFallback = false) {
    const warnings = [];
    if (item.costWarning) warnings.push(item.costWarning);
    if (item.oddLotOnly) warnings.push("零股流動性較差");
    const head = isFallback
      ? `<b>${esc(item.code)} ${esc(item.name)}</b><span class="small muted">補足部分</span>`
      : `<b>${esc(item.code)} ${esc(item.name)}</b><span class="tag">${Math.round(item.score)} ${esc(item.label || "")}</span>`;
    const reasons = isFallback
      ? `<ul class="plain small">${item.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
      : `<div class="small reasons" data-reasons="${esc(item.code)}"><span class="muted">載入符合的投資大師原則…</span></div>`;
    const process = table(["項目", "數值"], [
      ...(isFallback ? [] : [["每檔目標金額", money(plan.target)], ["已持有", money(item.held)]]),
      ["本次分配", money(item.allotted)],
      [`收盤價（${date}）`, price(item.price)],
      ["股數", `${count(item.shares)}（分配金額內、含手續費可買的最多股數）`],
      ["金額", money(item.amount)],
      ["預估手續費", money(item.fee)],
      ...(isFallback ? [] : [["來回成本占比", `${(item.costRatio * 100).toFixed(2)}%`]]),
    ]);
    return `<section class="card">
        <div class="card-head">${head}</div>
        <p class="big2">建議買 ${count(item.shares)} 股</p>
        <p class="small muted">${lotsText(item)}・金額 ${money(item.amount)}・手續費約 ${money(item.fee)}</p>
        ${warnings.map((w) => `<p class="warn"><i class="dot"></i><span>${esc(w)}</span></p>`).join("")}
        ${reasons}
        <details class="fw"><summary><span class="grow">計算過程</span>${CHEVRON}</summary><div class="body" style="padding:0 0 10px">${process}</div></details>
      </section>`;
  }

  async function loadReasons(codes) {
    await Promise.all(
      codes.map(async (code) => {
        if (!(code in pf.principles)) {
          try {
            const stock = await getJSON(`data/stocks/${encodeURIComponent(code)}.json`);
            pf.principles[code] = stock.principles || [];
          } catch {
            pf.principles[code] = null;
          }
        }
        const box = detailPane.querySelector(`[data-reasons="${CSS.escape(code)}"]`);
        if (!box) return;
        const list = pf.principles[code];
        if (!list) {
          box.innerHTML = '<span class="muted">無法載入原則說明</span>';
          return;
        }
        const met = list.filter((f) => f.passed && f.passed.length);
        box.innerHTML = met.length
          ? `<p class="muted">符合的投資大師原則：</p><ul class="plain">${met.map((f) => `<li><strong>${esc(f.name)}</strong>：${f.passed.map((n) => `「${esc(n)}」`).join("")}</li>`).join("")}</ul>`
          : '<span class="muted">沒有符合的原則條件</span>';
      })
    );
  }

  // --- events ------------------------------------------------------------------------------------

  function rerender() {
    if (location.hash.startsWith("#/c")) showCapital();
    else {
      const match = location.hash.match(/^#\/p\/([0-9A-Z]{4,7})$/);
      showHoldings(match ? match[1] : null);
    }
  }

  async function onClick(event) {
    const button = event.target.closest("[data-action]");
    if (!button || !pf.market) return;
    const s = records();
    switch (button.dataset.action) {
      case "trade": return openSheet(tradeForm(null));
      case "deposit": return openSheet(flowForm("deposit"));
      case "withdraw": return openSheet(flowForm("withdraw"));
      case "edit-trade": return openSheet(tradeForm(s.trades.find((t) => t.id === button.dataset.id)));
      case "delete-trade": return remove("trade", button.dataset.id);
      case "edit-flow": return openSheet(flowForm(null, s.cash_flows.find((f) => f.id === button.dataset.id)));
      case "delete-flow": return remove("flow", button.dataset.id);
      case "export-json": return exportJSON();
      case "export-csv": return exportCSV();
      case "import-json": return importJSON();
      case "close": return closeSheet();
      default: return undefined;
    }
  }

  async function onSubmit(event) {
    const form = event.target.closest("[data-form]");
    if (!form) return;
    event.preventDefault();
    if (form.dataset.done) {
      closeSheet();
      rerender();
      return;
    }
    switch (form.dataset.form) {
      case "trade": return submitTrade(form);
      case "flow": return submitFlow(form);
      case "settings": return submitSettings(form);
      case "capital": return submitCapital(form);
      default: return undefined;
    }
  }

  function onInput(event) {
    const form = event.target.closest("[data-form=trade]");
    if (!form) return;
    if (event.target.name === "code") {
      const box = form.querySelector("[data-name]");
      const code = event.target.value.trim().toUpperCase();
      box.textContent = code ? nameOf(code) || "清單外或代號不存在：沒有收盤價時不計入市值" : "";
    }
    form.querySelector("[data-sell]").hidden = form.elements.side.value !== "sell";
  }

  for (const target of [listPane, detailPane, sheet]) {
    target.addEventListener("click", onClick);
    target.addEventListener("submit", onSubmit);
  }
  sheet.addEventListener("input", onInput);
  sheet.addEventListener("change", onInput);
  sheet.addEventListener("cancel", () => {
    sheet.innerHTML = "";
  });

  return { showHoldings, showCapital, headAction: () => '<button type="button" class="primary small-btn" data-action="trade">＋ 新增交易</button>' };
})();
