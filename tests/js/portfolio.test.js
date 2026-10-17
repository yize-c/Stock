// Unit tests for web/portfolio.js (Node's built-in test runner, no packages):
//   node --test tests/js/portfolio.test.js
// params.json is the export's data/params.json for the default config; a Python test keeps
// it equal to stock_research.export.market.params_payload.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const P = require("../../web/portfolio.js");
const params = require("./params.json");

const close = (actual, expected, digits = 6) =>
  assert.ok(Math.abs(actual - expected) < 10 ** -digits, `${actual} != ${expected}`);

let nextId = 0;
const id = () => `id-${(nextId += 1)}`;
const buy = (code, date, price, shares, extra = {}) => ({ id: id(), code, side: "buy", date, price, shares, ...extra });
const sell = (code, date, price, shares, extra = {}) => ({ id: id(), code, side: "sell", date, price, shares, ...extra });
const deposit = (date, amount) => ({ id: id(), date, type: "deposit", amount, note: "" });
const withdraw = (date, amount) => ({ id: id(), date, type: "withdraw", amount, note: "" });

function state(trades = [], flows = [], settings = {}) {
  const s = P.emptyState();
  return { ...s, trades, cash_flows: flows, settings: { ...s.settings, ...settings } };
}
function ctx(extra = {}) {
  return { params, events: [], dataDate: "2026-10-16", today: "2026-10-17", closes: {}, ...extra };
}

// --- costs ------------------------------------------------------------------------------------

test("fee has a 20 minimum and the viewer's discount; ETFs pay the lower tax", () => {
  close(P.feeFor(10000, 1, params), 20); // 14.25 → minimum
  close(P.feeFor(100000, 1, params), 142.5);
  close(P.feeFor(100000, 0.6, params), 85.5);
  close(P.taxFor(100000, "2330", params), 300);
  close(P.taxFor(100000, "0050", params), 100);
  close(P.taxFor(100000, "00981A", params), 100);
  const discounted = state([], [], { fee_discount: 0.6 });
  const est = P.tradeCosts(sell("0050", "2026-10-01", 100, 1000), discounted, params);
  close(est.fee, 85.5);
  close(est.tax, 100);
  assert.equal(est.estimated, true);
  const entered = P.tradeCosts(sell("2330", "2026-10-01", 100, 1000, { fee: 50, tax: 300 }), discounted, params);
  assert.deepEqual([entered.fee, entered.tax, entered.estimated], [50, 300, false]);
  const zeroFee = P.tradeCosts(buy("2330", "2026-10-01", 100, 10, { fee: 0 }), discounted, params);
  assert.equal(zeroFee.fee, 0); // an entered 0 is a real 0, not "missing"
});

// --- moving average cost ----------------------------------------------------------------------

function averaging() {
  return state(
    [
      buy("2330", "2026-09-02", 500, 1000),
      buy("2330", "2026-09-03", 600, 1000),
      sell("2330", "2026-09-04", 700, 500),
    ],
    [deposit("2026-09-01", 1000000)]
  );
}

test("buys in batches give a moving-average cost; a partial sell realizes against it", () => {
  const book = P.replay(averaging(), ctx());
  const p = book.positions["2330"];
  // cost = 500000 + 712.5 + 600000 + 855; the sell removes 500 shares at the average
  const average = (500712.5 + 600855) / 2000;
  assert.equal(p.shares, 1500);
  close(p.cost, average * 1500);
  close(book.realized, 350000 - 498.75 - 1050 - average * 500);
  close(book.cash, 1000000 - 500712.5 - 600855 + (350000 - 498.75 - 1050));
  assert.equal(book.estimatedCosts, 3);
});

test("unrealized P&L deducts the estimated selling fee and tax", () => {
  const { rows } = P.holdings(averaging(), ctx({ closes: { 2330: ["台積電", 700] } }));
  const row = rows[0];
  const average = (500712.5 + 600855) / 2000;
  close(row.average, average);
  close(row.value, 1050000);
  close(row.unrealized, 1050000 - 1496.25 - 3150 - average * 1500);
  assert.equal(row.name, "台積電");
});

test("a stock outside the closes is kept but has no value", () => {
  const s = state([buy("9999", "2026-09-02", 10, 1000)], [deposit("2026-09-01", 20000)]);
  const summary = P.summary(s, ctx(), { history: {} });
  assert.deepEqual(summary.unquoted, ["9999"]);
  assert.equal(summary.rows[0].value, null);
  assert.equal(summary.marketValue, 0);
});

// --- dividends ----------------------------------------------------------------------------------

test("cash dividends go to whoever held before the ex-date; stock dividends add shares at no cost", () => {
  const s = state(
    [buy("2330", "2026-09-02", 50, 1000, { fee: 0 }), buy("2330", "2026-09-10", 50, 100, { fee: 0 })],
    [deposit("2026-09-01", 100000)]
  );
  const events = [
    { code: "2330", date: "2026-09-01", cash: 9, stock_per_share: 0 }, // before the buy
    { code: "2330", date: "2026-09-10", cash: 2.5, stock_per_share: 0.1 },
    { code: "2330", date: "2026-09-20", cash: null, stock_per_share: null, note: "單位待確認" },
  ];
  const book = P.replay(s, ctx({ events }));
  const p = book.positions["2330"];
  // the 100 shares bought on the ex-date get nothing; 1000 held → 2500 cash, 100 new shares
  assert.equal(p.shares, 1000 + 100 + 100);
  close(p.dividends, 2500);
  close(p.cost, 55000);
  close(p.cost / p.shares, 55000 / 1200);
  close(book.cash, 100000 - 50000 + 2500 - 5000);
  assert.deepEqual(book.pendingStockDividends.map((x) => x.date), ["2026-09-20"]);
});

// --- validation and cash ------------------------------------------------------------------------

test("a trade dated after the data date is refused; so is selling more than held", () => {
  const s = averaging();
  const late = P.validateTrade(buy("2330", "2026-10-17", 700, 1), s, ctx());
  assert.ok(late.some((e) => e.includes("資料日期 2026-10-16")));
  assert.deepEqual(P.validateTrade(buy("2330", "2026-10-16", 700, 1), s, ctx()), []);
  const over = P.validateTrade(sell("2330", "2026-09-05", 700, 1501), s, ctx());
  assert.deepEqual(over, ["賣出股數超過當時的持股"]);
  const early = P.validateTrade(sell("2330", "2026-09-02", 700, 1001), s, ctx()); // before 2nd buy
  assert.deepEqual(early, ["賣出股數超過當時的持股"]);
  assert.ok(P.validateTrade({ ...buy("23", "2026-09-05", 0, 1.5), side: "x" }, s, ctx()).length >= 4);
});

test("a buy the recorded cash does not cover is a warning, not an error", () => {
  const s = state([], [deposit("2026-09-01", 1000)]);
  const trade = buy("2330", "2026-09-02", 500, 10);
  assert.deepEqual(P.validateTrade(trade, s, ctx()), []);
  assert.deepEqual(P.tradeWarnings(trade, s, ctx()), ["現金為負，請補記入金"]);
});

test("cash with deposits, withdrawals, buys, sells and dividends; over-withdrawals refused", () => {
  const first = deposit("2026-09-01", 100000);
  const flows = [first, deposit("2026-09-15", 20000), withdraw("2026-09-20", 30000)];
  const s = state(
    [buy("2330", "2026-09-02", 50, 1000), sell("2330", "2026-09-20", 60, 500)],
    flows
  );
  const events = [{ code: "2330", date: "2026-09-10", cash: 1, stock_per_share: 0 }];
  const c = ctx({ events });
  const book = P.replay(s, c);
  const expected =
    100000 - (50000 + 71.25) + 1000 + 20000 + (30000 - 42.75 - 90) - 30000;
  close(book.cash, expected);
  assert.deepEqual(P.validateCashFlows(flows, s, c), []);
  // the sale on the 20th pays for that day's withdrawal (sales before withdrawals)
  const tooMuch = [...flows, withdraw("2026-09-21", expected + 1)];
  assert.ok(P.validateCashFlows(tooMuch, s, c)[0].includes("2026-09-21"));
  // lowering a past deposit leaves the later withdrawal uncovered: refused
  const edited = flows.map((f) => (f.id === first.id ? { ...f, amount: 10000 } : f));
  assert.ok(P.validateCashFlows(edited, s, c)[0].includes("2026-09-20"));
  // a smaller edit is accepted and everything is recalculated from it
  const smaller = flows.map((f) => (f.id === first.id ? { ...f, amount: 90000 } : f));
  assert.deepEqual(P.validateCashFlows(smaller, s, c), []);
  close(P.replay({ ...s, cash_flows: smaller }, c).cash, expected - 10000);
  assert.ok(P.validateCashFlows([withdraw("2026-10-18", 1)], s, c)[0].includes("今天"));
});

test("an edit or deletion that breaks later records is reported", () => {
  const s = averaging();
  assert.deepEqual(P.ledgerErrors(s, ctx()), []);
  const noFirstBuy = { ...s, trades: s.trades.slice(1) }; // the sale of 500 still fits in 1000
  assert.deepEqual(P.ledgerErrors(noFirstBuy, ctx()), []);
  const noBuys = { ...s, trades: s.trades.slice(2) };
  assert.ok(P.ledgerErrors(noBuys, ctx())[0].includes("2026-09-04"));
  const noDeposit = { ...s, cash_flows: [withdraw("2026-09-05", 1)] };
  assert.ok(P.ledgerErrors(noDeposit, ctx())[0].includes("2026-09-05"));
});

// --- returns ------------------------------------------------------------------------------------

// 100 → 80 → 121: buying more at 80 helps the money-weighted return, not the time-weighted one
const PATH = {
  history: {
    2330: [["2025-01-01", 100], ["2025-01-02", 100], ["2025-06-30", 80], ["2025-07-01", 80], ["2025-12-31", 121]],
  },
};
const returnsCtx = ctx({ dataDate: "2025-12-31", closes: { 2330: ["台積電", 121] } });

test("time-weighted return does not depend on when money came in; XIRR does", () => {
  const once = state([buy("2330", "2025-01-02", 100, 100, { fee: 0 })], [deposit("2025-01-02", 10000)]);
  const twice = state(
    [buy("2330", "2025-01-02", 100, 100, { fee: 0 }), buy("2330", "2025-07-01", 80, 100, { fee: 0 })],
    [deposit("2025-01-02", 10000), deposit("2025-07-01", 8000)]
  );
  const out = state(
    [buy("2330", "2025-01-02", 100, 200, { fee: 0 }), sell("2330", "2025-07-01", 80, 100, { fee: 0, tax: 0 })],
    [deposit("2025-01-02", 20000), withdraw("2025-07-01", 8000)]
  );
  const results = [once, twice, out].map((s) => P.summary(s, returnsCtx, PATH));
  for (const r of results) close(r.twr, 0.21);
  const [a, b, c] = results.map((r) => r.xirr);
  assert.ok(b > a + 0.05 && c < a - 0.05, `${a} ${b} ${c}`); // more money at the dip, or less
  close(results[0].totalAssets, 12100);
  close(results[1].totalGain, 24200 - 18000);
  close(results[2].totalGain, 12100 - 20000 + 8000);
});

test("XIRR solves a one-year 10% gain and refuses a single day", () => {
  close(P.xirr([{ date: "2025-01-01", amount: -100 }, { date: "2026-01-01", amount: 110 }]), 0.1, 6);
  assert.equal(P.xirr([{ date: "2025-01-01", amount: -100 }, { date: "2025-01-01", amount: 110 }]), null);
});

test("a holding without a close on a flow date is valued at its trade price and flagged", () => {
  const s = state([buy("1234", "2025-01-02", 10, 1000, { fee: 0 })], [deposit("2025-01-02", 10000)]);
  const r = P.timeWeighted(s, returnsCtx, PATH);
  assert.equal(r.estimated, true);
  close(r.twr, 0);
});

test("the 0050 comparison buys whole shares with the same fees on the flow days", () => {
  const s = state([], [deposit("2025-01-02", 100000)]);
  const prices = { history: { "0050": [["2025-01-02", 100], ["2025-12-31", 121]] } };
  const etf = P.etfComparison(s, returnsCtx, prices);
  // 998 shares: 99800 + 142.215 fee fits in 100000, 999 do not
  assert.equal(etf.shares, 998);
  close(etf.cash, 100000 - 99800 - 99800 * 0.001425);
  close(etf.value, 998 * 121 + etf.cash);
  close(etf.twr, etf.value / 100000 - 1);
  const early = state([], [deposit("2024-12-01", 100000)]);
  assert.deepEqual(P.etfComparison(early, returnsCtx, prices), {
    code: "0050",
    missing: "2024-12-01",
    since: "2025-01-02",
  });
  const withdrawal = state([], [deposit("2025-01-02", 100000), withdraw("2025-12-31", 50000)]);
  const after = P.etfComparison(withdrawal, returnsCtx, prices);
  assert.ok(after.cash >= -1e-9 && after.shares < 998);
  close(after.gain, after.value - 50000);
  const tooMuch = state([], [deposit("2025-01-02", 100000), withdraw("2025-12-31", 130000)]);
  assert.equal(P.etfComparison(tooMuch, returnsCtx, prices).shortfall, "2025-12-31");
});

// --- snapshots and reminders ----------------------------------------------------------------------

const HISTORY = [
  ["2026-09-30", { status: "ok", tiers: { 300000: 82 }, aspects: { revenue: "STABLE", profitability: "STABLE", risk: "REGIME_SHIFT" } }],
  ["2026-10-15", { status: "blocked" }],
  ["2026-10-16", { status: "ok", tiers: { 300000: 64 }, aspects: { revenue: "STABLE", profitability: "UNSTABLE", risk: "UNSTABLE" } }],
];

test("the snapshot is the latest export on or before the date, never a later one", () => {
  assert.equal(P.snapshotFor(HISTORY, "2026-09-29"), null); // 無買進時紀錄
  const bought = P.snapshotFor(HISTORY, "2026-10-14");
  assert.equal(bought.as_of, "2026-09-30");
  assert.equal(P.scoreAt(bought, 300000), 82);
  assert.equal(P.scoreAt(bought, 50000), null);
  assert.equal(P.scoreAt(P.snapshotFor(HISTORY, "2026-10-15"), 300000), null);
});

test("sell reminders: own stop-loss, aspects turned unstable, score drop, blocked, concentration", () => {
  const params2 = params;
  const row = { returnPct: -0.2 };
  const bought = P.snapshotFor(HISTORY, "2026-10-01");
  const now = P.snapshotFor(HISTORY, "2026-10-16");
  const s = state([], [], { stop_loss: -0.15 });
  const alerts = P.sellAlerts(row, bought, now, 0.3, 300000, s, params2);
  const reasons = alerts.map((a) => a.reason);
  assert.equal(alerts[0].action, "考慮賣出");
  assert.equal(alerts[0].level, "high");
  assert.ok(reasons.some((r) => r.startsWith("獲利由買進時的穩定轉為不穩定")));
  assert.ok(reasons.some((r) => r.startsWith("風險由買進時的結構轉變轉為不穩定")));
  assert.ok(reasons.some((r) => r.includes("低 18 分（82 → 64）")));
  assert.ok(reasons.some((r) => r.includes("超過 25.0%")));
  assert.ok(!reasons.some((r) => r.startsWith("營收")));
  // a price fall alone is never a reason
  assert.deepEqual(P.sellAlerts(row, bought, bought, 0.1, 300000, state(), params2), []);
  const blocked = P.snapshotFor(HISTORY, "2026-10-15");
  assert.equal(P.sellAlerts(row, bought, blocked, 0.1, 300000, state(), params2)[0].reason, "最新的資料檢查未通過，無法判斷");
  const low = { status: "ok", tiers: { 300000: 44 }, aspects: {} };
  assert.ok(P.sellAlerts(row, null, low, 0.1, 300000, state(), params2)[0].reason.includes("低於 45"));
  const gain = P.sellAlerts({ returnPct: 0.31 }, null, null, null, 300000, state([], [], { take_profit: 0.3 }), params2);
  assert.ok(gain[0].reason.includes("停利"));
});

test("the alert log keeps each reason once per data date", () => {
  const alerts = [{ level: "low", action: "考慮減碼", reason: "x" }];
  const first = P.newAlertLog([], "2330", alerts, "2026-10-16");
  assert.equal(first.length, 1);
  assert.deepEqual(P.newAlertLog(first, "2330", alerts, "2026-10-16"), []);
  assert.equal(P.newAlertLog(first, "2330", alerts, "2026-10-17").length, 1);
});

// --- capital suggestions ----------------------------------------------------------------------------

const TIERS = [50000, 300000, 1000000, 3000000];
const candidate = (code, score, price, cap_reason = null) => ({
  code,
  name: code,
  price,
  tiers: Object.fromEntries(TIERS.map((c) => [String(c), { value: score, label: "中", cap_reason }])),
});
const fallbackCloses = { closes: { "0050": ["元大台灣50", 100] } };

test("allocation: lots plus odd shares, odd lots only when a lot does not fit, cash left exact", () => {
  const candidates = [
    candidate("1101", 90, 50),
    candidate("2330", 85, 1000),
    candidate("2412", 80, 120),
    candidate("2882", 75, 60),
    candidate("2886", 70, 40),
    candidate("9999", 99, 10, "0050 的歷史表現較好"), // capped: left out
    candidate("8888", 59, 10), // below 60: left out
  ];
  const plan = P.allocate(300000, candidates, ctx(fallbackCloses));
  assert.equal(plan.tier.holdings, 5);
  close(plan.reserve, 15000);
  close(plan.target, 57000);
  assert.deepEqual(plan.items.map((x) => x.code), ["1101", "2330", "2412", "2882", "2886"]);
  const cement = plan.items[0];
  assert.deepEqual([cement.shares, cement.lots, cement.odd, cement.oddLotOnly], [1138, 1, 138, false]);
  close(cement.fee, 50000 * 0.001425 + 20); // the odd shares are a second order
  const tsmc = plan.items[1];
  assert.deepEqual([tsmc.shares, tsmc.lots, tsmc.oddLotOnly], [56, 0, true]);
  for (const x of plan.items) assert.ok(x.amount + x.fee <= 57000 + 1e-9);
  assert.equal(plan.fallback, null);
  const spent = plan.items.reduce((s, x) => s + x.amount + x.fee, 0);
  close(plan.spent, spent);
  close(plan.cashLeft, 300000 - spent);
  assert.ok(plan.cashLeft >= 15000);
});

test("allocation: one stock is at most 25%; the rest of a small amount goes to 0050", () => {
  const candidates = [candidate("1101", 90, 20), candidate("2412", 80, 30), candidate("2882", 75, 25)];
  const plan = P.allocate(50000, candidates, ctx(fallbackCloses));
  assert.equal(plan.tier.holdings, 3);
  close(plan.target, 47500 * 0.25);
  for (const x of plan.items) assert.ok(x.amount + x.fee <= 50000 * 0.25);
  assert.equal(plan.fallback.code, "0050");
  assert.ok(plan.fallback.reasons[0].includes("25.0%"));
  assert.ok(plan.fallback.amount + plan.fallback.fee <= 47500 * 0.25 + 1e-9);
  assert.ok(plan.items.every((x) => x.costWarning === null || x.costRatio > params.allocation.cost_ratio_warning));
});

test("allocation: too few candidates → the missing part goes to 0050 with the reason", () => {
  const candidates = [candidate("1101", 90, 50), candidate("2412", 61, 120), candidate("2882", 50, 60)];
  const plan = P.allocate(300000, candidates, ctx(fallbackCloses));
  assert.equal(plan.items.length, 2);
  assert.ok(plan.fallback.reasons[0].includes("只有 2 檔，少 3 檔"));
  // 3 × 57000 for 0050: 1707 shares (1 lot + 707 odd) fit with two fees, 1708 do not
  assert.equal(plan.fallback.shares, 1707);
  close(plan.cashLeft, 300000 - plan.spent);
  assert.ok(plan.cashLeft >= 15000 - 1e-9);
  const none = P.allocate(300000, [], ctx(fallbackCloses));
  assert.ok(none.fallback.reasons[0].includes("只有 0 檔，少 5 檔"));
});

test("allocation with current holdings buys only the gap and never sells", () => {
  const candidates = ["1101", "2412", "2882", "2886", "1216"].map((c, i) => candidate(c, 90 - i, 50));
  const current = [
    { code: "1101", name: "台泥", value: 200000 },
    { code: "3008", name: "大立光", value: 10000 },
  ];
  const plan = P.allocate(90000, candidates, ctx(fallbackCloses), current);
  assert.equal(plan.tier.holdings, 5); // 300000 in total
  assert.ok(!plan.items.some((x) => x.code === "1101"));
  const reductions = Object.fromEntries(plan.reductions.map((r) => [r.code, r]));
  close(reductions["1101"].over, 200000 - plan.target);
  assert.equal(reductions["3008"].reason, "不在這次的建議名單");
  // not enough cash for every gap: all scaled alike, never more than the cash minus reserve
  assert.equal(plan.items.length, 4);
  assert.ok(plan.spent <= 90000 - 15000 + 1e-9);
  close(plan.cashLeft, 90000 - plan.spent);
});

test("allocation scores each candidate at the tier of the amount", () => {
  const c = candidate("1101", 70, 50);
  c.tiers["50000"] = { value: 55, label: "觀察", cap_reason: null }; // a small amount scores lower
  assert.equal(P.allocate(50000, [c], ctx(fallbackCloses)).items.length, 0);
  assert.equal(P.allocate(300000, [c], ctx(fallbackCloses)).items[0].score, 70);
  const capped = candidate("1101", 90, 50, "0050 的歷史表現較好");
  assert.equal(P.allocate(300000, [capped], ctx(fallbackCloses)).items.length, 0);
});

test("candidates come from analysed watchlist rows with a close", () => {
  const stocks = [
    { code: "2330", name: "台積電", status: "ok", score: { available: true, tiers: { 300000: { value: 80 } } } },
    { code: "2317", name: "鴻海", status: "ok", score: { available: false, reason: "x" } },
    { code: "9917", name: "中保科", status: "blocked" },
    { code: "1101", name: "台泥", status: "ok", score: { available: true, tiers: {} } }, // no close
  ];
  const out = P.candidatesFrom(stocks, { 2330: ["台積電", 1000], 2317: ["鴻海", 200] });
  assert.deepEqual(out, [{ code: "2330", name: "台積電", price: 1000, tiers: { 300000: { value: 80 } } }]);
});

test("the calculator works for any stock and amount", () => {
  const r = P.calculator(30000, 25, "2884", state(), params);
  assert.deepEqual([r.shares, r.lots, r.odd], [1197, 1, 197]); // 1198 would exceed by 5.6
  close(r.cashLeft, 30000 - 1197 * 25 - r.fee);
  assert.equal(P.calculator(10, 25, "2884", state(), params), null);
});

test("settings: discount as a fraction, stop-loss and take-profit in percent", () => {
  const base = P.emptyState().settings;
  const ok = P.settingsFromInput({ fee_discount: "0.6", stop_loss: "-15", take_profit: "30", up_color: "green" }, base);
  assert.deepEqual(ok.errors, []);
  assert.deepEqual(ok.settings, { fee_discount: 0.6, stop_loss: -0.15, take_profit: 0.3, up_color: "green" });
  assert.deepEqual(P.settingsForInput(ok.settings), { fee_discount: "0.6", stop_loss: "-15", take_profit: "30", up_color: "green" });
  const blank = P.settingsFromInput({ fee_discount: "", stop_loss: "", take_profit: "", up_color: "red" }, ok.settings);
  assert.deepEqual(blank.settings, base);
  const bad = P.settingsFromInput({ fee_discount: "6", stop_loss: "15", take_profit: "-1", up_color: "red" }, base);
  assert.equal(bad.errors.length, 3);
});

// --- backup -------------------------------------------------------------------------------------------

test("a backup exported and imported again is identical", () => {
  const s = averaging();
  s.settings = { ...s.settings, fee_discount: 0.6, stop_loss: -0.15 };
  s.trades[0].snapshot = { as_of: "2026-09-01", status: "ok", tiers: { 300000: 70 }, aspects: {} };
  s.alert_log = [{ date: "2026-10-16", code: "2330", level: "low", action: "考慮減碼", reason: "x" }];
  s.last_backup = "2026-10-01";
  assert.deepEqual(P.importJSON(P.exportJSON(s)), s);
  assert.throws(() => P.importJSON("{"), /不是有效的 JSON/);
  assert.throws(() => P.importJSON(JSON.stringify({ ...s, version: 2 })), /版本 2/);
  const broken = { ...s, trades: [{ ...s.trades[0], shares: -1 }] };
  assert.throws(() => P.importJSON(JSON.stringify(broken)), /交易紀錄格式不正確/);
  const flows = { ...s, cash_flows: [{ id: "x", date: "2026-01-01", type: "deposit", amount: 0 }] };
  assert.throws(() => P.importJSON(JSON.stringify(flows)), /入出金紀錄格式不正確/);
});

test("CSV lists trades and cash flows and quotes cells that need it", () => {
  const s = state([buy("2330", "2026-09-02", 500, 1000, { fee: 712 })], [{ ...deposit("2026-09-01", 1000), note: '初始, "本金"' }]);
  const lines = P.toCSV(s).trim().split("\n");
  assert.equal(lines.length, 3);
  assert.equal(lines[1], "交易,2026-09-02,2330,買進,500,1000,500000,712,,");
  assert.equal(lines[2], '資金,2026-09-01,,入金,,,1000,,,"初始, ""本金"""');
});

// --- the holdings page --------------------------------------------------------------------------

test("review: rows with alerts at the tier of the total, backup reminder, new cash", () => {
  const first = buy("2330", "2026-09-01", 100, 1000, { fee: 0 });
  first.snapshot = { as_of: "2026-08-31", status: "ok", tiers: { 300000: 90 }, aspects: {} };
  const s = state([first, buy("0050", "2026-09-01", 100, 100, { fee: 0 })], [deposit("2026-09-01", 300000)]);
  const data = {
    params,
    events: [],
    closes: { 2330: ["台積電", 110], "0050": ["元大台灣50", 100] },
    dataDate: "2026-10-16",
    today: "2026-10-17",
    scoresAsOf: "2026-10-16",
    history: { 2330: [["2026-10-16", 110]], "0050": [["2026-08-31", 100], ["2026-10-16", 100]] },
    scores: { 2330: [["2026-10-16", { status: "ok", tiers: { 300000: 70 }, aspects: {} }]] },
  };
  const r = P.review(s, data);
  assert.equal(r.capital, 300000); // 300000 + 10000 gain
  const tsmc = r.held.find((x) => x.code === "2330");
  close(tsmc.share, 110000 / 120000);
  assert.ok(tsmc.alerts.some((a) => a.reason.includes("90 → 70"))); // the saved snapshot
  assert.ok(tsmc.alerts.some((a) => a.reason.includes("占總市值")));
  assert.equal(r.held.find((x) => x.code === "0050").now, null); // no scores for it
  assert.equal(r.backupDue, true);
  assert.equal(r.newCash, null); // the deposit was used by the day's buys
  const later = { ...s, cash_flows: [...s.cash_flows, deposit("2026-10-01", 50000)], last_backup: "2026-10-01" };
  const again = P.review(later, data);
  close(again.newCash, 300000 - 110000 + 50000);
  assert.equal(again.backupDue, false);
  assert.equal(P.backupDue({ ...later, last_backup: "2026-09-16" }, "2026-10-17", params), true); // 31 days
  assert.equal(P.backupDue(state(), "2026-10-17", params), false); // nothing to back up
});
