// Holdings calculations for the "我的持股" and "本金試算" pages (user decisions 2026-10-17).
//
// The only place in web/ that computes numbers, and only on the viewer's own records (trades,
// deposits, withdrawals), which never leave the browser. Every rate and threshold comes from
// the exported data/params.json; prices come from the exported official closes (TWSE / TPEx
// open data, closing prices, not real-time). Nothing here predicts prices.
//
// Works as a browser script (global `Portfolio`) and as a Node module for the unit tests
// (tests/js/portfolio.test.js).
"use strict";

const Portfolio = (() => {
  const VERSION = 1;
  const EPS = 1e-9;

  function emptyState() {
    return {
      version: VERSION,
      trades: [],
      cash_flows: [],
      settings: { fee_discount: null, stop_loss: null, take_profit: null, up_color: "red" },
      alert_log: [],
      last_backup: null,
    };
  }

  const given = (x) => x !== null && x !== undefined && x !== "";
  const byDate = (a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0);

  // --- costs ---------------------------------------------------------------------------------

  function isEtf(code, params) {
    return params.etf_code_prefixes.some((prefix) => code.startsWith(prefix));
  }
  function discountOf(state, params) {
    const own = state.settings.fee_discount;
    return given(own) ? Number(own) : params.costs.default_fee_discount;
  }
  /** Broker fee on one order (spec: max(min_fee, amount × rate × discount)). */
  function feeFor(amount, discount, params) {
    return Math.max(params.costs.min_fee, amount * params.costs.fee_rate * discount);
  }
  /** Securities transaction tax on a sell. */
  function taxFor(amount, code, params) {
    return amount * (isEtf(code, params) ? params.costs.tax_etf : params.costs.tax_stock);
  }
  /** A trade's fee and tax: the entered ones, or estimates (flagged). */
  function tradeCosts(trade, state, params) {
    const amount = trade.price * trade.shares;
    const fee = given(trade.fee) ? Number(trade.fee) : feeFor(amount, discountOf(state, params), params);
    let tax = 0;
    if (trade.side === "sell") {
      tax = given(trade.tax) ? Number(trade.tax) : taxFor(amount, trade.code, params);
    }
    const estimated = !given(trade.fee) || (trade.side === "sell" && !given(trade.tax));
    return { amount, fee, tax, estimated };
  }

  // --- validation ----------------------------------------------------------------------------

  /** Problems that block saving a trade. */
  function validateTrade(trade, state, ctx) {
    const errors = [];
    if (!/^\d{4,6}[A-Z]?$/.test(trade.code || "")) errors.push("代號格式不正確");
    if (trade.side !== "buy" && trade.side !== "sell") errors.push("請選擇買進或賣出");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(trade.date || "")) errors.push("日期格式不正確");
    else if (trade.date > ctx.dataDate) errors.push(`日期不可晚於資料日期 ${ctx.dataDate}`);
    if (!(trade.price > 0)) errors.push("價格必須大於 0");
    if (!Number.isInteger(trade.shares) || trade.shares <= 0) errors.push("股數必須是正整數");
    if (given(trade.fee) && !(Number(trade.fee) >= 0)) errors.push("手續費不可為負");
    if (given(trade.tax) && !(Number(trade.tax) >= 0)) errors.push("稅不可為負");
    if (errors.length) return errors;
    const book = replay(withTrade(state, trade), ctx);
    if (book.oversold.length) errors.push("賣出股數超過當時的持股");
    return errors;
  }

  /** Warnings that do not block saving: a buy the recorded cash does not cover (the viewer
   *  may not have entered the deposit yet). */
  function tradeWarnings(trade, state, ctx) {
    const book = replay(withTrade(state, trade), ctx);
    return book.negativeCashDates.length ? ["現金為負，請補記入金"] : [];
  }

  function withTrade(state, trade) {
    return { ...state, trades: [...state.trades.filter((t) => t.id !== trade.id), trade] };
  }

  /** The whole list of deposits and withdrawals after an add, edit or delete: a withdrawal
   *  larger than the cash at its date is refused, also when it is caused by changing an
   *  earlier deposit. */
  function validateCashFlows(flows, state, ctx) {
    const errors = [];
    for (const f of flows) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(f.date || "")) errors.push("日期格式不正確");
      else if (ctx.today && f.date > ctx.today) errors.push(`日期不可晚於今天 ${ctx.today}`);
      if (!(f.amount > 0)) errors.push("金額必須大於 0");
      if (f.type !== "deposit" && f.type !== "withdraw") errors.push("請選擇入金或出金");
    }
    if (errors.length) return errors;
    const book = replay({ ...state, cash_flows: flows }, ctx);
    if (book.uncoveredWithdrawals.length) {
      errors.push(`出金金額大於當時的現金餘額（${book.uncoveredWithdrawals.join("、")}）`);
    }
    return errors;
  }

  // --- the ledger ----------------------------------------------------------------------------

  // Order within one day: deposits, ex-rights events (whoever held the shares before the
  // ex-date gets them), trades in entry order, then withdrawals (so a deposit can pay for the
  // day's buys and a sale for the day's withdrawal).
  function timeline(state, events) {
    const items = [];
    state.cash_flows.forEach((f, i) =>
      items.push({ date: f.date, rank: f.type === "deposit" ? 0 : 3, i, kind: "flow", f })
    );
    (events || []).forEach((e, i) => items.push({ date: e.date, rank: 1, i, kind: "event", e }));
    state.trades.forEach((t, i) => items.push({ date: t.date, rank: 2, i, kind: "trade", t }));
    return items.sort((a, b) => byDate(a, b) || a.rank - b.rank || a.i - b.i);
  }

  /**
   * Replays every record up to ``ctx.until`` (default: all). Moving-average cost: a buy adds
   * its amount and fee to the cost; a sell removes the average cost of the shares sold; a
   * stock dividend adds shares at no cost (the average falls). Cash dividends are booked on
   * the ex-date (they are paid some weeks later).
   */
  function replay(state, ctx) {
    const params = ctx.params;
    const positions = {};
    const book = {
      positions,
      cash: 0,
      deposits: 0,
      withdrawals: 0,
      realized: 0,
      dividends: 0,
      fees: 0,
      taxes: 0,
      negativeCashDates: [],
      uncoveredWithdrawals: [],
      oversold: [],
      pendingStockDividends: [],
      estimatedCosts: 0,
    };
    const pos = (code) =>
      (positions[code] ||= {
        code,
        shares: 0,
        cost: 0,
        realized: 0,
        dividends: 0,
        opened: null, // first buy since the position was last empty
        trades: [],
      });
    for (const item of timeline(state, ctx.events)) {
      if (ctx.until && item.date > ctx.until) break;
      if (item.kind === "flow") {
        const f = item.f;
        if (f.type === "deposit") {
          book.cash += f.amount;
          book.deposits += f.amount;
        } else {
          if (f.amount > book.cash + EPS) book.uncoveredWithdrawals.push(f.date);
          book.cash -= f.amount;
          book.withdrawals += f.amount;
        }
      } else if (item.kind === "event") {
        const e = item.e;
        const p = positions[e.code];
        if (!p || p.shares <= 0) continue;
        if (e.cash) {
          const amount = p.shares * e.cash;
          p.dividends += amount;
          book.dividends += amount;
          book.cash += amount;
        }
        if (!given(e.stock_per_share)) {
          book.pendingStockDividends.push({ code: e.code, date: e.date, note: e.note });
        } else if (e.stock_per_share > 0) {
          p.shares += Math.floor(p.shares * e.stock_per_share); // fractions are paid in cash
        }
      } else {
        const t = item.t;
        const p = pos(t.code);
        if (t.side === "sell" && t.shares > p.shares) {
          book.oversold.push(t.date);
          continue;
        }
        const c = tradeCosts(t, state, params);
        p.trades.push(t);
        book.fees += c.fee;
        book.taxes += c.tax;
        if (c.estimated) book.estimatedCosts += 1;
        if (t.side === "buy") {
          if (p.shares === 0) p.opened = t;
          p.shares += t.shares;
          p.cost += c.amount + c.fee;
          book.cash -= c.amount + c.fee;
          if (book.cash < -EPS) book.negativeCashDates.push(t.date);
        } else {
          const sold = (p.cost / p.shares) * t.shares;
          const proceeds = c.amount - c.fee - c.tax;
          p.realized += proceeds - sold;
          book.realized += proceeds - sold;
          p.cost -= sold;
          p.shares -= t.shares;
          if (p.shares === 0) p.cost = 0;
          book.cash += proceeds;
        }
      }
    }
    return book;
  }

  // --- valuation -----------------------------------------------------------------------------

  /** Last close on or before ``date`` from a history list of [date, close]. */
  function closeOn(history, date) {
    let found = null;
    for (const [d, c] of history || []) {
      if (d > date) break;
      found = c;
    }
    return found;
  }

  /** Market value of the positions on ``date``; a missing close falls back to the last
   *  trade price on or before that date (flagged), never to a later price. */
  function valueOn(book, date, prices) {
    let value = 0;
    let estimated = false;
    for (const p of Object.values(book.positions)) {
      if (p.shares <= 0) continue;
      let price = closeOn(prices.history[p.code], date);
      if (price === null) {
        const traded = p.trades.filter((t) => t.date <= date);
        price = traded.length ? traded[traded.length - 1].price : null;
        estimated = true;
      }
      if (price === null) return { value: null, estimated: true };
      value += price * p.shares;
    }
    return { value, estimated };
  }

  /**
   * Per holding: shares, average cost, market value at the latest close, unrealized P&L after
   * the estimated cost of selling (fee + tax), realized P&L and dividends.
   */
  function holdings(state, ctx) {
    const book = replay(state, ctx);
    const discount = discountOf(state, ctx.params);
    const rows = [];
    for (const p of Object.values(book.positions)) {
      const quote = ctx.closes[p.code];
      const close = quote ? quote[1] : null;
      const row = {
        code: p.code,
        name: quote ? quote[0] : "",
        shares: p.shares,
        cost: p.cost,
        average: p.shares > 0 ? p.cost / p.shares : null,
        realized: p.realized,
        dividends: p.dividends,
        opened: p.opened,
        close,
        quoted: close !== null,
        value: null,
        unrealized: null,
        returnPct: null,
        trades: p.trades,
      };
      if (p.shares > 0 && close !== null) {
        const value = close * p.shares;
        const sellCost = feeFor(value, discount, ctx.params) + taxFor(value, p.code, ctx.params);
        row.value = value;
        row.unrealized = value - sellCost - p.cost;
        row.returnPct = p.cost > 0 ? row.unrealized / p.cost : null;
      }
      rows.push(row);
    }
    return { book, rows };
  }

  // --- returns -------------------------------------------------------------------------------

  /** The day the returns run to: the data date, or a later deposit / withdrawal date. */
  function endDate(state, ctx) {
    return state.cash_flows.reduce((end, f) => (f.date > end ? f.date : end), ctx.dataDate);
  }

  function netFlow(flows, date) {
    return flows
      .filter((f) => f.date === date)
      .reduce((s, f) => s + (f.type === "deposit" ? f.amount : -f.amount), 0);
  }

  /** The calendar day before an ISO date. */
  function dayBefore(date) {
    return new Date(Date.parse(`${date}T00:00:00Z`) - 86400000).toISOString().slice(0, 10);
  }

  /** Chains period returns. Each point is a day with deposits or withdrawals, taken at the
   *  start of the day: ``before`` is the value at the previous close, ``after`` adds the
   *  day's net flow; ``end`` is the value at the last close. */
  function chain(points, end) {
    let growth = 1;
    let base = null;
    for (const p of points) {
      if (base !== null && base > EPS) growth *= p.before / base;
      base = p.after;
    }
    if (base !== null && base > EPS) growth *= end / base;
    return growth - 1;
  }

  /**
   * Time-weighted return: the period is cut at every deposit or withdrawal and the period
   * returns are chained, so the result does not depend on when money was added or taken out.
   * Flows count at the start of their day (valued at the previous close), so the day's
   * trades, fees and price moves belong to the money that is invested after the flow.
   */
  function timeWeighted(state, ctx, prices) {
    const flowDates = [...new Set(state.cash_flows.map((f) => f.date))].sort();
    if (!flowDates.length) return { twr: null, estimated: false };
    let estimated = false;
    const valueAt = (date) => {
      const book = replay(state, { ...ctx, until: date });
      const held = valueOn(book, date, prices);
      estimated ||= held.estimated;
      return held.value === null ? null : held.value + book.cash;
    };
    const points = [];
    for (const date of flowDates) {
      const before = valueAt(dayBefore(date));
      if (before === null) return { twr: null, estimated: true };
      points.push({ before, after: before + netFlow(state.cash_flows, date) });
    }
    const end = valueAt(endDate(state, ctx));
    if (end === null) return { twr: null, estimated: true };
    return { twr: chain(points, end), estimated };
  }

  /** Annual internal rate of ``flows`` ({date, amount}, sorted); null when it cannot be
   *  solved (e.g. all on one day). */
  function xirr(flows) {
    if (flows.length < 2) return null;
    const t0 = Date.parse(flows[0].date);
    const years = flows.map((f) => (Date.parse(f.date) - t0) / (365 * 86400000));
    if (Math.max(...years) <= 0) return null;
    const npv = (r) => flows.reduce((s, f, i) => s + f.amount / Math.pow(1 + r, years[i]), 0);
    let lo = -0.9999;
    let hi = 10;
    if (npv(lo) * npv(hi) > 0) return null;
    for (let k = 0; k < 200; k += 1) {
      const mid = (lo + hi) / 2;
      if (npv(lo) * npv(mid) <= 0) hi = mid;
      else lo = mid;
    }
    return (lo + hi) / 2;
  }

  /** Money-weighted return of the viewer's own flows: deposits out (-), withdrawals in (+),
   *  today's total assets in. */
  function moneyWeighted(state, totalAssets, ctx) {
    const flows = [...state.cash_flows]
      .sort(byDate)
      .map((f) => ({ date: f.date, amount: f.type === "deposit" ? -f.amount : f.amount }));
    flows.push({ date: endDate(state, ctx), amount: totalAssets });
    return xirr(flows);
  }

  /** The sale amount whose proceeds after fee and tax are ``net``. */
  function grossForNet(net, code, discount, params) {
    const tax = isEtf(code, params) ? params.costs.tax_etf : params.costs.tax_stock;
    const gross = net / (1 - params.costs.fee_rate * discount - tax);
    if (gross * params.costs.fee_rate * discount >= params.costs.min_fee) return gross;
    return (net + params.costs.min_fee) / (1 - tax);
  }

  /**
   * The same deposits and withdrawals made in the comparison ETF (0050) on the same days, at
   * its closes, in whole shares (odd lots allowed) with the same fee and the ETF tax; cash
   * that buys no whole share stays as cash. ``missing``: no close yet on that day;
   * ``shortfall``: the ETF account could not pay that withdrawal.
   */
  function etfComparison(state, ctx, prices) {
    const params = ctx.params;
    const code = params.compare_etf;
    const flows = [...state.cash_flows].sort(
      (a, b) => byDate(a, b) || (a.type === "deposit" ? -1 : 1)
    );
    if (!flows.length) return null;
    const history = prices.history[code] || [];
    const since = history.length ? history[0][0] : null;
    const discount = discountOf(state, params);
    const net = (n, price) =>
      n * price - feeFor(n * price, discount, params) - taxFor(n * price, code, params);
    let shares = 0;
    let cash = 0;
    const points = [];
    for (const date of [...new Set(flows.map((f) => f.date))]) {
      const price = closeOn(history, date);
      if (price === null) return { code, missing: date, since };
      const before = shares * (closeOn(history, dayBefore(date)) || 0) + cash;
      points.push({ before, after: before + netFlow(flows, date) });
      for (const f of flows.filter((x) => x.date === date)) {
        if (f.type === "deposit") {
          cash += f.amount;
          const buy = sharesFor(cash, price, discount, params);
          if (buy) {
            shares += buy.shares;
            cash -= buy.amount + buy.fee;
          }
          continue;
        }
        const short = f.amount - cash;
        if (short > EPS) {
          let sell = Math.min(shares, Math.ceil(grossForNet(short, code, discount, params) / price));
          while (sell < shares && net(sell, price) < short - EPS) sell += 1;
          if (net(sell, price) < short - EPS) return { code, shortfall: date };
          shares -= sell;
          cash += net(sell, price);
        }
        cash -= f.amount;
      }
    }
    const value = shares * closeOn(history, endDate(state, ctx)) + cash;
    const deposits = flows.filter((f) => f.type === "deposit").reduce((s, f) => s + f.amount, 0);
    const withdrawals = flows.filter((f) => f.type === "withdraw").reduce((s, f) => s + f.amount, 0);
    return { code, value, shares, cash, gain: value - deposits + withdrawals, twr: chain(points, value) };
  }

  /** Everything the holdings page's first screen shows. */
  function summary(state, ctx, prices) {
    const { book, rows } = holdings(state, ctx);
    const marketValue = rows.reduce((s, r) => s + (r.value || 0), 0);
    const totalAssets = marketValue + book.cash;
    const twr = timeWeighted(state, ctx, prices);
    return {
      book,
      rows,
      marketValue,
      cash: book.cash,
      totalAssets,
      totalGain: totalAssets - book.deposits + book.withdrawals,
      deposits: book.deposits,
      withdrawals: book.withdrawals,
      twr: twr.twr,
      twrEstimated: twr.estimated,
      xirr: moneyWeighted(state, totalAssets, ctx),
      etf: etfComparison(state, ctx, prices),
      unquoted: rows.filter((r) => r.shares > 0 && !r.quoted).map((r) => r.code),
    };
  }

  // --- snapshots and sell reminders ----------------------------------------------------------

  /** The latest export's entry on or before ``date`` (never a later one): {as_of, status,
   *  tiers, aspects}; null when the site has no record that early ("無買進時紀錄"). */
  function snapshotFor(scoreHistory, date) {
    let found = null;
    for (const [d, entry] of scoreHistory || []) {
      if (d > date) break;
      found = [d, entry];
    }
    if (!found) return null;
    const [asOf, entry] = found;
    return {
      as_of: asOf,
      status: entry.status,
      tiers: entry.tiers || null,
      aspects: entry.aspects || null,
    };
  }

  /** A snapshot's score for a capital tier, or null. */
  function scoreAt(snapshot, capital) {
    if (!snapshot || snapshot.status !== "ok" || !snapshot.tiers) return null;
    const score = snapshot.tiers[String(capital)];
    return score === undefined ? null : score;
  }

  /**
   * Reasons to look at a holding again (version A wording). ``bought`` is the snapshot at the
   * position's first buy, ``now`` the latest one, ``share`` the holding's share of the market
   * value, ``capital`` the tier whose scores are compared. Price moves are never a reason,
   * except the viewer's own stop-loss / take-profit.
   */
  function sellAlerts(row, bought, now, share, capital, state, params) {
    const out = [];
    const a = params.alerts;
    const s = state.settings;
    const add = (level, action, reason) => out.push({ level, action, reason });
    if (row.returnPct !== null) {
      if (given(s.stop_loss) && row.returnPct <= s.stop_loss) {
        add("high", "考慮賣出", `觸及自訂停損 ${pct(s.stop_loss)}（目前 ${pct(row.returnPct)}）`);
      }
      if (given(s.take_profit) && row.returnPct >= s.take_profit) {
        add("high", "考慮賣出", `觸及自訂停利 ${pct(s.take_profit)}（目前 ${pct(row.returnPct)}）`);
      }
    }
    if (now && now.status !== "ok") {
      add("mid", "考慮減碼", "最新的資料檢查未通過，無法判斷");
    } else if (now) {
      const names = { revenue: "營收", profitability: "獲利", risk: "風險" };
      const before = bought && bought.status === "ok" ? bought.aspects || {} : {};
      for (const [key, name] of Object.entries(names)) {
        const was = before[key];
        if ((was === "STABLE" || was === "REGIME_SHIFT") && now.aspects && now.aspects[key] === "UNSTABLE") {
          add("mid", "考慮減碼", `${name}由買進時的${was === "STABLE" ? "穩定" : "結構轉變"}轉為不穩定`);
        }
      }
      const then = scoreAt(bought, capital);
      const current = scoreAt(now, capital);
      if (then !== null && current !== null && then - current >= a.score_drop) {
        add("mid", "考慮減碼", `分數比買進時低 ${Math.round(then - current)} 分（${Math.round(then)} → ${Math.round(current)}）`);
      } else if (current !== null && current < a.score_floor) {
        add("mid", "考慮減碼", `分數 ${Math.round(current)}，低於 ${a.score_floor}`);
      }
    }
    if (share !== null && share > a.concentration) {
      add("low", "考慮減碼", `占總市值 ${pct(share)}，超過 ${pct(a.concentration)}`);
    }
    return out;
  }

  /** New entries for the alert log: each code and reason once per data date. */
  function newAlertLog(log, code, alerts, dataDate) {
    const seen = new Set(log.map((e) => `${e.date}|${e.code}|${e.reason}`));
    return alerts
      .filter((x) => !seen.has(`${dataDate}|${code}|${x.reason}`))
      .map((x) => ({ date: dataDate, code, level: x.level, action: x.action, reason: x.reason }));
  }

  function pct(x) {
    return `${(x * 100).toFixed(1)}%`;
  }

  // --- capital suggestions (version A) -------------------------------------------------------

  /** Holdings count for an amount: the largest capital tier not above it (spec 4.4). */
  function holdingsFor(amount, params) {
    const tiers = [...params.capital_tiers].sort((a, b) => a.capital - b.capital);
    let chosen = tiers[0];
    for (const t of tiers) if (t.capital <= amount) chosen = t;
    return chosen;
  }

  /** Fee of buying ``shares``: whole lots and the odd shares are separate orders, each with
   *  its own fee. */
  function orderFees(shares, price, discount, params) {
    const lot = params.costs.board_lot;
    const whole = Math.floor(shares / lot) * lot;
    const odd = shares - whole;
    return (
      (whole ? feeFor(whole * price, discount, params) : 0) +
      (odd ? feeFor(odd * price, discount, params) : 0)
    );
  }

  /** Shares for an allocation: whole lots when one fits, odd shares for the rest, fees
   *  included; null when not even one share fits. */
  function sharesFor(allocation, price, discount, params) {
    const lot = params.costs.board_lot;
    let shares = Math.floor(allocation / price);
    while (shares > 0 && shares * price + orderFees(shares, price, discount, params) > allocation + EPS) {
      shares -= 1;
    }
    if (shares <= 0) return null;
    return {
      shares,
      lots: Math.floor(shares / lot),
      odd: shares % lot,
      oddLotOnly: shares < lot,
      amount: shares * price,
      fee: orderFees(shares, price, discount, params),
    };
  }

  /**
   * Which stocks and how many shares for ``amount`` (version A). Candidates: score ≥
   * min_score and not capped for doing worse than the comparison ETF, highest first; equal
   * weights, each at most max_weight; a cash reserve kept; the part no stock fills (too few
   * candidates, or the weight cap) goes to the fallback ETF with the reason. With
   * ``current`` holdings ({code, name, value}) the targets cover cash plus holdings and only
   * the missing part is bought; positions above target are listed, never sold.
   */
  function allocate(amount, candidates, ctx, current = []) {
    const params = ctx.params;
    const rule = params.allocation;
    const discount = discountOf(ctx.state || emptyState(), params);
    const heldValue = (code) => current.filter((r) => r.code === code).reduce((s, r) => s + (r.value || 0), 0);
    const total = amount + current.reduce((s, r) => s + (r.value || 0), 0);
    const tier = holdingsFor(total, params);
    const reserve = Math.min(amount, total * rule.cash_reserve);
    const budget = amount - reserve;
    const investable = total - reserve;
    const picked = candidates
      .filter(
        (c) =>
          c.code !== rule.fallback_etf &&
          c.score !== null &&
          c.score >= rule.min_score &&
          c.cap_reason !== rule.excluded_cap_reason &&
          c.price > 0
      )
      .sort((a, b) => b.score - a.score)
      .slice(0, tier.holdings);
    const weight = Math.min(1 / tier.holdings, rule.max_weight);
    const target = investable * weight;
    const missing = tier.holdings - picked.length;
    const capLeft = Math.max(0, 1 - weight * tier.holdings);
    const fallbackTarget = investable * (missing * weight + capLeft);

    const reductions = [];
    const needs = picked.map((c) => {
      const over = heldValue(c.code) - target;
      if (over > EPS) reductions.push({ code: c.code, name: c.name, over, reason: "高於目標金額" });
      return Math.max(0, -over);
    });
    const fallbackOver = heldValue(rule.fallback_etf) - fallbackTarget;
    if (fallbackOver > EPS && current.some((r) => r.code === rule.fallback_etf)) {
      reductions.push({ code: rule.fallback_etf, name: "", over: fallbackOver, reason: "高於目標金額" });
    }
    const fallbackNeed = Math.max(0, -fallbackOver);
    for (const r of current) {
      if (r.code !== rule.fallback_etf && !picked.some((c) => c.code === r.code) && (r.value || 0) > 0) {
        reductions.push({ code: r.code, name: r.name, over: r.value, reason: "不在這次的建議名單" });
      }
    }
    // Not enough money for every target (holdings counted in): scale all down alike.
    const wanted = needs.reduce((s, x) => s + x, 0) + fallbackNeed;
    const scale = wanted > budget ? budget / wanted : 1;

    const items = [];
    let spent = 0;
    picked.forEach((c, i) => {
      const buy = needs[i] > 0 ? sharesFor(needs[i] * scale, c.price, discount, params) : null;
      if (!buy) return;
      const ratio = (2 * buy.fee + taxFor(buy.amount, c.code, params)) / buy.amount;
      items.push({
        code: c.code,
        name: c.name,
        score: c.score,
        price: c.price,
        ...buy,
        costRatio: ratio,
        costWarning: ratio > rule.cost_ratio_warning ? "這筆金額太小，手續費占比偏高" : null,
        reasons: c.reasons || [],
      });
      spent += buy.amount + buy.fee;
    });
    let fallback = null;
    const quote = ctx.closes[rule.fallback_etf];
    if (fallbackNeed > EPS && quote) {
      const buy = sharesFor(Math.min(fallbackNeed * scale, budget - spent), quote[1], discount, params);
      if (buy) {
        const why = [];
        if (missing > 0) {
          why.push(`分數 ≥ ${rule.min_score} 且未被標示「${rule.excluded_cap_reason}」的股票只有 ${picked.length} 檔，少 ${missing} 檔`);
        }
        if (capLeft > EPS) why.push(`單一個股上限 ${pct(rule.max_weight)}，其餘部分`);
        fallback = { code: rule.fallback_etf, name: quote[0], price: quote[1], ...buy, reasons: why };
        spent += buy.amount + buy.fee;
      }
    }
    return { tier, target, items, fallback, reductions, spent, reserve, cashLeft: amount - spent };
  }

  /** Version-independent calculator: shares, fee and cost ratio for one stock and amount. */
  function calculator(amount, price, code, state, params) {
    const buy = sharesFor(amount, price, discountOf(state, params), params);
    if (!buy) return null;
    const ratio = (2 * buy.fee + taxFor(buy.amount, code, params)) / buy.amount;
    return { ...buy, costRatio: ratio, cashLeft: amount - buy.amount - buy.fee };
  }

  // --- backup ----------------------------------------------------------------------------------

  function exportJSON(state) {
    return JSON.stringify(state, null, 2);
  }

  /** Parses a backup; refuses other versions and malformed records. */
  function importJSON(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("不是有效的 JSON 備份檔");
    }
    if (!data || data.version !== VERSION) {
      throw new Error(`備份檔版本 ${data && data.version} 不支援（需要 ${VERSION}）`);
    }
    if (!Array.isArray(data.trades) || !Array.isArray(data.cash_flows)) {
      throw new Error("備份檔缺少交易或入出金紀錄");
    }
    for (const t of data.trades) {
      const ok = t.id && t.code && t.date && t.price > 0 && Number.isInteger(t.shares) && t.shares > 0;
      if (!ok || !["buy", "sell"].includes(t.side)) {
        throw new Error(`交易紀錄格式不正確：${JSON.stringify(t).slice(0, 80)}`);
      }
    }
    for (const f of data.cash_flows) {
      if (!f.id || !f.date || !(f.amount > 0) || !["deposit", "withdraw"].includes(f.type)) {
        throw new Error(`入出金紀錄格式不正確：${JSON.stringify(f).slice(0, 80)}`);
      }
    }
    const base = emptyState();
    return { ...base, ...data, settings: { ...base.settings, ...(data.settings || {}) } };
  }

  function csvCell(value) {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  /** Trades and cash flows as CSV for a spreadsheet. */
  function toCSV(state) {
    const header = ["類型", "日期", "代號", "買賣/入出金", "價格", "股數", "金額", "手續費", "稅", "備註"];
    const lines = [header.join(",")];
    for (const t of state.trades) {
      const side = t.side === "buy" ? "買進" : "賣出";
      const row = ["交易", t.date, t.code, side, t.price, t.shares, t.price * t.shares, t.fee, t.tax, ""];
      lines.push(row.map(csvCell).join(","));
    }
    for (const f of state.cash_flows) {
      const kind = f.type === "deposit" ? "入金" : "出金";
      lines.push(["資金", f.date, "", kind, "", "", f.amount, "", "", f.note].map(csvCell).join(","));
    }
    return lines.join("\n") + "\n";
  }

  return {
    VERSION,
    emptyState,
    isEtf,
    discountOf,
    feeFor,
    taxFor,
    tradeCosts,
    validateTrade,
    tradeWarnings,
    validateCashFlows,
    replay,
    closeOn,
    holdings,
    endDate,
    timeWeighted,
    xirr,
    moneyWeighted,
    etfComparison,
    summary,
    snapshotFor,
    scoreAt,
    sellAlerts,
    newAlertLog,
    holdingsFor,
    orderFees,
    sharesFor,
    allocate,
    calculator,
    exportJSON,
    importJSON,
    toCSV,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Portfolio;
