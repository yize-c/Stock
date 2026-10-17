# stock-research

A command-line research assistant for Taiwan-listed stocks (TWSE and TPEx common shares).

It collects official and third-party data, checks the data carefully, computes fundamentals and
risk metrics, scores each stock against the published principles of well-known investors, and
tests whether those results held both before and after the 2020 pandemic regime change. An LLM
is used only to summarize filings and explain computed numbers. Every claim it makes must cite a
source, and code checks that each citation is real.

> **Status:** stage 10 of 11 (tools that validate the tool: preregistration, known cases,
> monthly walk-forward with a one-shot holdout, forward tracking, expert scoring) done.
> Full requirements: [`docs/stock-research-prompt.md`](docs/stock-research-prompt.md) (Traditional Chinese).

## Data sources and known limitations

| Source | Used for | Status |
|---|---|---|
| TPEx OpenAPI | TPEx listings, today's quotes, ex-rights reference prices, current-month index | Implemented |
| Yahoo Finance (`yfinance`) | Second price source for cross-checks, TAIEX closes | Implemented |
| FinMind API | Raw prices, ex-rights/capital-reduction/split reference prices, statements, monthly revenue, dividends, TAIEX closes and total-return index, trading calendar | Implemented |
| TWSE, MOPS | Official prices, calendar, filings | Not reachable from the cloud dev environment |

Consequences: historical trading days come from TAIEX dates in FinMind/Yahoo rather than the
official TWSE series, and TWSE-listed stocks are cross-checked between FinMind and Yahoo only.
The data-health report states these limitations. Yahoo closes are split-adjusted (Taiwan stock
dividends appear as splits), so periods around such events are excluded from cross-checks.

Further limits found while building the data layer:

- **Adjusted prices are computed, not downloaded.** FinMind's adjusted-price dataset needs a paid
  tier. Each ex-rights/ex-dividend, capital-reduction or split event has an exchange reference
  price; its factor is `reference_price / previous_close`, and earlier prices are multiplied by the
  factors of later events (the exchange's 還原權值 method). Events after `as_of` are not applied.
- **Prices start in 2003.** FinMind's reference prices and the TAIEX total-return index both
  start in 2003, so earlier prices cannot be adjusted or benchmarked and are not used.
- **Emerging-board (興櫃) trading is dropped.** FinMind and Yahoo include a stock's pre-listing
  emerging-board prices. The listing date is the first day in FinMind's PER/PBR series (which
  comes from the exchanges' files for listed stocks only) on or after the first trade. TPEx
  listing dates are checked against TPEx's official list. Stocks whose listing date cannot be
  derived need an entry in `markets.tw.listing.overrides`.
- **Publication dates are estimates.** Statements have no announcement date in FinMind and only
  the latest version is served, so `available_date` is the statutory deadline and every fetch is
  stored as an immutable snapshot. Monthly revenue uses the earlier of FinMind's storage date
  and the statutory day (the 10th).
- **Per-share figures are restated.** FinMind's past EPS is not restated for later stock
  dividends, splits or capital reductions. Each event's share multiplier comes from its
  reference price (`(previous close - cash dividend) / reference price`, which equals
  `1 + stock dividend / par` and is checked against it), and per-share figures are moved to one
  share basis before comparison. Share counts from capital / par are checked against
  net income / EPS, so a wrong par value fails loudly.
- **No central-bank data.** The CBC site is blocked from the development environment, so the
  Sharpe ratio uses the per-period rates in `markets.tw.risk_free.fallback_rates` and fails
  until they cover the window.
- **Trading halts for capital changes** (re-issuing shares after a reduction, split or par-value
  change) are derived from the event data and reported as expected gaps.
- **Capital changes before 2011 are missing from FinMind.** A stock-specific halt of three or
  more trading days before 2011 that no ex-rights event or known suspension explains (often
  followed by a close beyond the price limit) means prices before it cannot be adjusted. The
  price history then starts on the resumption day and the data-health report says so
  (`data/early_changes.py`).

## Design principles

1. **Code computes numbers; the LLM only interprets them.** The LLM never produces prices,
   financial figures or ratios.
2. **Fail loudly.** Missing, stale or anomalous data raises an error. The tool never silently
   falls back to old data or defaults.
3. **Traceable.** Every LLM conclusion cites a source document, and code checks the quote.
4. **No price prediction.** There are no forecasts, target prices, buy/sell signals or
   "probability of rising" numbers.
5. **Historical statistics, not probabilities.** Risk metrics are labeled as historical values.
6. **No look-ahead.** Every analysis takes an `as_of` date and sees only data published by then.
7. **Not investment advice.** Every report ends with a disclaimer.

## Non-goals

- No automated trading or broker integration
- No target prices and no claims that a price will rise or fall
- The 0-100 score and its labels (強／中／觀察／弱) are a **historical backtest reference**
  (rule changed by the user on 2026-10-17): calibrated out of sample on the whole market,
  shown with the group's historical average net of costs, its interval and sample size, and
  published only if the calibration passes its gate. They are historical statistics, not advice
- Framework scores and stability verdicts are descriptions, not recommendations
- The stability-verdict validation never uses subsequent price moves as its measure of success

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in FINMIND_TOKEN and ANTHROPIC_API_KEY
```

## Usage

```bash
uv run stock-research config show     # validate config.yaml and print it
uv run stock-research validate 2330   # run every data check and print the data-health report
uv run stock-research validate 2330 --as-of 2023-12-31
```

`validate` needs `FINMIND_TOKEN`. It exits with code 1 when any check fails; analysis must not
continue on that data.

```bash
uv run stock-research analyze 2330                       # full report -> reports/2330_<as_of>.md
uv run stock-research analyze 2330 2454 2317 --compare   # one report each + a comparison table
uv run stock-research analyze 2330 --window post_2020    # one window only
uv run stock-research analyze 2330 --as-of 2023-12-31    # point-in-time run
uv run stock-research analyze 2330 --only-unstable       # list only UNSTABLE / REGIME_SHIFT items
uv run stock-research analyze 2330 --no-peers            # skip industry medians (faster)
uv run stock-research analyze 2330 --llm                 # add the LLM sections (needs ANTHROPIC_API_KEY)
uv run stock-research breakpoints 2330                   # structural-break scan (information only)
```

`analyze` runs the data checks first; when any check fails it prints the data-health report and
exits with code 1 without writing an analysis. Reports are in Traditional Chinese and end with
the disclaimer. Sharpe ratios fail loudly until `markets.tw.risk_free.fallback_rates` covers the
window.

Validating the tool itself (spec §14; see [Validation](#validation)):

```bash
uv run stock-research validation run known-cases                 # expected vs tool, accuracy
uv run stock-research validation run walk-forward --phase dev     # one tuning attempt (max 30)
uv run stock-research validation run walk-forward --phase dev --set stability.relative_tolerance=0.2
uv run stock-research validation run walk-forward --phase holdout # once, after preregistration
uv run stock-research validation run point-in-time                # no data after as_of
uv run stock-research validation forward-compare --months 6       # forward log vs what happened
uv run stock-research validation export-expert-forms 2330 2412    # blind forms + tool verdicts
uv run stock-research validation score-experts validation/expert_forms/<date>
```

## Web app (iPhone home screen)

A static web app in `web/` shows the daily analysis of the watchlist (`app.watchlist` in
`config.yaml`). Nothing is computed in the browser: `stock-research export` analyses every
watchlist stock at today's date and writes `site/data/index.json` and
`site/data/stocks/<code>.json`; the app only lays them out. Stocks that fail the data checks
are listed with the failed checks, never with older data, and the app flags a dataset older
than `app.stale_after_hours`.

The app is dark only and shows little at first: the list has code and name, the score for
the chosen capital tier (each tier's score is precomputed by the export; the page only
switches), three dots for the revenue / profitability / risk verdicts, and a star; a stock's
page opens with the score, the three aspects, the overall verdict and up to three main
changes, with everything else in collapsed sections. Phones get one column and stacked tables,
tablets a two-column list, and desktops (1024 px and up) a fixed list on the left with the
stock on the right. The stale-data warning, failed checks and errors are always shown in full.

```bash
uv run stock-research export --out site       # all watchlist stocks (about 10 FinMind requests each)
uv run stock-research export 2330 2603 --out site
cp -r web/. site/ && python -m http.server --directory site   # preview at http://localhost:8000
```

`.github/workflows/daily-export.yml` runs the export at 04:17 Taipei time on weekdays and
deploys `site/` to GitHub Pages. One-time setup: make the repository public (free Pages), add
the `FINMIND_TOKEN` repository secret, set Settings → Pages → Source to "GitHub Actions", and
merge the workflow into the default branch (scheduled workflows only run there). On the
iPhone, open the Pages URL in Safari → Share → Add to Home Screen.

Industry peers are skipped in the daily export (`app.with_peers: false`): fetching every peer's
statements does not fit FinMind's free tier, so framework items that need industry medians
show as unavailable.

## 0-100 score and its calibration

The score (spec v2 §4) has a base part that does not depend on capital: investor principles
40 (the V1 framework composite), consistency 20 (V1/V3 agreement on the metrics whose
walk-forward is positive), valuation 15 (today's exchange P/E and P/B as percentiles of the
stock's own 10 years), risk 15 (3-year volatility and drawdown against the total-return
index) and chips 10 (the three institutional investors' buying or selling streak, the
investment trust's change in holdings, and margin use). Parts without data are left
out and the rest scaled to 100. Each capital tier (NT$50k, 300k, 1M, 3M) then multiplies the
base score by a capital factor (trading-cost ratio, odd lots, concentration) and applies the
calibration's cap.

`.github/workflows/calibration.yml` calibrates it on GitHub Actions with a second FinMind
token: every month end from 2021-06-30 to 2025-09-30, all TWSE/TPEx common stocks (including
those delisted since), the most traded half each month, ten score groups, and each stock's
next-12-month return net of one round trip for the tier's position. Base dates from
2024-01-31 on are out of sample; the group table, the caps (no better than the deposit rate →
40; worse than 0050 → 55) and the gate (top group above bottom group, rank correlation ≥ 0.7)
use them only. If the gate fails the score is not published. Rules are logged in
`calibration/CHANGELOG.md`; the result is `calibration/score_calibration.json`.

Robustness (spec v2 P1): the report and the site mark which metrics' walk-forward is positive
(only those count toward the consistency part); the V1 frameworks are re-run with every
half-life at 0.5x and 2x, and conclusions that change are listed (when the shortest and the
longest half-life disagree, the conclusion may only reflect the post-2023 AI rally); and an
in-sample result at least twice the out-of-sample one (the calibration's top-minus-bottom
spread, or the walk-forward AUC above 0.5 against the holdout) raises an overfitting warning.

Each stock's report and page also say, in plain words, which of each investor's principles
the stock meets, misses or cannot be judged on (V1). The expert scoring lists every stock and
question where the expert majority and the tool disagree (`disagreements.csv`).

## Official sources (probe)

The development environment cannot reach TWSE, MOPS, the central bank or DGBAS, but GitHub's
runners can. `.github/workflows/probe-sources.yml` runs `scripts/probe_sources.py` (read-only
GET requests, standard library) and prints status, content type and a preview of each
response. So far it has verified the hand-set listing dates against TWSE's official
`t187ap03_L` data and shown that TWSE's per-stock history starts in 2010 and its
capital-reduction data in 2011.

## Validation

Code in `src/stock_research/validation/`, records in `validation/`:

| File | Purpose |
|---|---|
| `validation/preregistration.md` | Success criteria and definitions, committed before any run. The holdout runs only when it says `Status: FINAL`, states the SHA-256 of the committed `config.yaml`, and is itself committed and unmodified |
| `validation/known_cases.yaml` | Expected changes per aspect (revenue, profitability, risk), written before running the tool |
| `validation/results/tuning_log.csv` | Every dev-phase attempt, append-only, at most 30 |
| `validation/results/walk_forward.parquet` | Every (stock, base date, metric) row of every run |
| `validation/results/holdout_lock.json` | Written when the holdout starts; a second holdout run is refused |
| `validation/forward_log/` | Verdicts of each analysis run at today's date, with git commit and config hash; never overwritten |
| `validation/expert_forms/<date>/` | Blank form, the tool's verdicts at the same date, filled forms and scores |

Walk-forward in brief: each month-end base date runs the tool with `--as-of`; for every aspect
metric the change over the next three months (next published TTM for statements) is divided by
the §5 tolerance. Success is never defined by price moves: UNSTABLE rows must show larger
normalized changes than STABLE / REGIME_SHIFT rows (one-sided Mann-Whitney, block bootstrap
by stock and quarter) and the tool's AUC must beat the 95th percentile of shuffled verdicts.
Dev outcomes stop at the end of the quarantine (checked when the config loads).

## Claude Code settings

`.claude/` holds the project settings for Claude Code:

| Item | Purpose |
|---|---|
| `settings.json` | Allows pytest/ruff/uv sync; asks before `git push`, `uv add`, network tools and CLI commands that fetch data; denies reading `.env` |
| `hooks/format_and_test.py` | After a Python edit: `ruff format` and the tests that import the module |
| `hooks/guard_secrets.py` | Blocks `git commit` if the changes contain `.env`, a token value, or a key-shaped string |
| `hooks/protect_forward_log.py` | Blocks changes to existing files in `validation/forward_log/` |
| `agents/data-auditor.md` | Read-only audit of the data layer for look-ahead and missing checks |
| `agents/report-reviewer.md` | Checks reports for predictions, advice, unverified claims and the disclaimer |
| `agents/validation-reviewer.md` | Checks validation runs against `validation/preregistration.md` |
| `skills/stock-analysis/SKILL.md` | Standard procedure for analyzing one stock |
| `commands/analyze.md` | `/analyze <code>`: run the analysis, then the report reviewer |

The hooks use only the Python standard library and are tested in `tests/test_hooks.py`.

## Configuration

All thresholds, half-lives, tolerances, framework rules and weights live in
[`config.yaml`](config.yaml). Python code does not hard-code them. The file is validated on load,
and unknown keys, missing keys or inconsistent values (for example, weights that do not sum to 1)
raise an error. Ratios are stored as fractions (`0.15` = 15%).

Secrets are read from `.env` or the environment, never from `config.yaml`.

## Methodology

### Time windows

| Window | Name | Prices | Financial statements |
|---|---|---|---|
| V1 | `post_2020` | 2020-01-01 to `as_of` | Periods ending after 2020-01-01, published by `as_of` |
| V2 | `full` | Earliest to `as_of` | 2013 (IFRS) onward, published by `as_of` |
| V3 | `pre_2020` | Earliest to 2019-12-31 | 2013 to periods ending by 2019-12-31 |

V1 and V3 do not overlap and are the primary comparison. V2 contains V1, so it is shown for
reference only. Financial figures are assigned to a window by period end. Text sent to the LLM is
assigned by publication date, so V3 never sees documents that may mention the pandemic. A TTM
value belongs to a window only if all four of its quarters do.

### Stability check

Each metric gets one verdict: `STABLE`, `REGIME_SHIFT`, `UNSTABLE` or `INSUFFICIENT`. Two
comparisons decide it:

- **A. V1 vs V3.** The economic gap uses relative tolerance for level metrics and percentage-point
  tolerance for ratios. Statistical tests are Welch's t-test (mean returns), Brown-Forsythe
  (volatility), Chow test with HAC errors (beta), and a HAC regression with a period dummy
  (monthly revenue YoY). Benjamini-Hochberg controls the false discovery rate. Accounting metrics
  use the economic gap only (`ECON_ONLY`).
- **B. Inside V1.** V1 is split at its midpoint and compared the same way, both with and without
  the 2020 crash.

### LLM layer

The LLM summarises filings, gives information-only moat and inversion notes, and explains
computed numbers in plain Traditional Chinese. It never produces numbers of its own.

- Documents are local files `data/documents/<ticker>/<YYYY-MM-DD>_<title>.(pdf|txt|md)` (the
  date is the publication date; MOPS is not reachable from the development environment). PDF
  text that fails quality checks is not sent. V3 only reads documents published by 2019-12-31.
- Output follows a JSON schema; every claim carries a verbatim quote, document id and date.
- Code checks each quote against the cited document (after NFKC, whitespace and punctuation
  normalisation; never Traditional/Simplified conversion) and each number, including Chinese
  numerals and 兆/億/萬 units, against the document or computed values. Failing claims are
  marked unverified and kept for the report appendix.
- `claude-opus-5-5` takes no temperature; `effort` is configured instead. Refusals use the
  API's default fallback; a failed call is retried once, then reported as failed. Every call
  is logged to `logs/llm/`.

### Investor frameworks

Seven rule checklists run once per window: Warren Chou, Guyu and Chen Chung-Ming (Taiwan, 60%)
and Buffett, Graham, Lynch and Munger (international, 40%). Each scored item ends as PASS, FAIL,
INSUFFICIENT or NOT_APPLICABLE; a framework's score is PASS / (PASS + FAIL). N/A frameworks are
renormalised inside their group before the groups are combined. LLM judgements (moat, inversion)
are information only. The score describes fit with these investors' principles; it is not a
recommendation.

- Level rules use each window's weighted TTM value (V2 unweighted); "every year" rules use all
  fiscal years in V1/V3 (at least 5) and the latest 10 in V2; valuation rules use the window's
  valuation date (V3: 2019-12-31, with only figures published by then).
- Industry medians (Chen Chung-Ming, Munger) are computed from every listed peer in the same
  exchange industry, the same way and in the same window as the stock. Peer statements are
  cached for 7 days; a peer that cannot be loaded makes the median unavailable rather than
  being skipped. The first run for a large industry can take hours under FinMind's free quota.
- Buffett's debt rule uses interest-bearing borrowings / equity; Warren Chou's core EPS is
  operating income per share.
- Warren Chou's inflation comparison needs CPI. The statistics bureau (DGBAS) is blocked from
  the development environment, so that item reports insufficient data until a CPI source exists.

### Metric definitions

Ratios are fractions. TTM and annual values sum four consecutive quarters; balance-sheet
averages use only the quarter ends inside the same period, so no value reaches outside its
window. Monthly revenue is always compared year over year, and a YoY value belongs to a window
only if both months do (V1's first is January 2021).

| Metric | Definition |
|---|---|
| ROE | net income attributable to parent / mean parent equity |
| ROIC | operating income x (1 - effective tax rate) / mean (equity + borrowings) |
| Free cash flow | operating cash flow - purchases of property, plant and equipment |
| Interest coverage | (pre-tax income + interest expense) / interest expense |
| Volatility | daily-return standard deviation x sqrt(252); V1/V3 also EWMA (90-day half-life) |
| Beta | cov(stock, TAIEX total-return index) / var(index), daily returns |
| VaR 95% | 5th percentile of daily returns, reported as a loss |

Risk metrics are historical statistics over the stated dates, not probabilities.

Implementation details:

- Gaps: percentage points for ratios (per-metric `absolute_tolerance`), relative to the larger
  magnitude for levels, `score_tolerance` for framework and composite scores, equality for
  pass/fail items. A gap exactly at the tolerance counts as small; opposite signs never do.
- Benjamini-Hochberg runs separately for step A, step B with the crash and step B without it.
- Step B: V1 is unstable only when both the with-crash and without-crash comparisons are; it is
  stable when either is; otherwise UNKNOWN (which makes an inconsistent metric UNSTABLE).
  Framework items, scores and single-date valuation metrics have no crash-excluded variant.
- Framework items in V1's halves usually lack the five fiscal years the year rule needs, so an
  item that differs between V1 and V3 is usually UNSTABLE rather than REGIME_SHIFT.
- `LOW_SAMPLE` marks single-date metrics and metrics with fewer than 8 observations in V1 or V3;
  `ECON_ONLY` marks metrics without a test (or with too few observations for one).
- HAC standard errors use the Newey-West lag rule `floor(4 (n / 100) ^ (2/9))`.

A Quandt-Andrews sup-F test also reports the most likely break date. It is shown for information
only and does not change the verdict.

### References

**Stability testing**
- Haggard, K. S. & Witte, H. D. (2012). Subperiod robustness checks: testing for effect mean
  stationarity. *Managerial Finance* 38(5). https://doi.org/10.1108/03074351211217841
- McLean, R. D. & Pontiff, J. (2016). Does Academic Research Destroy Stock Return
  Predictability? *Journal of Finance* 71(1). https://doi.org/10.1111/jofi.12365
- Wielechowski, M. (2020). Chow-test evidence of a pandemic structural break in the S&P 500.
  https://doi.org/10.22630/esare.2020.4.4
- Structural Breaks in Interactive Effects Panels and the Stock Market Reaction to COVID-19.
  arXiv:2111.03035. https://arxiv.org/pdf/2111.03035
- Andrews, D. W. K. (1993). Tests for Parameter Instability and Structural Change with Unknown
  Change Point. *Econometrica* 61(4).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. *Journal of the
  Royal Statistical Society B* 57(1).
- FactSet. Understanding Regime Changes for Robustness in Backtesting.
  https://insight.factset.com/understanding-regime-changes-for-robustness-in-backtesting

**Taiwanese investor principles**
- Warren Chou (周文偉): Mirror Media, 2019-12-25. https://www.mirrormedia.mg/story/20191225fin010
- Guyu (股魚): Business Today.
  https://www.businesstoday.com.tw/article/category/80394/post/201502050042/
- Chen Chung-Ming (陳重銘): Smart.
  https://smart.businessweekly.com.tw/Reading/IndepArticle.aspx?id=34617

**Validation**
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov=stock_research
```

Tests never call real APIs. They use fixtures and mocks only.

## Disclaimer

This tool is for research only and does not constitute investment advice.
本工具僅供研究，不構成投資建議。
