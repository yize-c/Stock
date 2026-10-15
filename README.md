# stock-research

A command-line research assistant for Taiwan-listed stocks (TWSE and TPEx common shares).

It collects official and third-party data, checks the data carefully, computes fundamentals and
risk metrics, scores each stock against the published principles of well-known investors, and
tests whether those results held both before and after the 2020 pandemic regime change. An LLM
is used only to summarize filings and explain computed numbers. Every claim it makes must cite a
source, and code checks that each citation is real.

> **Status:** stage 3 of 11 (data layer) in progress. Analysis is not implemented yet.
> Full requirements: [`docs/stock-research-prompt.md`](docs/stock-research-prompt.md) (Traditional Chinese).

## Data sources and known limitations

| Source | Used for | Status |
|---|---|---|
| TPEx OpenAPI | TPEx listings, today's quotes, ex-rights reference prices, current-month index | Implemented |
| Yahoo Finance (`yfinance`) | Second price source for cross-checks, TAIEX closes | Implemented |
| FinMind API | Adjusted prices, statements, monthly revenue, dividends, total-return index | Envelope only; datasets pending |
| TWSE, MOPS | Official prices, calendar, filings | Not reachable from the cloud dev environment |

Consequences: historical trading days come from TAIEX dates in FinMind/Yahoo rather than the
official TWSE series, and TWSE-listed stocks are cross-checked between FinMind and Yahoo only.
The data-health report states these limitations. Yahoo closes are split-adjusted (Taiwan stock
dividends appear as splits), so periods around such events are excluded from cross-checks.

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
- No price predictions, target prices or buy/sell recommendations
- Framework scores and stability verdicts are descriptions, not recommendations
- Validation never uses subsequent price moves as the measure of success

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in FINMIND_TOKEN and ANTHROPIC_API_KEY
```

## Usage

```bash
uv run stock-research config show     # validate config.yaml and print it
```

Planned commands (later stages):

```bash
stock-research analyze 2330
stock-research analyze 2330 2454 2317 --compare
stock-research analyze 2330 --window post_2020
stock-research analyze 2330 --as-of 2023-12-31
stock-research analyze 2330 --only-unstable
stock-research breakpoints 2330
stock-research validate 2330
stock-research validation run known-cases
stock-research validation run point-in-time
stock-research validation export-expert-forms
stock-research validation score-experts
```

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
