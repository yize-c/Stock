# 逐月 walk-forward 驗證：開發期

- 基準日：2023-12-31 ~ 2025-06-30，共 19 個
- 調參嘗試編號：1
- 目前累計調參嘗試次數：1
- 成功標準不使用股價漲跌（spec §14.4）；後續變動是指標本身相對容忍度的變化。

## 結果

| 項目 | 值 |
|---|---|
| 可用樣本（股票 × 基準日 × 指標） | 3058 |
| 區塊數（股票 × 季度） | 154 |
| 排除：工具判定 INSUFFICIENT | 38 |
| 排除：沒有後續資料 | 0 |
| 判定 UNSTABLE | 1252 |
| 後續變動超過容忍度 | 1679 |
| 工具 AUC（95% 區塊自助法信賴區間） | 0.630（0.603–0.656） |
| 基準 A（全部 STABLE）AUC | 0.500 |
| 基準 B（隨機判定）AUC 的 95% 分位數 | 0.514 |
| Mann-Whitney 效果量 P(UNSTABLE 變動較大)（95% 信賴區間） | 0.695（0.658–0.728） |
| 單尾 p 值（區塊自助法） | 0.000 |
| 結果 | **通過** |

## 各股票 × 基準日的執行狀態

| 狀態 | 次數 |
|---|---|
| OK | 387 |
| DATA_CHECK_FAILED | 69 |

### 未執行的原因

- 2603 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2603 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2603 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2382 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2382 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2382 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 9921 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 9921 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 9921 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2412 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2412 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2412 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 1216 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 1216 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 1216 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2023-12-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-01-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-02-29：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-06-30：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-07-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-08-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-09-30：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-10-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-11-30：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2024-12-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-01-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-02-28：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 9917 2025-06-30：DATA_CHECK_FAILED（prices.cross_source）
- 2317 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2317 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2317 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2454 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2454 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2454 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 1101 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 1101 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 1101 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2023-12-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-01-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-02-29：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-06-30：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-07-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-08-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-09-30：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-10-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-11-30：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2024-12-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-01-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-02-28：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 2881 2025-06-30：DATA_CHECK_FAILED（prices.cross_source）
- 6488 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 6488 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 6488 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 5347 2024-03-31：DATA_CHECK_FAILED（prices.cross_source）
- 5347 2024-04-30：DATA_CHECK_FAILED（prices.cross_source）
- 5347 2024-05-31：DATA_CHECK_FAILED（prices.cross_source）
- 5347 2024-09-30：DATA_CHECK_FAILED（prices.cross_source）

## 依指標

| 指標 | 樣本 | UNSTABLE | 超過容忍度 | 變動中位數 |
|---|---|---|---|---|
| revenue_yoy | 387 | 211 | 279 | 2.23 |
| roe | 387 | 148 | 201 | 1.10 |
| gross_margin | 368 | 111 | 165 | 0.72 |
| operating_margin | 368 | 135 | 165 | 0.78 |
| eps | 387 | 236 | 255 | 1.69 |
| volatility | 387 | 152 | 247 | 1.47 |
| beta | 387 | 168 | 283 | 1.86 |
| max_drawdown | 387 | 91 | 84 | 0.49 |

## 更正

`tuning_log.csv` 記錄的 git commit（5953bbb）是執行結束時的版本；實際執行的程式為 2ef3098。詳見 `tuning_log_corrections.md`。
