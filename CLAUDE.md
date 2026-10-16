# CLAUDE.md — 台股研究助理（stock-research）

## 專案目的
個人投資研究工具兼作品集專案：從零建立的台股研究 CLI。重視**資料正確性、可測試性、可追溯性、可驗證性**。
完整規格見 [`docs/stock-research-prompt.md`](docs/stock-research-prompt.md)（需求變更時一併更新）；本檔整理實作時必須隨時遵守的部分。

## 工作規則
- 依規格第 18 節的階段逐步完成，每階段結束前測試全部通過，並 commit（Conventional Commits）。
- **新增任何第三方套件前先問使用者。**
- 需求不確定或規格互相矛盾時，停下來問，不要自行假設或挑一個。
- API key、token 只放 `.env`，絕對不進 git。
- git push 與網路請求需先詢問。

## 核心原則（所有模組都要遵守）
1. **數字由程式計算，LLM 只負責解讀。** LLM 不能自行產生股價、財務數字或比率。
2. **Fail loudly**：資料沒抓到、過期或異常就明確報錯，不可靜默使用舊資料或預設值。
3. LLM 的每個結論都要能追溯到來源，並由程式驗證引用真實存在。
4. **不做股價預測**：不輸出漲跌預測、目標價、買賣建議、上漲機率。
5. 風險指標一律標註為「歷史統計值」，不是未來機率。
6. **不偷看未來**：所有分析函式接受 `as_of`，只能看到 `available_date <= as_of` 的資料。
7. 所有報告結尾附免責聲明：本工具僅供研究，不構成投資建議。

## 台股特性
- 範圍：上市（TWSE）、上櫃（TPEx）普通股。內部代號 `2330` + 市場別；yfinance 用 `.TW` / `.TWO`。
- 漲跌幅：2015-06-01 前 ±7%、之後 ±10%；新上市（櫃）前 5 個交易日無限制。
- 報酬、波動、回撤一律用**還原權息價格**；顯示現價用原始價格。兩者不可混用。
- 基準：發行量加權股價**報酬**指數（含息）；取不到改用還原後的 0050 並註明。不可用不含息 TAIEX。
- 2013 年起 IFRS，之前的財報不可比較，不納入任何版本。
- 月營收每月 10 日前公告；財報期限 Q1 5/15、Q2 8/14、Q3 11/14、Q4 次年 3/31（放 config）。
- 歷史交易日 = 官方有加權指數收盤的日期（自動排除過去颱風休市）；今天與未來用證交所公告 + config 手動休市。
- 每股數據用調整後股數（分割、面額變更）。

## 資料來源的已知限制（使用者已同意）
- 雲端開發環境連不到**證交所**（防火牆回傳 HTTP 200 的封鎖頁）與 **MOPS**（連線被切斷），相關 fetcher 暫不實作。
- 因此：歷史交易日改用 FinMind／Yahoo 的加權指數日期（與官方櫃買指數比對過一致）；上市股票的交叉比對只有 FinMind vs Yahoo，沒有官方基準。資料健康報告必須註明這些限制。
- 櫃買中心 OpenAPI 只有當日／當月資料；Yahoo 的 Close 已做分割調整（台股配股也被當成分割），成交量與官方不一致，不可當準。
- 已知停牌放 `markets.tw.known_suspensions`，清單內缺資料視為正常，清單外仍報錯。

## 第 3 階段實作時發現的資料限制（使用者已同意）
- FinMind 免費版拿不到還原股價（`TaiwanStockPriceAdj` 需付費），改由程式用除權息、減資、分割的參考價自行還原（`data/adjustment.py`）；`as_of` 之後的事件不套用。
- FinMind 除權息參考價與加權報酬指數都從 2003 年開始 → 股價只用 2003 年起（`data_sources.finmind.price_history_start`）。
- FinMind／Yahoo 股價含上市前的興櫃交易，一律剔除。上市日 = 首次交易後 `TaiwanStockPER` 的第一天（只涵蓋上市櫃股票）；上櫃股票再與櫃買中心官方上櫃日比對；推不出來的放 `markets.tw.listing.overrides`。
- 財報沒有公告日、只有最新版本 → `available_date` 用法定期限、標 `latest_only`，每次抓取存不可覆寫快照。月營收公告日用 FinMind 建檔日與法定期限較早者。
- 減資、分割、面額變更前的停止買賣期間由事件資料推得，視為已知停牌（INFO）。
- 未來休市日取自 FinMind `TaiwanStockTradingDate`（轉載證交所行事曆），歷史部分必須與加權指數收盤日一致。

## 第 4 階段的做法（使用者已同意）
- 每股數據用事件回溯調整（`data/shares.py`）：股數倍數 =（前一日收盤 − 現金股利）÷ 參考價，與「1 + 股票股利 ÷ 面額」交叉驗證；面額上櫃用櫃買官方值、上市預設 10 元（config）並隨面額變更事件調整；「普通股股本 ÷ 面額」與「歸屬母公司淨利 ÷ EPS」的股數比對不一致就報錯。只從 IFRS 起點（2013）開始調整。
- 中央銀行網站（cbc.gov.tw）被雲端環境的網路政策擋下 → Sharpe 用 `markets.tw.risk_free.fallback_rates`，沒填滿版本期間就報錯。
- （待使用者確認）月營收年增率：比較月與基期月都要在同一版本內（V1 第一個為 2021-01，V3 最後一個為 2019-12），與 TTM 規則一致。

## 第 5 階段的做法（使用者已同意）
- CPI：使用者要開通主計總處網域；目前本 session 仍被網路政策擋下 → 華倫「年增率高於通膨」在沒有新高時判未通過，有新高時為「資料不足」，CPI fetcher 等能連線、取得真實回應後再寫。
- 同產業中位數：抓同產業全部上市櫃普通股（不跳過；任一家抓取失敗 → 中位數不可用），同業財報快取 7 天（`markets.tw.industries.peers`）；產業別取最細分類，略過「電子工業」等大類，上市／上櫃不同寫法用 aliases 合併。
- Buffett 負債權益比 = 有息負債（短借＋長借＋公司債）÷ 權益，門檻 0.5。
- 華倫本業 EPS = 營業利益 ÷ 股數（同一股數基準），去年同期 ≤ 0 為不適用。
- 產業群組（金融、營建、半導體、成熟電子、傳產、民生、電信）放 `markets.tw.industries.groups`，為待檢視的預設值。

## 三個版本（TimeWindow）
| 版本 | 名稱 | 股價 | 財報 |
|---|---|---|---|
| V1 | `post_2020` | 2020-01-01 ~ `as_of` | period_end 在 2020-01-01 後且已公告 |
| V2 | `full` | 最早 ~ `as_of` | 2013 起、`as_of` 前已公告 |
| V3 | `pre_2020` | 最早 ~ 2019-12-31 | 2013 起、period_end ≤ 2019-12-31 |

邊界規則：
- 財報**數字**依 `period_end` 分版本；LLM 讀的**文字**依**公告日**分版本（V3 只讀 2019-12-31 前公告的文件）。
- TTM 四季必須全部落在同一版本，否則排除（V1 第一個 TTM 截至 2020Q4，V3 最後一個截至 2019Q4）。
- V1 與 V3 是主要比較；V2 只當參考、不加權、不影響判定。
- 版本資料不足 → `INSUFFICIENT_HISTORY`，不可用其他版本補。跨範圍存取要報錯。
- 只用一個 `TimeWindow` 物件傳入分析函式，不可為三個版本複製程式碼。
- 加權：`weight = 0.5 ** (age_days / half_life_days)`，V1 以 `as_of`、V3 以 2019-12-31 為基準。
  - 數值指標的穩定性判定（經濟差距與統計檢定）用**未加權值**。
  - 框架計分項的通過／未通過用**各版本的加權值**（V2 用未加權值）。
- 季節性：季度一律用 TTM 或 YoY，月營收一律用年增率。

## 穩定性判定表（§5.3）
| V1 vs V3 | V1 內部 | 判定 |
|---|---|---|
| 一致 | — | `STABLE` |
| 不一致 | 穩定 | `REGIME_SHIFT` |
| 不一致 | 不穩定 | `UNSTABLE` |
| 不一致 | `UNKNOWN` | `UNSTABLE` |
| V1 或 V3 資料不足 | — | `INSUFFICIENT` |

- 一致 = 經濟差距小 **且**（有檢定時）BH 校正後不顯著。
- V1 內部：只有「含崩跌」與「排除崩跌」兩種算法**都**不穩定，才算不穩定。
- 財報類指標不做統計檢定，標 `ECON_ONLY`；觀察值 < 8 標 `LOW_SAMPLE`。
- 彙整取最差：`UNSTABLE` > `REGIME_SHIFT` > `STABLE`；`INSUFFICIENT` 另計。

## 框架權重（§6.4）
| 群組 | 群組權重 | 成員 |
|---|---|---|
| 台灣代表 | 60% | 華倫、股魚、陳重銘 各 20% |
| 國外代表 | 40% | Buffett、Graham、Lynch、Munger 各 10% |

- 框架分數 = 通過 ÷（通過 + 未通過）；資料不足、不適用不計入分母；分母為 0 → `N/A`。
- `N/A` 框架先在群組內重新正規化，再以 60/40 合併；整個群組 `N/A` 時只用另一群組並註明。
- LLM 的質化判斷一律是資訊項，不計分。

## 驗證：逐月 walk-forward（§14.4）
- 目的是驗證穩定性判定準不準，**不可用股價漲跌當成功標準**。
- 預測期 H 預設 3 個月；開發期基準日 2023-12-31 ~ 2025-06-30（每月月底）；隔離期 2025-07 ~ 2025-09 不用；保留期 2025-10-31 ~ 2026-06-30。基準日 + H > 今天就不納入。日期全部放 config。
- 每個基準日用 `--as-of` 跑；後續變動 = 股價／月營收看 H 個月後、財報看下一次公告的 TTM，再除以第 5 節容忍度標準化。
- 調參只能用開發期；每次嘗試都寫進 `validation/results/tuning_log.csv`（不可覆寫、不可只留最好的），上限 30 組，優先選穩健的參數區域。
- `preregistration.md` 寫好最終參數與成功標準並 commit 之前，**禁止執行保留期**（程式或 hook 檢查）。保留期只跑一次；沒通過不可回頭調參，改用 §14.5 前瞻追蹤。
- 比較基準：A 全部 STABLE、B 依相同比例隨機判定（1000 次）。工具 AUC 要高於基準 B 的 95 百分位。
- 預測期重疊 → 顯著性用 block bootstrap（以股票、季度為區塊），每個結果附樣本數與信賴區間。
- 輸出 `validation/results/walk_forward.parquet`；CLI：`stock-research validation run walk-forward --phase dev|holdout`。

## 常用指令
```bash
uv sync                                  # 安裝依賴
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov=stock_research       # 測試 + 覆蓋率
uv run stock-research config show        # 驗證並顯示 config
uv run stock-research validate 2330      # 資料驗證 + 資料健康報告（需 FINMIND_TOKEN）
```

## 程式風格
- 程式碼、註解、docstring、README 一律英文；產出的分析報告用繁體中文。CLAUDE.md 用繁體中文。
- Python 3.11+，完整型別標註；ruff（line-length 100）。
- **所有門檻、半衰期、權重、日期都放 `config.yaml`**，透過 `stock_research.config.load_config()` 取得，不可寫死。
- 比率在 config 與程式中一律用小數（0.15 = 15%）。
- 錯誤用自訂例外並附 ticker、欄位、實際值、預期條件。
- 測試不可呼叫真實 API，一律用 `tests/fixtures/` 或 mock。
- 核心模組（validation、windows、calendar、weighting、risk、stability、breakpoints、frameworks、scoring、verification）覆蓋率目標 90%。
