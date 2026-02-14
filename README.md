# ATR Drop — Right Side of V Strategy

## Use this prompt to resume work on this strategy:

> I have an ATR Drop + Right Side of V intraday backtest in `/Users/kooshall/Repo/lance_script/right_of_v/`. The main file is `atr_drop_backtest.py` with an interactive dashboard at `atr_drop_dashboard.html`. Here's the full strategy:

---

## Strategy Overview

**Goal**: Find stocks that have dropped hard (≥4x ATR over 3–6 days), then trade the intraday V-shaped reversal on the signal day using 2-min bars.

## Daily Signal (Filter)

- Compute **14-period ATR** on daily bars
- Signal fires when: `High[N days ago] - Today's Low ≥ 4 × ATR`
- Drop must occur over **3 to 6 consecutive days**
- Only signals from **Jan 2023+** are traded

## Intraday Entry (2-min chart)

### V-Bottom Detection
1. Skip first **10 minutes** (5 bars) of open chaos
2. Find the **day's low** in the first half of the day (before 1:00 PM)
3. Verify a **sharp drop ≥1%** from recent high into the low
4. V-bottom must be **below VWAP** (confirms capitulatory move)

### Two Entry Methods (checked in order)

#### 1. Signal Bar (Hammer) — 4x position ($400k)
- Look for a **hammer/reversal candle** at or within 2 bars of the V-bottom
- Hammer criteria: **lower wick ≥50%** of bar range, **close in upper 60%** of range
- Entry: break above the signal bar's **high**
- This is the highest-conviction setup → **4x account size**

#### 2. Break of Prior High (default) — 2x position ($200k)
- After V-bottom, scan for the first green bar that breaks above its prior bar's **high**
- Prior bar must have a **meaningfully higher low** (≥0.05% above V-bottom)
- Entry bar must be **green** (close > open)
- Fallback when no hammer is present → **2x account size**

### Entry Filters
- Not too late in the day (before 3:00 PM)
- 1-day **cooldown** after each trade (skip next-day signals)

## Exit Rules

- **Stop loss**: Prior 2-min candle's low (set at entry)
- **Trailing stop**: Prior bar's low, ratchets up only (never down)
- **EOD exit**: Close all at 3:55 PM
- **No partial profit** — full position rides the trailing stop (tested: removing partials increased P&L by +29%)

## Risk Management

- Max risk per trade: **5%** of entry price
- Min risk per trade: **0.05%** (allows tight stops on capitulatory days)
- Risk = `(entry_price - stop_loss) / entry_price`

## Position Sizing

- **Account size**: $100,000
- **Normal trades** (break of prior high): **2x** = $200,000
- **Signal bar trades** (hammer): **4x** = $400,000

## Tickers (19)

AMZN, NVDA, MSFT, TSLA, AAPL, GOOGL, META, NFLX, BA, TSM, AVGO, LLY, WMT, V, ASML, MU, AMD, PLTR, AMAT

## Backtest Results (Jan 2023 – Feb 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 136 |
| Win Rate | 82% |
| Total P&L | +$203,485 |
| Return on $100k | +203.5% |
| Avg P&L/Trade | +$1,496 |
| Period | ~3 years |

### By Ticker (sorted by P&L)
| Ticker | Trades | Win% | P&L |
|--------|--------|------|-----|
| MU | 10 | 90% | +$28,142 |
| TSM | 8 | 88% | +$23,283 |
| AMD | 13 | 100% | +$21,667 |
| BA | 9 | 78% | +$21,044 |
| LLY | 8 | 100% | +$20,454 |
| AMZN | 11 | 64% | +$17,367 |
| AMAT | 7 | 57% | +$11,760 |
| PLTR | 8 | 50% | +$9,994 |
| TSLA | 10 | 80% | +$7,796 |
| V | 3 | 100% | +$7,557 |
| NVDA | 7 | 100% | +$7,203 |
| META | 6 | 83% | +$5,392 |
| NFLX | 5 | 80% | +$4,436 |
| AVGO | 8 | 62% | +$4,168 |
| MSFT | 3 | 100% | +$3,849 |
| AAPL | 7 | 100% | +$3,686 |
| GOOGL | 6 | 100% | +$2,972 |
| ASML | 4 | 75% | +$1,904 |
| WMT | 3 | 67% | +$812 |

## Key Files

- `atr_drop_backtest.py` — Main backtest script
- `atr_drop_chart.html` — Interactive intraday/daily chart with entry/exit markers (serve with `python3 -m http.server`)
- `atr_drop_dashboard.html` — Portfolio dashboard with equity curve, P&L breakdown, trade log
- `atr_drop_variations.py` — Tested 5 entry variations (Higher Low, Signal Bar, Spring, Combined)
- `data/atr_drop/dashboard_data.json` — Dashboard data
- `data/atr_drop/{TICKER}_atr_drop_chart_data.json` — Per-ticker chart data

## Key Design Decisions & Why

1. **Entry on prior candle's HIGH (not close)** — You break *above* a candle by exceeding its high
2. **Skip first 10 min for V-bottom** — Open chaos creates false lows
3. **Higher-low threshold (0.05%)** — Filters out bars barely above the dip (still part of the V)
4. **V-bottom must be below VWAP** — Confirms the move is capitulatory, not just a pullback
5. **No partial profit** — Tested both; full trail produced +29% more P&L
6. **Signal bar gets 4x sizing** — 100% win rate in backtest; hammer at V-bottom = highest conviction
7. **1-day cooldown** — Prevents stacking trades on consecutive signal days after a capitulatory move
8. **Data source**: Alpaca IEX feed, cached locally in `data/` directories
