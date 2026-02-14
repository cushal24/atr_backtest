"""
ATR Drop Backtest — 4x ATR Drop in >= 3 Days + Right-of-V Entry

Strategy Logic:
  DAILY FILTER:
    - Compute 14-period ATR on daily bars
    - ATR% = ATR / Close * 100 (percentage of stock price)
    - Signal: stock drops >= 4x ATR over >= 3 consecutive days
    - The "drop" is measured as: high of N days ago minus today's low >= 4 * ATR

  INTRADAY ENTRY (2-min chart, Right-of-V):
    - After a qualifying multi-day ATR drop, scan the NEXT trading day's 2-min chart
    - Find the intraday V-bottom: sharp selloff followed by bounce
    - Entry: first green higher-low bar after the dip low (below VWAP)

  STOP LOSS:
    - Break of prior 2-min candle's low

  PROFIT TAKING:
    - Sell 50% at +0.5% from entry (partial profit)
    - Trail remaining position with prior 2-min candle low as stop
    - Close all at 3:55 PM EOD

  POSITION SIZING:
    - $50,000 per trade (fixed notional)

  TEST:
    - 19 tickers, Jan 2023 to current

Usage:
    python3 atr_drop_backtest.py
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

EST = ZoneInfo('US/Eastern')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'atr_drop')
SHARED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = ['AMZN', 'NVDA', 'MSFT', 'TSLA', 'AAPL', 'GOOGL', 'META', 'NFLX', 'BA', 'TSM', 'AVGO', 'LLY', 'WMT', 'V', 'ASML', 'MU', 'AMD', 'PLTR', 'AMAT']
ATR_PERIOD = 14
ATR_DROP_MULTIPLIER = 4.0      # Stock must drop >= 4x ATR
MIN_DROP_DAYS = 3              # Drop must occur over >= 3 days
MAX_DROP_DAYS = 6              # Drop must occur in no more than 6 days
ACCOUNT_SIZE = 100000          # $100k account
POSITION_SIZE = ACCOUNT_SIZE   # Base unit for multipliers
NORMAL_SIZE_MULT = 2           # 2x account ($200k) on normal break-of-high setups
SIGNAL_BAR_SIZE_MULT = 4       # 4x account ($400k) on signal bar (hammer) setups
MAX_RISK_PCT = 0.05            # Max 5% risk per trade
MIN_RISK_PCT = 0.0005          # Min 0.05% risk (allow tight stops on capitulatory days)
COOLDOWN_DAYS = 1              # Skip signals within N trading days of last trade

# ============================================================================
# DATA FETCHING (with caching)
# ============================================================================

def get_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_daily(client, ticker, start, end):
    """Fetch daily bars with caching. Also checks shared data dir."""
    cache_dir = os.path.join(DATA_DIR, 'daily')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{ticker}_daily_{start}_{end}.csv')

    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['Date'])
        if len(df) > 0:
            return df

    # Check shared data directory (from other backtests)
    shared_daily = os.path.join(SHARED_DATA_DIR, 'daily')
    if os.path.isdir(shared_daily):
        import glob
        pattern = os.path.join(shared_daily, f'{ticker}_daily_*.csv')
        matches = glob.glob(pattern)
        for m in matches:
            df = pd.read_csv(m, parse_dates=['Date'])
            if len(df) > 0:
                print(f"   (using cached daily data from {os.path.basename(m)})")
                return df

    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=EST),
            end=datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=EST),
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        data = bars.df
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['timestamp']).dt.tz_convert(EST).dt.tz_localize(None),
            'Open': data['open'].values,
            'High': data['high'].values,
            'Low': data['low'].values,
            'Close': data['close'].values,
            'Volume': data['volume'].values,
        })
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"  ERROR fetching daily: {e}")
        return pd.DataFrame()


def fetch_intraday_2min(client, ticker, date_str):
    """Fetch 2-min intraday bars for a single day with caching."""
    cache_dir = os.path.join(DATA_DIR, 'intraday')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{ticker}_2min_{date_str}.csv')

    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['Date'])
        if len(df) > 0:
            return df

    # Check shared intraday cache
    shared_cache = os.path.join(SHARED_DATA_DIR, 'intraday', f'{ticker}_2min_{date_str}.csv')
    if os.path.exists(shared_cache):
        df = pd.read_csv(shared_cache, parse_dates=['Date'])
        if len(df) > 0:
            return df

    try:
        trade_date = datetime.strptime(date_str, '%Y-%m-%d')
        start = trade_date.replace(hour=9, minute=30, tzinfo=EST)
        end = trade_date.replace(hour=16, minute=0, tzinfo=EST)

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(2, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        data = bars.df
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['timestamp']).dt.tz_convert(EST).dt.tz_localize(None),
            'Open': data['open'].values,
            'High': data['high'].values,
            'Low': data['low'].values,
            'Close': data['close'].values,
            'Volume': data['volume'].values,
        })
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"  ERROR fetching intraday for {date_str}: {e}")
        return pd.DataFrame()


# ============================================================================
# ATR CALCULATION & MULTI-DAY DROP DETECTION
# ============================================================================

def compute_atr(daily_df, period=14):
    """Compute Average True Range on daily bars."""
    df = daily_df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()
    df['ATR_pct'] = df['ATR'] / df['Close'] * 100
    return df


def find_atr_drop_signals(daily_df, atr_mult=4.0, min_days=3, max_days=6):
    """
    Find days where the stock has dropped >= atr_mult * ATR over >= min_days.

    For each day i, look back min_days to max_days and check if:
      high[i - N] - low[i] >= atr_mult * ATR[i]
    where N is between min_days and max_days.

    Returns list of signal dicts with the qualifying day info.
    """
    df = daily_df.copy()
    signals = []

    for i in range(max(ATR_PERIOD, max_days), len(df)):
        row = df.iloc[i]
        atr = row['ATR']
        if pd.isna(atr) or atr <= 0:
            continue

        target_drop = atr_mult * atr
        today_low = row['Low']

        # Check lookback windows from min_days to max_days
        best_drop = 0
        best_days = 0
        for lookback in range(min_days, min(max_days + 1, i + 1)):
            past_row = df.iloc[i - lookback]
            drop = past_row['High'] - today_low
            if drop >= target_drop and drop > best_drop:
                best_drop = drop
                best_days = lookback

        if best_drop >= target_drop:
            drop_pct = best_drop / df.iloc[i - best_days]['High'] * 100
            signals.append({
                'date': row['Date'],
                'date_str': row['Date'].strftime('%Y-%m-%d'),
                'close': row['Close'],
                'low': today_low,
                'atr': round(atr, 4),
                'atr_pct': round(row['ATR_pct'], 2),
                'drop_amount': round(best_drop, 2),
                'drop_atr_mult': round(best_drop / atr, 2),
                'drop_pct': round(drop_pct, 2),
                'drop_days': best_days,
                'high_before': round(df.iloc[i - best_days]['High'], 2),
            })

    return signals


# ============================================================================
# INTRADAY V-SETUP DETECTION (2-min chart) — Right of V
# ============================================================================

def compute_vwap(df):
    """Compute cumulative VWAP on intraday data."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = df['Volume'].cumsum()
    cum_tp_vol = (typical_price * df['Volume']).cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap


def _find_v_bottom(intraday_df):
    """
    Common V-bottom detection shared by all entry methods.
    Returns (df_with_vwap, dip_low_idx, dip_low, sharp_drop_pct, open_price) or None.
    """
    if intraday_df.empty or len(intraday_df) < 15:
        return None

    df = intraday_df.copy()
    df['VWAP'] = compute_vwap(df)

    open_price = df.iloc[0]['Open']

    skip_bars = 5
    cutoff_idx = len(df)
    for i in range(len(df)):
        if df.iloc[i]['Date'].hour >= 13:
            cutoff_idx = i
            break

    if cutoff_idx < skip_bars + 5:
        return None

    search_range = df.iloc[skip_bars:cutoff_idx]
    dip_low_idx = search_range['Low'].idxmin()
    dip_low = df.iloc[dip_low_idx]['Low']

    lookback = min(15, dip_low_idx)
    lookback_start = max(0, dip_low_idx - lookback)
    high_before_dip = df.iloc[lookback_start:dip_low_idx + 1]['High'].max()
    sharp_drop_pct = (high_before_dip - dip_low) / high_before_dip * 100

    if sharp_drop_pct < 1.0:
        return None

    dip_bar = df.iloc[dip_low_idx]
    if dip_low >= dip_bar['VWAP']:
        return None

    return df, dip_low_idx, dip_low, sharp_drop_pct, open_price


def _find_signal_bar_entry(df, dip_low_idx, dip_low, sharp_drop_pct, open_price):
    """
    Signal Bar (hammer reversal) entry — 4x position size.
    Look for a hammer candle at/near the V-bottom, enter on break of its high.
    """
    signal_bar_idx = None
    signal_bar_high = None

    for i in range(dip_low_idx, min(dip_low_idx + 3, len(df))):
        bar = df.iloc[i]
        bar_range = bar['High'] - bar['Low']
        if bar_range <= 0:
            continue

        body_bottom = min(bar['Open'], bar['Close'])
        lower_wick = body_bottom - bar['Low']
        lower_wick_pct = lower_wick / bar_range
        close_position = (bar['Close'] - bar['Low']) / bar_range

        is_hammer = lower_wick_pct >= 0.50 and close_position >= 0.60

        if is_hammer:
            signal_bar_idx = i
            signal_bar_high = bar['High']
            break

    if signal_bar_idx is None:
        return None

    for i in range(signal_bar_idx + 1, len(df)):
        bar = df.iloc[i]
        if bar['Date'].hour >= 15:
            return None

        is_breakout = bar['High'] > signal_bar_high
        is_green = bar['Close'] > bar['Open']

        if is_breakout and is_green:
            drop_from_open = (open_price - dip_low) / open_price * 100
            return {
                'entry_idx': i,
                'entry_price_level': signal_bar_high,
                'dip_low': dip_low,
                'dip_low_idx': dip_low_idx,
                'dip_pct': round(drop_from_open, 2),
                'sharp_drop_pct': round(sharp_drop_pct, 2),
                'dip_bars': i - dip_low_idx,
                'open_price': open_price,
                'setup_type': 'signal_bar',
                'size_multiplier': SIGNAL_BAR_SIZE_MULT,
            }

    return None


def _find_break_prior_high_entry(df, dip_low_idx, dip_low, sharp_drop_pct, open_price):
    """
    Default entry — break above prior candle's high. Normal 1x position size.
    """
    min_higher_low_pct = 0.0005
    for i in range(dip_low_idx + 2, len(df)):
        bar = df.iloc[i]

        if bar['Date'].hour >= 15:
            return None

        prior_bar = df.iloc[i - 1]
        prior_high = prior_bar['High']

        prior_low_above_dip = (prior_bar['Low'] - dip_low) / dip_low
        is_prior_higher_low = prior_low_above_dip >= min_higher_low_pct
        is_breakout = bar['High'] > prior_high
        is_green = bar['Close'] > bar['Open']

        if is_prior_higher_low and is_breakout and is_green:
            drop_from_open = (open_price - dip_low) / open_price * 100
            return {
                'entry_idx': i,
                'entry_price_level': prior_high,
                'dip_low': dip_low,
                'dip_low_idx': dip_low_idx,
                'dip_pct': round(drop_from_open, 2),
                'sharp_drop_pct': round(sharp_drop_pct, 2),
                'dip_bars': i - dip_low_idx,
                'open_price': open_price,
                'setup_type': 'break_prior_high',
                'size_multiplier': NORMAL_SIZE_MULT,
            }

    return None


def find_intraday_v_setup(intraday_df):
    """
    Scan a single day's 2-min bars for a Right-of-V entry.

    Two entry methods (checked in order):
    1. Signal Bar (hammer) at V-bottom → 4x position size
    2. Break above prior candle's high → 1x position size (default)

    Returns dict with setup info including 'size_multiplier', or None.
    """
    bottom = _find_v_bottom(intraday_df)
    if bottom is None:
        return None

    df, dip_low_idx, dip_low, sharp_drop_pct, open_price = bottom

    # Try signal bar first (4x size)
    setup = _find_signal_bar_entry(df, dip_low_idx, dip_low, sharp_drop_pct, open_price)
    if setup is not None:
        return setup

    # Fall back to break of prior high (1x size)
    return _find_break_prior_high_entry(df, dip_low_idx, dip_low, sharp_drop_pct, open_price)


# ============================================================================
# TRADE EXECUTION — 0.5% partial, trailing stop on prior 2-min low
# ============================================================================

def execute_trade(intraday_df, date_str, v_setup, position_size_dollars=50000):
    """
    Execute intraday trade on 2-min bars.

    Entry: Break above prior candle's high (the breakout level)
    Stop: Prior 2-min candle's low, trailing up (ratchets only)
    Exit: Stop hit or EOD at 3:55 PM
    No partial profit — full position rides the trail.
    """
    if intraday_df.empty or len(intraday_df) < 10:
        return None

    df = intraday_df.copy()
    df['VWAP'] = compute_vwap(df)

    entry_idx = v_setup['entry_idx']
    entry_bar = df.iloc[entry_idx]
    entry_time = entry_bar['Date']

    # Entry price = break above prior candle's high (the breakout level)
    entry_price = v_setup['entry_price_level']
    vwap_at_entry = df.iloc[entry_idx]['VWAP']

    # Apply size multiplier (4x for signal bar, 1x for default)
    size_mult = v_setup.get('size_multiplier', 1)
    position_size_dollars = position_size_dollars * size_mult

    # Stop loss: prior candle's low (the bar before the entry bar)
    if entry_idx < 2:
        return None
    prior_bar = df.iloc[entry_idx - 1]
    stop_loss = prior_bar['Low']

    # Risk check
    risk_pct = (entry_price - stop_loss) / entry_price
    if risk_pct > MAX_RISK_PCT or risk_pct < MIN_RISK_PCT:
        return None

    # Position sizing
    shares = int(position_size_dollars / entry_price)
    if shares < 1:
        return None

    trail_stop = stop_loss
    final_exit_price = None
    final_exit_time = None
    final_exit_reason = None
    high_water = entry_price

    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        bar_time = bar['Date']

        # EOD exit at 3:55 PM
        if bar_time.hour >= 15 and bar_time.minute >= 54:
            final_exit_price = bar['Close']
            final_exit_time = bar_time
            final_exit_reason = "EOD close"
            break

        # Check trail stop hit
        if bar['Low'] <= trail_stop:
            final_exit_price = trail_stop
            final_exit_time = bar_time
            final_exit_reason = "Stop hit (prior bar low)"
            break

        # Update trailing stop: prior bar's low (only ratchet up)
        if i > entry_idx + 1:
            prev_low = df.iloc[i-1]['Low']
            if prev_low > trail_stop:
                trail_stop = prev_low

        # Track high water mark
        if bar['High'] > high_water:
            high_water = bar['High']

    # Safety: if we never exited
    if final_exit_price is None:
        last_bar = df.iloc[-1]
        final_exit_price = last_bar['Close']
        final_exit_time = last_bar['Date']
        final_exit_reason = "EOD close"

    total_pnl = (final_exit_price - entry_price) * shares

    return {
        'date': date_str,
        'entry_price': round(entry_price, 2),
        'entry_time': entry_time.strftime('%H:%M'),
        'stop_loss': round(stop_loss, 2),
        'shares': shares,
        'vwap_at_entry': round(vwap_at_entry, 2),
        'dip_pct': v_setup['dip_pct'],
        'sharp_drop_pct': v_setup.get('sharp_drop_pct', 0),
        'dip_low': round(v_setup['dip_low'], 2),
        'final_exit_price': round(final_exit_price, 2),
        'final_exit_time': final_exit_time.strftime('%H:%M') if final_exit_time else None,
        'final_exit_reason': final_exit_reason,
        'total_pnl': round(total_pnl, 2),
        'pnl_pct': round(total_pnl / (entry_price * shares) * 100, 2),
        'high_water': round(high_water, 2),
    }


# ============================================================================
# MAIN BACKTEST
# ============================================================================

def run_backtest_for_ticker(client, ticker):
    """Run the full ATR drop backtest for a single ticker."""
    print(f"\n{'='*80}")
    print(f"  ATR DROP BACKTEST — {ticker} — Jan 2023 to Current")
    print(f"{'='*80}")
    print(f"  Signal:  Stock drops >= {ATR_DROP_MULTIPLIER}x ATR over >= {MIN_DROP_DAYS} days")
    print(f"  Entry:   Right-of-V on 2-min chart (below VWAP)")
    print(f"  Stop:    Prior 2-min candle low")
    print(f"  Trail:   Prior 2-min candle low (ratchets up)")
    print(f"  Size:    ${POSITION_SIZE*NORMAL_SIZE_MULT:,} normal (2x) | ${POSITION_SIZE*SIGNAL_BAR_SIZE_MULT:,} signal bar (4x)")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Fetch daily data — start from mid-2022 for ATR warmup
    print(f"\n1. Fetching {ticker} daily data (2022-07-01 to 2026-02-14)...")
    daily_df = fetch_daily(client, ticker, '2022-07-01', '2026-02-14')

    if daily_df.empty:
        print("   ERROR: Could not fetch daily data")
        return [], []

    # Compute ATR
    daily_df = compute_atr(daily_df, period=ATR_PERIOD)
    daily_df['Date_str'] = daily_df['Date'].dt.strftime('%Y-%m-%d')

    print(f"   Got {len(daily_df)} daily bars")
    print(f"   ATR(14) range: {daily_df['ATR'].min():.2f} - {daily_df['ATR'].max():.2f}")
    print(f"   ATR% range: {daily_df['ATR_pct'].min():.2f}% - {daily_df['ATR_pct'].max():.2f}%")

    # Find ATR drop signals (only in 2025+)
    print(f"\n2. Scanning for {ATR_DROP_MULTIPLIER}x ATR drops over >= {MIN_DROP_DAYS} days...")
    all_signals = find_atr_drop_signals(daily_df, atr_mult=ATR_DROP_MULTIPLIER, min_days=MIN_DROP_DAYS, max_days=MAX_DROP_DAYS)

    # Filter to Jan 2023+
    signals = [s for s in all_signals if s['date'] >= datetime(2023, 1, 1)]
    print(f"   Found {len(signals)} qualifying ATR drop signals in 2023+")

    if not signals:
        print("   No signals found. Try lowering ATR_DROP_MULTIPLIER or MIN_DROP_DAYS.")
        return [], signals

    for s in signals:
        print(f"   {s['date_str']}: Drop {s['drop_pct']:.1f}% ({s['drop_atr_mult']:.1f}x ATR) "
              f"over {s['drop_days']} days | ATR={s['atr']:.2f} ({s['atr_pct']:.1f}%) "
              f"| High={s['high_before']} → Low={s['low']:.2f}")

    # For each signal day, look for intraday V-entry on the signal day itself
    print(f"\n3. Scanning intraday 2-min charts for Right-of-V entries...")
    all_trades = []
    intraday_data = {}
    v_setup_count = 0
    no_setup_count = 0
    skipped_risk = 0
    skipped_cooldown = 0
    traded_dates = set()  # Prevent duplicate trades on the same day
    last_trade_idx = -999  # Index of last traded day in all_trading_days

    # Build list of all trading days for cooldown lookup
    all_trading_days = daily_df['Date_str'].tolist()

    for sig in signals:
        sig_date = sig['date_str']

        # Only trade on the signal day itself
        trade_date_str = sig_date

        # Skip if we already traded this day from a prior signal
        if trade_date_str in traded_dates:
            continue

        # Cooldown: skip if within COOLDOWN_DAYS of last trade
        try:
            current_idx = all_trading_days.index(trade_date_str)
        except ValueError:
            continue
        if current_idx - last_trade_idx <= COOLDOWN_DAYS:
            skipped_cooldown += 1
            print(f"   {trade_date_str}: Skipped (cooldown — {current_idx - last_trade_idx}d after last trade)")
            continue

        intraday_df = fetch_intraday_2min(client, ticker, trade_date_str)
        if intraday_df.empty:
            continue

        v_setup = find_intraday_v_setup(intraday_df)
        if v_setup is None:
            no_setup_count += 1
            continue

        v_setup_count += 1

        # Store intraday data for charting
        idf = intraday_df.copy()
        idf['VWAP'] = compute_vwap(idf)
        intraday_data[trade_date_str] = idf

        # Execute the trade
        trade = execute_trade(intraday_df, trade_date_str, v_setup, POSITION_SIZE)
        if trade is None:
            skipped_risk += 1
            print(f"   {trade_date_str}: V-setup found but risk too large/small — skipped")
            continue

        # Attach signal info to trade
        trade['signal_date'] = sig_date
        trade['atr'] = sig['atr']
        trade['atr_pct'] = sig['atr_pct']
        trade['drop_atr_mult'] = sig['drop_atr_mult']
        trade['drop_days'] = sig['drop_days']
        trade['daily_drop_pct'] = sig['drop_pct']
        trade['setup_type'] = v_setup.get('setup_type', 'break_prior_high')
        trade['size_multiplier'] = v_setup.get('size_multiplier', 1)

        all_trades.append(trade)
        traded_dates.add(trade_date_str)
        last_trade_idx = current_idx

        pnl_str = f"+${trade['total_pnl']:.2f}" if trade['total_pnl'] >= 0 else f"-${abs(trade['total_pnl']):.2f}"
        size_tag = f" [SIGNAL BAR {trade['size_multiplier']}x]" if trade['setup_type'] == 'signal_bar' else ""
        print(f"   Signal {sig_date} → Trade {trade_date_str}: "
              f"Entry {trade['entry_time']} @ ${trade['entry_price']:.2f} | "
              f"Exit {trade['final_exit_time']} @ ${trade['final_exit_price']:.2f} "
              f"({trade['final_exit_reason']}) | P&L: {pnl_str} ({trade['pnl_pct']:+.2f}%){size_tag}")

        time.sleep(0.15)

    print(f"\n   Summary: {len(signals)} signals, {v_setup_count} V-setups found, "
          f"{len(all_trades)} trades taken, {skipped_risk} skipped (risk), "
          f"{skipped_cooldown} skipped (cooldown), {no_setup_count} no V-setup")

    # ========================================================================
    # RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"  RESULTS — {ticker}")
    print(f"{'='*80}")

    if not all_trades:
        print("\n  No trades executed.")
        return [], signals

    df = pd.DataFrame(all_trades)
    total_pnl = df['total_pnl'].sum()
    winning = df[df['total_pnl'] > 0]
    losing = df[df['total_pnl'] <= 0]
    win_rate = len(winning) / len(df) * 100 if len(df) > 0 else 0

    print(f"\n  Ticker: {ticker}")
    print(f"  Period: Jan 2023 to Current")
    print(f"  ATR Drop Signals: {len(signals)}")
    print(f"  Position Size: ${POSITION_SIZE:,} per trade")
    print(f"\n  {'─'*50}")
    print(f"  Total Trades:    {len(df)}")
    print(f"  Winning Trades:  {len(winning)}")
    print(f"  Losing Trades:   {len(losing)}")
    print(f"  Win Rate:        {win_rate:.1f}%")
    print(f"  {'─'*50}")
    print(f"  Total P&L:       $ {total_pnl:>10,.2f}")
    print(f"  Avg P&L/Trade:   $ {total_pnl/len(df):>10,.2f}")
    if len(winning) > 0:
        print(f"  Avg Win:         $ {winning['total_pnl'].mean():>10,.2f}")
        print(f"  Largest Win:     $ {winning['total_pnl'].max():>10,.2f}")
    if len(losing) > 0:
        print(f"  Avg Loss:        $ {losing['total_pnl'].mean():>10,.2f}")
        print(f"  Largest Loss:    $ {losing['total_pnl'].min():>10,.2f}")


    print(f"\n  TRADE LOG:")
    for t in all_trades:
        pnl_str = f"+${t['total_pnl']:,.2f}" if t['total_pnl'] >= 0 else f"-${abs(t['total_pnl']):,.2f}"
        print(f"    {t['date']}  Signal: {t['signal_date']}  "
              f"Drop: {t['daily_drop_pct']:.1f}% ({t['drop_atr_mult']:.1f}x ATR / {t['drop_days']}d)  "
              f"Entry: {t['entry_time']} @ ${t['entry_price']:<8.2f}  "
              f"Exit: {t['final_exit_time']} @ ${t['final_exit_price']:<8.2f}  "
              f"P&L: {pnl_str:>12} ({t['pnl_pct']:+.2f}%)")

    # Save results
    results_file = os.path.join(DATA_DIR, f'{ticker}_atr_drop_results.csv')
    df.to_csv(results_file, index=False)
    print(f"\n  Results saved to: {results_file}")

    # Save chart data
    save_chart_data(ticker, intraday_data, all_trades, daily_df, signals)

    return all_trades, signals


def save_chart_data(ticker, intraday_data, trades, daily_df, signals):
    """Save all data needed for review/charting."""
    chart_data = {
        'ticker': ticker,
        'trades': trades,
        'signals': signals,
        'intraday': {},
        'daily': [],
    }

    # Daily data (2025+)
    daily_backtest = daily_df[daily_df['Date'] >= '2023-01-01'].copy()
    for _, row in daily_backtest.iterrows():
        d = {
            'date': row['Date'].strftime('%Y-%m-%d'),
            'open': round(row['Open'], 2),
            'high': round(row['High'], 2),
            'low': round(row['Low'], 2),
            'close': round(row['Close'], 2),
            'volume': int(row['Volume']),
            'atr': round(row['ATR'], 4) if not pd.isna(row['ATR']) else None,
            'atr_pct': round(row['ATR_pct'], 2) if not pd.isna(row['ATR_pct']) else None,
        }
        chart_data['daily'].append(d)

    # Intraday data with VWAP
    for date_str, idf in intraday_data.items():
        bars = []
        for _, row in idf.iterrows():
            vwap_val = row['VWAP'] if 'VWAP' in row else 0
            bars.append({
                'time': row['Date'].strftime('%H:%M'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume']),
                'vwap': round(vwap_val, 2),
            })
        chart_data['intraday'][date_str] = bars

    chart_file = os.path.join(DATA_DIR, f'{ticker}_atr_drop_chart_data.json')
    with open(chart_file, 'w') as f:
        json.dump(chart_data, f, default=str)
    print(f"  Chart data saved to: {chart_file}")


# ============================================================================
# COMBINED SUMMARY
# ============================================================================

def build_dashboard_data(all_results, all_signals_map):
    """Build dashboard JSON with portfolio-level stats."""
    starting_capital = ACCOUNT_SIZE
    ticker_stats = []
    all_trades_list = []

    for ticker in TICKERS:
        trades = all_results.get(ticker, [])
        signals = all_signals_map.get(ticker, [])
        n = len(trades)
        if n == 0:
            ticker_stats.append({
                'ticker': ticker, 'trades': 0, 'signals': len(signals),
                'wins': 0, 'losses': 0, 'win_rate': 0,
                'total_pnl': 0, 'avg_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                'largest_win': 0, 'largest_loss': 0, 'profit_factor': 0,
                'signal_bar_trades': 0, 'normal_trades': 0,
                'gross_profit': 0, 'gross_loss': 0,
            })
            continue

        pnls = [t['total_pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        sb_trades = [t for t in trades if t.get('setup_type') == 'signal_bar']

        ticker_stats.append({
            'ticker': ticker,
            'trades': n,
            'signals': len(signals),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / n * 100, 1),
            'total_pnl': round(sum(pnls), 2),
            'avg_pnl': round(sum(pnls) / n, 2),
            'avg_win': round(np.mean(wins), 2) if wins else 0,
            'avg_loss': round(np.mean(losses), 2) if losses else 0,
            'largest_win': round(max(wins), 2) if wins else 0,
            'largest_loss': round(min(losses), 2) if losses else 0,
            'profit_factor': round(pf, 2) if pf != float('inf') else 999,
            'signal_bar_trades': len(sb_trades),
            'normal_trades': n - len(sb_trades),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        })

        for t in trades:
            all_trades_list.append({**t, 'ticker': ticker})

    # Sort all trades by date for equity curve
    all_trades_list.sort(key=lambda x: x['date'])

    equity_curve = [{'date': 'Start', 'balance': starting_capital, 'pnl': 0, 'ticker': '', 'cumulative_pnl': 0}]
    running_balance = starting_capital
    cumulative_pnl = 0
    for t in all_trades_list:
        cumulative_pnl += t['total_pnl']
        running_balance = starting_capital + cumulative_pnl
        equity_curve.append({
            'date': t['date'],
            'balance': round(running_balance, 2),
            'pnl': round(t['total_pnl'], 2),
            'ticker': t['ticker'],
            'cumulative_pnl': round(cumulative_pnl, 2),
            'setup_type': t.get('setup_type', 'break_prior_high'),
            'size_mult': t.get('size_multiplier', 1),
            'entry_time': t.get('entry_time', ''),
            'entry_price': t.get('entry_price', 0),
            'exit_price': t.get('final_exit_price', 0),
            'pnl_pct': t.get('pnl_pct', 0),
        })

    # Portfolio-level stats
    all_pnls = [t['total_pnl'] for t in all_trades_list]
    all_wins = [p for p in all_pnls if p > 0]
    all_losses = [p for p in all_pnls if p <= 0]
    total_gross_profit = sum(all_wins) if all_wins else 0
    total_gross_loss = abs(sum(all_losses)) if all_losses else 0
    total_pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')

    # Max drawdown
    peak = starting_capital
    max_dd = 0
    max_dd_pct = 0
    for pt in equity_curve:
        if pt['balance'] > peak:
            peak = pt['balance']
        dd = peak - pt['balance']
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    portfolio = {
        'starting_capital': starting_capital,
        'ending_balance': round(starting_capital + sum(all_pnls), 2) if all_pnls else starting_capital,
        'total_pnl': round(sum(all_pnls), 2) if all_pnls else 0,
        'total_return_pct': round(sum(all_pnls) / starting_capital * 100, 2) if all_pnls else 0,
        'total_trades': len(all_trades_list),
        'wins': len(all_wins),
        'losses': len(all_losses),
        'win_rate': round(len(all_wins) / len(all_trades_list) * 100, 1) if all_trades_list else 0,
        'profit_factor': round(total_pf, 2) if total_pf != float('inf') else 999,
        'avg_pnl': round(sum(all_pnls) / len(all_pnls), 2) if all_pnls else 0,
        'avg_win': round(np.mean(all_wins), 2) if all_wins else 0,
        'avg_loss': round(np.mean(all_losses), 2) if all_losses else 0,
        'largest_win': round(max(all_wins), 2) if all_wins else 0,
        'largest_loss': round(min(all_losses), 2) if all_losses else 0,
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 2),
        'gross_profit': round(total_gross_profit, 2),
        'gross_loss': round(total_gross_loss, 2),
        'signal_bar_trades': sum(1 for t in all_trades_list if t.get('setup_type') == 'signal_bar'),
        'normal_trades': sum(1 for t in all_trades_list if t.get('setup_type') != 'signal_bar'),
        'position_size': POSITION_SIZE,
        'signal_bar_mult': SIGNAL_BAR_SIZE_MULT,
    }

    dashboard = {
        'portfolio': portfolio,
        'ticker_stats': ticker_stats,
        'equity_curve': equity_curve,
        'trades': all_trades_list,
        'config': {
            'tickers': TICKERS,
            'atr_period': ATR_PERIOD,
            'atr_drop_mult': ATR_DROP_MULTIPLIER,
            'min_drop_days': MIN_DROP_DAYS,
            'max_drop_days': MAX_DROP_DAYS,
            'position_size': POSITION_SIZE,
            'signal_bar_size_mult': SIGNAL_BAR_SIZE_MULT,
            'cooldown_days': COOLDOWN_DAYS,
        },
    }

    dash_file = os.path.join(DATA_DIR, 'dashboard_data.json')
    with open(dash_file, 'w') as f:
        json.dump(dashboard, f, default=str, indent=2)
    print(f"\n  Dashboard data saved to: {dash_file}")
    return dashboard


def main():
    print("=" * 80)
    print("  ATR DROP BACKTEST — 4x ATR Drop in >= 3 Days + Right-of-V Entry")
    print(f"  Tickers: {', '.join(TICKERS)} | Period: Jan 2023 to Current")
    print("=" * 80)

    client = get_client()

    all_results = {}
    all_signals_map = {}

    for ticker in TICKERS:
        trades, signals = run_backtest_for_ticker(client, ticker)
        all_results[ticker] = trades
        all_signals_map[ticker] = signals

    # Combined summary
    print(f"\n\n{'='*80}")
    print("  COMBINED SUMMARY — ALL TICKERS")
    print(f"{'='*80}")

    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_losses = 0

    for ticker in TICKERS:
        trades = all_results[ticker]
        signals = all_signals_map[ticker]
        if not trades:
            print(f"\n  {ticker}: No trades ({len(signals)} signals)")
            continue

        df = pd.DataFrame(trades)
        t_pnl = df['total_pnl'].sum()
        wins = len(df[df['total_pnl'] > 0])
        losses = len(df[df['total_pnl'] <= 0])
        wr = wins / len(df) * 100

        total_trades += len(df)
        total_pnl += t_pnl
        total_wins += wins
        total_losses += losses

        print(f"\n  {ticker}: {len(df)} trades | Win Rate: {wr:.0f}% | "
              f"P&L: ${t_pnl:+,.2f} | Signals: {len(signals)}")

    if total_trades > 0:
        overall_wr = total_wins / total_trades * 100
        print(f"\n  {'─'*50}")
        print(f"  TOTAL:  {total_trades} trades | Win Rate: {overall_wr:.0f}% | "
              f"P&L: ${total_pnl:+,.2f}")
        print(f"  Avg P&L/Trade: ${total_pnl/total_trades:+,.2f}")
    else:
        print(f"\n  No trades across any ticker.")

    # Build and save dashboard data
    dashboard = build_dashboard_data(all_results, all_signals_map)

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
