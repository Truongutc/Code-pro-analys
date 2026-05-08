"""
Module: Elliott Engine
======================
Implements Elliott Wave smoothing (DEMA/TEMA) and Linear Regression Channels (AUTO SEC).
Optimized for performance to handle large datasets.
"""

import pandas as pd
import numpy as np

def calculate_zigzag(series, percent_change):
    """
    AmiBroker-style ZigZag implementation.
    Identifies peaks and troughs where price moves by >= percent_change.
    """
    if len(series) < 2:
        return series
    
    change = percent_change / 100.0
    data = series.values
    n = len(data)
    zigzag = np.full(n, np.nan)
    
    last_piv_val = data[0]
    last_piv_idx = 0
    trend = 0 
    
    zigzag[0] = last_piv_val
    
    for i in range(1, n):
        val = data[i]
        if last_piv_val == 0:
            diff = 0.0
            if val != 0:
                diff = float('inf') if val > 0 else float('-inf')
        else:
            diff = (val - last_piv_val) / last_piv_val
        
        if trend == 0:
            if diff >= change:
                trend = 1
                last_piv_val = val
                last_piv_idx = i
            elif diff <= -change:
                trend = -1
                last_piv_val = val
                last_piv_idx = i
        elif trend == 1:
            if val > last_piv_val:
                last_piv_val = val
                last_piv_idx = i
            elif diff <= -change:
                zigzag[last_piv_idx] = last_piv_val
                trend = -1
                last_piv_val = val
                last_piv_idx = i
        elif trend == -1:
            if val < last_piv_val:
                last_piv_val = val
                last_piv_idx = i
            elif diff >= change:
                zigzag[last_piv_idx] = last_piv_val
                trend = 1
                last_piv_val = val
                last_piv_idx = i
                
    zigzag[last_piv_idx] = last_piv_val
    zigzag[-1] = data[-1]
    
    zz_series = pd.Series(zigzag).interpolate(method='linear')
    return zz_series

def dema(series, n):
    """Double Exponential Moving Average."""
    e1 = series.ewm(span=n, adjust=False).mean()
    e2 = e1.ewm(span=n, adjust=False).mean()
    return 2 * e1 - e2

def tema(series, n):
    """Triple Exponential Moving Average."""
    e1 = series.ewm(span=n, adjust=False).mean()
    e2 = e1.ewm(span=n, adjust=False).mean()
    e3 = e2.ewm(span=n, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3

def exrem(buy, sell):
    """
    Excludes redundant signals. Optimized via Vectorization/Fast Loop.
    """
    buy_raw = buy.values if hasattr(buy, 'values') else buy
    sell_raw = sell.values if hasattr(sell, 'values') else sell
    n = len(buy_raw)
    
    result_buy = np.zeros(n, dtype=bool)
    result_sell = np.zeros(n, dtype=bool)
    
    state = 0 
    for i in range(n):
        if buy_raw[i] and state <= 0:
            result_buy[i] = True
            state = 1
        elif sell_raw[i] and state >= 0:
            result_sell[i] = True
            state = -1
            
    idx = buy.index if hasattr(buy, 'index') else None
    return pd.Series(result_buy, index=idx), pd.Series(result_sell, index=idx)

def calculate_elliott_wave_system(df):
    """
    Calculates Elliott Wave logic and AUTO SEC Channel (Anchored Linear Regression).
    Precisely replicates AFL logic for signal placement and channel projection.
    """
    n = len(df)
    if n < 10:
        return pd.DataFrame(index=df.index)

    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    typical = (h + l) / 2.0
    
    res = pd.DataFrame(index=df.index)
    
    # --- 1. Smaller Wave (Zig 1.0%) ---
    # Buy1 = DEMA(Zig, 1) > MA(Zig, 2)
    # Sell1 = DEMA(Zig, 2) > Zig
    zz_small = calculate_zigzag(pd.Series(typical), 1.0).values
    ew_small_1 = dema(pd.Series(zz_small), 1).values
    ew_small_2 = dema(pd.Series(zz_small), 2).values
    
    # MA(Zig, 2)
    ma_zz_2 = pd.Series(zz_small).rolling(2).mean().fillna(pd.Series(zz_small)).values
    
    raw_buy1 = ew_small_1 > ma_zz_2
    raw_sell1 = ew_small_2 > zz_small
    
    res['EW_Small_Buy'], res['EW_Small_Sell'] = exrem(raw_buy1, raw_sell1)
    res['EW_Small_1'] = ew_small_1
    res['EW_Small_2'] = ew_small_2
    
    # --- 2. Larger Wave (Zig 4.5%) ---
    zz_large = calculate_zigzag(pd.Series(typical), 4.5).values
    res['EW_Large_1'] = tema(pd.Series(zz_large), 1)
    res['EW_Large_2'] = tema(pd.Series(zz_large), 2)
    
    # --- 3. AUTO SEC VERSION 1.2 (Anchored Linear Regression) ---
    sens = 4.75
    zz_sec_series = calculate_zigzag(pd.Series(c), sens)
    zz_sec = zz_sec_series.values
    
    # Identify Peaks and Troughs same as AFL
    # A turnpoint occurs where the slope of the ZigZag changes
    diffs = zz_sec_series.diff().values
    is_turnpoint = np.zeros(n, dtype=bool)
    for i in range(1, n-1):
        if abs(diffs[i] - diffs[i+1]) > 1e-10:
            is_turnpoint[i] = True
            
    # Handle the absolute last bar as a turnpoint to close the last wave calculation
    is_turnpoint[-1] = True 
    is_turnpoint[0] = True # Root
    
    pivot_indices = np.where(is_turnpoint)[0]
    
    sec_mid = np.full(n, np.nan)
    sec_up = np.full(n, np.nan)
    sec_lo = np.full(n, np.nan)
    sec_slope = np.zeros(n)
    
    # Active regression parameters
    aa = np.nan
    bb = np.nan
    std_e = np.nan
    last_piv_idx = 0
    daysback = 0
    
    for i in range(n):
        if is_turnpoint[i]:
            # At turnpoint, update Daysback and Regression params from the FINISHED wave
            # Wave length = distance to PREVIOUS pivot
            daysback = i - last_piv_idx + 1
            
            if daysback > 1:
                # Calculate regression on Close[last_piv_idx : i+1]
                y_window = c[last_piv_idx : i + 1]
                x_window = np.arange(len(y_window))
                
                # LinRegSlope and Intercept
                # y = intercept + slope * x
                slope, intercept = np.polyfit(x_window, y_window, 1)
                
                # StdErr(C, Daysback)
                preds = intercept + slope * x_window
                errs = y_window - preds
                # AFL StdErr is sample standard deviation of residuals
                # Note: AFL StdErr uses N-1 in denominator
                std_e = np.sqrt(np.sum(errs**2) / (len(y_window) - 1)) if len(y_window) > 1 else 0
                
                aa = intercept
                bb = slope
                
            last_piv_idx = i
            
        # Every bar, calculate projected lines
        # y = Aa + bb * ( x - (Lastx - DaysBack + 1) )
        if not np.isnan(aa):
            # x_rel is the distance from the START of the previous wave
            # If current wave started at last_piv_idx.
            # The previous wave started at (last_piv_idx - daysback + 1)
            # Actually, per AFL: aa is intercept at start of previous wave.
            # So x_rel = distance from start of PREVIOUS wave.
            start_prev_wave = last_piv_idx - daysback + 1
            x_rel = i - start_prev_wave
            
            y_val = aa + bb * x_rel
            sec_up[i] = y_val + 2 * std_e
            sec_lo[i] = y_val - 2 * std_e
            
            # Mback = Level + slope * Turnpoint
            # Turnpoint is BarsSince(pivot)
            turnpoint = i - last_piv_idx
            sec_mid[i] = aa + bb * turnpoint # Note: Mback resets at every pivot in AFL
            sec_slope[i] = bb
            
    res['EW_SEC_Mid'] = sec_mid
    res['EW_SEC_Upper'] = sec_up
    res['EW_SEC_Lower'] = sec_lo
    res['EW_SEC_Slope'] = sec_slope
    
    # Signals (Mega Wave)
    # Buy  = Cross(C,eU) OR C > eU;       
    # Sell = Cross(eL,C) OR C < eU;
    
    # We use eU for both based on the AFL provided
    # The exrem function ensures we only take the first transition
    raw_mega_buy = pd.Series(c > res['EW_SEC_Upper'])
    raw_mega_sell = pd.Series(c < res['EW_SEC_Upper'])
    
    res['EW_Strong_Buy'], res['EW_Strong_Sell'] = exrem(raw_mega_buy, raw_mega_sell)
    
    return res
