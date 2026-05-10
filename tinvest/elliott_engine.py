import numpy as np
import pandas as pd

# =========================================================
# AFL STYLE EMA / DEMA / TEMA
# =========================================================

def ema_afl(arr, period):
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full(len(arr), np.nan)
    if len(arr) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(out[i-1]):
            out[i] = arr[i]
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def dema_afl(arr, period):
    e1 = ema_afl(arr, period)
    e2 = ema_afl(e1, period)
    return 2 * e1 - e2

def tema_afl(arr, period):
    e1 = ema_afl(arr, period)
    e2 = ema_afl(e1, period)
    e3 = ema_afl(e2, period)
    return 3 * e1 - 3 * e2 + e3

# =========================================================
# AFL STYLE MA
# =========================================================

def ma_afl(arr, period):
    return pd.Series(arr).rolling(int(period), min_periods=1).mean().values

# =========================================================
# AFL STYLE CROSS
# =========================================================

def cross(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2: return out
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out

# =========================================================
# AFL STYLE EXREM
# =========================================================

def exrem(buy, sell):
    buy = np.asarray(buy, dtype=bool)
    sell = np.asarray(sell, dtype=bool)
    rb = np.zeros(len(buy), dtype=bool)
    rs = np.zeros(len(sell), dtype=bool)
    state = 0
    for i in range(len(buy)):
        if buy[i] and state != 1:
            rb[i] = True
            state = 1
        elif sell[i] and state != -1:
            rs[i] = True
            state = -1
    return rb, rs

# =========================================================
# AFL STYLE VALUEWHEN
# =========================================================

def valuewhen(cond, values):
    out = np.full(len(values), np.nan)
    last = np.nan
    for i in range(len(values)):
        if cond[i]:
            last = values[i]
        out[i] = last
    return out

# =========================================================
# AFL STYLE BARSSINCE
# =========================================================

def barssince(cond):
    out = np.full(len(cond), np.nan)
    last_true = -1
    for i in range(len(cond)):
        if cond[i]:
            last_true = i
        if last_true == -1:
            out[i] = np.nan
        else:
            out[i] = i - last_true
    return out

# =========================================================
# AFL STYLE PEAK / TROUGH
# =========================================================

def peak(arr, change_pct):
    arr = np.asarray(arr)
    out = np.full(len(arr), np.nan)
    change = change_pct / 100.0
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] >= arr[i+1]:
            retrace = False
            for j in range(i+1, len(arr)):
                dd = (arr[j] - arr[i]) / arr[i]
                if dd <= -change:
                    retrace = True
                    break
            if retrace:
                out[i] = arr[i]
    return out

def trough(arr, change_pct):
    arr = np.asarray(arr)
    out = np.full(len(arr), np.nan)
    change = change_pct / 100.0
    for i in range(1, len(arr)-1):
        if arr[i] < arr[i-1] and arr[i] <= arr[i+1]:
            bounce = False
            for j in range(i+1, len(arr)):
                up = (arr[j] - arr[i]) / arr[i]
                if up >= change:
                    bounce = True
                    break
            if bounce:
                out[i] = arr[i]
    return out

# =========================================================
# AFL STYLE ZIG
# =========================================================

def zig_afl(series, percent):
    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)
    zz = np.full(n, np.nan)
    change = percent / 100.0
    trend = 0
    last_pivot_idx = 0
    last_pivot_price = arr[0]
    zz[0] = arr[0]
    for i in range(1, n):
        move = (arr[i] - last_pivot_price) / last_pivot_price
        if trend == 0:
            if move >= change:
                trend = 1
                last_pivot_idx = i
                last_pivot_price = arr[i]
            elif move <= -change:
                trend = -1
                last_pivot_idx = i
                last_pivot_price = arr[i]
        elif trend == 1:
            if arr[i] >= last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = arr[i]
            elif move <= -change:
                zz[last_pivot_idx] = last_pivot_price
                trend = -1
                last_pivot_idx = i
                last_pivot_price = arr[i]
        elif trend == -1:
            if arr[i] <= last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = arr[i]
            elif move >= change:
                zz[last_pivot_idx] = last_pivot_price
                trend = 1
                last_pivot_idx = i
                last_pivot_price = arr[i]
    zz[last_pivot_idx] = last_pivot_price
    zz = pd.Series(zz).interpolate().bfill().ffill().values
    return zz

# =========================================================
# AFL STYLE LINEAR REGRESSION
# =========================================================

def linreg_point_afl(arr):
    """Calculates Linear Regression for a specific window, returning end values."""
    n = len(arr)
    if n < 2: return 0.0, arr[-1], 0.0
    x = np.arange(n)
    xm, ym = x.mean(), arr.mean()
    ssxy = np.sum((x - xm) * (arr - ym))
    ssxx = np.sum((x - xm) ** 2)
    slope = ssxy / ssxx if ssxx != 0 else 0
    intercept_start = ym - slope * xm
    intercept_end = intercept_start + slope * (n - 1)
    pred = intercept_start + slope * x
    stderr = np.sqrt(np.mean((arr - pred) ** 2))
    return slope, intercept_end, stderr

# =========================================================
# MAIN ENGINE
# =========================================================

def calculate_elliott_wave_system(df):
    """
    Main Engine: Replicates AFL Elliott Wave Super System.
    """
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    n = len(c)
    out = pd.DataFrame(index=df.index)

    # =====================================================
    # SMALLER WAVE (ZIG 1.0%)
    # =====================================================
    zz1 = zig_afl(c, 1.0)
    ew1 = dema_afl(zz1, 1)
    ew2 = dema_afl(zz1, 2)
    ma2 = ma_afl(zz1, 2)
    ma1 = ma_afl(zz1, 1)

    buy1 = cross(ew1, ma2) | (ew1 > ma2)
    sell1 = cross(ew2, ma1) | (ew2 > ma1)
    buy1, sell1 = exrem(buy1, sell1)

    out['EW_Small_Buy'] = buy1
    out['EW_Small_Sell'] = sell1
    out['BuyPrice1'] = valuewhen(buy1, (h + l) / 2)
    out['SellPrice1'] = valuewhen(sell1, (h + l) / 2)
    out['EW_Small_1'] = ew1
    out['EW_Small_2'] = ew2

    # =====================================================
    # LARGE WAVE (ZIG 4.5%)
    # =====================================================
    zz2 = zig_afl(c, 4.5)
    out['EW_Large_1'] = tema_afl(zz2, 1)
    out['EW_Large_2'] = tema_afl(zz2, 2)

    # =====================================================
    # AUTO SEC (ZIG 4.75%)
    # =====================================================
    sens = 4.75
    zz_sec = zig_afl(c, sens)
    pk = peak(c, sens)
    tr = trough(c, sens)

    # Use small epsilon or exact match for pivot detection
    peak_cond = np.isclose(zz_sec, np.nan_to_num(pk, nan=-999999))
    trough_cond = np.isclose(zz_sec, np.nan_to_num(tr, nan=-999999))

    barpk = barssince(peak_cond)
    bartr = barssince(trough_cond)
    turnpoint = np.fmin(barpk, bartr)

    sec_mid = np.full(n, np.nan)
    sec_up = np.full(n, np.nan)
    sec_lo = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(turnpoint[i]): continue
        daysback = int(turnpoint[i]) + 1
        if daysback < 2: continue
        
        window = c[i - daysback + 1 : i + 1]
        slope, intercept, stderr = linreg_point_afl(window)
        
        sec_mid[i] = intercept
        sec_up[i] = intercept + 2 * stderr
        sec_lo[i] = intercept - 2 * stderr

    out['EW_SEC_Mid'] = sec_mid
    out['EW_SEC_Upper'] = sec_up
    out['EW_SEC_Lower'] = sec_lo

    # =====================================================
    # MEGA BUY SELL
    # =====================================================
    buy = cross(c, sec_up) | (c > sec_up)
    sell = cross(sec_lo, c) | (c < sec_lo) # breakdown lower band
    
    buy, sell = exrem(buy, sell)
    out['EW_Strong_Buy'] = buy
    out['EW_Strong_Sell'] = sell
    out['MegaBuyPrice'] = valuewhen(buy, c)
    out['MegaSellPrice'] = valuewhen(sell, c / 4)

    return out
