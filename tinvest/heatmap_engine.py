"""
Module: Heatmap Engine
======================
Converts AmiBroker AFL logic into Python status matrices (Green/Red/Teal).
Implements 9 layers of technical confluence.
"""

import pandas as pd
import numpy as np

def calculate_stc(close_prices, ma1=23, ma2=50, tc_len=10, factor=0.5):
    """
    Schaff Trend Cycle (STC) - Ported from AFL.
    """
    # 1. MACD
    ema1 = close_prices.ewm(span=ma1, adjust=False).mean()
    ema2 = close_prices.ewm(span=ma2, adjust=False).mean()
    x_mac = ema1 - ema2
    
    # 2. 1st Stochastic
    val1 = x_mac.rolling(window=tc_len).min()
    val2 = x_mac.rolling(window=tc_len).max() - val1
    
    frac1 = np.zeros(len(close_prices))
    for i in range(1, len(close_prices)):
        if val2.iloc[i] > 0:
            frac1[i] = ((x_mac.iloc[i] - val1.iloc[i]) / val2.iloc[i]) * 100
        else:
            frac1[i] = frac1[i-1]
            
    # Smoothed PF
    pf = np.zeros(len(close_prices))
    pf[0] = frac1[0]
    for i in range(1, len(close_prices)):
        pf[i] = pf[i-1] + (factor * (frac1[i] - pf[i-1]))
        
    pf_series = pd.Series(pf)
    
    # 3. 2nd Stochastic
    val3 = pf_series.rolling(window=tc_len).min()
    val4 = pf_series.rolling(window=tc_len).max() - val3
    
    frac2 = np.zeros(len(close_prices))
    for i in range(1, len(close_prices)):
        if val4.iloc[i] > 0:
            frac2[i] = ((pf[i] - val3.iloc[i]) / val4.iloc[i]) * 100
        else:
            frac2[i] = frac2[i-1]
            
    # Smoothed PFF
    pff = np.zeros(len(close_prices))
    pff[0] = frac2[0]
    for i in range(1, len(close_prices)):
        pff[i] = pff[i-1] + (factor * (frac2[i] - pff[i-1]))
        
    return pd.Series(pff)

def calculate_pfe(close_prices, pds=10, smooth=5):
    """
    Polarized Fractal Efficiency (PFE).
    """
    roc_9 = (close_prices.diff(9))
    # Approximation of sqrt((ROC(9)^2) + 100)
    x = np.sqrt((roc_9**2) + 100)
    
    roc_1 = close_prices.diff(1).abs()
    y = np.sqrt((roc_1**2) + 1).rolling(window=pds).sum()
    
    z = x / (y + 1e-10)
    
    direction = np.where(close_prices > close_prices.shift(9), z, -z) * 100
    pfe = pd.Series(direction).ewm(span=smooth, adjust=False).mean()
    return pfe

def calculate_heatmap_matrix(df_orig):
    """
    Main entry point: Returns a DataFrame with status codes (1: Green, -1: Red, 0: Teal/Neutral).
    """
    df = df_orig.copy()
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    
    # Pre-calculate common indicators if missing
    if 'ATR14' not in df.columns:
        # Simple ATR approximation for heatmap
        high_low = h - l
        high_pc = (h - c.shift(1)).abs()
        low_pc = (l - c.shift(1)).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()

    res = pd.DataFrame(index=df.index)
    
    # 1. PFE Signal
    pfe = calculate_pfe(c)
    pfe_up = (pfe > 10) & (pfe > pfe.shift(1))
    pfe_dn = (pfe < -10) & (pfe < pfe.shift(1))
    res['PFE'] = np.select([pfe_up, pfe_dn], [1, -1], default=0)
    
    # 2. STC Signal
    stc = calculate_stc(c)
    stc_up = stc > 98
    stc_dn = stc < 2
    res['STC'] = np.select([stc_up, stc_dn], [1, -1], default=0)
    
    # 3. RSI(7) Signal
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(7).mean()
    roll_down = down.abs().rolling(7).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi7 = 100.0 - (100.0 / (1.0 + rs))
    res['RSI(7)'] = np.select([rsi7 > 70, rsi7 < 30], [1, -1], default=0)
    
    # 4. ATR Rays Signal (Simplified HHV LLV logic)
    # Pp1=3; Pp2=2; CS33 = HHV(LLV(H, 3)-ATR(2), 4)
    atr2 = df['ATR14'].rolling(2).mean() # Approx
    llv_h3 = h.rolling(3).min()
    cs33 = (llv_h3 - atr2).rolling(4).max()
    res['Rays'] = np.select([c > cs33, cs33 > c], [1, -1], default=0)
    
    # 5. Exit Beast Signal
    # EntrySig = C > ( LLV( L, 10 ) + 1.9 * ATR( 10 ) )
    # ExitSig = C < ( HHV( H, 10 ) - 1.9 * ATR( 10 ) )
    llv_l10 = l.rolling(10).min()
    hhv_h10 = h.rolling(10).max()
    atr10 = df['ATR14'].rolling(10).mean()
    res['Beast'] = np.select([c > (llv_l10 + 1.9 * atr10), c < (hhv_h10 - 1.9 * atr10)], [1, -1], default=0)
    
    # 6. CCI Signal (CCI 9 vs CCI 8)
    # AFL uses CCI(9) > 0 and CCI(8) < 0
    # TP = (H+L+C)/3; CCI = (TP - MA(TP))/ (0.015 * MD(TP))
    def cci(data_h, data_l, data_c, period):
        tp = (data_h + data_l + data_c) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md + 1e-10)
    
    cci9 = cci(h, l, c, 9)
    cci8 = cci(h, l, c, 8)
    res['CCI'] = np.select([cci9 > 0, cci8 < 0], [1, -1], default=0)
    
    # 7. %BB Signal
    # x = ((C+2*StDev(C,7)-MA(C,7))/(4*StDev(C,7)))*100
    ma7 = c.rolling(7).mean()
    sd7 = c.rolling(7).std()
    bb_x = ((c + 2*sd7 - ma7) / (4*sd7 + 1e-10)) * 100
    res['%BB'] = np.select([bb_x > 40, bb_x < 40], [1, -1], default=0)
    
    # 8. MACD Bull/Bear State
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    m_up = (macd > 0) & (macd > signal)
    m_dn = (macd < 0) & (macd < signal)
    res['MACD'] = np.select([m_up, m_dn], [1, -1], default=0)
    
    # 9. Volume Spiker (Money Flow Convergence) - AFL Spiker Logic
    c1 = c.shift(1)
    green_vol = np.where((c > df['Open']) & (c > c1), v, 0)
    blue_vol = np.where((c <= df['Open']) & (c > c1), v, 0)
    red_vol = np.where((c <= df['Open']) & (c <= c1), v, 0)
    yellow_vol = np.where((c > df['Open']) & (c <= c1), v, 0)
    
    uv = pd.Series(green_vol + blue_vol)
    dv = pd.Series(red_vol + yellow_vol)
    
    def tema(series, p):
        e1 = series.ewm(span=p, adjust=False).mean()
        e2 = e1.ewm(span=p, adjust=False).mean()
        e3 = e2.ewm(span=p, adjust=False).mean()
        return 3*e1 - 3*e2 + e3
        
    mauv = tema(uv, 10)
    madv = tema(dv, 10)
    converge = tema(mauv - madv, 4)
    rising = (converge > converge.shift(1)) & (converge > 0)
    falling = (converge <= converge.shift(1)) & (converge > 0)
    
    res['MoneyFlow'] = np.select([rising, falling], [1, 0.5], default=-1)

    # ── 10. Flower OHLC (Smoothed Candles) ──────────────────────────────────
    # fC=EMA((O+H+L+C)/4,3); fO=EMA((Ref(O,-1)+Ref(fC,-1))/2,3);
    typical = (df['Open'] + h + l + c) / 4.0
    res['Flower_Close'] = typical.ewm(span=3, adjust=False).mean()
    
    n = len(df)
    f_open = np.zeros(n)
    f_open[0] = (df['Open'].iloc[0] + res['Flower_Close'].iloc[0]) / 2.0
    for i in range(1, n):
        # fO=EMA((Ref(O,-1)+Ref(fC,-1))/2,3);
        # Use simple EMA(prev_val, 3) approximation from the chain
        prev_mix = (df['Open'].iloc[i-1] + res['Flower_Close'].iloc[i-1]) / 2.0
        f_open[i] = f_open[i-1] + (2.0/4.0) * (prev_mix - f_open[i-1]) # EMA(3) alpha = 2/(3+1)
        
    res['Flower_Open'] = pd.Series(f_open, index=df.index)
    res['Flower_High'] = pd.concat([h, res['Flower_Open'], res['Flower_Close']], axis=1).max(axis=1).ewm(span=3, adjust=False).mean()
    res['Flower_Low'] = pd.concat([l, res['Flower_Open'], res['Flower_Close']], axis=1).min(axis=1).ewm(span=3, adjust=False).mean()

    # ── 11. Bands Cloud (Heatmap Clouds) ──────────────────────────────────
    # bk = 40
    bk = 40
    hi_40 = h.rolling(bk).max()
    lo_40 = l.rolling(bk).min()
    
    res['Band_Hi'] = hi_40
    res['Band_Lo'] = lo_40
    res['Band_KM'] = (hi_40 + lo_40) / 2
    res['Band_KH'] = (res['Band_KM'] + hi_40) / 2
    res['Band_KL'] = (res['Band_KM'] + lo_40) / 2
    
    # Bands Long (Background Backdrop)
    sut = 60; ret = 160
    res['Band_Long_Hr'] = h.rolling(ret).max()
    res['Band_Long_Ls'] = l.rolling(sut).min()

    return res
