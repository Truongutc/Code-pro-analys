"""
Super strategy (4 setups, MA khối lượng 10 phiên > 200.000 cp)
================================================================
A. Super gốc      : 2Trend SMA vừa chuyển L + ADX xanh/trắng + DI+ > DI- + RS14 > 70 + Heikin tăng + mây Ichimoku xanh -> bán LAI
B. Nhánh Octopus  : Octopus xanh đậm -> xanh nhạt + (MCDX > 0 hoặc 2Trend L) + RS14 > 70 + Heikin tăng + mây xanh    -> bán LAI
C. Chạm Bollinger : cổ phiếu mạnh (RS14 > 70, RS52 > 70, MA50 đi lên) + hôm qua đóng cửa dưới dải BB dưới,
                    hôm nay đóng cửa quay lại trong dải                                                                -> bán CŨ
D. Two-Pole sâu   : cổ phiếu mạnh (RS14 & RS52 > 70 hoặc RS rating > 80; MA50 đi lên) + Two-Pole cắt lên dưới -1     -> bán CŨ
Ưu tiên A > B > C > D nếu trùng phiên; cùng mã không mở lệnh mới trong 5 phiên kể từ tín hiệu trước.
Mua giá mở cửa phiên sau tín hiệu (bỏ qua nếu mở cửa tăng > 6,5%). Chỉ bán từ T+3; tín hiệu bán / đóng cửa dưới mức dừng
trước T+3 -> bán giá mở cửa phiên T+3.
Bán LAI : cắt lỗ 7%; giá từng chạm +7% -> chặn lãi +3%; trước khi chạm +10% bán khi Heikin chuyển giảm;
          đã chạm +10% -> bỏ Heikin, bám sóng Chandelier (giá đóng cửa cao nhất - 3 x ATR22).
Bán CŨ  : cắt lỗ 7%; giá từng chạm +5% -> chặn lãi +1%; bán khi Heikin chuyển giảm.
Backtest giá điều chỉnh 2006-10/2026: ~99 lệnh/năm (2022-26), thắng 58,9%, lệnh thắng TB +12,6%, lệnh thua TB -7,1%, +4,5%/lệnh.
"""
import numpy as np
import pandas as pd

LOOKBACK = None          # None = lấy tín hiệu mua gần nhất trong toàn bộ lịch sử
MIN_VOL_MA10 = 200_000
CUTLOSS = 0.07
MAX_GAP_UP = 0.065
MIN_HOLD = 3             # chỉ bán từ phiên T+3
DEDUPE = 5               # cùng mã không mở lệnh mới trong 5 phiên

SETUPS = {
    'A': dict(name='Super gốc (2Trend L)', exit='hybrid'),
    'B': dict(name='Nhánh Octopus', exit='hybrid'),
    'C': dict(name='Chạm Bollinger dưới', exit='classic'),
    'D': dict(name='Two-Pole quá bán sâu', exit='classic'),
}
EXIT_RULES = {
    'hybrid': dict(protect_trigger=0.07, protect_lock=0.03, trail_from=0.10, chandelier=3.0),
    'classic': dict(protect_trigger=0.05, protect_lock=0.01, trail_from=None, chandelier=None),
}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'T2_SMA_Trend', 'T2_ST_Trend', 'DI_Plus', 'DI_Minus', 'RS14', 'RS52',
                'HK_Trend', 'SpanA', 'SpanB', 'MCDX_Banker', 'MA50', 'RS_RATING']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def two_pole(close: np.ndarray, length: int = 15) -> np.ndarray:
    """Two-Pole Oscillator [BigBeluga]: normalized (close - SMA25), smoothed twice with alpha = 2/(length+1)."""
    c = pd.Series(close, dtype=float)
    x = c - c.rolling(25).mean()
    norm = ((x - x.rolling(25).mean()) / x.rolling(25).std(ddof=0)).to_numpy()
    alpha = 2.0 / (length + 1)
    out = np.full(len(c), np.nan)
    s1 = s2 = np.nan
    for i, v in enumerate(norm):
        if np.isnan(v):
            continue
        s1 = v if np.isnan(s1) else (1 - alpha) * s1 + alpha * v
        s2 = s1 if np.isnan(s2) else (1 - alpha) * s2 + alpha * s1
        out[i] = s2
    return out


def _shift(a: np.ndarray, k: int = 1) -> np.ndarray:
    return np.r_[np.full(k, np.nan), a[:-k]]


def _signals(df: pd.DataFrame) -> np.ndarray:
    """Setup letter ('A'..'D') on each signal bar, '' elsewhere (after priority and 5-session dedupe)."""
    n = len(df)
    setup = np.array([''] * n, dtype=object)
    need = ['T2_SMA_Trend', 'T2_ST_Trend', 'ADX_Color', 'DI_Plus', 'DI_Minus', 'RS14', 'RS52', 'HK_Trend', 'SpanA', 'SpanB',
            'OCT_Color', 'MCDX_Banker', 'MA50', 'Volume']
    if n < 60 or any(col not in df.columns for col in need):
        return setup
    c = df['Close'].to_numpy(float)
    liq = (df['Volume'].rolling(10).mean() > MIN_VOL_MA10).to_numpy()
    t2 = df['T2_SMA_Trend'].to_numpy(float)
    rs14, rs52 = df['RS14'].to_numpy(float), df['RS52'].to_numpy(float)
    base = (rs14 > 70) & (df['HK_Trend'] > 0).to_numpy() & (df['SpanA'] > df['SpanB']).to_numpy()
    # A. 2Trend SMA turns L + Super filter
    adx_ok = (df['ADX_Color'].astype(str).str.upper().isin(['WHITE', 'GREEN']) & (df['DI_Plus'] > df['DI_Minus'])).to_numpy()
    sig_a = (t2 > 0) & (_shift(t2) <= 0) & adx_ok & base
    # B. Octopus dark green -> light green + money flow
    oc = df['OCT_Color'].astype(str).str.upper().to_numpy()
    oc_prev = np.r_[[''], oc[:-1]]
    money = (df['MCDX_Banker'] > 0).to_numpy() | (t2 > 0) | (df['T2_ST_Trend'].to_numpy(float) > 0)
    sig_b = (oc == '#00FF00') & (oc_prev == '#008000') & money & base
    # C. strong stock + close back inside the lower Bollinger band
    ma50 = df['MA50'].to_numpy(float)
    ma50_up = ma50 / _shift(ma50, 10) - 1 > 0
    rs_strong = (rs14 > 70) & (rs52 > 70) & ma50_up
    cs = pd.Series(c)
    ma20, sd20 = cs.rolling(20).mean().to_numpy(), cs.rolling(20).std().to_numpy()
    bb_pos = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    sig_c = rs_strong & (_shift(bb_pos) < 0) & (bb_pos > 0)
    # D. strong stock + Two-Pole crosses up below -1
    rsr = df['RS_RATING'].to_numpy(float) if 'RS_RATING' in df.columns else np.full(n, np.nan)
    tp = two_pole(c)
    sig_d = (rs_strong | ((rsr > 80) & ma50_up)) & (tp > _shift(tp, 4)) & (_shift(tp) <= _shift(tp, 5)) & (tp < -1.0)
    for letter, sig in (('D', sig_d), ('C', sig_c), ('B', sig_b), ('A', sig_a)):   # later overrides earlier: A wins
        setup[np.asarray(sig, bool) & liq] = letter
    last = -10 ** 9
    for i in np.flatnonzero(setup != ''):
        if i - last <= DEDUPE:
            setup[i] = ''
        else:
            last = i
    return setup


def _arrays(df: pd.DataFrame):
    o, h, l, c = (df[x].to_numpy(float) for x in ['Open', 'High', 'Low', 'Close'])
    pc = _shift(c)
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    atr22 = pd.Series(tr).rolling(22).mean().to_numpy()
    hk_down = (df['HK_Trend'] < 0).to_numpy(bool)
    hk_down_evt = hk_down & ~np.r_[False, hk_down[:-1]]
    return o, h, l, c, atr22, hk_down_evt


def _simulate(arrs, t: int, exit_kind: str) -> dict:
    """Simulate the trade opened by the signal at bar t (raw price units)."""
    o, h, l, c, atr22, hk_down_evt = arrs
    rule = EXIT_RULES[exit_kind]
    n = len(c)
    b = t + 1
    if b >= n:
        return {"state": "WAIT"}
    px = o[b]
    if not px > 0 or px / c[t] - 1 > MAX_GAP_UP:
        return {"state": "SKIP", "b": b, "px": px}
    stop = px * (1 - CUTLOSS)
    lock = px * (1 + rule['protect_lock'])
    pend = None              # sell reason queued for the next allowed open
    hc = peak = px
    protect_active = trailing = False

    def reason():
        if stop > lock + 1e-9:
            return "Bám sóng (Chandelier)"
        return "Chặn lãi" if stop > px else "Cắt lỗ"

    for k in range(b, n):
        s = k - b
        if s >= MIN_HOLD:
            if pend:
                return dict(state="SOLD", b=b, px=px, k=k, sp=o[k], why=pend, protect_active=protect_active,
                            trailing=trailing, peak=peak, stop=stop)
            if l[k] <= stop:
                return dict(state="SOLD", b=b, px=px, k=k, sp=min(o[k], stop), why=reason(), protect_active=protect_active,
                            trailing=trailing, peak=peak, stop=stop)
        elif c[k] <= stop and not pend:
            pend = reason()
        hc = max(hc, c[k])
        peak = max(peak, h[k])
        if h[k] >= px * (1 + rule['protect_trigger']):
            protect_active = True
            stop = max(stop, lock)
        if rule['trail_from'] is not None:
            if h[k] >= px * (1 + rule['trail_from']):
                trailing = True
            if trailing and not np.isnan(atr22[k]):
                stop = max(stop, hc - rule['chandelier'] * atr22[k])
        if hk_down_evt[k] and not pend and not trailing:
            pend = "Heikin chuyển giảm"
    return dict(state="HOLD", b=b, px=px, pend=pend, protect_active=protect_active, trailing=trailing, peak=peak, stop=stop)


def _fmt_date(d) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def super_trades(df: pd.DataFrame) -> list:
    """Every Super trade in the history, for chart markers. Raw price units.
    Compact keys: s=setup, sd=signal date, bd/bp=buy date/price, xd/xp=sell date/price, why=sell reason."""
    if df is None or len(df) < 60:
        return []
    df = _prepare(df)
    setup = _signals(df)
    if not (setup != '').any():
        return []
    arrs = _arrays(df)
    dates = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d").tolist()
    out = []
    for t in np.flatnonzero(setup != ''):
        r = _simulate(arrs, int(t), SETUPS[setup[t]]['exit'])
        if r["state"] == "SKIP":
            continue
        tr = {"s": setup[t], "sd": dates[t]}
        if r["state"] != "WAIT":
            tr.update(bd=dates[r["b"]], bp=round(float(r["px"]), 4))
        if r["state"] == "SOLD":
            tr.update(xd=dates[r["k"]], xp=round(float(r["sp"]), 4), why=r["why"])
        out.append(tr)
    return out


def _combo_text(df: pd.DataFrame, t: int, letter: str) -> str:
    g = lambda col: df[col].iloc[t] if col in df.columns else np.nan
    if letter == 'A':
        return (f"A · 2Trend L (SMA) · ADX {str(g('ADX_Color')).capitalize()} · DI+ {g('DI_Plus'):.0f} > DI- {g('DI_Minus'):.0f}"
                f" · RS14 {g('RS14'):.0f} · Heikin tăng · Mây xanh")
    if letter == 'B':
        return f"B · Octopus xanh đậm→nhạt · MCDX {g('MCDX_Banker'):.1f} · RS14 {g('RS14'):.0f} · Heikin tăng · Mây xanh"
    if letter == 'C':
        return f"C · Đóng cửa dưới BB dưới rồi quay lại · RS14 {g('RS14'):.0f} · RS52 {g('RS52'):.0f} · MA50 đi lên"
    rr = g('RS_RATING')
    return (f"D · Two-Pole cắt lên dưới -1 · RS14 {g('RS14'):.0f} · RS52 {g('RS52'):.0f}"
            + (f" · RS rating {rr:.0f}" if not pd.isna(rr) else "") + " · MA50 đi lên")


def evaluate_super(df: pd.DataFrame, price_scale: float = 1000.0) -> "dict | None":
    """Return the latest Super signal (whole history unless LOOKBACK is set) and the state of its trade, or None."""
    if df is None or len(df) < 60:
        return None
    df = _prepare(df)
    setup = _signals(df)
    n = len(df)
    start = 0 if LOOKBACK is None else max(0, n - LOOKBACK)
    recent = np.flatnonzero(setup[start:] != '') + start
    if len(recent) == 0:
        return None
    t = int(recent[-1])
    letter = setup[t]
    exit_kind = SETUPS[letter]['exit']
    rule = EXIT_RULES[exit_kind]
    arrs = _arrays(df)
    c = arrs[3]
    last = df.iloc[-1]
    rec = {
        "setup": letter,
        "setup_name": SETUPS[letter]['name'],
        "exit_kind": exit_kind,
        "protect_trigger_pct": rule['protect_trigger'] * 100,
        "protect_lock_pct": rule['protect_lock'] * 100,
        "trail_from_pct": None if rule['trail_from'] is None else rule['trail_from'] * 100,
        "signal_date": _fmt_date(df['Date'].iloc[t]),
        "signal_close": round(c[t] * price_scale, 2),
        "combo": _combo_text(df, t, letter),
        "last_date": _fmt_date(last['Date']),
        "last_price": round(float(last['Close']) * price_scale, 2),
        "sessions_since_signal": int(n - 1 - t),
    }
    r = _simulate(arrs, t, exit_kind)
    if r["state"] == "WAIT":
        ref = c[t]
        rec.update(status="CHỜ MUA", status_note="Mua ở giá mở cửa phiên tới (bỏ qua nếu mở cửa tăng > 6,5%)",
                   buy_date=None, buy_price=None,
                   cutloss=round(ref * (1 - CUTLOSS) * price_scale, 2),
                   protect_trigger=round(ref * (1 + rule['protect_trigger']) * price_scale, 2),
                   protect_stop=round(ref * (1 + rule['protect_lock']) * price_scale, 2),
                   protect_active=False, trailing=False, levels_ref="Giá đóng cửa phiên tín hiệu (tạm tính)")
        return rec
    b, px = r["b"], r["px"]
    if r["state"] == "SKIP":
        rec.update(status="BỎ QUA", status_note="Phiên mua mở cửa tăng > 6,5% — chiến lược không mua đuổi",
                   buy_date=_fmt_date(df['Date'].iloc[b]), buy_price=round(px * price_scale, 2))
        return rec

    t3 = b + MIN_HOLD
    rec.update(
        buy_date=_fmt_date(df['Date'].iloc[b]),
        buy_price=round(px * price_scale, 2),
        cutloss=round(px * (1 - CUTLOSS) * price_scale, 2),
        protect_trigger=round(px * (1 + rule['protect_trigger']) * price_scale, 2),
        protect_stop=round(px * (1 + rule['protect_lock']) * price_scale, 2),
        protect_active=r["protect_active"],
        trailing=r["trailing"],
        current_stop=round(r["stop"] * price_scale, 2),
        peak_price=round(r["peak"] * price_scale, 2),
        t3_date=_fmt_date(df['Date'].iloc[t3]) if t3 < n else None,
    )
    if r["state"] == "SOLD":
        k, sp = r["k"], r["sp"]
        rec.update(status="ĐÃ BÁN", sell_date=_fmt_date(df['Date'].iloc[k]), sell_price=round(sp * price_scale, 2),
                   sell_reason=r["why"], pnl_pct=round((sp / px - 1) * 100, 2))
    else:
        pend = r["pend"]
        rec.update(status="ĐANG NẮM GIỮ", pnl_pct=round((c[-1] / px - 1) * 100, 2))
        if pend:
            rec["status_note"] = (f"{pend}: bán ở giá mở cửa phiên tới" if n - 1 - b >= MIN_HOLD - 1
                                  else f"{pend}: bán ở giá mở cửa phiên T+3 ({rec['t3_date'] or 'chưa tới'})")
        elif n - 1 - b < MIN_HOLD:
            rec["status_note"] = "Chưa về hàng (chưa tới T+3)"
        elif r["trailing"]:
            rec["status_note"] = f"Đã chạm +10%: bám sóng, bán nếu giá về {rec['current_stop']:,.0f}"
    return rec


def compute_rs_rating(dfs: dict) -> dict:
    """IBD-style RS rating (0-100): per date, percentile rank across all stocks of
    0.4*ROC60 + 0.2*ROC125 + 0.2*ROC190 + 0.2*ROC250. Returns {ticker: pd.Series indexed by Date}."""
    parts = []
    for tk, d in dfs.items():
        if d is None or len(d) < 60 or 'INDEX' in str(tk).upper() or 'Close' not in d.columns or 'Date' not in d.columns:
            continue
        cl = pd.to_numeric(d['Close'], errors='coerce').reset_index(drop=True)
        score = (0.4 * (cl / cl.shift(60) - 1) + 0.2 * (cl / cl.shift(125) - 1)
                 + 0.2 * (cl / cl.shift(190) - 1) + 0.2 * (cl / cl.shift(250) - 1))
        parts.append(pd.DataFrame({'Date': pd.to_datetime(d['Date']).dt.normalize().to_numpy(), 'T': tk, 'v': score.to_numpy()}))
    if not parts:
        return {}
    allp = pd.concat(parts, ignore_index=True).dropna(subset=['v'])
    allp['r'] = allp.groupby('Date')['v'].rank(pct=True) * 100
    return {tk: g.drop_duplicates('Date').set_index('Date')['r'] for tk, g in allp.groupby('T')}


def attach_rs_rating(df: pd.DataFrame, series: "pd.Series | None") -> pd.DataFrame:
    """Add the RS_RATING column (NaN where unknown) to a ticker frame."""
    if df is None or 'Date' not in df.columns:
        return df
    if series is None or len(series) == 0:
        df['RS_RATING'] = np.nan
    else:
        df['RS_RATING'] = pd.to_datetime(df['Date']).dt.normalize().map(series).to_numpy(float)
    return df
