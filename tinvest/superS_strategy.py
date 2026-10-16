"""
SuperS strategy (MA khối lượng 10 phiên > 200.000 cp)
=====================================================
Bộ chính có 4 điểm mua, cùng một kiểu bán GIỮ LÃI. Cùng mã không mua thêm khi đang giữ lệnh.
Mua giá mở cửa phiên liền sau tín hiệu (bỏ qua nếu mở cửa tăng > 6,5%). Chỉ bán từ T+3; tín hiệu bán /
đóng cửa dưới mức dừng trước T+3 -> bán giá mở cửa phiên T+3. Trùng phiên: ưu tiên S1 > N3 > N2 > S2.

S1 · Sóng mạnh: 2Trend SMA vừa chuyển L + ADX xanh/trắng + DI+ > DI- + RS14 > 70 + Heikin trên đường NW
     + mây Ichimoku xanh + giá đóng cửa trên mây + Octopus xanh (đậm hoặc nhạt)
S2 · Thị trường quay lại + nền chặt: VNINDEX vừa vào xu hướng tăng trong 3 phiên gần nhất (đóng cửa > MA50 và
     MA20 > MA50) + 2Trend SuperTrend L + Bollinger thắt chặt (độ rộng BB20 thuộc 20% thấp nhất 120 phiên) + RS14 > 50
N2 · Bứt phá nền chặt: ADX vừa vượt 25 + MACD > Signal + khối lượng > 1,5 x TB20 + Bollinger thắt chặt
N3 · Thị trường quay lại + bùng khối lượng: VNINDEX vào xu hướng tăng ĐÚNG phiên này + 2Trend SuperTrend L
     + khối lượng > 1,5 x TB20 + Bollinger thắt chặt

Bán GIỮ LÃI: cắt lỗ -7%; giá từng chạm +7% -> chặn lãi +2%; khi giá đóng cửa cao nhất đã lãi >= +10% -> không để
     trả lại quá 50% phần lãi cao nhất đó; luôn kèm bám sóng Chandelier (giá đóng cửa cao nhất - 4 x ATR22).
     Mức dừng = mức cao nhất trong các mức trên; giá thấp nhất phiên chạm mức dừng -> bán.

Backtest giá điều chỉnh (CafeF, 2006-10/2026, phí mua 0,15%, bán 0,25%): ~119 lệnh/năm (2022–2026), thắng ~60%.
"""
import numpy as np
import pandas as pd

MIN_VOL_MA10 = 200_000
CUTLOSS = 0.07
MAX_GAP_UP = 0.065
MIN_HOLD = 3                     # chỉ bán từ phiên T+3
VN_TURN_WINDOW = 3               # S2: VNINDEX vừa vào xu hướng tăng trong N phiên gần nhất
BB_SQUEEZE_PCT = 0.20            # độ rộng BB thuộc 20% thấp nhất của 120 phiên
VOL_SURGE = 1.5                  # khối lượng > 1,5 x TB20

SETUPS = {
    'S1': 'Sóng mạnh',
    'N3': 'Thị trường quay lại + bùng khối lượng',
    'N2': 'Bứt phá nền chặt',
    'S2': 'Thị trường quay lại + nền chặt',
}
PRIORITY = ['S2', 'N2', 'N3', 'S1']          # later overrides earlier: S1 wins
EXIT = dict(protect_trigger=0.07, protect_lock=0.02, giveback_from=0.10, giveback=0.50, chandelier=4.0)


def _num(df, col):
    return pd.to_numeric(df[col], errors='coerce').to_numpy(float) if col in df.columns else np.full(len(df), np.nan)


def _shift(a: np.ndarray, k: int = 1) -> np.ndarray:
    return np.r_[np.full(k, np.nan), a[:-k]]


def _onset(state: np.ndarray) -> np.ndarray:
    return state & ~np.r_[False, state[:-1]]


def vn_uptrend(vn_df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """Per VNINDEX date: 'on' = switched into the uptrend (Close > MA50 and MA20 > MA50) this session,
    'recent' = switched within the last VN_TURN_WINDOW sessions."""
    if vn_df is None or len(vn_df) < 60 or 'Close' not in vn_df.columns:
        return None
    d = vn_df[['Date', 'Close']].copy()
    d['Date'] = pd.to_datetime(d['Date']).dt.normalize()
    d = d.drop_duplicates('Date').sort_values('Date')
    c = pd.to_numeric(d['Close'], errors='coerce')
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    up = (c > ma50) & (ma20 > ma50)
    on = up & ~up.shift(1, fill_value=False)
    recent = on.astype(float).rolling(VN_TURN_WINDOW, min_periods=1).max() > 0
    return pd.DataFrame({'on': on.to_numpy(), 'recent': recent.to_numpy()}, index=d['Date'])


def _signals(df: pd.DataFrame, vn: "pd.DataFrame | None") -> np.ndarray:
    """Setup code on each signal bar, '' elsewhere (before the no-overlap rule)."""
    n = len(df)
    main = np.array([''] * n, dtype=object)
    if n < 60:
        return main
    c, v = _num(df, 'Close'), _num(df, 'Volume')
    vs = pd.Series(v)
    volma10 = vs.rolling(10).mean().to_numpy()
    liq = volma10 > MIN_VOL_MA10
    vol_surge = v > VOL_SURGE * vs.rolling(20).mean().to_numpy()
    rs14 = _num(df, 'RS14')
    cs = pd.Series(c)
    ma20, sd20 = cs.rolling(20, min_periods=16).mean(), cs.rolling(20, min_periods=16).std()
    squeeze = ((4 * sd20 / ma20).rolling(120, min_periods=60).rank(pct=True) < BB_SQUEEZE_PCT).to_numpy()
    st_l = _num(df, 'T2_ST_Trend') > 0
    # S1 · Sóng mạnh
    t2 = _num(df, 'T2_SMA_Trend')
    adx_col = df['ADX_Color'].astype(str).str.upper().to_numpy() if 'ADX_Color' in df.columns else np.array([''] * n)
    oct_col = df['OCT_Color'].astype(str).str.upper().to_numpy() if 'OCT_Color' in df.columns else np.array([''] * n)
    cloud_top = np.fmax(_num(df, 'SpanA'), _num(df, 'SpanB'))
    s1 = ((t2 > 0) & (_shift(t2) <= 0)
          & np.isin(adx_col, ['GREEN', 'WHITE']) & (_num(df, 'DI_Plus') > _num(df, 'DI_Minus'))
          & (rs14 > 70) & (_num(df, 'HK_Trend') > 0) & (_num(df, 'SpanA') > _num(df, 'SpanB'))
          & (c > cloud_top) & np.isin(oct_col, ['#00FF00', '#008000']))
    # N2 · Bứt phá nền chặt
    n2 = _onset(_num(df, 'ADX') > 25) & (_num(df, 'MACD') > _num(df, 'MACD_Signal')) & vol_surge & squeeze
    # S2 / N3 need the VNINDEX regime
    s2 = n3 = np.zeros(n, bool)
    if vn is not None:
        dates = pd.to_datetime(df['Date']).dt.normalize()
        vn_on = dates.map(vn['on']).fillna(False).to_numpy(bool)
        vn_recent = dates.map(vn['recent']).fillna(False).to_numpy(bool)
        s2 = vn_recent & st_l & squeeze & (rs14 > 50)
        n3 = vn_on & st_l & vol_surge & squeeze
    for code, sig in zip(PRIORITY, (s2, n2, n3, s1)):
        main[np.asarray(sig, bool) & liq] = code
    return main


def _arrays(df: pd.DataFrame):
    o, h, l, c = (_num(df, x) for x in ['Open', 'High', 'Low', 'Close'])
    pc = _shift(c)
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    atr22 = pd.Series(tr).rolling(22).mean().to_numpy()
    return o, h, l, c, atr22


def _simulate(arrs, t: int) -> dict:
    """Simulate the trade opened by the signal at bar t with the GIỮ LÃI exit (raw price units)."""
    o, h, l, c, atr22 = arrs
    n = len(c)
    b = t + 1
    if b >= n:
        return {"state": "WAIT"}
    px = o[b]
    if not px > 0 or px / c[t] - 1 > MAX_GAP_UP:
        return {"state": "SKIP", "b": b, "px": px}
    stop = px * (1 - CUTLOSS)
    protect = keep = chand = 0.0
    pend = pend_k = None
    hc = peak = px

    def reason():
        if stop < px:
            return "Cắt lỗ"
        best = max(protect, keep, chand)
        return "Bám sóng (Chandelier)" if best == chand else ("Giữ 50% lãi" if best == keep else "Chặn lãi")

    def result(state, **kw):
        return dict(state=state, b=b, px=px, peak=peak, best_close=hc, stop=stop, protect=protect, keep=keep, chand=chand, **kw)

    for k in range(b, n):
        if k - b >= MIN_HOLD:
            if pend:
                return result("SOLD", k=k, sp=o[k], why=pend)
            if l[k] <= stop:
                return result("SOLD", k=k, sp=min(o[k], stop), why=reason())
        elif c[k] <= stop and not pend:
            pend, pend_k = reason(), k
        hc = max(hc, c[k])
        peak = max(peak, h[k])
        if h[k] >= px * (1 + EXIT['protect_trigger']):
            protect = px * (1 + EXIT['protect_lock'])
        if hc >= px * (1 + EXIT['giveback_from']):
            keep = max(keep, px * (1 + (1 - EXIT['giveback']) * (hc / px - 1)))
        if h[k] >= px and not np.isnan(atr22[k]):
            chand = max(chand, hc - EXIT['chandelier'] * atr22[k])
        stop = max(stop, protect, keep, chand)
    return result("HOLD", pend=pend, pend_k=pend_k)


def _walk(df: pd.DataFrame, vn_df) -> list:
    """All trades in order, with the no-overlap rule: (bar t, setup, simulation result)."""
    setup = _signals(df, vn_uptrend(vn_df))
    arrs = _arrays(df)
    out, busy_until = [], -1
    for t in np.flatnonzero(setup != ''):
        if t <= busy_until:
            continue
        r = _simulate(arrs, int(t))
        out.append((int(t), setup[t], r))
        if r["state"] == "SOLD":
            busy_until = r["k"]
        elif r["state"] in ("HOLD", "WAIT"):
            busy_until = len(df)
    return out


def _fmt_date(d) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def superS_trades(df: pd.DataFrame, vn_df=None) -> list:
    """Every SuperS trade for chart markers. Raw price units.
    Keys: s=setup, sd=signal date, bd/bp=buy date/price, xd/xp=sell date/price, why=sell reason."""
    if df is None or len(df) < 60:
        return []
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d").tolist()
    out = []
    for t, code, r in _walk(df, vn_df):
        if r["state"] == "SKIP":
            continue
        tr = {"s": code, "sd": dates[t]}
        if r["state"] != "WAIT":
            tr.update(bd=dates[r["b"]], bp=round(float(r["px"]), 4))
        if r["state"] == "SOLD":
            tr.update(xd=dates[r["k"]], xp=round(float(r["sp"]), 4), why=r["why"])
        out.append(tr)
    return out


def _combo_text(df: pd.DataFrame, t: int, code: str) -> str:
    g = lambda col: pd.to_numeric(df[col], errors='coerce').iloc[t] if col in df.columns else np.nan
    if code == 'S1':
        return (f"S1 · 2Trend SMA vừa chuyển L · ADX {str(df['ADX_Color'].iloc[t]).capitalize()} · DI+ {g('DI_Plus'):.0f} > DI− {g('DI_Minus'):.0f}"
                f" · RS14 {g('RS14'):.0f} · Heikin trên NW · mây xanh, giá trên mây · Octopus xanh")
    if code == 'S2':
        return f"S2 · VNINDEX vừa quay lại xu hướng tăng · 2Trend SuperTrend L · Bollinger thắt chặt · RS14 {g('RS14'):.0f}"
    if code == 'N2':
        return f"N2 · ADX vừa vượt 25 ({g('ADX'):.0f}) · MACD > Signal · khối lượng > 1,5× TB20 · Bollinger thắt chặt"
    return "N3 · VNINDEX vào xu hướng tăng phiên này · 2Trend SuperTrend L · khối lượng > 1,5× TB20 · Bollinger thắt chặt"


def evaluate_superS(df: pd.DataFrame, vn_df=None, price_scale: float = 1000.0) -> "dict | None":
    """Latest SuperS trade of the ticker (whole history) and its state, or None if it never signalled."""
    if df is None or len(df) < 60:
        return None
    df = df.reset_index(drop=True)
    trades = _walk(df, vn_df)
    if not trades:
        return None
    t, code, r = trades[-1]
    c = _num(df, 'Close')
    n = len(df)
    P = lambda v: None if v is None else round(float(v) * price_scale, 2)
    rec = {
        "setup": code,
        "setup_name": SETUPS[code],
        "signal_date": _fmt_date(df['Date'].iloc[t]),
        "signal_close": P(c[t]),
        "combo": _combo_text(df, t, code),
        "last_date": _fmt_date(df['Date'].iloc[-1]),
        "last_price": P(c[-1]),
        "sessions_since_signal": int(n - 1 - t),
    }
    ref = c[t] if r["state"] == "WAIT" else r["px"]
    rec.update(cutloss=P(ref * (1 - CUTLOSS)), protect_trigger=P(ref * (1 + EXIT['protect_trigger'])),
               protect_stop=P(ref * (1 + EXIT['protect_lock'])), keep_trigger=P(ref * (1 + EXIT['giveback_from'])))
    if r["state"] == "WAIT":
        rec.update(status="CHỜ MUA", status_note="Mua ở giá mở cửa phiên tới (bỏ qua nếu mở cửa tăng > 6,5%)",
                   buy_date=None, buy_price=None, protect_active=False, keep_active=False)
        return rec
    b, px = r["b"], r["px"]
    if r["state"] == "SKIP":
        rec.update(status="BỎ QUA", status_note="Phiên mua mở cửa tăng > 6,5% — chiến lược không mua đuổi",
                   buy_date=_fmt_date(df['Date'].iloc[b]), buy_price=P(px))
        return rec
    t3 = b + MIN_HOLD
    rec.update(
        buy_date=_fmt_date(df['Date'].iloc[b]), buy_price=P(px),
        protect_active=bool(r["protect"] > 0), keep_active=bool(r["keep"] > 0),
        keep_stop=P(r["keep"]) if r["keep"] > 0 else None, chandelier_stop=P(r["chand"]) if r["chand"] > 0 else None,
        best_close=P(r["best_close"]), current_stop=P(r["stop"]), peak_price=P(r["peak"]),
        t3_date=_fmt_date(df['Date'].iloc[t3]) if t3 < n else None,
    )
    if r["state"] == "SOLD":
        k, sp = r["k"], r["sp"]
        rec.update(status="ĐÃ BÁN", sell_date=_fmt_date(df['Date'].iloc[k]), sell_price=P(sp), sell_reason=r["why"],
                   pnl_pct=round((sp / px - 1) * 100, 2))
        return rec
    rec.update(status="ĐANG NẮM GIỮ", pnl_pct=round((c[-1] / px - 1) * 100, 2))
    if r["pend"]:
        rec.update(pending_sell=r["pend"], pending_date=_fmt_date(df['Date'].iloc[r["pend_k"]]),
                   pending_exec="phiên tới" if n - 1 - b >= MIN_HOLD - 1 else f"phiên T+3 ({rec['t3_date'] or 'chưa tới'})")
        rec["status_note"] = f"{r['pend']}: bán ở giá mở cửa {rec['pending_exec']}"
    elif n - 1 - b < MIN_HOLD:
        rec["status_note"] = f"Chưa về hàng (chưa tới T+3) · mức dừng hiện tại {rec['current_stop']:,.0f}"
    else:
        rec["status_note"] = f"Bán nếu giá chạm {rec['current_stop']:,.0f}"
    return rec
