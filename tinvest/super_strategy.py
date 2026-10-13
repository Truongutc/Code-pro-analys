"""
Super strategy (chiến lược A)
=============================
Lọc cổ phiếu mạnh : ADX xanh/trắng + DI+ > DI- + RS14 > 70 + Heikin xu hướng tăng + mây Ichimoku xanh (SpanA > SpanB)
                    + MA khối lượng 10 phiên > 500.000 cp
Điểm mua          : 2Trend SMA vừa chuyển L (T2_SMA_Trend cắt lên 0)
Mua               : giá mở cửa phiên liền sau tín hiệu (bỏ qua nếu mở cửa tăng > 6,5% so với giá đóng cửa phiên tín hiệu)
Bán (từ T+3)      : - Cắt lỗ: giá chạm -7% so với giá mua
                    - Chặn lãi: khi giá từng chạm +5%, nếu quay về +1% thì bán
                    - Chốt lãi: Heikin chuyển xu hướng giảm -> bán ở giá mở cửa phiên sau
                    - Tín hiệu bán / giá đóng cửa dưới mức cắt lỗ xảy ra trước T+3 -> bán ở giá mở cửa phiên T+3
Backtest 2006-2026 (526 mã): thắng 69,7%, lãi TB lệnh thắng +12,7%, lỗ TB lệnh thua -6,8%, lãi ròng TB +6,8%/lệnh.
"""
import numpy as np
import pandas as pd

LOOKBACK = 30            # chỉ lưu nếu có tín hiệu trong 30 phiên gần nhất
MIN_VOL_MA10 = 500_000
CUTLOSS = 0.07
PROTECT_TRIGGER = 0.05
PROTECT_LOCK = 0.01
MAX_GAP_UP = 0.065
MIN_HOLD = 3             # chỉ bán từ phiên T+3


def _signal_mask(df: pd.DataFrame) -> np.ndarray:
    need = ['T2_SMA_Trend', 'ADX_Color', 'DI_Plus', 'DI_Minus', 'RS14', 'HK_Trend', 'SpanA', 'SpanB', 'Volume']
    if any(c not in df.columns for c in need):
        return np.zeros(len(df), bool)
    t2 = df['T2_SMA_Trend']
    vol_ma10 = df['Volume'].rolling(10).mean()
    mask = (
        (t2 > 0) & (t2.shift(1) <= 0)
        & df['ADX_Color'].astype(str).str.upper().isin(['WHITE', 'GREEN'])
        & (df['DI_Plus'] > df['DI_Minus'])
        & (df['RS14'] > 70)
        & (df['HK_Trend'] > 0)
        & (df['SpanA'] > df['SpanB'])
        & (vol_ma10 > MIN_VOL_MA10)
    )
    return mask.fillna(False).to_numpy(bool)


def _fmt_date(d) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'T2_SMA_Trend', 'DI_Plus', 'DI_Minus', 'RS14', 'HK_Trend', 'SpanA', 'SpanB']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _simulate(o, h, l, c, hk_down_evt, t: int) -> dict:
    """Simulate the trade opened by the signal at bar t. Returns buy/sell bars and prices (raw price units)."""
    n = len(c)
    b = t + 1
    if b >= n:
        return {"state": "WAIT"}
    px = o[b]
    if not px > 0 or px / c[t] - 1 > MAX_GAP_UP:
        return {"state": "SKIP", "b": b, "px": px}
    stop = px * (1 - CUTLOSS)
    protect_active = False
    pend = None              # sell reason queued for the next allowed open
    peak = px
    for k in range(b, n):
        s = k - b
        if s >= MIN_HOLD:
            if pend:
                return {"state": "SOLD", "b": b, "px": px, "k": k, "sp": o[k], "why": pend,
                        "protect_active": protect_active, "peak": peak}
            if l[k] <= stop:
                return {"state": "SOLD", "b": b, "px": px, "k": k, "sp": min(o[k], stop),
                        "why": "Chặn lãi" if protect_active else "Cắt lỗ", "protect_active": protect_active, "peak": peak}
        elif c[k] <= stop and not pend:
            pend = "Chặn lãi" if protect_active else "Cắt lỗ"
        peak = max(peak, h[k])
        if not protect_active and h[k] >= px * (1 + PROTECT_TRIGGER):
            protect_active = True
            stop = max(stop, px * (1 + PROTECT_LOCK))
        if hk_down_evt[k] and not pend:
            pend = "Heikin chuyển giảm"
    return {"state": "HOLD", "b": b, "px": px, "pend": pend, "protect_active": protect_active, "peak": peak}


def _arrays(df: pd.DataFrame):
    o, h, l, c = (df[x].to_numpy(float) for x in ['Open', 'High', 'Low', 'Close'])
    hk_down = (df['HK_Trend'] < 0).to_numpy(bool)
    hk_down_evt = hk_down & ~np.r_[False, hk_down[:-1]]
    return o, h, l, c, hk_down_evt


def super_trades(df: pd.DataFrame) -> list:
    """Every Super trade in the history (each buy signal is its own trade), for chart markers.
    Compact keys: sd=signal date, bd/bp=buy date/price, xd/xp=sell date/price, why=sell reason. Raw price units."""
    if df is None or len(df) < 60:
        return []
    df = _prepare(df)
    sig = _signal_mask(df)
    if not sig.any():
        return []
    arrs = _arrays(df)
    dates = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d").tolist()
    out = []
    for t in np.flatnonzero(sig):
        r = _simulate(*arrs, int(t))
        if r["state"] in ("WAIT", "SKIP"):
            if r["state"] == "WAIT":
                out.append({"sd": dates[t]})
            continue
        tr = {"sd": dates[t], "bd": dates[r["b"]], "bp": round(float(r["px"]), 4)}
        if r["state"] == "SOLD":
            tr.update(xd=dates[r["k"]], xp=round(float(r["sp"]), 4), why=r["why"])
        out.append(tr)
    return out


def evaluate_super(df: pd.DataFrame, price_scale: float = 1000.0) -> "dict | None":
    """Return the latest Super signal within the last LOOKBACK sessions and the state of its trade, or None."""
    if df is None or len(df) < 60:
        return None
    df = _prepare(df)
    sig = _signal_mask(df)
    n = len(df)
    recent = np.flatnonzero(sig[max(0, n - LOOKBACK):]) + max(0, n - LOOKBACK)
    if len(recent) == 0:
        return None
    t = int(recent[-1])

    o, h, l, c, hk_down_evt = _arrays(df)
    last = df.iloc[-1]
    rec = {
        "signal_date": _fmt_date(df['Date'].iloc[t]),
        "signal_close": round(c[t] * price_scale, 2),
        "combo": (f"2Trend L (SMA) · ADX {str(df['ADX_Color'].iloc[t]).capitalize()} · DI+ {df['DI_Plus'].iloc[t]:.0f} > DI- {df['DI_Minus'].iloc[t]:.0f}"
                  f" · RS14 {df['RS14'].iloc[t]:.0f} · Heikin tăng · Mây xanh"),
        "last_date": _fmt_date(last['Date']),
        "last_price": round(float(last['Close']) * price_scale, 2),
        "sessions_since_signal": int(n - 1 - t),
    }
    r = _simulate(o, h, l, c, hk_down_evt, t)
    if r["state"] == "WAIT":
        ref = c[t]
        rec.update(status="CHỜ MUA", status_note="Mua ở giá mở cửa phiên tới (bỏ qua nếu mở cửa tăng > 6,5%)",
                   buy_date=None, buy_price=None,
                   cutloss=round(ref * (1 - CUTLOSS) * price_scale, 2),
                   protect_trigger=round(ref * (1 + PROTECT_TRIGGER) * price_scale, 2),
                   protect_stop=round(ref * (1 + PROTECT_LOCK) * price_scale, 2),
                   protect_active=False, levels_ref="Giá đóng cửa phiên tín hiệu (tạm tính)")
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
        protect_trigger=round(px * (1 + PROTECT_TRIGGER) * price_scale, 2),
        protect_stop=round(px * (1 + PROTECT_LOCK) * price_scale, 2),
        protect_active=r["protect_active"],
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
    return rec
