"""
gold_engine.py — محرك الاستراتيجيات الذهبية + مستويات ذكية + Backtest
مع TP1 ونقل الوقف.
"""
import numpy as np
import pandas as pd

from constants import CONFLUENCE, RSI2, SQUEEZE, TURTLE, WYCKOFF, STRATEGY_FAMILIES

# gold_engine يعيد تصدير هذه الأسماء ضمن namespace الخاص به
NEW_STRATEGIES = [CONFLUENCE, SQUEEZE, TURTLE, RSI2, WYCKOFF]

SL_MIN_ATR, SL_MAX_ATR, SL_BUFFER_ATR = 1.0, 2.5, 0.25
TP1_R, TP2_R = 1.5, 3.0
MIN_RR_TP1 = 1.0


def _safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------ مؤشرات إضافية
def add_extra_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["SMA5"] = c.rolling(5).mean()
    df["SMA20"] = c.rolling(20).mean()
    df["SMA200"] = c.rolling(200).mean()

    # RSI(2)
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    df["RSI2"] = 100 - 100 / (1 + g / l.replace(0, np.nan))

    # Keltner + Squeeze
    kel_u = df["EMA20"] + 1.5 * df["ATR14"]
    kel_l = df["EMA20"] - 1.5 * df["ATR14"]
    df["SQUEEZE_ON"] = (df["BB_Upper"] < kel_u) & (df["BB_Lower"] > kel_l)

    # Donchian 55 / 10
    df["Don55_Up"] = df["High"].rolling(55).max()
    df["Don55_Dn"] = df["Low"].rolling(55).min()
    df["Don10_Up"] = df["High"].rolling(10).max()
    df["Don10_Dn"] = df["Low"].rolling(10).min()

    # MACD histogram
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()

    # قمم/قيعان قريبة
    df["Low5"] = df["Low"].rolling(5).min()
    df["High5"] = df["High"].rolling(5).max()
    return df


# ------------------------------------------------ الإشارات
def extra_signals(df, base_sigs):
    """base_sigs = خرج strategy_signals الأصلي. يرجع قاموسًا بالإشارات الجديدة."""
    c = df["Close"]

    def to_sig(buy, sell):
        return pd.Series(
            np.select([buy.fillna(False), sell.fillna(False)], [1, -1], default=0),
            index=df.index,
        )

    was_sq = (
        df["SQUEEZE_ON"].astype(float).shift(1).rolling(5).max().fillna(0).astype(bool)
    )
    squeeze = to_sig(
        was_sq & (c > df["BB_Upper"]) & (df["MACD_H"] > 0) & (c > df["EMA50"]),
        was_sq & (c < df["BB_Lower"]) & (df["MACD_H"] < 0) & (c < df["EMA50"]),
    )

    turtle = to_sig(
        (c > df["Don55_Up"].shift(1)) & (c > df["EMA200"]) & (df["ADX14"] > 20),
        (c < df["Don55_Dn"].shift(1)) & (c < df["EMA200"]) & (df["ADX14"] > 20),
    )

    rsi2 = to_sig(
        (c > df["SMA200"]) & (df["RSI2"] < 10) & (c < df["SMA5"]),
        (c < df["SMA200"]) & (df["RSI2"] > 90) & (c > df["SMA5"]),
    )

    # Confluence: تصويت بحسب العائلة، لا بحسب الاستراتيجية الفردية
    all_signals = {}
    for k in (
        "Breakout (اختراق)",
        "Trend + Pullback (اتجاه+ارتداد)",
        "Mean Reversion (عودة للمتوسط)",
        "Trend Follow (تتبع الاتجاه)",
    ):
        if k in base_sigs:
            all_signals[k] = base_sigs[k]
    all_signals.update({SQUEEZE: squeeze, TURTLE: turtle, RSI2: rsi2})

    fam_series = {}
    for name, sig in all_signals.items():
        fam = STRATEGY_FAMILIES.get(name)
        if not fam or fam == "ensemble":
            continue
        fam_series.setdefault(fam, []).append(sig)

    fam_votes = {}
    for fam, sigs in fam_series.items():
        stacked = pd.concat(sigs, axis=1)
        ups = (stacked == 1).sum(axis=1)
        dns = (stacked == -1).sum(axis=1)
        fam_votes[fam] = pd.Series(
            np.select([ups > dns, dns > ups], [1, -1], default=0),
            index=df.index,
        )

    if fam_votes:
        fam_df = pd.concat(fam_votes.values(), axis=1)
        ups_f = (fam_df == 1).sum(axis=1)
        dns_f = (fam_df == -1).sum(axis=1)
        conf = pd.Series(
            np.select(
                [(ups_f >= 2) & (dns_f == 0), (dns_f >= 2) & (ups_f == 0)],
                [1, -1],
                default=0,
            ),
            index=df.index,
        )
    else:
        conf = pd.Series(0, index=df.index)

    return {SQUEEZE: squeeze, TURTLE: turtle, RSI2: rsi2, CONFLUENCE: conf,
            WYCKOFF: wyckoff_signals(df)}


# ------------------------------------------------ Wyckoff (الجزء الميكانيكي فقط)
# نطاق جانبي واضح ثم كسر كاذب لحدّه يعود الإغلاق بعده داخله:
#   Spring  (شراء): يخترق القاع قليلاً ثم يُغلق داخل النطاق في النصف الأعلى من الشمعة.
#   Upthrust (بيع): يخترق القمة قليلاً ثم يُغلق داخل النطاق في النصف الأدنى.
# لا مراحل (A/B/C) ولا حجم: الأول حكم شخصي والثاني غير موثوق في الفوركس. مرشّحة تُحاكَم
# بالأرقام خارج العيّنة؛ ليست داخل AUTO ولا Confluence.
WYCKOFF_LOOKBACK = 40
WYCKOFF_RANGE_MAX_ATR = 10.0     # أقصى اتساع للنطاق بوحدات ATR (نطاق "ضيق" فعلاً)
WYCKOFF_ADX_MAX = 22.0           # لا نطاق إن كان السوق في اتجاه
WYCKOFF_PIERCE_MAX_ATR = 1.0     # أعمق اختراق مقبول (أعمق = كسر حقيقي لا فخ)


def wyckoff_signals(df):
    """إشارات سببية: كل شمعة تعتمد على ما قبلها فقط (النطاق مُزاح شمعة)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    atr = df["ATR14"].shift(1)
    rh = h.shift(1).rolling(WYCKOFF_LOOKBACK).max()
    rl = l.shift(1).rolling(WYCKOFF_LOOKBACK).min()
    in_range = ((rh - rl) <= WYCKOFF_RANGE_MAX_ATR * atr) & (df["ADX14"].shift(1) < WYCKOFF_ADX_MAX)
    mid = (h + l) / 2.0
    spring = in_range & (l < rl) & (c > rl) & ((rl - l) <= WYCKOFF_PIERCE_MAX_ATR * atr) & (c > mid)
    upthrust = in_range & (h > rh) & (c < rh) & ((h - rh) <= WYCKOFF_PIERCE_MAX_ATR * atr) & (c < mid)
    return pd.Series(np.select([spring, upthrust], [1, -1], default=0), index=df.index)


# ------------------------------------------------ مستويات ذكية
def smart_levels(signal, entry, df, i=-1, mean_rev=False):
    """Return deterministic SL/TP levels from completed-bar data only."""
    if signal not in ("BUY", "SELL"):
        return {
            "sl": np.nan,
            "tp1": np.nan,
            "tp2": np.nan,
            "risk_dist": np.nan,
            "rr1": np.nan,
            "rr2": np.nan,
            "valid": False,
        }
    row = df.iloc[i]
    atr = _safe_float(row.get("ATR14"))
    entry = _safe_float(entry)
    low5 = _safe_float(row.get("Low5"))
    high5 = _safe_float(row.get("High5"))
    if (
        not np.isfinite(atr)
        or atr <= 0
        or not np.isfinite(entry)
        or not np.isfinite(low5)
        or not np.isfinite(high5)
    ):
        return {
            "sl": np.nan,
            "tp1": np.nan,
            "tp2": np.nan,
            "risk_dist": np.nan,
            "rr1": np.nan,
            "rr2": np.nan,
            "valid": False,
        }

    s = 1 if signal == "BUY" else -1
    swing = low5 - SL_BUFFER_ATR * atr if s == 1 else high5 + SL_BUFFER_ATR * atr
    sd = min(max(abs(entry - swing), SL_MIN_ATR * atr), SL_MAX_ATR * atr)
    if not np.isfinite(sd) or sd <= 0:
        return {
            "sl": np.nan,
            "tp1": np.nan,
            "tp2": np.nan,
            "risk_dist": np.nan,
            "rr1": np.nan,
            "rr2": np.nan,
            "valid": False,
        }

    sl = entry - s * sd
    tp1 = entry + s * TP1_R * sd
    tp2 = entry + s * TP2_R * sd

    if mean_rev:
        mean = _safe_float(row.get("SMA20"))
        if np.isfinite(mean) and s * (mean - entry) >= sd * MIN_RR_TP1:
            tp1 = mean
        tp2 = entry + s * 2.5 * sd
        if s * (tp2 - tp1) <= 0:
            tp2 = tp1 + s * 0.5 * sd

    rr1 = abs(tp1 - entry) / sd
    rr2 = abs(tp2 - entry) / sd
    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_dist": sd,
        "rr1": rr1,
        "rr2": rr2,
        "valid": bool(rr1 >= MIN_RR_TP1),
    }


# ---------------------------------------------------------------- محرك الصفقة المشترك
# مصدر حقيقة واحد: الباكتست والمحاكي الورقي يستدعيان الدوال نفسها، فلا يمكن أن يختلف سلوكهما.
def new_trade_state(s, entry, sl, tp1, tp2, partial=0.5):
    return {"s": int(s), "entry": float(entry), "sl": float(sl), "sl0": float(sl),
            "tp1": float(tp1) if tp1 is not None else np.nan, "tp2": float(tp2),
            "partial": float(min(max(partial, 0.0), 1.0)), "stage": 0, "realized_gross": 0.0}


def step_trade(t, o, h, l):
    """يقدّم الصفقة شمعة واحدة (يعدّل t في مكانه). يرجع (سبب, سعر) عند الخروج، وإلا None.
    ترتيب تحفّظي داخل الشمعة: الوقف أولاً؛ وبعد TP1 إن لامست الشمعة نفسها الدخول نفترض BE.
    الفجوات: الوقف يُنفَّذ عند الافتتاح إن كان أسوأ. tp1=NaN => صفقة بهدف واحد (سجلات قديمة)."""
    s, entry = t["s"], t["entry"]
    if (l <= t["sl"]) if s == 1 else (h >= t["sl"]):
        px = min(t["sl"], o) if s == 1 else max(t["sl"], o)
        return ("BE" if t["stage"] == 1 else "SL"), px
    if t["stage"] == 0 and ((h >= t["tp1"]) if s == 1 else (l <= t["tp1"])):
        t["realized_gross"] = t["partial"] * s * (t["tp1"] - entry) / entry * 100.0
        t["stage"] = 1
        t["sl"] = entry
        if (l <= entry) if s == 1 else (h >= entry):
            return "BE", entry
    single_target = t["tp1"] != t["tp1"]
    if (t["stage"] == 1 or single_target) and ((h >= t["tp2"]) if s == 1 else (l <= t["tp2"])):
        return "TP", t["tp2"]
    return None


def trade_pnl(t, exit_price, cost_pct=0.0):
    """(gross%, net%, R صافٍ, R خام) لصفقة خرجت عند exit_price. R بالنسبة للوقف الأولي."""
    s, entry = t["s"], t["entry"]
    rem = (1.0 - t["partial"]) if t["stage"] == 1 else 1.0
    gross = t["realized_gross"] + rem * s * (exit_price - entry) / entry * 100.0
    net = gross - cost_pct
    risk_pct = abs(entry - t["sl0"]) / entry * 100.0
    if risk_pct > 0:
        return gross, net, net / risk_pct, gross / risk_pct
    return gross, net, np.nan, np.nan


# ---------------------------------------------------------------- Backtest v2
def run_backtest_v2(df, signals, cost_pct=0.05, max_hold=30, start=200, mean_rev=False, partial=0.5):
    """
    Backtest: دخول عند افتتاح الشمعة التالية، وقف ذكي، TP1 جزئي ثم BE (عبر المحرك المشترك).

    mean_rev: bool، أو مصفوفة bool بطول df (لكل شمعة إشارة) — لازم لـ AUTO الذي يخلط استراتيجيات
    اتجاهية وانعكاسية. ``pnl`` و``r`` صافيان بعد التكلفة؛ ``r_gross`` قبلها.
    """
    required = ["Open", "High", "Low", "Close", "ATR14", "Low5", "High5", "SMA20"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"أعمدة الباكتيست مفقودة: {missing}")
    if len(df) == 0 or len(signals) != len(df):
        return pd.DataFrame()
    partial = min(max(float(partial), 0.0), 1.0)
    cost_pct = max(float(cost_pct), 0.0)
    max_hold = max(int(max_hold), 1)
    start = max(int(start), 1)

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    atr = df["ATR14"].to_numpy(dtype=float)
    low5 = df["Low5"].to_numpy(dtype=float)
    high5 = df["High5"].to_numpy(dtype=float)
    sma20 = df["SMA20"].to_numpy(dtype=float)
    sig = pd.Series(signals, index=df.index).fillna(0).to_numpy(dtype=int)
    n = len(df)
    mr_arr = np.broadcast_to(np.asarray(mean_rev, dtype=bool), (n,))
    trades, i = [], min(start, n - 1)

    while i < n - 1:
        s = int(sig[i])
        if s not in (-1, 1) or not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(low5[i]) or not np.isfinite(high5[i]):
            i += 1
            continue

        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        swing = low5[i] - SL_BUFFER_ATR * atr[i] if s == 1 else high5[i] + SL_BUFFER_ATR * atr[i]
        sd = min(max(abs(entry - swing), SL_MIN_ATR * atr[i]), SL_MAX_ATR * atr[i])
        if not np.isfinite(sd) or sd <= 0:
            i += 1
            continue

        sl = entry - s * sd
        tp1 = entry + s * TP1_R * sd
        tp2 = entry + s * TP2_R * sd
        mr = bool(mr_arr[i])
        if mr and np.isfinite(sma20[i]):
            if s * (sma20[i] - entry) >= sd * MIN_RR_TP1:
                tp1 = sma20[i]
            tp2 = entry + s * 2.5 * sd
            if s * (tp2 - tp1) <= 0:
                tp2 = tp1 + s * 0.5 * sd
        if abs(tp1 - entry) / sd < MIN_RR_TP1:
            i += 1
            continue

        t = new_trade_state(s, entry, sl, tp1, tp2, partial)
        exit_idx = min(i + max_hold, n - 1)
        exit_price, reason = c[exit_idx], "Time"
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            res = step_trade(t, o[j], h[j], l[j])
            if res is not None:
                reason, exit_price = res
                exit_idx = j
                break

        gross_pnl, net_pnl, r_net, r_gross = trade_pnl(t, exit_price, cost_pct)
        trades.append({
            "entry_date": df.index[i + 1], "exit_date": df.index[exit_idx],
            "direction": "BUY" if s == 1 else "SELL", "entry": entry,
            "exit": exit_price, "reason": reason, "pnl": net_pnl,
            "gross_pnl": gross_pnl, "cost_pct": cost_pct, "r": r_net, "r_gross": r_gross,
            "bars": exit_idx - i, "sl0": sl, "tp1": tp1, "tp2": tp2, "mean_rev": mr,
        })
        i = exit_idx

    return pd.DataFrame(trades)