"""
strategies.py — توليد إشارات كل استراتيجية، واختيار الاستراتيجية
التلقائي حسب نظام السوق (ADX)، ومصفوفة الإشارات الكاملة للباكتست.

مصدر الحقيقة الوحيد للإشارات هو strategy_signals (متجهي وسريع)،
وgenerate_signal يقرأ آخر قيمة منه فقط، فيتطابق التنبيه الحي مع الباكتست.
"""
import numpy as np
import pandas as pd

from constants import (AUTO, BREAKOUT, CONFLUENCE, CONTRARIAN, GANN, MEAN_REV,
                       PULLBACK, RSI2, SQUEEZE, TREND, TURTLE)
from data_engine import apply_bias_filter
from gold_engine import extra_signals

_REASONS = {
    BREAKOUT: ("اختراق صعودي فوق قمة 20 يوم + اتجاه صاعد",
               "كسر هبوطي تحت قاع 20 يوم + اتجاه هابط",
               "لا يوجد اختراق واضح حاليا"),
    PULLBACK: ("ارتداد من EMA50 في اتجاه صاعد",
               "ارتداد من EMA50 في اتجاه هابط",
               "لا يوجد ارتداد واضح"),
    MEAN_REV: ("تشبع بيعي: السعر تحت بولنجر السفلي + RSI < 35",
               "تشبع شرائي: السعر فوق بولنجر العلوي + RSI > 65",
               "السعر في المنطقة الطبيعية"),
    GANN: ("انقلاب صاعد: الإغلاق فوق متوسط القمم (Gann HiLo)",
           "انقلاب هابط: الإغلاق تحت متوسط القيعان (Gann HiLo)",
           "لا يوجد انقلاب اتجاه في Gann HiLo"),
    TREND: ("اتجاه صاعد مؤكد: فوق EMA200 + EMA20>EMA50 + الميل إيجابي",
            "اتجاه هابط مؤكد: تحت EMA200 + EMA20<EMA50 + الميل سلبي",
            "لا يوجد اتجاه واضح (مؤشرات متضاربة)"),
}


def generate_signal(df, strategy, sigs=None):
    """إشارة آخر شمعة. مرّر sigs (خرج strategy_signals) لتفادي إعادة الحساب."""
    if sigs is None:
        sigs = strategy_signals(df)
    series = sigs.get(strategy)
    v = int(series.iloc[-1]) if series is not None and len(series) else 0
    nm = strategy.split(" (")[0]

    if strategy in _REASONS:
        buy_r, sell_r, none_r = _REASONS[strategy]
    else:
        buy_r, sell_r, none_r = (f"إشارة شراء من {nm}", f"إشارة بيع من {nm}",
                                 f"لا توجد إشارة من {nm}")
    if v == 1:
        return "BUY", buy_r
    if v == -1:
        return "SELL", sell_r
    return "No Signal", none_r


def select_strategy(df, mode, bias=None):
    adx = float(df["ADX14"].iloc[-1])
    if adx >= 25:
        regime = "اتجاه قوي 📈"
        order = [CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE, PULLBACK]
    elif adx < 20:
        regime = "سوق عرضي ↔️"
        order = [RSI2, MEAN_REV]
    else:
        regime = "انتقالي (اتجاه ضعيف) 🔄"
        order = [CONFLUENCE, TREND, PULLBACK, SQUEEZE, BREAKOUT, RSI2, MEAN_REV]

    if mode != AUTO:
        order = [mode]

    sigs = strategy_signals(df)  # مرة واحدة فقط
    checks = []
    for strat in order:
        sig, reason = generate_signal(df, strat, sigs)
        if bias is not None and sig != "No Signal" and strat not in CONTRARIAN:
            allowed = (sig == "BUY" and bias == "UP") or (sig == "SELL" and bias == "DOWN")
            if not allowed:
                reason = f"{reason} — ❌ مرفوضة: تعاكس اتجاه اليومي"
                sig = "No Signal"
        checks.append((strat, sig, reason))
    chosen = next((c for c in checks if c[1] != "No Signal"), None)
    return regime, adx, checks, chosen


def strategy_signals(df):
    """كل الإشارات الخام (بدون فلتر اتجاه اليومي). متجهي — لا يحتاج كاش."""
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]

    def to_sig(buy, sell):
        return pd.Series(np.select([buy, sell], [1, -1], default=0), index=df.index)

    sigs_ = {
        BREAKOUT: to_sig(
            (c > df["Donchian_Upper"].shift(1)) & (c > df["EMA200"]),
            (c < df["Donchian_Lower"].shift(1)) & (c < df["EMA200"])),
        PULLBACK: to_sig(
            (c > df["EMA200"]) & (l <= df["EMA50"]) & (c > o),
            (c < df["EMA200"]) & (h >= df["EMA50"]) & (c < o)),
        MEAN_REV: to_sig(
            (c < df["BB_Lower"]) & (df["RSI14"] < 35),
            (c > df["BB_Upper"]) & (df["RSI14"] > 65)),
        TREND: to_sig(
            (c > df["EMA200"]) & (df["EMA20"] > df["EMA50"]) & (df["EMA20"] > df["EMA20"].shift(1)),
            (c < df["EMA200"]) & (df["EMA20"] < df["EMA50"]) & (df["EMA20"] < df["EMA20"].shift(1))),
    }
    sigs_.update(extra_signals(df, sigs_))
    return sigs_


def auto_signals(df, sigs, bias=None):
    adx = pd.to_numeric(df["ADX14"], errors="coerce")
    orders = {
        "trend": [CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE, PULLBACK],
        "range": [RSI2, MEAN_REV],
        "transition": [CONFLUENCE, TREND, PULLBACK, SQUEEZE, BREAKOUT, RSI2, MEAN_REV],
    }

    if bias is not None:
        trend_part = apply_bias_filter(
            {k: v for k, v in sigs.items() if k not in CONTRARIAN}, bias)
        contra_part = {k: v for k, v in sigs.items() if k in CONTRARIAN}
        local = {**trend_part, **contra_part}
    else:
        local = sigs

    zero = pd.Series(0, index=df.index)

    def choose(order):
        chosen = pd.Series(0, index=df.index, dtype=int)
        for name in order:
            s = pd.to_numeric(local.get(name, zero), errors="coerce").fillna(0).astype(int)
            chosen = chosen.where(chosen != 0, s)
        return chosen

    out = choose(orders["trend"]).where(adx >= 25, choose(orders["transition"]))
    out = out.where(adx >= 20, choose(orders["range"]))
    return out.fillna(0).astype(int)


def strategy_matrix(df, bias=None):
    """كل الاستراتيجيات + AUTO، مع تطبيق فلتر اليومي على غير الانعكاسية.
    نقطة واحدة تُستعمل في الباكتست وWalk-Forward والمقارنة."""
    raw = strategy_signals(df)
    out = dict(raw)
    out[AUTO] = auto_signals(df, raw, bias)
    if bias is not None:
        filtered = apply_bias_filter(
            {k: v for k, v in raw.items() if k not in CONTRARIAN}, bias)
        out.update(filtered)
    return out