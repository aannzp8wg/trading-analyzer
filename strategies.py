"""
strategies.py — توليد إشارات كل استراتيجية، واختيار الاستراتيجية
التلقائي حسب نظام السوق (ADX)، ومصفوفة الإشارات الكاملة للباكتست.
"""
import numpy as np
import pandas as pd
import streamlit as st

from constants import AUTO, BREAKOUT, CONTRARIAN, MEAN_REV, PULLBACK, TREND, WYCKOFF
from data_engine import apply_bias_filter
from gold_engine import (CONFLUENCE, NEW_STRATEGIES, RSI2, SQUEEZE,
                          TURTLE, extra_signals)


def generate_signal(df, strategy):
    latest = df.iloc[-1]
    signal, reason = "No Signal", ""

    if strategy in NEW_STRATEGIES:
        v = int(strategy_signals(df)[strategy].iloc[-1])
        nm = strategy.split(" (")[0]
        if v == 1:
            return "BUY", f"إشارة شراء من {nm}"
        if v == -1:
            return "SELL", f"إشارة بيع من {nm}"
        return "No Signal", f"لا توجد إشارة من {nm}"

    if strategy == BREAKOUT:
        if latest["Close"] > df["Donchian_Upper"].iloc[-2] and latest["Close"] > latest["EMA200"]:
            signal, reason = "BUY", "اختراق صعودي فوق قمة 20 يوم + اتجاه صاعد"
        elif latest["Close"] < df["Donchian_Lower"].iloc[-2] and latest["Close"] < latest["EMA200"]:
            signal, reason = "SELL", "كسر هبوطي تحت قاع 20 يوم + اتجاه هابط"
        else:
            reason = "لا يوجد اختراق واضح حاليا"

    elif strategy == PULLBACK:
        if latest["Close"] > latest["EMA200"] and latest["Low"] <= latest["EMA50"] and latest["Close"] > latest["Open"]:
            signal, reason = "BUY", "ارتداد من EMA50 في اتجاه صاعد"
        elif latest["Close"] < latest["EMA200"] and latest["High"] >= latest["EMA50"] and latest["Close"] < latest["Open"]:
            signal, reason = "SELL", "ارتداد من EMA50 في اتجاه هابط"
        else:
            reason = "لا يوجد ارتداد واضح"

    elif strategy == MEAN_REV:
        if latest["Close"] < latest["BB_Lower"] and latest["RSI14"] < 35:
            signal, reason = "BUY", "تشبع بيعي: السعر تحت بولنجر السفلي + RSI < 35"
        elif latest["Close"] > latest["BB_Upper"] and latest["RSI14"] > 65:
            signal, reason = "SELL", "تشبع شرائي: السعر فوق بولنجر العلوي + RSI > 65"
        else:
            reason = "السعر في المنطقة الطبيعية"

    elif strategy == TREND:
        try:
            ema20_slope = float(latest["EMA20"]) - float(df["EMA20"].iloc[-2])
        except Exception:
            ema20_slope = 0.0

        above_200 = latest["Close"] > latest["EMA200"]
        below_200 = latest["Close"] < latest["EMA200"]
        ema_stack_up = latest["EMA20"] > latest["EMA50"]
        ema_stack_dn = latest["EMA20"] < latest["EMA50"]

        if above_200 and ema_stack_up and ema20_slope > 0:
            signal, reason = "BUY", "اتجاه صاعد مؤكد: فوق EMA200 + EMA20>EMA50 + الميل إيجابي"
        elif below_200 and ema_stack_dn and ema20_slope < 0:
            signal, reason = "SELL", "اتجاه هابط مؤكد: تحت EMA200 + EMA20<EMA50 + الميل سلبي"
        else:
            reason = "لا يوجد اتجاه واضح (مؤشرات متضاربة)"

    return signal, reason


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

    checks = []
    for strat in order:
        sig, reason = generate_signal(df, strat)
        if bias is not None and sig != "No Signal" and strat not in CONTRARIAN:
            allowed = (sig == "BUY" and bias == "UP") or (sig == "SELL" and bias == "DOWN")
            if not allowed:
                reason = f"{reason} — ❌ مرفوضة: تعاكس اتجاه اليومي"
                sig = "No Signal"
        checks.append((strat, sig, reason))
    chosen = next((c for c in checks if c[1] != "No Signal"), None)
    return regime, adx, checks, chosen


@st.cache_data(ttl=300, show_spinner=False)
def strategy_signals(df):
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]

    def to_sig(buy, sell):
        return pd.Series(
            np.select([buy, sell], [1, -1], default=0),
            index=df.index)

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


def auto_signals_ex(df, sigs, bias=None):
    """يرجع (إشارات AUTO، قناع mean_rev) — القناع True حيث جاءت الإشارة من استراتيجية انعكاسية.
    كان الباكتست يمرّر mean_rev=False لـ AUTO فتُقيَّم صفقات الارتداد بأهداف الاتجاه، بينما
    المسار الحي يستخدم هدف المتوسط (SMA20) لها: تضارب بين الباكتست والواقع."""
    adx = pd.to_numeric(df["ADX14"], errors="coerce")
    orders = {
        "trend": [CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE, PULLBACK],
        "range": [RSI2, MEAN_REV],
        "transition": [CONFLUENCE, TREND, PULLBACK, SQUEEZE, BREAKOUT, RSI2, MEAN_REV],
    }

    if bias is not None:
        trend_part = {k: v for k, v in sigs.items() if k not in CONTRARIAN}
        contra_part = {k: v for k, v in sigs.items() if k in CONTRARIAN}
        trend_part = apply_bias_filter(trend_part, bias)
        local = {**trend_part, **contra_part}
    else:
        local = sigs

    def choose(order):
        chosen = pd.Series(0, index=df.index, dtype=int)
        mr = pd.Series(False, index=df.index)
        for name in order:
            s_ = pd.to_numeric(local.get(name, pd.Series(0, index=df.index)),
                               errors="coerce").fillna(0).astype(int)
            take = (chosen == 0) & (s_ != 0)
            chosen = chosen.where(~take, s_)
            mr = mr.where(~take, name in CONTRARIAN)
        return chosen, mr

    (trend, mr_t), (rng, mr_r), (transition, mr_x) = (choose(orders[k]) for k in ("trend", "range", "transition"))
    out = trend.where(adx >= 25, transition).where(adx >= 20, rng)
    mr_out = mr_t.where(adx >= 25, mr_x).where(adx >= 20, mr_r)
    return out.fillna(0).astype(int), mr_out.fillna(False).astype(bool)


def auto_signals(df, sigs, bias=None):
    return auto_signals_ex(df, sigs, bias)[0]


ALL_STRATEGIES = [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE, PULLBACK, RSI2, MEAN_REV, WYCKOFF]


def all_strategy_signals(df, bias=None):
    """مصدر واحد لإشارات الباكتست والـ Walk-Forward: يرجع (sigs, mr)
    sigs[name] = إشارات (مفلترة باتجاه اليومي للاتجاهية)، mr[name] = bool أو مصفوفة bool (لـ AUTO)."""
    raw = {k: v.copy() for k, v in strategy_signals(df).items()}
    auto, auto_mr = auto_signals_ex(df, raw, bias)
    out = dict(raw)
    if bias is not None:
        trend_part = {k: v for k, v in raw.items() if k not in CONTRARIAN}
        contra_part = {k: v for k, v in raw.items() if k in CONTRARIAN}
        out.update({**apply_bias_filter(trend_part, bias), **contra_part})
    out[AUTO] = auto
    mr = {k: (k in CONTRARIAN) for k in out}
    mr[AUTO] = auto_mr.to_numpy()
    return out, mr