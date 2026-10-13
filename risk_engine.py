"""
risk_engine.py — كل منطق المال والمخاطر: مستويات الوقف/الهدف،
بناء خطة الصفقة، تحكيم الوكلاء، نقاط الجودة، القرار النهائي،
وحارس الارتباط.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from constants import (AUTO, BREAKOUT, CONTRARIAN, DEFAULT_COST, MEAN_REV,
                        PULLBACK, SL_ATR, TF_D1, TP1_ATR, TP2_ATR, TREND)
from data_engine import apply_bias_filter, load_prepared
from gold_engine import (CONFLUENCE, RSI2, SQUEEZE, TURTLE,
                          run_backtest_v2, smart_levels)
from strategies import auto_signals, strategy_signals


def P(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax >= 10:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.1:
        return f"{x:.5f}"
    return f"{x:.6f}"


def truthy(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1", "نعم")


def as_list(v):
    if isinstance(v, list):
        return v
    return [v] if v else []


def clean_price_levels(v, limit=6):
    out = []
    for item in as_list(v):
        try:
            n = float(item)
            if np.isfinite(n):
                out.append(n)
        except (TypeError, ValueError):
            continue
    return out[:limit]


def guess_contract_size(ticker):
    t = ticker.strip().upper()
    known = {
        "GC=F": 100.0, "SI=F": 5000.0, "CL=F": 1000.0,
        "NG=F": 10000.0, "BTC-USD": 1.0, "ETH-USD": 1.0,
        "DX-Y.NYB": 1000.0,
    }
    if t in known:
        return known[t]
    if t.endswith("=X"):
        return 100000.0
    return 1.0


def trade_levels(signal, price, atr, df=None, mean_rev=False, levels=None):
    if levels is not None:
        return levels["sl"], levels["tp1"], levels["tp2"]
    if df is not None:
        L = smart_levels(signal, price, df, mean_rev=mean_rev)
        return L["sl"], L["tp1"], L["tp2"]
    sign = 1 if signal == "BUY" else -1
    return (
        price - sign * SL_ATR * atr,
        price + sign * TP1_ATR * atr,
        price + sign * TP2_ATR * atr,
    )


def news_policy(news):
    if not isinstance(news, dict):
        return {"status": "missing", "impact": "unknown",
                "event_risk": "unknown", "mult": 0.5}
    impact = str(news.get("impact_on_signal", "neutral")).lower()
    event_risk = str(news.get("event_risk", "unknown")).lower()
    if impact == "conflicts":
        mult = 0.0
    elif event_risk == "high":
        mult = 0.0
    elif impact == "supports" and event_risk in ("low", "unknown"):
        mult = 1.0
    elif event_risk == "medium":
        mult = 0.5
    else:
        mult = 0.75
    return {"status": "ok", "impact": impact,
            "event_risk": event_risk, "mult": mult}


def arbitrate(signal, tech, news, risk):
    if signal not in ("BUY", "SELL"):
        return {"action": "انتظر — لا توجد إشارة مناسبة الآن",
                "level": "info", "mult": 0.0, "notes": []}

    notes = []
    if tech is None:
        return {"action": "لا تدخل — الوكيل الفني غير متاح/غير صالح",
                "level": "error", "mult": 0.0,
                "notes": ["لا يتم فتح صفقة عند فشل التحقق من رد الوكيل الفني."]}
    if risk is None:
        return {"action": "لا تدخل — مدير المخاطر غير متاح/غير صالح",
                "level": "error", "mult": 0.0,
                "notes": ["النظام يعمل بمبدأ fail-closed عند غياب تقييم المخاطر."]}

    tech_ok = (
        truthy(tech.get("supports_signal"))
        and str(tech.get("verdict", "")).upper() == signal
    )
    risk_dec = str(risk.get("decision", "")).upper()
    npol = news_policy(news)
    news_conflict = npol["impact"] == "conflicts"
    event_high = npol["event_risk"] == "high"

    if not tech_ok:
        return {"action": "لا تدخل — التأكيد الفني لا يدعم الإشارة",
                "level": "error", "mult": 0.0,
                "notes": [str(tech.get("market_context", "لا يوجد تأكيد فني."))]}
    if risk_dec == "VETO":
        return {"action": "لا تدخل — مدير المخاطر اعترض",
                "level": "error", "mult": 0.0,
                "notes": [str(c) for c in as_list(risk.get("concerns"))]}

    if risk_dec == "REDUCE":
        notes += [f"مدير المخاطر: {c}" for c in as_list(risk.get("concerns"))] \
                 or ["مدير المخاطر يطلب تخفيض الحجم"]
    if news_conflict:
        notes.append("الأخبار تعاكس اتجاه الإشارة")
    if event_high:
        notes.append("يوجد خطر أحداث مرتفع بحسب العناوين المتاحة")
    if news is None:
        notes.append("الأخبار غير متاحة؛ لن تُستخدم كدليل تأكيد")

    if risk_dec == "REDUCE" or news_conflict or event_high or news is None:
        return {"action": "دخول بحذر — نصف الحجم",
                "level": "warning", "mult": 0.5, "notes": notes}
    return {"action": "تأكيد — الفني والمخاطر متوافقان",
            "level": "success", "mult": 1.0, "notes": notes}


def build_plan(signal, price, atr, arb, balance, risk_pct,
               contract_size=1.0, df=None, mean_rev=False,
               volume_step=0.01, min_lot=0.01, levels=None):
    plan = {"direction": None, "entry": float(price), "sl": None,
            "tp1": None, "tp2": None, "size": None, "lots": None,
            "risk_amount": 0.0, "action": arb["action"],
            "level": arb["level"]}
    if signal not in ("BUY", "SELL") or arb.get("mult", 0) <= 0:
        return plan
    plan["direction"] = signal
    plan["sl"], plan["tp1"], plan["tp2"] = trade_levels(
        signal, price, atr, df, mean_rev, levels=levels
    )
    sl_dist = abs(price - plan["sl"])
    risk_amount = float(balance) * float(risk_pct) / 100.0 * float(arb["mult"])
    plan["risk_amount"] = risk_amount
    if sl_dist > 0 and risk_amount > 0 and contract_size > 0:
        units = risk_amount / sl_dist
        lots_raw = units / contract_size
        step = max(float(volume_step), 1e-8)
        lots = math.floor(lots_raw / step + 1e-12) * step
        lots = round(lots, 8)
        plan["size"] = lots * contract_size
        plan["lots"] = lots if lots >= min_lot else None
        if plan["lots"] is None:
            plan["size"] = 0.0
    return plan


def history_evidence(t, tf, df, bias_s, used, use_closed, cost=None, hold=30):
    cost = DEFAULT_COST.get(t.strip().upper(), 0.1) if cost is None else cost
    try:
        if tf == TF_D1:
            dfh, _, err = load_prepared(t, TF_D1, use_closed, "5y")
            bias_h = None
            if err:
                return None
        else:
            dfh, bias_h = df, bias_s
        if dfh is None or len(dfh) < 300:
            return None

        sigs = strategy_signals(dfh)
        if used == AUTO:
            sig = auto_signals(dfh, sigs, bias_h)
        else:
            if bias_h is not None and used not in CONTRARIAN:
                sigs_filtered = apply_bias_filter({used: sigs[used]}, bias_h)
                sig = sigs_filtered.get(used)
            else:
                sig = sigs.get(used)
        if sig is None:
            return None

        tr = run_backtest_v2(dfh, sig, cost, hold,
                             mean_rev=(used in (MEAN_REV, RSI2)))
        bars = max(len(dfh) - 200, 0)
        if tr.empty:
            return {**backtest_stats(tr), "bars": bars,
                    "out_of_sample": True, "full_sample_trades": 0}

        n = len(dfh)
        start = 200
        span = (n - start) // 3
        windows = []
        for i in range(3):
            seg_s = start + i * span
            seg_e = seg_s + span if i < 2 else n
            t_len = max(int(span * 0.30), 1)
            t_s = max(seg_e - t_len, seg_s + 1)
            if t_s < seg_e:
                windows.append((pd.Timestamp(dfh.index[t_s]),
                                pd.Timestamp(dfh.index[seg_e - 1])))
        if not windows:
            return None

        dates = pd.to_datetime(tr["entry_date"]).reset_index(drop=True)
        tr_r = tr.reset_index(drop=True)
        mask = pd.Series(False, index=tr_r.index)
        for d0, d1 in windows:
            mask |= (dates >= d0) & (dates <= d1)
        tr_oos = tr_r[mask]

        stats = backtest_stats(tr_oos)
        stats.update({"bars": bars, "out_of_sample": True,
                      "folds": len(windows),
                      "full_sample_trades": len(tr)})
        return stats
    except Exception:
        return None


def compute_quality(signal, df, adx, used, bias, tech, news, risk, ev):
    parts = []
    latest = df.iloc[-1]
    close, e50, e200 = float(latest["Close"]), float(latest["EMA50"]), float(latest["EMA200"])
    buy = signal == "BUY"

    is_reversal = used in CONTRARIAN
    if is_reversal:
        rsi = float(latest.get("RSI14", np.nan))
        bb_u = float(latest.get("BB_Upper", np.nan))
        bb_l = float(latest.get("BB_Lower", np.nan))
        if buy:
            rsi_pts = 6 if np.isfinite(rsi) and rsi < 35 else 0
            loc_pts = 6 if np.isfinite(bb_l) and close <= bb_l else 0
        else:
            rsi_pts = 6 if np.isfinite(rsi) and rsi > 65 else 0
            loc_pts = 6 if np.isfinite(bb_u) and close >= bb_u else 0
        pts = rsi_pts + loc_pts
        parts.append(("توافق الانعكاس (RSI/BB)", pts, 12,
                      "تشبع/موقع انعكاسي" if pts >= 6 else "دليل انعكاسي ضعيف"))
    else:
        side = (close > e200) if buy else (close < e200)
        slope = (e50 > e200) if buy else (e50 < e200)
        pts = 7 * side + 5 * slope
        parts.append(("توافق الاتجاه (EMA)", pts, 12,
                      "السعر وEMA50 في اتجاه الصفقة" if pts == 12 else "الصفقة ضد الاتجاه العام جزئيًا"))

    if used in (BREAKOUT, PULLBACK, TREND, SQUEEZE, TURTLE, CONFLUENCE):
        pts = 10 if adx >= 25 else 5 if adx >= 20 else 0
    else:
        pts = 10 if adx < 20 else 5 if adx < 25 else 0
    parts.append(("ملاءمة الاستراتيجية لحالة السوق", pts, 10, f"ADX = {adx:.1f}"))

    if bias is None or used in CONTRARIAN:
        note = "معفى (استراتيجية انعكاسية)" if used in CONTRARIAN else "غير مستخدم في هذا الإطار"
        parts.append(("اتجاه اليومي", 6, 13, note))
    else:
        ok = (bias == "UP" and buy) or (bias == "DOWN" and not buy)
        parts.append(("اتجاه اليومي", 13 if ok else 0, 13,
                      "متوافق" if ok else "معاكس"))

    if tech is None:
        parts.append(("الوكيل الفني", 0, 15, "غير متاح"))
    else:
        supports = (truthy(tech.get("supports_signal"))
                    and str(tech.get("verdict", "")).upper() == signal)
        try:
            conf = float(tech.get("confidence", 50))
        except (TypeError, ValueError):
            conf = 50.0
        pts = round(15 * min(max(conf, 0), 90) / 90, 1) if supports else 0
        parts.append(("الوكيل الفني", pts, 15,
                      f"يؤيد (ثقة {conf:.0f}%)" if supports else "لا يؤيد"))

    npol = news_policy(news)
    if npol["status"] == "missing":
        news_pts = 3.0
    else:
        news_pts = round(10.0 * npol["mult"], 1)
    note = f"الأثر: {npol['impact']} | مخاطر: {npol['event_risk']}"
    if npol["status"] == "missing":
        note += " | بيانات غير متاحة (نقاط جزئية)"
    parts.append(("الأخبار", news_pts, 10, note))

    dec = None if risk is None else str(risk.get("decision", "")).upper()
    pts = {"APPROVE": 15, "REDUCE": 7}.get(dec, 0)
    parts.append(("وكيل المخاطر", pts, 15,
                  {"APPROVE": "موافقة", "REDUCE": "تخفيض"}.get(dec, "اعتراض/غير متاح")))

    if not ev or ev["trades"] == 0:
        parts.append(("الأداء التاريخي (OOS)", 0, 25,
                      "لا بيانات كافية خارج العينة"))
    else:
        f = min(ev["trades"], 30) / 30
        avg_r = ev["avg_r"] if np.isfinite(ev["avg_r"]) else 0.0
        exp_pts = 17 * min(max(avg_r, 0) / 0.3, 1)
        pf = ev["pf"] if np.isfinite(ev["pf"]) else 3.0
        pf_pts = 8 * min(max((pf - 1) / 0.5, 0), 1)
        pts = round((exp_pts + pf_pts) * f, 1)
        note = f"{ev['trades']} صفقة (OOS) | متوسط R = {avg_r:.2f} | PF = {pf:.2f}"
        if ev.get("folds"):
            note += f" | {ev['folds']} نوافذ"
        if ev["trades"] < 30:
            note += " | عينة صغيرة"
        parts.append(("الأداء التاريخي (OOS)", pts, 25, note))

    total = int(round(sum(p[1] for p in parts)))
    return total, parts


def final_decision(signal, arb, score, threshold):
    if signal not in ("BUY", "SELL") or arb["mult"] <= 0:
        return {**arb, "enter": False}
    if score >= threshold:
        return {**arb, "enter": True,
                "action": f"🔔 {arb['action']} — الجودة {score}/100"}
    return {"action": f"لا تدخل الآن — الجودة {score}/100 أقل من الحد {threshold}",
            "level": "warning", "mult": 0.0, "enter": False,
            "notes": arb["notes"] + ["الوكلاء لا يعترضون، لكن نقاط الجودة أقل من الحد."]}


def format_alert(t, tf, used, plan, score):
    d = "شراء 🟢" if plan["direction"] == "BUY" else "بيع 🔴"
    lots = f"{plan['lots']:.2f} لوت" if plan.get("lots") else "أقل من 0.01 لوت"
    return (
        f"🔔 إشارة دخول — {t}\n"
        f"الاتجاه: {d}\nالجودة: {score}/100\n"
        f"الدخول: {P(plan['entry'])}\nالوقف: {P(plan['sl'])}\n"
        f"الهدف 1: {P(plan['tp1'])}\nالهدف 2: {P(plan['tp2'])}\n"
        f"الحجم: {lots} (مخاطرة {plan['risk_amount']:,.2f})\n"
        f"{used.split(' (')[0]} | {tf.split(' (')[0]}\n"
        f"⚠️ لأغراض تعليمية، وليست نصيحة مالية."
    )


def backtest_stats(tr):
    if tr.empty:
        return {"trades": 0, "win_rate": 0.0, "total": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd": 0.0, "avg_bars": 0.0}
    gross_win = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(tr.loc[tr["pnl"] <= 0, "pnl"].sum())
    cum = tr["pnl"].cumsum()
    return {
        "trades": len(tr),
        "win_rate": (tr["pnl"] > 0).mean() * 100,
        "total": tr["pnl"].sum(),
        "avg_r": tr["r"].mean() if len(tr) else 0.0,
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "max_dd": abs(float((cum - cum.cummax()).min())),
        "avg_bars": tr["bars"].mean(),
    }


_DOLLAR_TRIPLE = {
    "GC=F": {"DX-Y.NYB", "EURUSD=X"},
    "EURUSD=X": {"DX-Y.NYB", "GC=F"},
    "DX-Y.NYB": {"GC=F", "EURUSD=X"},
}


def correlation_exposure_check(ticker, trades, current_position, hours=48):
    t = ticker.strip().upper()
    partners = _DOLLAR_TRIPLE.get(t, set())
    if not partners:
        return False, ""

    if current_position:
        cp = str(current_position.get("ticker", "")).strip().upper()
        if cp in partners:
            return True, (
                f"لديك صفقة مفتوحة على **{cp}** — فتح صفقة على {t} قد يكون "
                f"الرهان نفسه مضاعفًا على الدولار، لا صفقتين مستقلّتين."
            )

    cutoff = datetime.now() - timedelta(hours=hours)
    recent = []
    for tr in trades or []:
        try:
            tk = str(tr.get("ticker", "")).strip().upper()
            if tk not in partners:
                continue
            dt = datetime.strptime(str(tr.get("exit_time", "")), "%Y-%m-%d %H:%M")
            if dt >= cutoff:
                recent.append((tk, tr.get("exit_time"), tr.get("direction")))
        except Exception:
            continue

    if recent:
        sample = "، ".join(f"{tk} ({d})" for tk, _, d in recent[-2:])
        return True, (
            f"أُغلقت {len(recent)} صفقة على أصول مرتبطة خلال آخر {hours}س "
            f"({sample}). احذر تكرار الرهان نفسه على الدولار."
        )
    return False, ""