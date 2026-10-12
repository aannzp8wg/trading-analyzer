"""
risk_engine.py — كل منطق المال والمخاطر: مستويات الوقف/الهدف،
بناء خطة الصفقة، تحكيم الوكلاء، نقاط الجودة، القرار النهائي،
Walk-Forward، حارس الارتباط.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from constants import (ADAPTIVE_EVIDENCE_HAIRCUT, AUTO, BLOCK_ON_HIGH_EVENT, BREAKOUT, CONTRARIAN,
                        DEFAULT_COST, FRESH_D1_STOP_BD, FRESH_D1_WARN_BD, FRESH_H4_STOP_H,
                        FRESH_H4_WARN_H, MEAN_REV, PULLBACK, SL_ATR, THROTTLE_COOLDOWN_H,
                        THROTTLE_DD_HALF, THROTTLE_DD_STOP, THROTTLE_LOSS_HALF, THROTTLE_LOSS_STOP,
                        TF_D1, TP1_ATR, TP2_ATR, TREND, USED_OOS_MIN_TRADES, WYCKOFF)
from data_engine import load_prepared
from gold_engine import (CONFLUENCE, RSI2, SQUEEZE, TURTLE,
                          run_backtest_v2, smart_levels)
from strategies import all_strategy_signals


def P(x):
    """تنسيق سعر آمن."""
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
    """إذا مُرِّرت levels (خرج smart_levels محسوب مسبقًا) نعيد استخدامها."""
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
    """سياسة أخبار واحدة مشتركة بين التحكيم وحساب نقاط الجودة."""
    if not isinstance(news, dict):
        return {"status": "missing", "impact": "unknown",
                "event_risk": "unknown", "mult": 0.5}
    impact = str(news.get("impact_on_signal", "neutral")).lower()
    event_risk = str(news.get("event_risk", "unknown")).lower()
    if impact == "conflicts":
        mult = 0.0
    elif event_risk == "high":
        mult = 0.0
    elif impact == "supports" and event_risk == "low":
        mult = 1.0
    elif impact == "supports" and event_risk == "unknown":
        mult = 0.85            # لا نقطة كاملة بينما التقويم غير متاح
    elif event_risk == "medium":
        mult = 0.5
    else:
        mult = 0.75
    if news.get("degraded"):
        mult = min(mult, 0.5)  # وضع التقويم فقط (بلا تحليل عناوين)
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

    if event_high and BLOCK_ON_HIGH_EVENT:
        ev_txt = [f"{e['title']} ({e['currency']}) خلال {e['hours']:+.0f}س"
                  for e in ((news or {}).get("calendar") or {}).get("events", []) if e.get("level") == "high"]
        return {"action": "لا تدخل — حدث اقتصادي عالي التأثير قريب", "level": "warning", "mult": 0.0,
                "notes": ev_txt or ["خطر أحداث مرتفع بحسب الأخبار/التقويم."]}

    if risk_dec == "REDUCE":
        notes += [f"مدير المخاطر: {c}" for c in as_list(risk.get("concerns"))] \
                 or ["مدير المخاطر يطلب تخفيض الحجم"]
    if news_conflict:
        notes.append("الأخبار تعاكس اتجاه الإشارة")
    if event_high:
        notes.append("يوجد خطر أحداث مرتفع بحسب العناوين المتاحة")
    if news is None:
        notes.append("الأخبار غير متاحة؛ لن تُستخدم كدليل تأكيد")
    degraded = isinstance(news, dict) and bool(news.get("degraded"))
    if degraded:
        notes.append("تحليل العناوين غير متاح — اعتُمد التقويم الاقتصادي فقط")
    for n_ in ((news or {}).get("notes", []) if isinstance(news, dict) else []):
        notes.append(f"الأخبار: {n_}")

    if risk_dec == "REDUCE" or news_conflict or event_high or news is None or degraded:
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
    sl_, tp1_, tp2_ = trade_levels(signal, price, atr, df, mean_rev, levels=levels)
    if not all(v is not None and np.isfinite(v) for v in (sl_, tp1_, tp2_)):
        plan["action"], plan["level"] = "لا تدخل — تعذّر حساب مستويات وقف/هدف صالحة", "warning"
        return plan
    plan["direction"] = signal
    plan["sl"], plan["tp1"], plan["tp2"] = sl_, tp1_, tp2_
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


_ALL_STRATEGIES = [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE,
                   TURTLE, PULLBACK, RSI2, MEAN_REV, WYCKOFF]


def _slice(tr, d0, d1=None, exit_before=None):
    """صفقات دخلت في [d0, d1). exit_before: يستبعد الصفقات التي لم تنتهِ قبل هذا التاريخ (purging)."""
    if tr is None:
        return pd.DataFrame()
    if tr.empty:
        return tr.iloc[0:0]
    dates = pd.to_datetime(tr["entry_date"])
    m = dates >= d0
    if d1 is not None:
        m &= dates < d1
    if exit_before is not None:
        m &= pd.to_datetime(tr["exit_date"]) < exit_before
    return tr[m].copy()


def walk_forward_evidence(t, tf, df, bias_s, used_fallback, cost=None, hold=30,
                          n_folds=5, train_frac=0.15, test_frac=0.15,
                          embargo=5, min_train_trades=8):
    """Walk-forward باختيار استراتيجية متجدد، مع:
      • فترة عزل ≥ أطول صفقة (hold): كان embargo=5 أقل من hold=30 فكانت صفقات التدريب تنتهي
        داخل نافذة الاختبار (تسريب: تغيير بيانات الاختبار كان يغيّر اختيار التدريب).
      • purging: لا تُحتسب صفقة تدريب لم تنتهِ قبل بداية الاختبار.
      • لا طيّ صامت عند te_e == n (كان IndexError يُبتلع فيسقط الدليل كله إلى التقسيم الواحد).
      • لا تداول إجباري: إن لم تنجح أي استراتيجية في التدريب تبقى الطيّة نقداً بدل فرض استراتيجية.
    """
    cost = DEFAULT_COST.get(t.strip().upper(), 0.1) if cost is None else cost
    try:
        if tf == TF_D1:
            dfh, _, err = load_prepared(t, TF_D1, True, "5y")
            if err:
                return None
            bias_h = None
        else:
            dfh, bias_h = df, bias_s

        if dfh is None or len(dfh) < 400:
            return None

        sigs, mr = all_strategy_signals(dfh, bias_h)
        all_tr = {}
        for name in _ALL_STRATEGIES:
            sig = sigs.get(name)
            if sig is None:
                continue
            try:
                all_tr[name] = run_backtest_v2(dfh, sig, cost, hold,
                                               mean_rev=mr.get(name, name in CONTRARIAN))
            except Exception:
                continue
        if not all_tr:
            return None

        start, n = 200, len(dfh)
        gap = max(int(embargo), int(hold))
        usable = n - start
        window = (usable - gap) // (n_folds + 1)
        if window < 30:
            return None

        picks, per_fold, test_chunks, used_chunks = [], [], [], []
        for k in range(n_folds):
            tr_s = start + k * window
            tr_e = tr_s + window
            te_s = tr_e + gap
            te_e = te_s + window
            if te_e > n:
                break
            d0_tr, d1_tr = pd.Timestamp(dfh.index[tr_s]), pd.Timestamp(dfh.index[tr_e])
            d0_te = pd.Timestamp(dfh.index[te_s])
            d1_te = pd.Timestamp(dfh.index[te_e]) if te_e < n else None

            best_name, best_score, best_tr_st = None, -np.inf, None
            for name, tr in all_tr.items():
                st_ = backtest_stats(_slice(tr, d0_tr, d1_tr, exit_before=d0_te))
                if st_["trades"] < min_train_trades or not st_["avg_r"] > 0:
                    continue                       # لا نختار استراتيجية غير رابحة في التدريب
                score = st_["avg_r"] if st_["pf"] > 1 else st_["avg_r"] - 1.0
                if score > best_score:
                    best_score, best_name, best_tr_st = score, name, st_

            if used_fallback in all_tr:
                used_chunks.append(_slice(all_tr[used_fallback], d0_te, d1_te))

            if best_name is None:
                picks.append(None)
                per_fold.append({"fold": k + 1, "pick": "— (لا تداول: لا استراتيجية رابحة في التدريب)",
                                 "train_r": 0.0, "train_n": 0, "test_r": 0.0, "test_n": 0,
                                 "test_total": 0.0})
                continue

            test_slice = _slice(all_tr[best_name], d0_te, d1_te)
            test_st = backtest_stats(test_slice)
            if not test_slice.empty:
                test_chunks.append(test_slice)
            picks.append(best_name)
            per_fold.append({
                "fold": k + 1, "pick": best_name,
                "train_r": round(float(best_tr_st["avg_r"]), 2), "train_n": int(best_tr_st["trades"]),
                "test_r": round(float(test_st["avg_r"]), 2), "test_n": int(test_st["trades"]),
                "test_total": round(float(test_st["total"]), 2),
            })

        bars = max(n - 200, 0)
        used_df = pd.concat([c for c in used_chunks if not c.empty], ignore_index=True) \
            if any(not c.empty for c in used_chunks) else pd.DataFrame()
        used_st = backtest_stats(used_df)
        base = {"bars": bars, "out_of_sample": True, "walk_forward": True,
                "folds": len(per_fold), "picks": picks, "per_fold": per_fold,
                "cash_folds": sum(1 for p in picks if p is None),
                "used_oos": {"trades": used_st["trades"], "avg_r": used_st["avg_r"], "pf": used_st["pf"]},
                "full_sample_trades": sum(len(x) for x in all_tr.values())}
        if not test_chunks:
            return {**backtest_stats(pd.DataFrame()), **base}
        stats = backtest_stats(pd.concat(test_chunks, ignore_index=True))
        stats.update(base)
        return stats
    except Exception:
        return None


def _single_split_evidence(t, tf, df, bias_s, used, use_closed, cost=None,
                           hold=30, n_folds=3, oos_frac=0.30):
    """نسخة احتياطية (تقسيم واحد) — تُستخدم إن فشل walk-forward."""
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

        sigs_all, mr_all = all_strategy_signals(dfh, bias_h)
        sig = sigs_all.get(used)
        if sig is None:
            return None

        tr = run_backtest_v2(dfh, sig, cost, hold,
                             mean_rev=mr_all.get(used, used in CONTRARIAN))
        bars = max(len(dfh) - 200, 0)
        if tr.empty:
            return {**backtest_stats(tr), "bars": bars,
                    "out_of_sample": True, "full_sample_trades": 0,
                    "folds": 0, "walk_forward": False}

        n = len(dfh)
        start = 200
        span = (n - start) // n_folds
        windows = []
        for i in range(n_folds):
            seg_s = start + i * span
            seg_e = seg_s + span if i < n_folds - 1 else n
            t_len = max(int(span * oos_frac), 1)
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
                      "walk_forward": False, "folds": len(windows),
                      "full_sample_trades": len(tr)})
        return stats
    except Exception:
        return None


def history_evidence(t, tf, df, bias_s, used, use_closed, cost=None, hold=30):
    """نقطة دخول موحدة: Walk-Forward أولًا، ثم التقسيم الواحد."""
    wf = walk_forward_evidence(t, tf, df, bias_s, used, cost, hold)
    if wf is not None:
        return wf
    return _single_split_evidence(t, tf, df, bias_s, used, use_closed, cost, hold)


def compute_quality(signal, df, adx, used, bias, tech, news, risk, ev):
    """نقاط توافق إرشادية (0-100).
    التوزيع: تقني 12 / ADX 10 / اتجاه يومي 13 / فني 15 / أخبار 10
              / مخاطر 15 / OOS 25 = 100
    """
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
        # الدليل المتكيّف يقيس "عملية الاختيار" لا هذه الاستراتيجية بالذات. فإن توفّر ما يكفي من
        # صفقات الاستراتيجية الحالية خارج العيّنة نعتمدها هي؛ وإلا نخصم من الدليل المتكيّف (تحفّظ).
        eff, haircut, basis = ev, 1.0, ""
        uo = ev.get("used_oos")
        if ev.get("walk_forward"):
            if uo and uo.get("trades", 0) >= USED_OOS_MIN_TRADES:
                eff = {"trades": uo["trades"], "avg_r": uo["avg_r"], "pf": uo["pf"]}
                basis = "الدليل: الاستراتيجية الحالية نفسها"
            else:
                haircut = ADAPTIVE_EVIDENCE_HAIRCUT
                basis = f"الدليل متكيّف لا يخص هذه الاستراتيجية (خصم ×{haircut:g})"
        f = min(eff["trades"], 30) / 30
        avg_r = eff["avg_r"] if np.isfinite(eff["avg_r"]) else 0.0     # NaN كان يُسقط الدالة كلها
        exp_pts = 17 * min(max(avg_r, 0) / 0.3, 1)
        pf = eff["pf"] if np.isfinite(eff["pf"]) else 3.0
        pf_pts = 8 * min(max((pf - 1) / 0.5, 0), 1)
        pts = round((exp_pts + pf_pts) * f * haircut, 1)
        tag = "WF" if ev.get("walk_forward") else "OOS"
        note = f"{eff['trades']} صفقة ({tag}) | متوسط R (صافٍ) = {avg_r:.2f} | PF = {pf:.2f}"
        if basis:
            note += f" | {basis}"
        if ev.get("folds"):
            note += f" | {ev['folds']} نوافذ"
        if eff["trades"] < 30:
            note += " | عينة صغيرة"
        picks_ = [p for p in (ev.get("picks") or []) if p]
        if picks_ and len(set(picks_)) > 1:
            note += f" | اختيار متكيّف ({len(set(picks_))} استراتيجيات)"
        if ev.get("cash_folds"):
            note += f" | {ev['cash_folds']} نافذة بلا تداول"
        if uo and uo.get("trades") and eff is ev:
            note += f" | الاستراتيجية الحالية وحدها: {uo['trades']} صفقة، R={uo['avg_r']:.2f}"
        parts.append(("الأداء التاريخي (OOS)", pts, 25, note))

    total = int(round(sum(p[1] for p in parts)))
    return total, parts


def risk_throttle(trades, now=None):
    """مخفّف المخاطر: يقرأ سجل صفقاتك (الأثر على الحساب %) ويخفّض الحجم بعد تراجع أو سلسلة خسائر.
    لا يمنع للأبد: بعد THROTTLE_COOLDOWN_H ساعة من آخر صفقة يعود الإيقاف إلى نصف حجم (لإعادة البناء)."""
    now = now or datetime.now()
    pnls, last_exit = [], None
    for tr in trades or []:
        v = tr.get("account_pnl_percent")
        if v is None:
            v = tr.get("pnl_percent")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        pnls.append(v)
        try:
            dt = datetime.strptime(str(tr.get("exit_time", "")), "%Y-%m-%d %H:%M")
            last_exit = dt if (last_exit is None or dt > last_exit) else last_exit
        except ValueError:
            pass
    base = {"mult": 1.0, "level": "ok", "reason": "", "dd": 0.0, "streak": 0}
    if not pnls:
        return base
    eq = 100.0 * np.cumprod(1.0 + np.asarray(pnls) / 100.0)
    peak = np.maximum.accumulate(np.r_[100.0, eq])[1:]
    dd = float((peak[-1] - eq[-1]) / peak[-1] * 100.0)
    streak = 0
    for v in reversed(pnls):
        if v < 0:
            streak += 1
        else:
            break
    hours = (now - last_exit).total_seconds() / 3600.0 if last_exit else float("inf")
    cooled = hours >= THROTTLE_COOLDOWN_H
    mult, reasons = 1.0, []
    if dd >= THROTTLE_DD_STOP:
        mult = min(mult, 0.5 if cooled else 0.0)
        reasons.append(f"تراجع {dd:.1f}% من القمة (≥{THROTTLE_DD_STOP:g}%)")
    elif dd >= THROTTLE_DD_HALF:
        mult = min(mult, 0.5)
        reasons.append(f"تراجع {dd:.1f}% من القمة (≥{THROTTLE_DD_HALF:g}%)")
    if streak >= THROTTLE_LOSS_STOP:
        mult = min(mult, 0.5 if cooled else 0.0)
        reasons.append(f"{streak} خسائر متتالية")
    elif streak >= THROTTLE_LOSS_HALF:
        mult = min(mult, 0.5)
        reasons.append(f"{streak} خسائر متتالية")
    level = "stop" if mult <= 0 else "half" if mult < 1 else "ok"
    return {"mult": mult, "level": level, "dd": dd, "streak": streak,
            "reason": " + ".join(reasons)}


def apply_throttle(arb, thr):
    """يطبّق المخفّف على نتيجة التحكيم قبل final_decision/build_plan."""
    if thr["level"] == "ok" or arb["mult"] <= 0:
        return arb
    if thr["mult"] <= 0:
        return {"action": f"لا تدخل — وضع حماية الحساب ({thr['reason']})", "level": "warning",
                "mult": 0.0, "notes": arb.get("notes", []) + [
                    f"توقّف مؤقت عن الدخول الجديد حتى تمر {THROTTLE_COOLDOWN_H} ساعة على آخر صفقة."]}
    return {**arb, "mult": arb["mult"] * thr["mult"],
            "action": arb["action"] + " (حجم مخفَّض: وضع حماية الحساب)",
            "notes": arb.get("notes", []) + [f"مخفّف المخاطر: {thr['reason']} ⇒ ×{thr['mult']:g}"]}


def data_freshness(df, tf, now=None):
    """حارس حداثة البيانات: ok / warn / stale. السوق المغلق (نهاية الأسبوع) لا يُعدّ تأخراً."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    try:
        last = pd.Timestamp(df.index[-1])
    except Exception:
        return {"status": "warn", "age_h": None, "msg": "تعذّر تحديد وقت آخر شمعة."}
    last_utc = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    wd, hr = now.weekday(), now.hour
    closed = wd == 5 or (wd == 4 and hr >= 22) or (wd == 6 and hr < 22)

    if tf == TF_D1:
        today = np.datetime64(now.date())
        # آخر شمعة يومية مكتملة متوقعة: أمس إن كان اليوم يوم عمل، وإلا آخر جمعة (لا الخميس)
        expected = (np.busday_offset(today, -1, roll="backward") if np.is_busday(today)
                    else np.busday_offset(today, 0, roll="backward"))
        last_d = np.datetime64(last_utc.date())
        lag = int(np.busday_count(last_d, expected)) if last_d < expected else 0
        age_h = (now - last_utc).total_seconds() / 3600.0
        if lag >= FRESH_D1_STOP_BD:
            return {"status": "stale", "age_h": age_h,
                    "msg": f"آخر شمعة يومية متأخرة {lag} أيام عمل — بيانات قديمة (أو عطلة طويلة)."}
        if lag >= FRESH_D1_WARN_BD:
            return {"status": "warn", "age_h": age_h, "msg": "آخر شمعة يومية أقدم من المتوقع بيوم عمل (قد تكون عطلة)."}
        return {"status": "ok", "age_h": age_h, "msg": ""}

    bar_close = last_utc + pd.Timedelta(hours=4)
    age_h = max((now - bar_close).total_seconds() / 3600.0, 0.0)
    if closed:
        return {"status": "ok", "age_h": age_h, "msg": "السوق مغلق (نهاية الأسبوع)."}
    if age_h > FRESH_H4_STOP_H:
        return {"status": "stale", "age_h": age_h,
                "msg": f"آخر شمعة 4H مكتملة قبل {age_h:.0f} ساعة والسوق مفتوح — بيانات قديمة."}
    if age_h > FRESH_H4_WARN_H:
        return {"status": "warn", "age_h": age_h, "msg": f"آخر شمعة 4H قبل {age_h:.0f} ساعة."}
    return {"status": "ok", "age_h": age_h, "msg": ""}


def apply_data_guard(arb, fresh):
    if fresh["status"] == "stale" and arb["mult"] > 0:
        return {"action": "لا تدخل — بيانات السوق قديمة", "level": "warning", "mult": 0.0,
                "notes": arb.get("notes", []) + [fresh["msg"]]}
    if fresh["status"] == "warn" and arb["mult"] > 0:
        return {**arb, "notes": arb.get("notes", []) + [f"⚠️ {fresh['msg']}"]}
    return arb


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
        "avg_r": float(np.nan_to_num(tr["r"].mean(), nan=0.0)),
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
    """حارس الارتباط: يحذّر من مضاعفة الرهان على الدولار."""
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