"""
journal.py — السجل الدائم للصفقات (محلي أو GitHub Gist) ومحاكي
التداول الورقي مع دعم TP1 ونقل الوقف والإغلاق الجزئي.
"""
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from constants import SL_ATR, TP2_ATR
from risk_engine import P

JOURNAL_FILE = os.path.join(tempfile.gettempdir(), "trading_journal.json")


class JournalConflict(Exception):
    """Gist تغيّر من جلسة أخرى."""


def _gist_cfg():
    try:
        tok, gid = st.secrets.get("GITHUB_TOKEN"), st.secrets.get("GIST_ID")
    except Exception:
        return None
    return (str(tok), str(gid)) if tok and gid else None


def storage_mode():
    return "gist" if _gist_cfg() else "local"


def _jdump(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
    )


def _gist_req(url, tok, body=None, method="GET", etag=None):
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "trading-journal",
        "Content-Type": "application/json",
    }
    if etag:
        headers["If-Match"] = etag
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            new_etag = r.headers.get("ETag") or r.headers.get("etag")
            return data, new_etag
    except urllib.error.HTTPError as e:
        if e.code == 412:
            raise JournalConflict(
                "Gist تغيّر من جلسة أخرى — لم يُحفظ تعديلك لتجنّب الكتابة فوقه."
            ) from e
        raise


def _scrub(msg, cfg):
    return (msg.replace(cfg[0], "***") if cfg else msg)[:160]


def journal_load():
    empty = {"trades": [], "position": None}
    cfg = _gist_cfg()
    try:
        if cfg:
            data, new_etag = _gist_req(
                f"https://api.github.com/gists/{cfg[1]}", cfg[0]
            )
            st.session_state["journal_etag"] = new_etag
            f = (data.get("files") or {}).get("trades.json")
            if not f:
                return empty, ""
            txt = (f.get("content") or "").strip()
            obj = json.loads(txt) if txt else {}
        else:
            if not os.path.exists(JOURNAL_FILE):
                return empty, ""
            with open(JOURNAL_FILE, encoding="utf-8") as fh:
                obj = json.load(fh)
        if isinstance(obj, list):
            obj = {"trades": obj}
        if not isinstance(obj, dict):
            obj = {}
        trades = obj.get("trades")
        return {
            "trades": trades if isinstance(trades, list) else [],
            "position": obj.get("position"),
        }, ""
    except Exception as e:
        return empty, _scrub(f"{type(e).__name__}: {e}", cfg)


def journal_save(trades, position):
    obj = {"version": 1, "trades": trades, "position": position}
    cfg = _gist_cfg()
    try:
        if cfg:
            body = json.dumps(
                {"files": {"trades.json": {"content": _jdump(obj)}}}
            ).encode()
            etag = st.session_state.get("journal_etag")
            _, new_etag = _gist_req(
                f"https://api.github.com/gists/{cfg[1]}",
                cfg[0],
                body,
                "PATCH",
                etag=etag,
            )
            st.session_state["journal_etag"] = new_etag
        else:
            with open(JOURNAL_FILE, "w", encoding="utf-8") as fh:
                fh.write(_jdump(obj))
        return ""
    except JournalConflict as e:
        return f"⚠️ {e} صفقاتك محفوظة في الجلسة الحالية فقط — أعد فتح الصفحة لتزامنها."
    except Exception as e:
        return _scrub(f"{type(e).__name__}: {e}", cfg)


def persist(force=False, min_interval=3.0):
    """حفظ مع debounce خفيف."""
    if not st.session_state.get("journal_ok", True):
        st.session_state["journal_save_err"] = "الحفظ متوقف لأن تحميل السجل فشل."
        return

    now = time.time()
    last = st.session_state.get("journal_last_save_ts", 0.0)

    if not force and (now - last) < min_interval:
        st.session_state["journal_dirty"] = True
        return

    err = journal_save(st.session_state.trades, st.session_state.current_position)
    st.session_state["journal_save_err"] = err
    st.session_state["journal_last_save_ts"] = now
    st.session_state["journal_dirty"] = bool(err)


def journal_groups(tdf):
    if tdf is None or tdf.empty or "pnl_percent" not in tdf:
        return []
    d = tdf.copy()
    d["win"] = d["pnl_percent"] > 0
    for c in ("r_multiple", "score"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in ("used", "tf", "source", "direction", "regime"):
        if c not in d:
            d[c] = None
    first = lambda x: str(x).split(" (")[0] if isinstance(x, str) and x else None
    d["الاستراتيجية"] = d["used"].map(first)
    d["الإطار"] = d["tf"].map(first)
    d["المصدر"] = d["source"].map({"plan": "حسب الخطة", "manual": "يدوي"})
    d["الاتجاه"] = d["direction"]
    d["حالة السوق"] = d["regime"].map(
        lambda x: x if isinstance(x, str) and x else None
    )
    try:
        d["نقاط الجودة"] = pd.cut(
            d["score"],
            [-1, 69, 79, 84, 100],
            labels=["أقل من 70", "70–79", "80–84", "85 فأكثر"],
        ).astype(object)
    except (ValueError, TypeError):
        d["نقاط الجودة"] = None
    out = []
    for col in ("نقاط الجودة", "الاستراتيجية", "الإطار",
                "المصدر", "الاتجاه", "حالة السوق"):
        sub = d.dropna(subset=[col])
        if sub.empty:
            continue
        g = sub.groupby(col).agg(
            n=("win", "size"), wr=("win", "mean"),
            r=("r_multiple", "mean"), tot=("pnl_percent", "sum"),
        ).reset_index()
        g["wr"] = (g["wr"] * 100).round(1)
        g["r"] = g["r"].round(2)
        g["tot"] = g["tot"].round(2)
        g.columns = [col, "الصفقات", "نسبة الربح %", "متوسط R", "الإجمالي %"]
        out.append((col, g))
    return out


def get_live_price(ticker, fallback, max_age_minutes=3):
    """يرجع (price, is_live). is_live فقط إذا كان الطابع الزمني حديثًا."""
    try:
        h = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not h.empty:
            last_ts = pd.Timestamp(h.index[-1])
            if last_ts.tz is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
            age_min = (
                pd.Timestamp.now(tz="UTC") - last_ts
            ).total_seconds() / 60.0
            if age_min <= max_age_minutes:
                return float(h["Close"].iloc[-1]), True
    except Exception:
        pass
    return fallback, False


def calc_pnl(pos, price):
    if pos["direction"] == "BUY":
        return (price - pos["entry_price"]) / pos["entry_price"] * 100
    return (pos["entry_price"] - price) / pos["entry_price"] * 100


def today_realized_account_pnl_pct(trades):
    today = datetime.now().strftime("%Y-%m-%d")
    total = 0.0
    for tr in trades or []:
        if str(tr.get("exit_time") or "").startswith(today):
            try:
                val = tr.get("account_pnl_percent")
                val = val if val is not None else tr.get("pnl_percent")
                total += float(val or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def daily_breaker_status(trades, max_loss_pct):
    pnl_today = today_realized_account_pnl_pct(trades)
    tripped = pnl_today <= -abs(max_loss_pct)
    return tripped, pnl_today


def meta_from(a, source):
    if not a:
        return {"source": source}
    return {
        "source": source, "used": a.get("used"), "tf": a.get("tf"),
        "score": a.get("score"),
        "adx": round(float(a["adx"]), 1) if a.get("adx") is not None else None,
        "regime": a.get("regime"), "signal": a.get("signal"),
        "decision": (a.get("arb") or {}).get("action", "")[:70],
    }


def open_position(direction, price, ticker, atr, meta=None, sl=None, tp=None,
                  tp1=None, account_risk_pct=None, partial=0.5):
    sign = 1 if direction == "BUY" else -1
    original_sl = sl if sl is not None else price - sign * SL_ATR * atr
    final_tp = tp if tp is not None else price + sign * TP2_ATR * atr
    resolved_tp1 = (
        tp1 if tp1 is not None else (price + (final_tp - price) * 0.5)
    )

    st.session_state.current_position = {
        "direction": direction,
        "entry_price": price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "sl": original_sl,
        "original_sl": original_sl,
        "tp": final_tp,
        "tp1": resolved_tp1,
        "partial": float(partial),
        "stage": 0,
        "realized_partial": 0.0,
        "atr": atr,
        "account_risk_pct": account_risk_pct,
        "meta": meta or {},
    }
    persist(force=True)


def compute_trade_result(pos, price):
    """دالة خالصة تحسب نتيجة إغلاق صفقة."""
    pnl = calc_pnl(pos, price)
    sl_dist_pct = abs(pos["entry_price"] - pos["sl"]) / pos["entry_price"] * 100
    r_multiple = (pnl / sl_dist_pct) if sl_dist_pct else None
    acct_risk = pos.get("account_risk_pct")
    if r_multiple is not None and acct_risk:
        account_pnl_percent = r_multiple * acct_risk
    else:
        account_pnl_percent = pnl
    return {
        "pnl": pnl,
        "r_multiple": r_multiple,
        "account_pnl_percent": account_pnl_percent,
    }


def close_trade(price, exit_reason, pnl_override=None, realized_partial=0.0):
    if st.session_state.current_position is None:
        return
    pos = dict(st.session_state.current_position)
    meta = pos.pop("meta", None) or {}

    if pnl_override is not None:
        pnl = float(pnl_override)
        original_sl = float(pos.get("original_sl") or pos["sl"])
        sl_dist_pct = (
            abs(pos["entry_price"] - original_sl) / pos["entry_price"] * 100
        )
        r_multiple = (pnl / sl_dist_pct) if sl_dist_pct else None
    else:
        result = compute_trade_result(pos, price)
        pnl = result["pnl"]
        r_multiple = result["r_multiple"]

    acct_risk = pos.get("account_risk_pct")
    account_pnl_percent = (
        r_multiple * acct_risk if (r_multiple is not None and acct_risk) else pnl
    )

    cleaned = {
        k: v for k, v in pos.items()
        if k not in ("stage", "partial", "realized_partial")
    }
    st.session_state.trades.append({
        **cleaned, **meta,
        "exit_price": price,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exit_reason": exit_reason,
        "pnl_percent": pnl,
        "r_multiple": r_multiple,
        "account_pnl_percent": account_pnl_percent,
        "partial_realized": realized_partial,
    })
    st.session_state.current_position = None
    why = {
        "SL": "ضُرب وقف الخسارة",
        "TP": "تحقق الهدف",
        "BE": "أُغلق عند الدخول (بعد TP1)",
        "Manual": "أُغلقت يدوياً",
    }[exit_reason]
    kind = "success" if pnl > 0 else "error"
    st.session_state.notice = (
        kind,
        f"{'✅' if pnl > 0 else '❌'} {why} عند {P(price)} | النتيجة: {pnl:+.2f}%",
    )
    persist(force=True)


def _bar_touches(direction, hi, lo, level, kind):
    if direction == "BUY":
        return lo <= level if kind == "sl" else hi >= level
    return hi >= level if kind == "sl" else lo <= level


def check_exit_candles(pos, df_bars):
    """يفحص كل شمعة منذ وقت الدخول، ويطابق منطق run_backtest_v2."""
    events = []
    if df_bars is None or df_bars.empty or pos is None:
        return events

    direction = pos["direction"]
    entry = float(pos["entry_price"])
    sl = float(pos["original_sl"])
    tp = float(pos["tp"])
    tp1 = float(pos.get("tp1") or tp)
    stage = int(pos.get("stage", 0))

    try:
        entry_ts = pd.Timestamp(pos["entry_time"])
        idx_tz = df_bars.index.tz
        if idx_tz is not None and entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize(idx_tz)
        elif idx_tz is None and entry_ts.tz is not None:
            entry_ts = entry_ts.tz_localize(None)
    except Exception:
        entry_ts = None

    for ts, bar in df_bars.iterrows():
        if entry_ts is not None and pd.Timestamp(ts) <= entry_ts:
            continue
        hi, lo, op = (
            float(bar["High"]), float(bar["Low"]), float(bar["Open"])
        )

        cur_sl = sl if stage == 0 else entry
        if _bar_touches(direction, hi, lo, cur_sl, "sl"):
            fill = (
                min(cur_sl, op) if direction == "BUY" else max(cur_sl, op)
            )
            events.append({
                "type": "sl" if stage == 0 else "be",
                "price": fill, "time": ts,
            })
            return events

        if stage == 0 and _bar_touches(direction, hi, lo, tp1, "tp"):
            events.append({"type": "tp1", "price": tp1, "time": ts})
            stage = 1

        if stage == 1 and _bar_touches(direction, hi, lo, tp, "tp"):
            events.append({"type": "tp2", "price": tp, "time": ts})
            return events

    return events


def fetch_bars_since(ticker, entry_time, max_days=90):
    """يجلب شموع منذ وقت الدخول. 5m/1h/1d حسب العمر."""
    try:
        entry_ts = pd.Timestamp(entry_time)
    except Exception:
        return None
    age_days = (pd.Timestamp.now() - entry_ts).days
    if age_days <= 5:
        interval, period = "5m", "5d"
    elif age_days <= 30:
        interval, period = "1h", "30d"
    else:
        interval, period = "1d", f"{min(max_days, 90)}d"
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            return None
        cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
        return df[cols]
    except Exception:
        return None


def _apply_exit_events(events):
    pos = st.session_state.current_position
    if pos is None:
        return
    realized = float(pos.get("realized_partial", 0.0))
    partial = float(pos.get("partial", 0.5))

    for ev in events:
        if ev["type"] == "tp1":
            realized += calc_pnl(pos, ev["price"]) * partial
            pos["stage"] = 1
            pos["sl"] = pos["entry_price"]
            pos["realized_partial"] = realized
            st.session_state.current_position = pos
        elif ev["type"] == "tp2":
            rem = (1.0 - partial) if pos.get("stage") == 1 else 1.0
            final = calc_pnl(pos, ev["price"]) * rem
            close_trade(
                ev["price"], "TP",
                pnl_override=realized + final,
                realized_partial=realized,
            )
            return
        elif ev["type"] in ("sl", "be"):
            rem = (1.0 - partial) if pos.get("stage") == 1 else 1.0
            final = calc_pnl(pos, ev["price"]) * rem
            close_trade(
                ev["price"],
                "SL" if ev["type"] == "sl" else "BE",
                pnl_override=realized + final,
                realized_partial=realized,
            )
            return


def check_exit(price):
    """فحص الخروج بناءً على شموع منذ الدخول. fallback: نقطة السعر."""
    pos = st.session_state.current_position
    if pos is None:
        return

    df_bars = fetch_bars_since(pos["ticker"], pos["entry_time"])
    if df_bars is not None and not df_bars.empty:
        events = check_exit_candles(pos, df_bars)
        if events:
            _apply_exit_events(events)
            return
        return

    if pos["direction"] == "BUY":
        hit_sl, hit_tp = price <= pos["sl"], price >= pos["tp"]
    else:
        hit_sl, hit_tp = price >= pos["sl"], price <= pos["tp"]
    if hit_sl:
        close_trade(price, "SL")
    elif hit_tp:
        close_trade(price, "TP")


def should_alert_event(news, ticker, last_alerts, cooldown_hours=6.0):
    """تنبيه حدث اقتصادي كبير."""
    if not isinstance(news, dict):
        return False
    if str(news.get("event_risk", "")).lower() != "high":
        return False
    key = f"event_{ticker.strip().upper()}"
    last = (last_alerts or {}).get(key)
    if last:
        try:
            dt = datetime.strptime(str(last), "%Y-%m-%d %H:%M")
            if (datetime.now() - dt).total_seconds() / 3600.0 < cooldown_hours:
                return False
        except Exception:
            pass
    return True


def should_alert_position(pos, price, last_alert_time=None, cooldown_hours=6.0):
    """تنبيه قرب الوقف/الهدف."""
    atr = float(pos.get("atr") or 0)
    if atr <= 0:
        return False, ""
    try:
        sl = float(pos.get("original_sl") or pos["sl"])
        tp = float(pos["tp"])
        direction = pos["direction"]
    except (KeyError, TypeError, ValueError):
        return False, ""

    if last_alert_time:
        try:
            last = datetime.strptime(str(last_alert_time), "%Y-%m-%d %H:%M")
            if (datetime.now() - last).total_seconds() / 3600.0 < cooldown_hours:
                return False, ""
        except Exception:
            pass

    if direction == "BUY":
        near_sl, near_tp = price <= sl + 0.3 * atr, price >= tp - 0.3 * atr
    else:
        near_sl, near_tp = price >= sl - 0.3 * atr, price <= tp + 0.3 * atr

    if near_sl:
        return True, "⚠️ السعر يقترب من وقف الخسارة"
    if near_tp:
        return True, "🎯 السعر يقترب من الهدف"
    return False, ""


def compute_equity_curve(trades, initial=100.0):
    """يبني منحنى رأس المال من سجل الصفقات."""
    if not trades:
        return pd.DataFrame()
    rows = []
    eq, peak = float(initial), float(initial)
    for tr in trades:
        try:
            pnl = float(
                tr.get("account_pnl_percent")
                if tr.get("account_pnl_percent") is not None
                else tr.get("pnl_percent") or 0.0
            )
        except (TypeError, ValueError):
            pnl = 0.0
        eq *= (1 + pnl / 100.0)
        peak = max(peak, eq)
        rows.append({
            "exit_time": tr.get("exit_time") or tr.get("entry_time") or "",
            "pnl_percent": pnl,
            "equity": eq,
            "drawdown": (eq / peak - 1) * 100 if peak > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def monte_carlo(trades, n_sims=1000, n_trades=None, initial=100.0, seed=42):
    """محاكاة مونت كارلو: إعادة سحب عشوائي للصفقات."""
    if not trades:
        return None
    pnls = []
    for tr in trades:
        try:
            val = tr.get("account_pnl_percent")
            val = val if val is not None else tr.get("pnl_percent")
            pnls.append(float(val or 0.0))
        except (TypeError, ValueError):
            continue
    if not pnls:
        return None
    pnls = np.array(pnls, dtype=float)
    k = int(n_trades or len(pnls))
    rng = np.random.default_rng(seed)
    finals = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    for i in range(n_sims):
        sample = rng.choice(pnls, size=k, replace=True)
        eq = initial * np.cumprod(1 + sample / 100.0)
        finals[i] = eq[-1]
        peak = np.maximum.accumulate(eq)
        max_dds[i] = float(np.min(eq / peak - 1) * 100)
    return {
        "n_sims": n_sims, "n_trades": k, "initial": initial,
        "p05": float(np.percentile(finals, 5)),
        "p50": float(np.percentile(finals, 50)),
        "p95": float(np.percentile(finals, 95)),
        "dd_p50": float(np.percentile(max_dds, 50)),
        "dd_p05": float(np.percentile(max_dds, 5)),
    }