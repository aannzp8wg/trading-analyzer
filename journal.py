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
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from constants import SL_ATR, TP1_ATR, TP2_ATR
from gold_engine import new_trade_state, step_trade, trade_pnl
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
    for c in ("news_impact", "news_mode"):
        if c not in d:
            d[c] = None
    d["أثر الأخبار"] = d["news_impact"].map({"supports": "تدعم الإشارة", "neutral": "محايدة", "conflicts": "تعاكس"})
    d["وضع الأخبار"] = d["news_mode"].map({"full": "كامل", "degraded": "تقويم فقط", "none": "بلا أخبار"})
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
                "المصدر", "الاتجاه", "حالة السوق", "أثر الأخبار", "وضع الأخبار"):
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
    news = (a.get("agents") or {}).get("news")
    return {
        "source": source, "used": a.get("used"), "tf": a.get("tf"),
        "score": a.get("score"),
        "adx": round(float(a["adx"]), 1) if a.get("adx") is not None else None,
        "regime": a.get("regime"), "signal": a.get("signal"),
        "decision": (a.get("arb") or {}).get("action", "")[:70],
        # لقياس أثر طبقة الأخبار لاحقاً (news_value_report)
        "news_impact": (news or {}).get("impact_on_signal") if news else None,
        "news_event_risk": (news or {}).get("event_risk") if news else None,
        "news_mode": ("degraded" if news.get("degraded") else "full") if news else "none",
    }


def open_position(direction, price, ticker, atr, meta=None, sl=None, tp=None,
                  tp1=None, account_risk_pct=None, partial=0.5, cost_pct=0.0):
    """sl / tp (=TP2) / tp1 من الخطة. أي مستوى غير منتهٍ أو في الجهة الخطأ يُستبدل بمستوى ATR:
    كان sl=NaN يُحفظ كما هو، وكل مقارنة مع NaN خاطئة فلا يُضرب الوقف أبدًا.
    cost_pct: تكلفة الصفقة % تُخصم من النتيجة (كما في الباكتست)."""
    sign = 1 if direction == "BUY" else -1

    def fin(v):
        try:
            return v is not None and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    sl_ok = fin(sl) and sign * (price - float(sl)) > 0
    tp_ok = fin(tp) and sign * (float(tp) - price) > 0
    fallback = not (sl_ok and tp_ok)
    sl_v = float(sl) if sl_ok else price - sign * SL_ATR * atr
    tp_v = float(tp) if tp_ok else price + sign * TP2_ATR * atr
    if fallback:
        tp1_v = price + sign * TP1_ATR * atr
    elif fin(tp1) and sign * (float(tp1) - price) > 0 and sign * (tp_v - float(tp1)) > 0:
        tp1_v = float(tp1)
    else:
        tp1_v = price + (tp_v - price) * 0.5

    st.session_state.current_position = {
        "direction": direction,
        "entry_price": price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "entry_ts": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "sl": sl_v,
        "original_sl": sl_v,
        "tp": tp_v,
        "tp1": tp1_v,
        "partial": float(partial),
        "stage": 0,
        "realized_partial": 0.0,
        "cost_pct": float(cost_pct or 0.0),
        "last_scan_ts": None,
        "levels_fallback": fallback,
        "atr": atr,
        "account_risk_pct": account_risk_pct,
        "meta": meta or {},
    }
    persist(force=True)


# ---------------------------------------------------------------- المحرك الورقي (مشترك مع الباكتست)
def _fnum(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def pos_to_state(pos):
    """حالة المحرك المشترك من صفقة محفوظة (نفس أسماء الحقول القديمة: original_sl/realized_partial/stage)."""
    s = 1 if pos["direction"] == "BUY" else -1
    sl0 = _fnum(pos.get("original_sl"), _fnum(pos["sl"]))
    t = new_trade_state(s, pos["entry_price"], sl0, _fnum(pos.get("tp1"), np.nan), pos["tp"],
                        pos.get("partial", 0.5))
    t["sl"] = _fnum(pos["sl"], sl0)
    t["stage"] = int(pos.get("stage") or 0)
    t["realized_gross"] = _fnum(pos.get("realized_partial"), 0.0)
    return t


def _to_utc(x):
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:                     # entry_time القديم: توقيت السيرفر المحلي بلا منطقة
        return pd.Timestamp(ts.to_pydatetime().astimezone(timezone.utc))
    return ts.tz_convert("UTC")


def _entry_ts_utc(pos):
    return _to_utc(pos.get("entry_ts") or pos["entry_time"])


def filter_bars(df, entry_ts, last_scan_ts, bar_td=None, now=None):
    """شموع للفحص: (1) بتوقيت UTC موحّد (كان entry_time المحلي يُفسَّر بتوقيت البورصة فتضيع ساعات)،
    (2) تبدأ بعد الدخول (الشمعة التي تحتوي لحظة الدخول مستبعدة)، (3) بعد آخر فحص،
    (4) مكتملة فقط إن عُرفت مدتها (الشمعة الجارية تُغطّيها لقطة السعر)."""
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    idx = pd.DatetimeIndex(d.index)
    d.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    mask = d.index >= _to_utc(entry_ts)
    if last_scan_ts:
        mask &= d.index > _to_utc(last_scan_ts)
    if bar_td is not None:
        mask &= (d.index + bar_td) <= (_to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC"))
    cols = [c for c in ("Open", "High", "Low", "Close") if c in d.columns]
    return d.loc[mask, cols].dropna()


def _replay(pos, bars=None, live_price=None):
    """ينفّذ الشموع بالترتيب عبر step_trade (نفس منطق الباكتست). لقطة السعر تُقيَّم على نسخة:
    لا تُثبَّت إلا إن أغلقت الصفقة، حتى لا تتقدّم الحالة عن نقطة آخر شمعة فُحصت (وإلا أُعيد
    فحص شمعة قديمة بوقف BE لم يكن قائماً وقتها)."""
    t = pos_to_state(pos)
    events, outcome, last_ts = [], None, pos.get("last_scan_ts")
    names = {"SL": "sl", "BE": "be", "TP": "tp2"}
    if bars is not None and len(bars):
        arr = bars[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
        for ts, r in zip(bars.index, arr):
            stage_before = t["stage"]
            res = step_trade(t, r[0], r[1], r[2])
            last_ts = pd.Timestamp(ts).isoformat()
            if stage_before == 0 and t["stage"] == 1:
                events.append({"type": "tp1", "price": t["tp1"], "time": ts})
            if res is not None:
                outcome = {"reason": res[0], "price": float(res[1])}
                events.append({"type": names[res[0]], "price": float(res[1]), "time": ts})
                break
    if outcome is None and live_price is not None and np.isfinite(live_price):
        t2 = dict(t)
        res = step_trade(t2, live_price, live_price, live_price)
        if res is not None:
            t = t2
            outcome = {"reason": res[0], "price": float(res[1])}
            events.append({"type": names[res[0]], "price": float(res[1]), "time": "live"})
    return t, outcome, events, last_ts


def advance_position(pos, bars=None, live_price=None):
    """دالة خالصة: ترجع (pos_جديد، خروج|None، أحداث). لا تلمس session_state."""
    t, outcome, events, last_ts = _replay(pos, bars, live_price)
    new = dict(pos)
    new.update(sl=t["sl"], stage=t["stage"], realized_partial=t["realized_gross"], last_scan_ts=last_ts)
    return new, outcome, events


def check_exit_candles(pos, df_bars):
    """متوافقة مع الاسم القديم: أحداث الشموع (sl/be/tp1/tp2) دون تعديل الصفقة."""
    if pos is None or df_bars is None or len(df_bars) == 0:
        return []
    try:
        bars = filter_bars(df_bars, _entry_ts_utc(pos), pos.get("last_scan_ts"))
    except (ValueError, TypeError, KeyError):
        return []                      # وقت دخول تالف: لا نخمّن، ولا ننهار
    return _replay(pos, bars, None)[2]


def fetch_bars_since(ticker, entry_time, max_days=700):
    """شموع منذ وقت الدخول: 5m (≤5 أيام) أو 1h (حتى سنتين) أو 1d. المدة في df.attrs['bar_td']."""
    try:
        age_days = (pd.Timestamp.now(tz="UTC") - _to_utc(entry_time)).days
    except Exception:
        return None
    if age_days <= 5:
        interval, period, td = "5m", "5d", pd.Timedelta(minutes=5)
    elif age_days <= max_days:
        interval, period, td = "1h", f"{min(age_days + 3, 729)}d", pd.Timedelta(hours=1)
    else:
        interval, period, td = "1d", "max", pd.Timedelta(days=1)
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df[[c for c in ("Open", "High", "Low", "Close") if c in df.columns]].copy()
        df.attrs["bar_td"] = td
        return df
    except Exception:
        return None


def floating_pnl(pos, price):
    """الربح الحالي % (خام) متضمنًا الجزء المُحقَّق عند TP1 (calc_pnl يتجاهله)."""
    return trade_pnl(pos_to_state(pos), price, 0.0)[0]


def compute_trade_result(pos, price):
    """نتيجة إغلاق صفقة بنفس معادلات الباكتست (إغلاق جزئي + تكلفة). R بالنسبة للوقف الأولي.
    كان الإغلاق اليدوي بعد TP1 يتجاهل الجزء المُحقَّق ويحسب R من وقف BE (=0) فيصير R=None."""
    gross, net, r_net, _ = trade_pnl(pos_to_state(pos), price, _fnum(pos.get("cost_pct"), 0.0))
    r_multiple = float(r_net) if np.isfinite(r_net) else None
    acct_risk = pos.get("account_risk_pct")
    account = r_multiple * acct_risk if (r_multiple is not None and acct_risk) else net
    return {"pnl": net, "gross_pnl": gross, "r_multiple": r_multiple, "account_pnl_percent": account}


def close_trade(price, exit_reason, pnl_override=None, realized_partial=0.0):
    if st.session_state.current_position is None:
        return
    pos = dict(st.session_state.current_position)
    meta = pos.pop("meta", None) or {}
    result = compute_trade_result(pos, price)
    pnl, r_multiple, account_pnl_percent = result["pnl"], result["r_multiple"], result["account_pnl_percent"]
    if pnl_override is not None:                       # توافق مع الاستدعاءات القديمة
        pnl = float(pnl_override)
        original_sl = float(pos.get("original_sl") or pos["sl"])
        sl_pct = abs(pos["entry_price"] - original_sl) / pos["entry_price"] * 100
        r_multiple = (pnl / sl_pct) if sl_pct else None
        acct = pos.get("account_risk_pct")
        account_pnl_percent = r_multiple * acct if (r_multiple is not None and acct) else pnl
    partial_realized = float(pos.get("realized_partial") or realized_partial or 0.0)

    cleaned = {k: v for k, v in pos.items() if k not in ("stage", "partial", "realized_partial", "last_scan_ts")}
    st.session_state.trades.append({
        **cleaned, **meta,
        "exit_price": price,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exit_reason": exit_reason,
        "pnl_percent": pnl,
        "gross_pnl_percent": result["gross_pnl"],
        "r_multiple": r_multiple,
        "account_pnl_percent": account_pnl_percent,
        "partial_realized": partial_realized,
    })
    st.session_state.current_position = None
    why = {
        "SL": "ضُرب وقف الخسارة",
        "TP": "تحقق الهدف",
        "BE": "أُغلق عند الدخول (بعد TP1)",
        "Manual": "أُغلقت يدوياً",
    }.get(exit_reason, exit_reason)
    kind = "success" if pnl > 0 else "error"
    st.session_state.notice = (
        kind,
        f"{'✅' if pnl > 0 else '❌'} {why} عند {P(price)} | النتيجة: {pnl:+.2f}%",
    )
    persist(force=True)


def check_exit(price, bars_fn=None):
    """يفحص الصفقة بالشموع المكتملة منذ الدخول (يلتقط ذيول الشموع) ثم بلقطة السعر الحالية، ويغلقها
    عند الوقف/الهدف. يرجع تقريرًا {closed, reason, events, bars_ok}."""
    pos = st.session_state.current_position
    rep = {"closed": False, "reason": None, "events": [], "bars_ok": False}
    if pos is None:
        return rep
    bars = None
    try:
        raw = (bars_fn or fetch_bars_since)(pos["ticker"], pos.get("entry_ts") or pos["entry_time"])
        if raw is not None and len(raw):
            bars = filter_bars(raw, _entry_ts_utc(pos), pos.get("last_scan_ts"),
                               getattr(raw, "attrs", {}).get("bar_td"))
            rep["bars_ok"] = True
    except Exception:
        bars = None
    new, outcome, events = advance_position(pos, bars, price)
    st.session_state.current_position = new
    rep["events"] = [e["type"] for e in events]
    if outcome:
        close_trade(outcome["price"], outcome["reason"])
        rep.update(closed=True, reason=outcome["reason"])
    elif any(e["type"] == "tp1" for e in events):
        st.session_state.notice = ("info", "🎯 تحقق الهدف 1: أُغلق جزء من الصفقة ونُقل الوقف إلى سعر الدخول.")
        persist(force=True)          # الحالة + نقطة الفحص تُحفظان معاً (ضروري لسلامة إعادة الفحص)
    else:
        persist()
    return rep


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
        sl = float(pos["sl"])          # الوقف الحالي (يصير سعر الدخول بعد TP1)
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


def news_value_report(trades, min_group=10, n_perm=5000, seed=7):
    """هل تفيد طبقة الأخبار فعلاً؟ يقارن متوسط R للصفقات التي دعمتها الأخبار بالباقي (محايدة/معاكسة)،
    باختبار تبديل (permutation) لا يفترض توزيعاً. لا استنتاج بعيّنة صغيرة."""
    rows = []
    for t in trades or []:
        imp, r = t.get("news_impact"), t.get("r_multiple")
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if imp in ("supports", "neutral", "conflicts") and np.isfinite(r):
            rows.append((imp, r))
    groups = {}
    for name in ("supports", "neutral", "conflicts"):
        rs = np.array([r for i, r in rows if i == name], dtype=float)
        groups[name] = {"n": int(len(rs)), "avg_r": float(rs.mean()) if len(rs) else float("nan"),
                        "wr": float((rs > 0).mean() * 100) if len(rs) else float("nan")}
    sup = np.array([r for i, r in rows if i == "supports"], dtype=float)
    oth = np.array([r for i, r in rows if i != "supports"], dtype=float)
    out = {"groups": groups, "n": len(rows), "diff": float("nan"), "p": float("nan"), "verdict": ""}
    if len(sup) < min_group or len(oth) < min_group:
        out["verdict"] = (f"عيّنة غير كافية: تحتاج {min_group} صفقات على الأقل في كل من «تدعم» و«غيرها» "
                          f"(الآن {len(sup)} و{len(oth)}).")
        return out
    diff = float(sup.mean() - oth.mean())
    allr = np.r_[sup, oth]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(allr)
        if abs(allr[:len(sup)].mean() - allr[len(sup):].mean()) >= abs(diff):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    out.update(diff=diff, p=float(p))
    if p < 0.05:
        out["verdict"] = ("الأخبار تُحسّن النتائج بدلالة إحصائية." if diff > 0
                          else "الصفقات التي تدعمها الأخبار أسوأ بدلالة إحصائية — راجع طبقة الأخبار.")
    else:
        out["verdict"] = f"لا فرق ذو دلالة إحصائية بعد {len(rows)} صفقة (p={p:.2f}): لا دليل بعد أن الأخبار تفيد أو تضر."
    return out