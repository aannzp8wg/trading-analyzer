"""
monitor.py — مراقب مجدول للصفقة المفتوحة (للقراءة فقط) + تنبيهات تيليجرام بلا تكرار.

لماذا للقراءة فقط؟ التطبيق يحمي سجل الـGist بـ If-Match على ETag الـGist كله. أي كتابة من
هنا (حتى في ملف آخر داخل الـGist) تغيّر الـETag فقد يرفض التطبيق حفظ صفقاتك بحجة "تعارض".
لذلك التطبيق هو الكاتب الوحيد للسجل، والمراقب ينبّه فقط. وحين تفتح التطبيق وتضغط «تحديث السعر»
تُعاد معالجة الشموع من نقطة الفحص المحفوظة بنفس المنطق، فيُسجَّل الإغلاق بنفس السعر.

ذاكرة منع التكرار: ملف .monitor_state.json يحفظه كاش GitHub Actions (انظر الـworkflow).

المتغيرات (أسرار المستودع):
  GITHUB_TOKEN + GIST_ID        نفس أسرار التطبيق (قراءة الـGist يكفي)
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
الأكواد: 0 سليم | 1 فشل إرسال تيليجرام | 2 إعداد ناقص/السجل غير مقروء | 3 تعذّر فحص السوق بالكامل
(أي كود غير 0 يُفشل المهمة فيصلك بريد GitHub — هذا هو "نبض" المراقب نفسه).
"""
import json
import os
import sys
import types
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

STATE_FILE = ".monitor_state.json"
STATE_KEEP_DAYS = 14
EVENT_WINDOW_H = 1.05          # تنبيه الحدث الاقتصادي قبل ساعة تقريباً


def _ensure_streamlit_stub():
    """المراقب يستورد دوالاً خالصة من journal/risk_engine وهما يستوردان streamlit للتزيين فقط.
    في بيئة المراقب الخفيفة (بلا streamlit) نحقن بديلاً لا يفعل شيئاً."""
    try:
        import streamlit  # noqa: F401
        return
    except ImportError:
        pass

    class _S(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v

    m = types.ModuleType("streamlit")
    m.session_state, m.secrets = _S(), {}
    m.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["streamlit"] = m


_ensure_streamlit_stub()

import pandas as pd  # noqa: E402

import news_engine as nw  # noqa: E402
from journal import (advance_position, compute_trade_result, fetch_bars_since,  # noqa: E402
                     filter_bars, get_live_price, should_alert_position)


# ---------------------------------------------------------------- جلب السجل (قراءة فقط)
def load_position(token, gist_id, opener=None):
    """يرجع (position|None, error). GET فقط."""
    opener = opener or (lambda req: urllib.request.urlopen(req, timeout=15))
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "User-Agent": "position-monitor"})
    try:
        with opener(req) as r:
            data = json.loads(r.read())
        f = (data.get("files") or {}).get("trades.json")
        if not f:
            return None, ""
        txt = (f.get("content") or "").strip()
        obj = json.loads(txt) if txt else {}
        pos = obj.get("position") if isinstance(obj, dict) else None
        return (pos if isinstance(pos, dict) else None), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e).replace(token, '***')[:120]}"


# ---------------------------------------------------------------- تيليجرام
def send_telegram(token, chat, text, opener=None):
    opener = opener or (lambda req: urllib.request.urlopen(req, timeout=10))
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        with opener(req) as r:
            return (r.status == 200), ("" if r.status == 200 else f"HTTP {r.status}")
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).replace(str(token), '***')[:100]}"


# ---------------------------------------------------------------- ذاكرة منع التكرار
def load_state(path=STATE_FILE):
    try:
        with open(path, encoding="utf-8") as fh:
            st_ = json.load(fh)
        return st_ if isinstance(st_.get("sent"), dict) else {"sent": {}}
    except Exception:
        return {"sent": {}}


def save_state(state, now, path=STATE_FILE):
    cut = (now - timedelta(days=STATE_KEEP_DAYS)).isoformat()
    state["sent"] = {k: v for k, v in state["sent"].items() if str(v) >= cut}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)


# ---------------------------------------------------------------- منطق التنبيهات (دالة خالصة)
_REASON = {"SL": "ضُرب وقف الخسارة", "BE": "أُغلقت عند الدخول بعد TP1 (تعادل)", "TP": "تحقق الهدف 2"}


def _fmt(x):
    try:
        return f"{float(x):,.2f}" if abs(float(x)) >= 10 else f"{float(x):.5f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "—"


def build_alerts(pos, bars, live_price, calendar, now):
    """يرجع قائمة {id, text}. كل id ثابت لنفس الحدث عبر التشغيلات (لمنع التكرار)."""
    ets = str(pos.get("entry_ts") or pos.get("entry_time"))
    tk, side = pos.get("ticker", "?"), pos.get("direction", "?")
    head = f"{tk} {side}"
    new, outcome, events = advance_position(pos, bars, live_price)
    out = []

    for e in events:
        if e["type"] == "tp1":
            part = int(round(float(pos.get("partial", 0.5)) * 100))
            out.append({"id": f"{ets}|tp1|{e['time']}", "text":
                        f"🎯 {head}: تحقق الهدف 1 عند {_fmt(e['price'])}.\n"
                        f"أُغلق {part}% ونُقل الوقف إلى الدخول ({_fmt(pos['entry_price'])}).\n"
                        f"المتبقي يتجه للهدف 2 ({_fmt(pos['tp'])})."})

    if outcome:
        res = compute_trade_result(new, outcome["price"])
        r_txt = f" | R={res['r_multiple']:+.2f}" if res["r_multiple"] is not None else ""
        out.append({"id": f"{ets}|{outcome['reason']}", "text":
                    f"{'✅' if res['pnl'] > 0 else '❌'} {head}: {_REASON.get(outcome['reason'], outcome['reason'])} "
                    f"عند {_fmt(outcome['price'])}\nالنتيجة التقديرية: {res['pnl']:+.2f}%{r_txt}\n"
                    f"السجل ما زال مفتوحاً في التطبيق: افتحه واضغط «تحديث السعر» لتسجيل الإغلاق."})
        return out                                   # لا معنى لتنبيهات القرب بعد الإغلاق

    price = live_price
    if price is None and bars is not None and len(bars):
        price = float(bars["Close"].iloc[-1])
    if price is not None:
        near, msg = should_alert_position(new, price)
        if near:
            bucket = f"{now:%Y%m%d}-{now.hour // 6}"        # مرة كل 6 ساعات لكل مرحلة
            out.append({"id": f"{ets}|near|{new.get('stage', 0)}|{bucket}", "text":
                        f"{msg}\n{head} | السعر {_fmt(price)} | الوقف {_fmt(new['sl'])} | الهدف {_fmt(new['tp'])}"})

    for e in (calendar or {}).get("events", []):
        if e.get("level") == "high" and 0 <= e.get("hours", 99) <= EVENT_WINDOW_H:
            mins = max(int(round(e["hours"] * 60)), 0)
            out.append({"id": f"{ets}|event|{e['title']}|{e['when']}", "text":
                        f"⏰ حدث عالي التأثير خلال ~{mins} دقيقة: {e['title']} ({e['currency']})\n"
                        f"عندك صفقة {head} مفتوحة. فكّر في تقليل المخاطرة أو تأمين الوقف قبل الخبر."})
    return out


# ---------------------------------------------------------------- التشغيل
def run(env=None, now=None, deps=None, state_path=STATE_FILE):
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    d = deps or {}
    gh, gid = env.get("GITHUB_TOKEN"), env.get("GIST_ID")
    tg, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not all((gh, gid, tg, chat)):
        print("إعداد ناقص: GITHUB_TOKEN/GIST_ID/TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return 2, {}

    pos, err = (d.get("load_position") or load_position)(gh, gid)
    if err:
        print("تعذّر قراءة السجل:", err)
        return 2, {}
    if pos is None:
        print("لا صفقة مفتوحة.")
        return 0, {"sent": 0}

    tk = pos["ticker"]
    entry_ref = pos.get("entry_ts") or pos["entry_time"]
    bars, bars_ok = None, False
    try:
        raw = (d.get("bars") or fetch_bars_since)(tk, entry_ref)
        if raw is not None and len(raw):
            from journal import _entry_ts_utc
            bars = filter_bars(raw, _entry_ts_utc(pos), pos.get("last_scan_ts"),
                               getattr(raw, "attrs", {}).get("bar_td"), now)
            bars_ok = True
    except Exception as e:
        print("تعذّر جلب الشموع:", type(e).__name__)
    live, live_ok = (d.get("live") or (lambda t: get_live_price(t, None)))(tk)
    live = live if live_ok else None
    if not bars_ok and live is None:
        print("تعذّر فحص السوق (لا شموع ولا سعر لحظي).")
        return 3, {}

    cal = (d.get("calendar") or (lambda t, n: nw.fetch_calendar(nw.currencies_for(t), n)))(tk, now)
    alerts = build_alerts(pos, bars, live, cal, now)
    if not bars_ok:                       # نعمل بلقطة السعر فقط: قد يفوتنا ذيل شمعة، والمستخدم يجب أن يعلم
        alerts.append({"id": f"{entry_ref}|nobars|{now:%Y%m%d}-{now.hour // 12}", "text":
                       f"⚠️ المراقب لم يستطع جلب شموع {tk}؛ يفحص صفقتك بلقطة السعر فقط الآن "
                       f"(قد يفوته ذيل شمعة لمس الوقف). تأكد من وقف حقيقي عند وسيطك."})
    state = load_state(state_path)
    sender = d.get("send") or (lambda text: send_telegram(tg, chat, text))
    sent, failed = 0, 0
    for a in alerts:
        if a["id"] in state["sent"]:
            continue
        ok, why = sender(a["text"])
        if ok:
            state["sent"][a["id"]] = now.isoformat()
            sent += 1
        else:
            failed += 1
            print("فشل إرسال تنبيه:", why)
    if sent:
        save_state(state, now, state_path)
    print(f"فُحصت {tk}: تنبيهات جديدة={sent} | فشل={failed} | شموع={'نعم' if bars_ok else 'لا'} | لحظي={'نعم' if live else 'لا'}")
    return (1 if failed else 0), {"sent": sent, "failed": failed}


if __name__ == "__main__":
    code, summary = run()
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"changed={'true' if summary.get('sent') else 'false'}\n")
    sys.exit(code)