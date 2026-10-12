"""test_monitor.py — اختبارات المراقب بلا إنترنت:  python test_monitor.py"""
import json, os, sys, tempfile, types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import yfinance  # noqa
except ImportError:
    m = types.ModuleType("yfinance"); sys.modules["yfinance"] = m

import monitor as M  # noqa: E402  (يحقن بديل streamlit عند غيابه)
import journal  # noqa: E402
import streamlit as st  # noqa: E402

PASSED = []
def check(name, cond, extra=""):
    PASSED.append(bool(cond)); print(("✅" if cond else "❌"), name, "" if cond else extra)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
def mkpos(**kw):
    p = {"direction": "BUY", "entry_price": 100.0, "sl": 98.0, "original_sl": 98.0, "tp": 106.0, "tp1": 103.0,
         "partial": 0.5, "stage": 0, "realized_partial": 0.0, "cost_pct": 0.0, "entry_ts": "2026-10-05T08:00:00+00:00",
         "entry_time": "2026-10-05 08:00", "ticker": "GC=F", "atr": 1.0, "account_risk_pct": 1.0, "last_scan_ts": None}
    p.update(kw); return p
def bars(rows, start="2026-10-05 08:05", freq="5min"):
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.date_range(start, periods=len(rows), freq=freq, tz="UTC"))
    df.attrs["bar_td"] = pd.Timedelta(freq); return df
NOCAL = {"ok": True, "risk": "low", "events": []}

# ---------------- build_alerts
tp1_bars = bars([[100.2, 101.0, 99.7, 100.8], [100.8, 103.5, 100.6, 103.2]])
a1 = M.build_alerts(mkpos(), tp1_bars, None, NOCAL, NOW)
check("TP1 يُنبَّه به مرة واحدة بنص مفهوم", len(a1) == 1 and "الهدف 1" in a1[0]["text"] and "50%" in a1[0]["text"])
check("معرّف التنبيه ثابت عبر التشغيلات", a1[0]["id"] == M.build_alerts(mkpos(), tp1_bars, None, NOCAL, NOW + timedelta(minutes=30))[0]["id"])
sl_bars = bars([[100.2, 100.5, 97.5, 98.0]])
a2 = M.build_alerts(mkpos(), sl_bars, None, {"ok": True, "events": [{"level": "high", "hours": 0.5, "title": "NFP", "currency": "USD", "when": "w"}]}, NOW)
check("ضرب الوقف: تنبيه واحد بنتيجة سالبة ولا تنبيهات قرب/حدث بعد الإغلاق",
      len(a2) == 1 and "وقف" in a2[0]["text"] and "-" in a2[0]["text"].split("النتيجة التقديرية:")[1].split("%")[0], str(a2))
NB = bars([[99.0, 99.2, 98.1, 98.2]])          # الوقف 98 لم يُلمس؛ السعر 98.2 ≤ 98 + 0.3×ATR
near = M.build_alerts(mkpos(), NB, 98.2, NOCAL, NOW)
check("قرب الوقف يُنبَّه به", len(near) == 1 and "يقترب" in near[0]["text"])
near2 = M.build_alerts(mkpos(), NB, 98.2, NOCAL, NOW + timedelta(hours=1))
near3 = M.build_alerts(mkpos(), NB, 98.2, NOCAL, NOW + timedelta(hours=7))
check("تنبيه القرب مرة كل 6 ساعات", near[0]["id"] == near2[0]["id"] and near[0]["id"] != near3[0]["id"])
quiet = bars([[100.0, 100.8, 99.6, 100.4]])
check("لا شيء يحدث => لا تنبيهات", M.build_alerts(mkpos(), quiet, 100.4, NOCAL, NOW) == [])
ev = lambda h, lvl="high": {"ok": True, "events": [{"level": lvl, "hours": h, "title": "FOMC", "currency": "USD", "when": "2026-10-05 12:30 UTC"}]}  # noqa: E731
check("حدث عالٍ خلال 30 دقيقة => تنبيه", len(M.build_alerts(mkpos(), quiet, 100.4, ev(0.5), NOW)) == 1)
check("حدث بعد 3 ساعات أو متوسط الأهمية أو انتهى => لا تنبيه",
      all(M.build_alerts(mkpos(), quiet, 100.4, c, NOW) == [] for c in (ev(3.0), ev(0.5, "medium"), ev(-0.5))))

# تطابق التنبيه مع ما سيسجّله التطبيق عند الإغلاق (نفس المحرك)
st.session_state["trades"], st.session_state["current_position"] = [], None
st.session_state["journal_ok"] = False
journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0, cost_pct=0.1)
st.session_state["current_position"]["entry_ts"] = "2026-10-05T08:00:00+00:00"
mix = bars([[100.2, 101.0, 99.7, 100.8], [100.8, 103.5, 100.6, 103.2], [103.2, 106.5, 103.0, 106.0]])
journal.check_exit(None, bars_fn=lambda tk, et: mix)
rec = st.session_state["trades"][0]
al = M.build_alerts(mkpos(cost_pct=0.1), mix, None, NOCAL, NOW)
final = [a for a in al if "تحقق الهدف 2" in a["text"]]
check("المراقب والتطبيق يتفقان على الخروج (السبب والنتيجة)", rec["exit_reason"] == "TP" and len(final) == 1
      and f"{rec['pnl_percent']:+.2f}%" in final[0]["text"], f"{rec['pnl_percent']} | {final}")

# ---------------- run(): منع التكرار والفشل والأكواد
tmp = tempfile.mkdtemp(); sp = os.path.join(tmp, "state.json")
ENV = {"GITHUB_TOKEN": "ghp_SECRET", "GIST_ID": "g", "TELEGRAM_BOT_TOKEN": "tg_SECRET", "TELEGRAM_CHAT_ID": "1"}
sent = []
def deps(pos=None, raw=tp1_bars, live=(None, False), send=None, cal=NOCAL):
    return {"load_position": lambda t, g: (pos if pos is not None else mkpos(), ""), "bars": lambda tk, et: raw,
            "live": lambda tk: live, "calendar": lambda tk, n: cal, "send": send or (lambda text: (sent.append(text) or (True, "")))}
code, s1 = M.run(ENV, NOW, deps(), sp)
check("التشغيل الأول يرسل TP1", code == 0 and s1["sent"] == 1 and len(sent) == 1)
code, s2 = M.run(ENV, NOW + timedelta(minutes=30), deps(), sp)
check("التشغيل الثاني لا يكرر (ذاكرة منع التكرار)", code == 0 and s2["sent"] == 0 and len(sent) == 1)
sp2 = os.path.join(tmp, "s2.json"); n_before = len(sent)
flip = {"ok": False}
def flaky(text):
    if not flip["ok"]: return False, "HTTP 500"
    sent.append(text); return True, ""
code, s3 = M.run(ENV, NOW, deps(send=flaky), sp2)
check("فشل تيليجرام => كود 1 ولا يُعلَّم كمُرسَل", code == 1 and s3["failed"] == 1 and not os.path.exists(sp2))
flip["ok"] = True
code, s4 = M.run(ENV, NOW + timedelta(minutes=30), deps(send=flaky), sp2)
check("إعادة المحاولة تنجح في التشغيل التالي", code == 0 and s4["sent"] == 1 and len(sent) == n_before + 1)
check("إعداد ناقص => 2", M.run({"GIST_ID": "g"}, NOW, deps(), sp)[0] == 2)
check("خطأ قراءة السجل => 2", M.run(ENV, NOW, {**deps(), "load_position": lambda t, g: (None, "boom")}, sp)[0] == 2)
check("لا صفقة مفتوحة => 0 بلا إرسال", M.run(ENV, NOW, {**deps(), "load_position": lambda t, g: (None, "")}, sp) == (0, {"sent": 0}))
check("لا شموع ولا سعر لحظي => 3 (يُفشل المهمة فيصلك بريد)", M.run(ENV, NOW, deps(raw=None, live=(None, False)), sp)[0] == 3)
sp3 = os.path.join(tmp, "s3.json")
n0 = len(sent); M.run(ENV, NOW, deps(raw=None, live=(98.2, True)), sp3)
new_ = sent[n0:]
check("سقوط الشموع مع بقاء السعر اللحظي: يعمل بتنبيه القرب + تنبيه بأن الفحص بلا شموع",
      len(new_) == 2 and any("يقترب" in x for x in new_) and any("لم يستطع جلب شموع" in x for x in new_), str(new_))
n1 = len(sent); M.run(ENV, NOW + timedelta(minutes=30), deps(raw=None, live=(98.2, True)), sp3)
check("تنبيه «بلا شموع» لا يتكرر خلال 12 ساعة", len(sent) == n1)
n2 = len(sent); M.run(ENV, NOW + timedelta(hours=13), deps(raw=None, live=(98.2, True)), sp3)
check("ويتجدد بعد 12 ساعة إن استمر العطل", any("لم يستطع جلب شموع" in x for x in sent[n2:]))
n3 = len(sent); M.run(ENV, NOW, deps(raw=quiet, live=(100.4, True)), os.path.join(tmp, "s4.json"))
check("لا تنبيه «بلا شموع» حين تتوفر الشموع", len(sent) == n3)
old = {"sent": {"old": (NOW - timedelta(days=30)).isoformat(), "new": NOW.isoformat()}}
M.save_state(old, NOW, sp3)
check("تنظيف الذاكرة القديمة (>14 يوماً)", list(M.load_state(sp3)["sent"]) == ["new"])

# ---------------- الأمان
def boom(req): raise OSError(f"fail at {req.full_url}")
ok, why = M.send_telegram("tg_SECRET", "1", "x", opener=boom)
check("الرمز لا يظهر في رسالة خطأ تيليجرام", not ok and "tg_SECRET" not in why, why)
def boom2(req): raise OSError("401 with ghp_SECRET")
pos_, err = M.load_position("ghp_SECRET", "g", opener=boom2)
check("الرمز لا يظهر في رسالة خطأ الـGist", pos_ is None and "ghp_SECRET" not in err, err)
class R:
    def __init__(s, b): s.b = b
    def read(s): return s.b
    def __enter__(s): return s
    def __exit__(s, *a): return False
good = json.dumps({"files": {"trades.json": {"content": json.dumps({"trades": [], "position": mkpos()})}}}).encode()
check("قراءة position من الـGist", M.load_position("t", "g", opener=lambda r: R(good))[0]["ticker"] == "GC=F")
check("بلا ملف trades.json => None بلا خطأ", M.load_position("t", "g", opener=lambda r: R(b'{"files":{}}')) == (None, ""))
src = open(os.path.join(os.path.dirname(os.path.abspath(M.__file__)), "monitor.py"), encoding="utf-8").read()
check("ضمان القراءة فقط: لا PATCH ولا journal_save ولا persist", all(w not in src for w in ("PATCH", "journal_save", "persist(")))

print(f"\n{sum(PASSED)}/{len(PASSED)} اختبار ناجح"); sys.exit(0 if all(PASSED) else 1)