"""test_upgrades.py — اختبارات الإضافات: Wyckoff، مخفّف المخاطر، حارس الحداثة، أدلة الجودة، أثر الأخبار."""
import os, sys, time, types
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def _stub(n, **a):
    m = types.ModuleType(n); m.__dict__.update(a); sys.modules[n] = m; return m


try:
    import streamlit  # noqa
except ImportError:
    class _S(dict):
        __getattr__ = dict.get
        def __setattr__(s, k, v): s[k] = v
    _stub("streamlit", session_state=_S(), secrets={},
          cache_data=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)))
for mod in ("yfinance", "groq", "google", "google.genai", "google.genai.types"):
    try:
        __import__(mod)
    except ImportError:
        if mod == "groq": _stub(mod, Groq=type("Groq", (), {"__init__": lambda s, *a, **k: None}))
        elif mod == "google": _stub(mod, __path__=[])
        elif mod == "google.genai":
            _stub(mod, Client=type("Client", (), {"__init__": lambda s, *a, **k: None}), __path__=[]); sys.modules["google"].genai = sys.modules[mod]
        elif mod == "google.genai.types":
            _stub(mod, HttpOptions=object, GenerateContentConfig=object, Tool=object, GoogleSearch=object); sys.modules["google.genai"].types = sys.modules[mod]
        else: _stub(mod)

import journal  # noqa: E402
import risk_engine as RE  # noqa: E402
import strategies  # noqa: E402
import gold_engine as GE  # noqa: E402
from constants import (ADAPTIVE_EVIDENCE_HAIRCUT, AUTO, STRATEGY_FAMILIES, TF_D1, TF_H4, TREND, WYCKOFF)  # noqa: E402
from data_engine import calculate_indicators  # noqa: E402

PASSED = []
def check(name, cond, extra=""):
    PASSED.append(bool(cond)); print(("✅" if cond else "❌"), name, "" if cond else extra)

# ============================================================ Wyckoff
def range_df(n=120):
    idx = pd.date_range("2024-01-01", periods=n)
    d = pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 100.0}, index=idx)
    d["ATR14"] = 2.0; d["ADX14"] = 15.0
    return d
def with_bar(d, i, high, low, close, adx_prev=None):
    d = d.copy(); d.loc[d.index[i], ["High", "Low", "Close"]] = [high, low, close]
    if adx_prev is not None: d.loc[d.index[i - 1], "ADX14"] = adx_prev
    return d

base = range_df()
check("نطاق هادئ بلا كسر => لا إشارات", (GE.wyckoff_signals(base) == 0).all())
sp = GE.wyckoff_signals(with_bar(base, 80, 100.0, 97.2, 99.5))
check("Spring: اختراق قاع النطاق ثم إغلاق داخله في النصف الأعلى => شراء في تلك الشمعة فقط",
      sp.iloc[80] == 1 and (sp.drop(sp.index[80]) == 0).all())
check("Spring: اختراق أعمق من 1×ATR = كسر حقيقي لا فخ => لا إشارة", GE.wyckoff_signals(with_bar(base, 80, 100.0, 95.0, 99.5)).iloc[80] == 0)
check("Spring: سوق في اتجاه (ADX 30) => لا نطاق => لا إشارة", GE.wyckoff_signals(with_bar(base, 80, 100.0, 97.2, 99.5, adx_prev=30)).iloc[80] == 0)
check("Spring: إغلاق في النصف الأدنى من الشمعة => لا إشارة", GE.wyckoff_signals(with_bar(base, 80, 100.0, 97.2, 98.2)).iloc[80] == 0)
check("Spring: إغلاق تحت القاع (استمرار الهبوط) => لا إشارة", GE.wyckoff_signals(with_bar(base, 80, 100.0, 97.2, 97.5)).iloc[80] == 0)
up = GE.wyckoff_signals(with_bar(base, 80, 102.8, 100.0, 100.5))
check("Upthrust: اختراق قمة النطاق ثم إغلاق داخله في النصف الأدنى => بيع", up.iloc[80] == -1 and (up.drop(up.index[80]) == 0).all())
wide = base.copy(); wide.loc[wide.index[50], "High"] = 140.0; wide.loc[wide.index[51], "Low"] = 60.0
check("نطاق أوسع من 10×ATR ليس نطاقاً ضيقاً => لا إشارات", (GE.wyckoff_signals(with_bar(wide, 80, 100.0, 57.0, 99.5)) == 0).all())
rng_ = np.random.default_rng(3); noisy = with_bar(base, 80, 100.0, 97.2, 99.5)
full = GE.wyckoff_signals(noisy)
fut = noisy.copy(); fut.iloc[81:, fut.columns.get_loc("Close")] = rng_.normal(100, 30, len(fut) - 81)
fut.iloc[81:, fut.columns.get_loc("Low")] = 1.0; fut.iloc[81:, fut.columns.get_loc("High")] = 500.0
check("سببية: تغيير المستقبل لا يغيّر إشارات الماضي", (GE.wyckoff_signals(fut).iloc[:81] == full.iloc[:81]).all()
      and (GE.wyckoff_signals(noisy.iloc[:81]).iloc[:81] == full.iloc[:81]).all())

r = np.random.default_rng(7); n = 900
c = 2000 + np.cumsum(r.normal(0, 6, n)); o = np.r_[c[0], c[:-1]]
raw = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + abs(r.normal(0, 4, n)), "Low": np.minimum(o, c) - abs(r.normal(0, 4, n)), "Close": c},
                   index=pd.date_range("2021-01-01", periods=n))
dfx = calculate_indicators(raw)
sg = strategies.strategy_signals(dfx)
check("Wyckoff مسجّلة: في الإشارات والقائمة الكاملة و generate_signal",
      WYCKOFF in sg and WYCKOFF in strategies.ALL_STRATEGIES and set(np.unique(sg[WYCKOFF])) <= {-1, 0, 1}
      and strategies.generate_signal(dfx, WYCKOFF)[0] in ("BUY", "SELL", "No Signal"))
check("Wyckoff مرشّحة فقط: ليست في AUTO ولا عائلات Confluence", WYCKOFF not in STRATEGY_FAMILIES
      and WYCKOFF not in strategies.auto_signals_ex.__code__.co_consts)
fakes = {k: v.copy() for k, v in sg.items()}; fakes2 = {k: v.copy() for k, v in sg.items()}
fakes2[WYCKOFF] = pd.Series(1, index=dfx.index)
check("AUTO لا يتأثر بإشارات Wyckoff إطلاقاً", strategies.auto_signals_ex(dfx, fakes)[0].equals(strategies.auto_signals_ex(dfx, fakes2)[0]))
sigs_all, mr_all = strategies.all_strategy_signals(dfx, None)
tr_w = GE.run_backtest_v2(dfx, sigs_all[WYCKOFF], 0.05, 30, mean_rev=mr_all[WYCKOFF])
check("Wyckoff تعمل في الباكتست بقناع mean_rev=True (استراتيجية نطاق)", mr_all[WYCKOFF] is True and isinstance(tr_w, pd.DataFrame)
      and (tr_w.empty or tr_w["mean_rev"].all()))
from constants import CONTRARIAN  # noqa: E402
check("مصدر واحد للقاعدة: قناع الباكتست = (الاسم ∈ CONTRARIAN) لكل استراتيجية ثابتة",
      all(mr_all[k] == (k in CONTRARIAN) for k in strategies.ALL_STRATEGIES if k != AUTO))
_app = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py"), encoding="utf-8").read()
check("الواجهة لا تحوي قوائم ارتداد مكتوبة يدوياً (كانت تنجرف عن CONTRARIAN)", "(MEAN_REV, RSI2)" not in _app and _app.count("in CONTRARIAN") >= 4)
bias_dn = pd.Series("DOWN", index=dfx.index)
sg_b, _ = strategies.all_strategy_signals(dfx, bias_dn)
check("Wyckoff معفاة من فلتر اتجاه اليومي بينما الاتجاهية تُفلتر", (sg_b[WYCKOFF] == sg[WYCKOFF]).all() and (sg_b[TREND] != 1).all())
_q, _parts = RE.compute_quality("BUY", dfx, 30, WYCKOFF, "DOWN", None, None, None, {"trades": 0})
check("نقاط Wyckoff بفرع الانعكاس (لا EMA الاتجاهي) واتجاه اليومي معفى",
      any(p_[0].startswith("توافق الانعكاس") for p_ in _parts) and any(p_[0] == "اتجاه اليومي" and "معفى" in p_[3] for p_ in _parts))
_rng = range_df(); _rng["Low5"] = _rng["Low"].rolling(5).min(); _rng["High5"] = _rng["High"].rolling(5).max(); _rng["SMA20"] = _rng["Close"].rolling(20).mean()
_rng = with_bar(_rng, 80, 100.0, 97.2, 99.5)
_t = GE.run_backtest_v2(_rng, GE.wyckoff_signals(_rng), 0.05, 30, start=60, mean_rev=True)
check("باكتست Spring كامل بقناع النطاق يعمل ويسجّل mean_rev", len(_t) == 1 and bool(_t.iloc[0]["mean_rev"]) and _t.iloc[0]["direction"] == "BUY", str(_t))

# ============================================================ أدلة الجودة
def oos_pts(ev):
    q, parts = RE.compute_quality("BUY", dfx, 30, TREND, "UP", None, None, None, ev)
    p = [x for x in parts if x[0].startswith("الأداء التاريخي")][0]
    return p[1], p[3], sum(x[2] for x in parts)
adaptive = {"trades": 40, "avg_r": 0.3, "pf": 1.5, "walk_forward": True, "folds": 5, "picks": ["a", "b"]}
pA, nA, tot = oos_pts({**adaptive, "used_oos": {"trades": 5, "avg_r": 0.9, "pf": 3.0}})
check("استراتيجية حالية بصفقات قليلة => خصم تحفظي ×0.6 على الدليل المتكيّف", np.isclose(pA, 25 * ADAPTIVE_EVIDENCE_HAIRCUT) and "خصم" in nA, f"{pA} {nA}")
pB, nB, _ = oos_pts({**adaptive, "used_oos": {"trades": 20, "avg_r": 0.0, "pf": 1.0}})
check("استراتيجية حالية بصفقات كافية وأداء صفري => صفر نقاط رغم جمال المتكيّف", pB == 0 and "الاستراتيجية الحالية نفسها" in nB, f"{pB} {nB}")
pC, _, _ = oos_pts({**adaptive, "used_oos": {"trades": 30, "avg_r": 0.3, "pf": 1.5}})
check("استراتيجية حالية جيدة وكافية => النقاط الكاملة بلا خصم", np.isclose(pC, 25.0))
pD, nD, _ = oos_pts({"trades": 40, "avg_r": 0.3, "pf": 1.5, "out_of_sample": True})
check("تقسيم واحد لنفس الاستراتيجية (غير متكيّف) => بلا خصم", np.isclose(pD, 25.0) and "خصم" not in nD)
check("مجموع الأوزان يبقى 100", tot == 100)

# ============================================================ مخفّف المخاطر
NOW = datetime(2026, 10, 5, 12, 0)
def T(pnls, hours_ago=1.0):
    return [{"account_pnl_percent": p, "exit_time": (NOW - timedelta(hours=hours_ago + (len(pnls) - i))).strftime("%Y-%m-%d %H:%M")} for i, p in enumerate(pnls)]
check("بلا صفقات => عادي", RE.risk_throttle([], NOW)["level"] == "ok" and RE.risk_throttle(None, NOW)["mult"] == 1.0)
check("ربح وخسارة صغيرة => عادي", RE.risk_throttle(T([1, -0.5, 0.8]), NOW)["level"] == "ok")
t1 = RE.risk_throttle(T([-3, -3, -3]), NOW)
check("تراجع ~8.7% => نصف الحجم", t1["level"] == "half" and 8 < t1["dd"] < 9.5 and t1["mult"] == 0.5, str(t1))
t2 = RE.risk_throttle(T([-4, -4, -4]), NOW)
check("تراجع ~11.5% وآخر صفقة حديثة => إيقاف مؤقت", t2["level"] == "stop" and t2["mult"] == 0.0, str(t2))
t3 = RE.risk_throttle(T([-4, -4, -4], hours_ago=60), NOW)
check("بعد 48 ساعة تهدئة يعود الإيقاف إلى نصف حجم (لا حبس للأبد)", t3["level"] == "half", str(t3))
check("4 خسائر متتالية صغيرة => نصف حجم", RE.risk_throttle(T([-0.5] * 4), NOW)["level"] == "half")
check("6 خسائر متتالية => إيقاف مؤقت", RE.risk_throttle(T([-0.3] * 6), NOW)["level"] == "stop")
check("ربح يقطع سلسلة الخسائر", RE.risk_throttle(T([-1, -1, -1, -1, 1]), NOW)["streak"] == 0)
check("العودة لقمة جديدة تُنهي التراجع", RE.risk_throttle(T([-5, 6]), NOW)["dd"] < 1e-9)
bad = T([-1, -1]) + [{"account_pnl_percent": None, "pnl_percent": "x", "exit_time": "??"}, {"pnl_percent": -0.5, "exit_time": "bad"}]
check("بيانات تالفة لا تُسقط الدالة", RE.risk_throttle(bad, NOW)["streak"] == 3)
arb1 = {"action": "ادخل", "level": "success", "mult": 1.0, "notes": []}
check("apply_throttle: عادي لا يغيّر شيئاً", RE.apply_throttle(arb1, RE.risk_throttle([], NOW)) is arb1)
ah = RE.apply_throttle(arb1, t1)
check("apply_throttle: نصف حجم يضاعف 0.5 ويضيف ملاحظة", ah["mult"] == 0.5 and "مخفّف" in ah["notes"][-1])
check("apply_throttle: إيقاف => لا دخول", RE.apply_throttle(arb1, t2)["mult"] == 0.0)
blocked = {"action": "لا تدخل", "level": "error", "mult": 0.0, "notes": []}
check("apply_throttle: لا يفتح ما أغلقه التحكيم", RE.apply_throttle(blocked, t3) is blocked)

# أثره الفعلي على حجم الصفقة
dfp = dfx.copy()
dec = RE.final_decision("BUY", arb1, 90, 70); dec_h = RE.final_decision("BUY", ah, 90, 70)
lv = GE.smart_levels("BUY", float(dfp["Close"].iloc[-1]), dfp)
pl_f = RE.build_plan("BUY", float(dfp["Close"].iloc[-1]), float(dfp["ATR14"].iloc[-1]), dec, 1_000_000, 1.0, 1.0, dfp, False, 0.0001, 0.0001, levels=lv)
pl_h = RE.build_plan("BUY", float(dfp["Close"].iloc[-1]), float(dfp["ATR14"].iloc[-1]), dec_h, 1_000_000, 1.0, 1.0, dfp, False, 0.0001, 0.0001, levels=lv)
check("المخفّف يخفض حجم الصفقة الفعلي إلى النصف تقريباً", pl_f.get("size") and abs(pl_h["size"] / pl_f["size"] - 0.5) < 0.02, f"{pl_f.get('size')} {pl_h.get('size')}")

# ============================================================ حداثة البيانات
def d1(last): return pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([last]))
def h4(open_utc): return pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp(open_utc, tz="UTC")]))
wed = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
check("يومي: آخر شمعة أمس => سليم", RE.data_freshness(d1("2026-10-06"), TF_D1, wed)["status"] == "ok")
check("يومي: متأخرة يوم عمل => تحذير", RE.data_freshness(d1("2026-10-05"), TF_D1, wed)["status"] == "warn")
check("يومي: متأخرة يومين عمل فأكثر => قديمة", RE.data_freshness(d1("2026-10-02"), TF_D1, wed)["status"] == "stale")
mon = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc); sat = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)
check("الاثنين: آخر شمعة الجمعة سليمة (عطلة نهاية الأسبوع)", RE.data_freshness(d1("2026-10-02"), TF_D1, mon)["status"] == "ok")
check("السبت: آخر شمعة الجمعة سليمة (كان سيُحسب الخميس خطأً)", RE.data_freshness(d1("2026-10-09"), TF_D1, sat)["status"] == "ok")
check("4H: شمعة حديثة => سليم", RE.data_freshness(h4("2026-10-07 04:00"), TF_H4, wed)["status"] == "ok")
check("4H: قبل 10 ساعات => تحذير", RE.data_freshness(h4("2026-10-06 20:00"), TF_H4, wed)["status"] == "warn")
check("4H: قبل 22 ساعة والسوق مفتوح => قديمة", RE.data_freshness(h4("2026-10-06 08:00"), TF_H4, wed)["status"] == "stale")
check("4H: عطلة نهاية الأسبوع لا تُعدّ تأخراً", RE.data_freshness(h4("2026-10-09 16:00"), TF_H4, sat)["status"] == "ok")
check("4H: الأحد 21:00 مغلق والأحد 23:00 مفتوح",
      RE.data_freshness(h4("2026-10-02 16:00"), TF_H4, datetime(2026, 10, 4, 21, 0, tzinfo=timezone.utc))["status"] == "ok"
      and RE.data_freshness(h4("2026-10-02 16:00"), TF_H4, datetime(2026, 10, 4, 23, 0, tzinfo=timezone.utc))["status"] == "stale")
g_stale = RE.apply_data_guard(arb1, {"status": "stale", "msg": "قديمة"})
check("الحارس: بيانات قديمة تمنع الدخول", g_stale["mult"] == 0.0 and "قديمة" in g_stale["notes"][-1])
g_warn = RE.apply_data_guard(arb1, {"status": "warn", "msg": "تأخر"})
check("الحارس: التحذير يضيف ملاحظة دون منع", g_warn["mult"] == 1.0 and "تأخر" in g_warn["notes"][-1])
check("الحارس: لا يغيّر شيئاً حين سليم أو حين مغلق أصلاً", RE.apply_data_guard(arb1, {"status": "ok", "msg": ""}) is arb1
      and RE.apply_data_guard(blocked, {"status": "stale", "msg": "x"}) is blocked)

# ============================================================ حماية وقت الدخول التالف
_pg = {"direction": "BUY", "entry_price": 100.0, "sl": 98.0, "original_sl": 98.0, "tp": 106.0, "tp1": 103.0, "entry_time": "garbage", "ticker": "GC=F"}
_bg = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5]}, index=pd.DatetimeIndex(["2026-10-05"], tz="UTC"))
try:
    _ev = journal.check_exit_candles(_pg, _bg)
    check("check_exit_candles بوقت دخول تالف لا ينهار (يرجع قائمة فارغة)", _ev == [])
except Exception as e:
    check("check_exit_candles بوقت دخول تالف لا ينهار", False, repr(e))

# ============================================================ أثر الأخبار
a_full = {"agents": {"news": {"impact_on_signal": "supports", "event_risk": "low"}}, "used": TREND, "tf": "x", "score": 80, "adx": 25.0,
          "regime": "r", "signal": "BUY", "arb": {"action": "ادخل"}}
m1 = journal.meta_from(a_full, "plan")
check("meta_from يسجّل حكم الأخبار", m1["news_impact"] == "supports" and m1["news_event_risk"] == "low" and m1["news_mode"] == "full")
m2 = journal.meta_from({**a_full, "agents": {"news": {"impact_on_signal": "neutral", "event_risk": "low", "degraded": True}}}, "plan")
m3 = journal.meta_from({**a_full, "agents": {"news": None}}, "plan")
check("وضع التقويم فقط والغياب يُسجَّلان", m2["news_mode"] == "degraded" and m3["news_mode"] == "none" and m3["news_impact"] is None)
check("meta_from يعمل بلا تحليل (يدوي)", journal.meta_from(None, "manual") == {"source": "manual"})
rr = np.random.default_rng(11)
def mk(imp, mu, n_, sd=1.0): return [{"news_impact": imp, "r_multiple": float(x), "pnl_percent": float(x), "direction": "BUY", "score": 80, "used": TREND, "tf": "x", "source": "plan", "regime": "r"} for x in rr.normal(mu, sd, n_)]
tdf = pd.DataFrame(mk("supports", .5, 12) + mk("neutral", 0, 12))
grp = dict(journal.journal_groups(tdf))
check("journal_groups يضيف جدول «أثر الأخبار»", "أثر الأخبار" in grp and set(grp["أثر الأخبار"]["أثر الأخبار"]) == {"تدعم الإشارة", "محايدة"})
few = journal.news_value_report(mk("supports", .5, 4) + mk("neutral", 0, 30))
check("عيّنة غير كافية => لا استنتاج", np.isnan(few["p"]) and "غير كافية" in few["verdict"])
good = journal.news_value_report(mk("supports", 1.0, 40, .6) + mk("neutral", 0.0, 40, .6) + mk("conflicts", -.2, 5, .6))
check("أخبار تفيد فعلاً (فرق كبير) => دلالة إحصائية", good["p"] < 0.05 and good["diff"] > 0 and "تُحسّن" in good["verdict"], str(good["verdict"]))
null = journal.news_value_report(mk("supports", 0.0, 40) + mk("neutral", 0.0, 40))
check("لا فرق حقيقي => لا ادّعاء فائدة", null["p"] >= 0.05 and "لا فرق" in null["verdict"], str(null["p"]))
bad_ = journal.news_value_report(mk("supports", -1.0, 40, .6) + mk("neutral", 0.0, 40, .6))
check("أخبار تضرّ => يُنبَّه", bad_["p"] < 0.05 and bad_["diff"] < 0 and "أسوأ" in bad_["verdict"])
_det = [{"news_impact": "supports", "r_multiple": float(i % 5)} for i in range(15)] + [{"news_impact": "neutral", "r_multiple": float(i % 3)} for i in range(15)]
_a, _b = journal.news_value_report(_det), journal.news_value_report(_det)
check("تقرير حتمي (نفس seed => نفس p والحكم)", _a["p"] == _b["p"] and _a["verdict"] == _b["verdict"] and np.isfinite(_a["p"]))
check("صفقات بلا R أو بلا حكم أخبار تُتجاهل", journal.news_value_report([{"news_impact": "supports", "r_multiple": None}, {"r_multiple": 1.0}])["n"] == 0)

print(f"\n{sum(PASSED)}/{len(PASSED)} اختبار ناجح"); sys.exit(0 if all(PASSED) else 1)