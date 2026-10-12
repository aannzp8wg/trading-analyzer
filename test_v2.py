"""test_v2.py — اختبارات بلا إنترنت لنسخة التطبيق الحالية.  python test_v2.py"""
import json, os, sys, time, types
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m


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
            _stub(mod, Client=type("Client", (), {"__init__": lambda s, *a, **k: None}), __path__=[])
            sys.modules["google"].genai = sys.modules[mod]
        elif mod == "google.genai.types":
            _stub(mod, HttpOptions=object, GenerateContentConfig=object, Tool=object, GoogleSearch=object)
            sys.modules["google.genai"].types = sys.modules[mod]
        else: _stub(mod)

import streamlit as st  # noqa: E402
import news_engine as nw  # noqa: E402
import journal  # noqa: E402
import risk_engine as RE  # noqa: E402
import strategies  # noqa: E402
import walkforward as wf  # noqa: E402
from agents import validate_news_json, validate_tech_json  # noqa: E402
from constants import (AUTO, BREAKOUT, CONTRARIAN, MEAN_REV, PULLBACK, RSI2, SQUEEZE, TREND, TURTLE,  # noqa: E402
                       STRATEGY_FAMILIES)
from data_engine import align_bias, calculate_indicators  # noqa: E402
from gold_engine import CONFLUENCE, new_trade_state, run_backtest_v2, step_trade, trade_pnl  # noqa: E402

PASSED = []


def check(name, cond, extra=""):
    PASSED.append(bool(cond)); print(("✅" if cond else "❌"), name, "" if cond else extra)


def make_df(n=700, seed=0, drift=0.0, vol=5.0):
    r = np.random.default_rng(seed)
    c = 2000 + np.cumsum(r.normal(drift, vol, n)); o = np.r_[c[0], c[:-1]] + r.normal(0, vol * .3, n)
    d = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + abs(r.normal(0, vol, n)),
                      "Low": np.minimum(o, c) - abs(r.normal(0, vol, n)), "Close": c},
                     index=pd.date_range("2022-01-01", periods=n, freq="D"))
    d["ATR14"] = d["Close"].diff().abs().ewm(alpha=1 / 14).mean().bfill() + vol
    d["Low5"] = d.Low.rolling(5).min(); d["High5"] = d.High.rolling(5).max(); d["SMA20"] = d.Close.rolling(20).mean()
    return d


NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)

# ================================================================ 1) البيانات
d = pd.Series(["UP", "DOWN", "UP"], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
h4 = pd.DatetimeIndex(pd.to_datetime(["2026-01-06 00:00", "2026-01-06 04:00", "2026-01-06 20:00", "2026-01-07 00:00"]), tz="UTC")
check("align_bias: لا تسريب مستقبل", list(align_bias(pd.DataFrame(index=h4), d).values) == ["UP", "UP", "UP", "DOWN"])

# ================================================================ 2) الباكتست
df = make_df(400, 1); sig = np.zeros(400, int); sig[210::15] = 1
tr = run_backtest_v2(df, sig, cost_pct=0.5)
check("R الصافي < R الخام مع التكلفة", (tr["r"] < tr["r_gross"]).all())
check("R الصافي = pnl الصافي / المخاطرة", np.allclose(tr["r"], tr["pnl"] / (tr["gross_pnl"] / tr["r_gross"])))
m = 260; px = np.full(m, 100.0)
d2 = pd.DataFrame({"Open": px, "High": px + .2, "Low": px - .2, "Close": px}); d2["ATR14"] = 1.0
d2["Low5"] = 99.8; d2["High5"] = 100.2; d2["SMA20"] = 100.0
s2 = np.zeros(m, int); s2[210] = 1; d2.loc[d2.index[212], ["High", "Low"]] = [102.0, 99.9]
t2 = run_backtest_v2(d2, s2, cost_pct=0.0, start=205)
check("BE في نفس شمعة TP1 (تحفّظ)", len(t2) == 1 and t2.iloc[0]["reason"] == "BE")

dfx = calculate_indicators(make_df(500, 3, 0.5)[["Open", "High", "Low", "Close"]])
sigs_all, mr_all = strategies.all_strategy_signals(dfx, pd.Series("DOWN", index=dfx.index))
check("all_strategy_signals: كل الاستراتيجيات + قناع AUTO بطول الإطار",
      set(sigs_all) == set(strategies.ALL_STRATEGIES) and len(mr_all[AUTO]) == len(dfx)
      and mr_all[RSI2] is True and mr_all[BREAKOUT] is False)
check("فلتر اليومي (DOWN) يلغي كل شراء اتجاهي",
      all((sigs_all[k] != 1).all() for k in (TREND, BREAKOUT, TURTLE, SQUEEZE, PULLBACK, CONFLUENCE)))
idx_ = pd.date_range("2024-01-01", periods=6); dfa = pd.DataFrame({"ADX14": [15, 30, 22, 15, 30, 22]}, index=idx_)
z = pd.Series(0, index=idx_)
sg = {k: z.copy() for k in (BREAKOUT, PULLBACK, MEAN_REV, TREND, SQUEEZE, TURTLE, RSI2, CONFLUENCE)}
sg[RSI2] = pd.Series([1, 0, -1, 0, 0, 0], index=idx_); sg[TREND] = pd.Series([0, 1, 0, 0, 1, 0], index=idx_)
a_sig, a_mr = strategies.auto_signals_ex(dfa, sg)
check("AUTO: صفقات الارتداد تحمل mean_rev=True والاتجاهية False",
      list(a_sig) == [1, 1, -1, 0, 1, 0] and list(a_mr) == [True, False, True, False, False, False])

# Confluence (عائلاتك): إعادة حساب مستقلة من المكوّنات مقابل المخرج
sg2 = strategies.strategy_signals(dfx)
fam_of = STRATEGY_FAMILIES
def vote(names):
    M = pd.concat([sg2[n] for n in names], axis=1); u = (M == 1).sum(axis=1); dn = (M == -1).sum(axis=1)
    return pd.Series(np.select([u > dn, dn > u], [1, -1], 0), index=dfx.index)
fams = {}
for n_ in (TREND, BREAKOUT, TURTLE, SQUEEZE, PULLBACK, MEAN_REV, RSI2):
    fams.setdefault(fam_of[n_], []).append(n_)
F = pd.concat([vote(v) for v in fams.values()], axis=1)
expect = pd.Series(np.select([((F == 1).sum(axis=1) >= 2) & ((F == -1).sum(axis=1) == 0),
                              ((F == -1).sum(axis=1) >= 2) & ((F == 1).sum(axis=1) == 0)], [1, -1], 0), index=dfx.index)
check("Confluence = اتفاق عائلتين فأكثر دون معارضة (إعادة حساب مستقلة)", (sg2[CONFLUENCE] == expect).all())

# ================================================================ 3) NaN
dfi = make_df(400, 1); dfi.loc[dfi.index[-1], "ATR14"] = np.nan
arb_ok = {"action": "x", "level": "success", "mult": 1.0, "notes": []}
plan = RE.build_plan("BUY", 2000.0, np.nan, arb_ok, 10000, 1.0, 100.0, df=dfi)
check("build_plan بمستويات NaN لا يعطي اتجاهًا", plan["direction"] is None and plan["level"] == "warning")
d3 = pd.DataFrame({"Close": [100.], "EMA50": [99.], "EMA200": [98.]})
try:
    q, _ = RE.compute_quality("BUY", d3, 30, TREND, "UP", None, None, None, {"trades": 10, "avg_r": float("nan"), "pf": 1.2})
    check("compute_quality لا ينهار مع avg_r=NaN", isinstance(q, int))
except Exception as e:
    check("compute_quality لا ينهار مع avg_r=NaN", False, repr(e))
check("رد فني متناقض => 'لا يؤيد' لا None",
      (lambda t: t is not None and t["supports_signal"] is False)(validate_tech_json({"verdict": "SELL", "confidence": 70, "supports_signal": True}, "BUY")))
check("evidence يُقصّ ويُزال تكراره",
      validate_news_json({"sentiment": "bullish", "event_risk": "low", "impact_on_signal": "supports",
                          "evidence": [1, 99, "2", "x", 1]}, n_items=3)["evidence"] == [1, 2])

# ================================================================ 4) الأخبار
RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Gold slips as dollar firms ahead of Fed - Reuters</title><link>http://a</link><pubDate>Mon, 05 Oct 2026 09:00:00 GMT</pubDate><source url="x">Reuters</source></item>
<item><title>Gold slips as dollar firms ahead of Fed - CNBC</title><link>http://b</link><pubDate>Mon, 05 Oct 2026 09:30:00 GMT</pubDate><source url="x">CNBC</source></item>
<item><title>Treasury yields climb on strong jobs data - Bloomberg</title><link>http://c</link><pubDate>Mon, 05 Oct 2026 06:00:00 GMT</pubDate><source url="x">Bloomberg</source></item>
<item><title>BUY GOLD NOW 1000% guaranteed - CryptoMoonBlog</title><link>http://d</link><pubDate>Mon, 05 Oct 2026 10:00:00 GMT</pubDate><source url="x">CryptoMoonBlog</source></item>
<item><title>Old story about gold - Reuters</title><link>http://e</link><pubDate>Mon, 28 Sep 2026 10:00:00 GMT</pubDate><source url="x">Reuters</source></item>
<item><title>Undated story - Reuters</title><link>http://f</link><source url="x">Reuters</source></item></channel></rss>"""
RDF = b"""<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<item><title>ECB keeps rates unchanged</title><link>http://ecb/1</link><dc:date>2026-10-05T08:00:00+00:00</dc:date></item></rdf:RDF>"""
def getter(url):
    if "news.google" in url: return RSS
    if "ecb.europa" in url: return RDF
    raise OSError("blocked")
CAL_ROWS = [{"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-10-05T17:00:00+00:00", "impact": "High", "forecast": "150K", "previous": "140K"},
            {"title": "German CPI", "country": "EUR", "date": "2026-10-05T13:00:00+00:00", "impact": "High"}]
def cal_getter(url):
    if "thisweek" in url: return json.dumps(CAL_ROWS).encode()
    raise OSError("no nextweek")
b = nw.build_bundle("GC=F", [("gold", 3, 10)], now=NOW, getter=getter, cal_getter=cal_getter)
titles = [i["title"] for i in b["items"]]
check("أخبار: يستبعد غير الموثوق والقديم وغير المؤرَّخ", b["dropped_untrusted"] == 1 and b["dropped_stale"] == 2)
check("أخبار: يزيل التكرار ويُبقي Tier1 ويقرأ RSS 1.0", sum("Gold slips" in x for x in titles) == 1 and any(i["tier"] == 1 and "ECB" in i["title"] for i in b["items"]))
check("تقويم: NFP بعد 5 ساعات => high وحدث EUR لا يُحتسب للذهب", b["calendar"]["risk"] == "high" and all(e["currency"] == "USD" for e in b["calendar"]["events"]))
check("تقويم معطّل => unknown لا low", nw.fetch_calendar({"USD"}, NOW, lambda u: (_ for _ in ()).throw(OSError("x")))["risk"] == "unknown")
BQ = {"items": [{"idx": 1, "tier": 1}, {"idx": 2, "tier": 2}], "trusted_count": 2, "tier1_count": 1, "dropped_stale": 0, "dropped_untrusted": 0,
      "calendar": {"ok": True, "risk": "low", "events": []}}
base = {"sentiment": "bullish", "event_risk": "low", "impact_on_signal": "supports", "evidence": [1]}
check("مطابقة: بلا أدلة => محايد، ومصدر واحد لا يكفي لـ'يدعم'",
      nw.reconcile_news({**base, "evidence": []}, BQ)["impact_on_signal"] == "neutral"
      and nw.reconcile_news(base, {**BQ, "trusted_count": 1})["impact_on_signal"] == "neutral")
check("مطابقة: النموذج لا يخفض خطر التقويم", nw.reconcile_news({**base, "event_risk": "low"}, {**BQ, "calendar": {"ok": True, "risk": "high", "events": []}})["event_risk"] == "high")
tech = {"supports_signal": True, "verdict": "BUY", "confidence": 70}; risk = {"decision": "APPROVE"}
news_hi = nw.reconcile_news(base, {**BQ, "calendar": {"ok": True, "risk": "high", "events": []}})
check("تحكيم: حدث عالي التأثير قريب => لا دخول", RE.arbitrate("BUY", tech, news_hi, risk)["mult"] == 0.0)
check("تحكيم: سليم => حجم كامل، وضع التقويم فقط => نصف", RE.arbitrate("BUY", tech, nw.reconcile_news(base, BQ), risk)["mult"] == 1.0
      and RE.arbitrate("BUY", tech, nw.calendar_only_news(b) | {"event_risk": "low"}, risk)["mult"] == 0.5)

# ================================================================ 5) المحاكي الورقي
def fresh():
    st.session_state["trades"], st.session_state["current_position"] = [], None
    st.session_state["journal_ok"] = False; st.session_state["notice"] = None

def barsdf(rows, start="2020-01-01 00:05", freq="5min", tz=None):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=tz)
    df_ = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx); df_.attrs["bar_td"] = pd.Timedelta(freq); return df_

# 5-أ) وقف NaN
fresh(); journal.open_position("BUY", 2000.0, "GC=F", 10.0, {}, float("nan"), float("nan"), account_risk_pct=1.0)
p = st.session_state["current_position"]
check("وقف NaN يُستبدل بمستوى ATR صالح", np.isfinite(p["sl"]) and p["sl"] < 2000 < p["tp"] and p["levels_fallback"])
st.session_state["current_position"]["entry_ts"] = "2020-01-01T00:00:00+00:00"
crash = barsdf([[2000, 2001, 1500, 1600]], "2020-01-01 00:05")
journal.check_exit(1600.0, bars_fn=lambda tk, et: crash)
check("انهيار -25% يضرب الوقف الآن", st.session_state["current_position"] is None and st.session_state["trades"][0]["exit_reason"] == "SL")
fresh(); journal.open_position("SELL", 2000.0, "GC=F", 10.0, {}, 1990.0, 2010.0, account_risk_pct=1.0)
p = st.session_state["current_position"]
check("مستويات في الجهة الخطأ تُستبدل", p["sl"] > 2000 > p["tp"] and p["tp1"] < 2000)

# 5-ب) لا BE خاطئ بعد TP1 (إعادة فحص الشموع)
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0)
st.session_state["current_position"]["entry_ts"] = "2020-01-01T00:00:00+00:00"
ball = barsdf([[100.2, 101.0, 99.6, 100.8], [100.8, 101.5, 100.5, 101.2], [102.5, 103.5, 102.0, 103.2], [103.8, 104.5, 103.5, 104.2]])
r1 = journal.check_exit(None, bars_fn=lambda tk, et: ball.iloc[:3])
p = st.session_state["current_position"]
check("TP1 يُسجَّل: stage=1، وقف=الدخول، محقق=1.5%", p["stage"] == 1 and p["sl"] == 100.0 and np.isclose(p["realized_partial"], 1.5) and "tp1" in r1["events"])
r2 = journal.check_exit(104.0, bars_fn=lambda tk, et: ball)
check("التحديث الثاني لا يُغلق الصفقة خطأً (كان يعيد مسح الشموع من الدخول بوقف BE)",
      st.session_state["current_position"] is not None and not r2["closed"], str(st.session_state["trades"]))
ball2 = pd.concat([ball, barsdf([[104.2, 106.5, 104.0, 106.0]], "2020-01-01 00:25")])
journal.check_exit(106.0, bars_fn=lambda tk, et: ball2)
tt = st.session_state["trades"]
check("ثم الخروج عند TP2 بنتيجة صحيحة (1.5 + 0.5×6 = 4.5%)", len(tt) == 1 and tt[0]["exit_reason"] == "TP" and np.isclose(tt[0]["pnl_percent"], 4.5)
      and np.isclose(tt[0]["r_multiple"], 4.5 / 2.0) and np.isclose(tt[0]["account_pnl_percent"], 2.25), str(tt))

# 5-ج) لقطة السعر لا تُثبّت TP1 قبل اكتمال الشمعة
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0)
st.session_state["current_position"]["entry_ts"] = "2020-01-01T00:00:00+00:00"
journal.check_exit(103.5, bars_fn=lambda tk, et: None)
check("لقطة السعر وحدها لا تغيّر المرحلة (تُثبَّت من الشموع المكتملة)", st.session_state["current_position"]["stage"] == 0)

# 5-د) فرق التوقيت: entry_ts بـUTC مقابل شموع بتوقيت نيويورك
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0)
st.session_state["current_position"]["entry_ts"] = "2026-10-05T15:00:00+00:00"
ny = barsdf([[100.0, 100.2, 99.0, 99.2], [99.0, 99.2, 97.5, 98.0], [98.5, 98.7, 98.0, 98.2]], "2026-10-05 11:05", "30min", tz="America/New_York")
journal.check_exit(None, bars_fn=lambda tk, et: ny)
check("وقف لُمس 15:35 UTC يُلتقط رغم اختلاف منطقة الشموع (كان يفوت)", st.session_state["current_position"] is None
      and st.session_state["trades"][0]["exit_reason"] == "SL")
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0)
pp = st.session_state["current_position"]; pp.pop("entry_ts"); pp["entry_time"] = "2026-10-05 15:00"   # سجل قديم (توقيت السيرفر UTC)
journal.check_exit(None, bars_fn=lambda tk, et: ny)
check("سجل قديم بلا entry_ts يعمل أيضًا", st.session_state["current_position"] is None)

# 5-هـ) الإغلاق اليدوي بعد TP1
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0)
p = st.session_state["current_position"]; p["stage"] = 1; p["sl"] = 100.0; p["realized_partial"] = 1.5
journal.close_trade(104.0, "Manual"); t = st.session_state["trades"][0]
check("إغلاق يدوي بعد TP1: pnl=3.5% وR=1.75 والحساب=1.75%",
      np.isclose(t["pnl_percent"], 3.5) and np.isclose(t["r_multiple"], 1.75) and np.isclose(t["account_pnl_percent"], 1.75), str(t))
fresh(); journal.open_position("BUY", 100.0, "GC=F", 1.0, {}, 98.0, 106.0, tp1=103.0, account_risk_pct=1.0, cost_pct=0.2)
journal.close_trade(99.0, "Manual"); t = st.session_state["trades"][0]
check("التكلفة تُخصم من نتيجة المحاكي (-1% -0.2%)", np.isclose(t["pnl_percent"], -1.2) and np.isclose(t["gross_pnl_percent"], -1.0))

# 5-و) تطابق المحاكي مع الباكتست — عبر خط التحديث الكامل بدفعات عشوائية ونقاط فحص
bad = {"exit": 0, "pnl": 0}; compared = 0
for seed in range(40):
    dfp = make_df(420, seed, drift=float(np.random.default_rng(seed).choice([-.5, 0, .5])), vol=8.0)
    rs = np.random.default_rng(seed + 1000); sg_ = rs.choice([-1, 0, 0, 0, 1], size=len(dfp))
    cost = float(rs.choice([0.0, 0.05, 0.3]))
    trs = run_backtest_v2(dfp, sg_, cost_pct=cost, max_hold=40, mean_rev=bool(seed % 2))
    for _, t_ in trs.iterrows():
        if t_["reason"] == "Time":
            continue
        j0, j1 = dfp.index.get_loc(t_["entry_date"]), dfp.index.get_loc(t_["exit_date"])
        allb = dfp.iloc[j0:j1 + 1][["Open", "High", "Low", "Close"]].copy(); allb.attrs["bar_td"] = pd.Timedelta(days=1)
        fresh(); journal.open_position(t_["direction"], t_["entry"], "GC=F", 1.0, {}, t_["sl0"], t_["tp2"], tp1=t_["tp1"],
                                       account_risk_pct=None, cost_pct=cost)
        st.session_state["current_position"]["entry_ts"] = (allb.index[0] - pd.Timedelta(hours=1)).isoformat()
        k = 0
        while st.session_state["current_position"] is not None and k < len(allb):
            k = min(k + int(rs.integers(1, 4)), len(allb))
            journal.check_exit(None, bars_fn=lambda tk, et, k=k: allb.iloc[:k])
        compared += 1
        if not st.session_state["trades"]:
            bad["exit"] += 1; continue
        rec = st.session_state["trades"][0]
        if rec["exit_reason"] != t_["reason"] or not np.isclose(rec["exit_price"], t_["exit"]):
            bad["exit"] += 1
        elif not np.isclose(rec["pnl_percent"], t_["pnl"]):
            bad["pnl"] += 1
check(f"المحاكي (مع التحديثات المتفرقة ونقاط الفحص) = الباكتست في {compared} صفقة: السبب والسعر", compared > 60 and bad["exit"] == 0, str(bad))
check("... وPnL الصافي", bad["pnl"] == 0, str(bad))

# 5-ز) التنبيهات
pos_ = {"direction": "BUY", "entry_price": 100.0, "sl": 100.0, "original_sl": 98.0, "tp": 110.0, "atr": 1.0}
check("تنبيه القرب من الوقف يستخدم الوقف الحالي (BE) لا الأولي", journal.should_alert_position(pos_, 100.2)[0] is True
      and journal.should_alert_position({**pos_, "sl": 98.0}, 100.2)[0] is False)
ec = journal.compute_equity_curve([{"account_pnl_percent": 1.0}, {"account_pnl_percent": -1.0}])
check("منحنى رأس المال مركّب", np.isclose(ec["equity"].iloc[-1], 100 * 1.01 * 0.99))
mc = journal.monte_carlo([{"account_pnl_percent": x} for x in (1, -1, 2, -1, 1.5, 0.5)], n_sims=200)
check("مونت كارلو حتمي ومرتّب", mc["p05"] <= mc["p50"] <= mc["p95"] and journal.monte_carlo([{"account_pnl_percent": x} for x in (1, -1, 2, -1, 1.5, 0.5)], n_sims=200) == mc)

# ================================================================ 6) Walk-Forward
dfw = make_df(1300, 11, drift=1.2, vol=6.0)
def fake(dfh, bias):
    ix = dfh.index; out, mr = {}, {}
    for k, nm in enumerate(RE._ALL_STRATEGIES):
        s_ = pd.Series(0, index=ix); s_.iloc[(5 + k)::11] = 1 if k % 2 == 0 else -1; out[nm] = s_; mr[nm] = False
    return out, mr
RE.all_strategy_signals = fake
r1 = RE.walk_forward_evidence("GC=F", "4h", dfw, None, "x", cost=0.05, hold=30)
te_lo = 200 + ((1300 - 200 - 30) // 6) + 30
dfb = dfw.copy()
for c in ("Open", "High", "Low", "Close"):
    dfb.iloc[te_lo:te_lo + 180, dfb.columns.get_loc(c)] *= 0.6
dfb["Low5"] = dfb.Low.rolling(5).min(); dfb["High5"] = dfb.High.rolling(5).max()
r2 = RE.walk_forward_evidence("GC=F", "4h", dfb, None, "x", cost=0.05, hold=30)
f1, f2 = r1["per_fold"][0], r2["per_fold"][0]
check("WF: لا تسريب — تخريب بيانات الاختبار لا يغيّر اختيار/أداء التدريب", (f1["pick"], f1["train_r"], f1["train_n"]) == (f2["pick"], f2["train_r"], f2["train_n"]), f"{f1} {f2}")
ok_all = all(RE.walk_forward_evidence("GC=F", "4h", make_df(n, 3, 0.8, 6.0), None, "x", cost=0.05, hold=10) is not None for n in range(560, 575))
check("WF: لا يرجع None بصمت لأي طول بيانات (كان يفشل عند te_e==n)", ok_all)
check("WF: كل الطيّات الخمس تُحسب", r1["folds"] == 5, str(r1["folds"]))
def only_neg(dfh, bias):
    ix = dfh.index; s_ = pd.Series(0, index=ix); s_.iloc[::11] = -1
    return {nm: s_.copy() for nm in RE._ALL_STRATEGIES}, {nm: False for nm in RE._ALL_STRATEGIES}
RE.all_strategy_signals = only_neg
r3 = RE.walk_forward_evidence("GC=F", "4h", make_df(1300, 11, 1.2, 6.0), None, "x", cost=0.05, hold=30)
check("WF: لا تداول إجباري — طيّات بلا استراتيجية رابحة تبقى نقداً", r3["cash_folds"] == r3["folds"] and r3["trades"] == 0, str(r3.get("picks")))
q, parts = RE.compute_quality("BUY", dfx, 30, TREND, "UP", None, None, None, {**r1, "trades": 40, "avg_r": 0.2, "pf": 1.4})
check("compute_quality يقبل picks فيها None ويحسب 100 نقطة كحد أقصى", isinstance(q, int) and sum(p_[2] for p_ in parts) == 100)

# وحدة walkforward المستقلة (حكم + فاصل ثقة)
SIGS = {"UP": pd.Series(0, index=dfw.index), "DOWN": pd.Series(0, index=dfw.index)}
SIGS["UP"].iloc[::12] = 1; SIGS["DOWN"].iloc[::12] = -1
W = wf.walk_forward(dfw, SIGS, {"UP": False, "DOWN": False}, cost_pct=0.05, max_hold=20)
check("walkforward.py: يعمل ويختار الرابحة فقط ولا يتداول خارج نوافذ الاختبار",
      W["ok"] and all(f["chosen"] == "UP" for f in W["folds"]) and len(W["oos_trades"]) > 0
      and all(any(dfw.index[lo] <= d_ <= dfw.index[hi - 1] for lo, hi in [f["test_range"] for f in W["folds"]]) for d_ in W["oos_trades"]["entry_date"]))
check("walkforward.verdict", (wf.verdict_for(50, .3, .05), wf.verdict_for(50, .3, -.05), wf.verdict_for(50, -.1, -.3), wf.verdict_for(10, .5, .2)) == ("supported", "unproven", "failed", "insufficient"))

# ================================================================ 7) محرك الصفقة
tt_ = new_trade_state(1, 100.0, 98.0, 103.0, 106.0, 0.5)
check("TP1 يغلق نصفاً ويرفع الوقف للدخول", step_trade(tt_, 100, 103.5, 100.5) is None and tt_["stage"] == 1 and tt_["sl"] == 100.0)
rr = step_trade(tt_, 101, 102, 99.5); g, nn, r_, _ = trade_pnl(tt_, rr[1], 0.1)
check("BE بعد TP1: ربح الجزء المُحقَّق (1.5% − 0.1 تكلفة)", rr[0] == "BE" and np.isclose(nn, 1.4) and np.isclose(r_, 0.7))
check("فجوة تحت الوقف تُنفَّذ عند الافتتاح", step_trade(new_trade_state(1, 100.0, 98.0, 103.0, 106.0, 0.5), 95.0, 96.0, 94.0) == ("SL", 95.0))

print(f"\n{sum(PASSED)}/{len(PASSED)} اختبار ناجح"); sys.exit(0 if all(PASSED) else 1)