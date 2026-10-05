import streamlit as st

st.set_page_config(
    page_title="محلل الذهب واليورو والدولار",
    page_icon="🤖",
    layout="wide",
)

import json
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

from agents import (
    build_context, call_gemini, call_groq, extract_json, fetch_news,
    get_gemini_models, get_groq_models, news_agent, run_agents, send_telegram,
    _list_gemini, _list_groq,
)
from constants import (
    ASSETS, AUTO, BREAKOUT, CONTRARIAN, CORRELATION_NOTES, DEFAULT_COST,
    MEAN_REV, PULLBACK, TF_D1, TF_DUAL, TF_H4, TREND,
)
from data_engine import (
    apply_bias_filter, calculate_indicators, load_data, load_data_h4,
    load_prepared,
)
from gold_engine import (
    CONFLUENCE, RSI2, SQUEEZE, TURTLE, run_backtest_v2, smart_levels,
)
from journal import (
    calc_pnl, check_exit, close_trade, compute_equity_curve,
    daily_breaker_status, fetch_bars_since, get_live_price, journal_groups,
    journal_load, journal_save, meta_from, monte_carlo, open_position,
    persist, should_alert_event, should_alert_position, storage_mode,
)
from risk_engine import (
    P, arbitrate, as_list, backtest_stats, build_plan, clean_price_levels,
    compute_quality, correlation_exposure_check, final_decision,
    format_alert, guess_contract_size, history_evidence, trade_levels,
    truthy, walk_forward_evidence,
)
from strategies import auto_signals, select_strategy, strategy_signals

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"خطأ: المفتاح {e} غير موجود في Streamlit Secrets.")
    st.stop()

st.title("محلل الذهب واليورو والدولار 🥇💶💵")
st.markdown("### فريق وكلاء: فني + أخبار + مخاطر")

defaults = {
    "trades": [],
    "current_position": None,
    "analysis": None,
    "live_price": None,
    "live_price_is_live": False,
    "notice": None,
    "backtest": None,
    "position_alert_last": None,
    "event_alerts": {},
    "mc_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "journal_loaded" not in st.session_state:
    _j, _jerr = journal_load()
    st.session_state.trades = _j["trades"]
    st.session_state.current_position = _j["position"]
    st.session_state.journal_ok = not _jerr
    st.session_state.journal_load_err = _jerr
    st.session_state.journal_loaded = True

if st.session_state.get("journal_dirty") and st.session_state.get("journal_loaded"):
    _now = time.time()
    _last = st.session_state.get("journal_last_save_ts", 0.0)
    if (_now - _last) >= 3.0:
        persist(force=True)


def create_chart(df, ticker, signal=None, sl=None, tp1=None, tp2=None, tf_label=""):
    chart_df = df.tail(100)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name="السعر"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"],
                             name="EMA20", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"],
                             name="EMA50", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200"],
                             name="EMA200", line=dict(color="red", width=1.5)))

    if signal in ("BUY", "SELL"):
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1]], y=[chart_df["Close"].iloc[-1]],
            mode="markers", name=signal,
            marker=dict(
                size=14,
                symbol="triangle-up" if signal == "BUY" else "triangle-down",
                color="green" if signal == "BUY" else "red"),
        ))
        if sl is not None:
            fig.add_hline(y=sl, line_dash="dash", line_color="red",
                          annotation_text="SL")
        if tp1 is not None:
            fig.add_hline(y=tp1, line_dash="dot", line_color="green",
                          annotation_text="TP1")
        if tp2 is not None:
            fig.add_hline(y=tp2, line_dash="dash", line_color="green",
                          annotation_text="TP2")

    fig.update_layout(title=f"{ticker} - {tf_label}", template="plotly_white",
                      height=500, xaxis_rangeslider_visible=False)
    return fig


TV_MAP = {
    "GC=F": "OANDA:XAUUSD", "SI=F": "OANDA:XAGUSD", "CL=F": "TVC:USOIL",
    "BZ=F": "TVC:UKOIL", "NG=F": "CAPITALCOM:NATURALGAS",
    "BTC-USD": "BINANCE:BTCUSDT", "ETH-USD": "BINANCE:ETHUSDT",
    "^GSPC": "SP:SPX", "^IXIC": "NASDAQ:IXIC", "^DJI": "DJ:DJI",
    "DX-Y.NYB": "TVC:DXY",
}


def guess_tv_symbol(t):
    u = t.strip().upper()
    if u in TV_MAP:
        return TV_MAP[u]
    if u.endswith("=X") and len(u) == 8:
        return f"FX:{u[:6]}"
    if u.endswith("-USD"):
        return f"BINANCE:{u[:-4]}USDT"
    return u


def tradingview_html(symbol, interval, height=520):
    cfg = {
        "autosize": True, "symbol": symbol, "interval": interval,
        "timezone": "Etc/UTC", "theme": "dark", "style": "1",
        "locale": "en", "allow_symbol_change": True,
        "withdateranges": True, "support_host": "https://www.tradingview.com",
    }
    payload = json.dumps(cfg).replace("</", "<\\/")
    return (
        f'<div class="tradingview-widget-container" '
        f'style="height:{height}px;width:100%">'
        f'<div class="tradingview-widget-container__widget" '
        f'style="height:{height - 32}px;width:100%"></div>'
        '<script type="text/javascript" '
        'src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'
        f'{payload}</script></div>'
    )


def render_tv_chart(a=None):
    sym_default = guess_tv_symbol(ticker)
    with st.expander("📺 الشارت الحي (TradingView)", expanded=True):
        c1, c2 = st.columns([2, 1])
        sym = c1.text_input(
            "رمز TradingView", value=sym_default,
            key=f"tv_sym_{ticker.strip().upper()}",
        )
        opts = ["15", "60", "240", "D", "W"]
        labels = {"15": "15 دقيقة", "60": "ساعة", "240": "4 ساعات",
                  "D": "يومي", "W": "أسبوعي"}
        iv = c2.selectbox(
            "الفريم", opts,
            index=opts.index("D") if timeframe == TF_D1 else opts.index("240"),
            format_func=lambda x: labels[x],
            key=f"tv_iv_{timeframe}",
        )
        components.html(
            tradingview_html(sym.strip() or sym_default, iv, 520),
            height=530,
        )
        pl = a.get("plan") if a else None
        if pl and pl.get("direction") and a.get("ticker") == ticker.strip():
            st.markdown(
                f"**مستويات الخطة:** الدخول `{P(pl['entry'])}` | "
                f"الوقف `{P(pl['sl'])}` | الهدف 1 `{P(pl['tp1'])}` | "
                f"الهدف 2 `{P(pl['tp2'])}`"
            )
            st.caption("أضف خطاً أفقياً على TradingView عند كل سعر.")


def create_trades_chart(df, tr, title, max_trades=25, max_bars=700):
    t = tr.tail(max_trades)
    i0 = i1 = 0
    while True:
        i0 = max(int(df.index.searchsorted(t["entry_date"].iloc[0])) - 15, 0)
        i1 = min(int(df.index.searchsorted(t["exit_date"].iloc[-1])) + 10, len(df))
        if i1 - i0 <= max_bars or len(t) <= 1:
            break
        t = t.iloc[1:]
    d = df.iloc[i0:i1]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="السعر", opacity=0.55))
    for win, color, name in ((True, "#2ecc71", "رابحة"),
                              (False, "#e74c3c", "خاسرة")):
        part = t[(t["pnl"] > 0) == win]
        xs, ys = [], []
        for _, r in part.iterrows():
            xs += [r["entry_date"], r["exit_date"], None]
            ys += [r["entry"], r["exit"], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=color, width=2), name=name))
    for direction, sym_, col in (("BUY", "triangle-up", "#2ecc71"),
                                  ("SELL", "triangle-down", "#e74c3c")):
        part = t[t["direction"] == direction]
        if len(part):
            fig.add_trace(go.Scatter(
                x=part["entry_date"], y=part["entry"],
                mode="markers",
                marker=dict(symbol=sym_, size=12, color=col,
                            line=dict(width=1, color="white")),
                name=f"دخول {direction}",
                text=[f"{direction} @ {P(e)}" for e in part["entry"]],
                hoverinfo="text",
            ))
    fig.add_trace(go.Scatter(
        x=t["exit_date"], y=t["exit"], mode="markers",
        marker=dict(symbol="x", size=10,
                    color=["#2ecc71" if v > 0 else "#e74c3c" for v in t["pnl"]]),
        name="خروج",
        text=[f"{r} @ {P(x)} | {p_:+.2f}%" for r, x, p_ in zip(t["reason"], t["exit"], t["pnl"])],
        hoverinfo="text",
    ))
    fig.update_layout(title=title, template="plotly_white",
                      height=520, xaxis_rangeslider_visible=False)
    return fig


def render_trust_strip(a):
    df = a.get("df")
    try:
        last_ts = pd.Timestamp(df.index[-1])
        now_ts = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz is not None else pd.Timestamp.now()
        age_h = (now_ts - last_ts).total_seconds() / 3600.0
    except Exception:
        age_h = None

    ag = a.get("agents") or {}
    tech_ok = ag.get("tech") is not None
    news_ok = ag.get("news") is not None
    risk_ok = ag.get("risk") is not None or ag.get("risk_skipped")

    def badge(ok, label):
        return f"{'🟢' if ok else '🔴'} {label}"

    cols = st.columns(4)
    if age_h is None:
        cols[0].markdown("⚪ **حداثة البيانات:** غير معروفة")
    elif age_h <= 26:
        cols[0].markdown(f"🟢 **حداثة البيانات:** قبل {age_h:.0f} ساعة")
    else:
        cols[0].markdown(f"🟠 **حداثة البيانات:** قبل {age_h:.0f} ساعة — قديمة نسبيًا")
    cols[1].markdown(badge(tech_ok, "الوكيل الفني"))
    cols[2].markdown(badge(news_ok, "وكيل الأخبار"))
    cols[3].markdown(badge(risk_ok, "وكيل المخاطر"))
    if not (tech_ok and news_ok and risk_ok):
        st.caption("⚠️ أحد الوكلاء غير متاح — القرار النهائي يعمل بمبدأ fail-closed.")


def run_diagnostics(t):
    out = []

    def step(title, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
            out.append((ok, title, f"{detail}  ({time.time() - t0:.1f}s)"))
        except Exception as e:
            out.append((False, title, f"{type(e).__name__}: {str(e)[:300]}"))

    def secrets():
        have = {
            k: bool(st.secrets.get(k))
            for k in ("GEMINI_API_KEY", "GROQ_API_KEY",
                      "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        }
        return have["GEMINI_API_KEY"] and have["GROQ_API_KEY"], str(have)

    def yd():
        df = load_data(t, "1y")
        return True, f"{len(df)} شمعة يومية، آخر تاريخ: {df.index[-1]}"

    def yh():
        df = load_data_h4(t)
        return True, f"{len(df)} شمعة 4H، آخر تاريخ: {df.index[-1]}"

    def ind():
        df = calculate_indicators(load_data(t, "1y"))
        l = df.iloc[-1]
        return (
            not l[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any(),
            f"ATR={l['ATR14']:.4f} ADX={l['ADX14']:.1f} RSI={l['RSI14']:.1f}",
        )

    def gl():
        raw = _list_gemini()
        return True, f"{len(raw)} نموذجاً | سيُجرَّب: {get_gemini_models()}"

    def gc():
        txt, _, err, model = call_gemini(
            "Reply with the single word OK", False, get_gemini_models(), budget=25)
        return (
            (not err and bool(txt)),
            f"{model}: {txt.strip()[:40]}" if not err else err,
        )

    def rss():
        items, ferr = fetch_news(t)
        return (
            bool(items),
            f"{len(items)} عنواناً | أولها: {items[0]['title'][:80]}"
            if items else f"فشل: {ferr}",
        )

    def nag():
        raw, src_, err, model = news_agent(
            t, "BUY", get_gemini_models(), get_groq_models())
        ok = bool(raw and extract_json(raw))
        return ok, (
            f"{model}: فُهم الرد، العناوين = {len(src_)}"
            if ok else (err or "الرد ليس JSON")
        )

    def ql():
        raw = _list_groq()
        return True, f"{len(raw)} نموذجاً | سيُجرَّب: {get_groq_models()}"

    def qc():
        txt, err, model = call_groq(
            "Reply briefly.", "Reply with the single word OK",
            0.0, get_groq_models(), budget=25)
        return (
            (not err and bool(txt.strip())),
            f"{model}: {txt.strip()[:40]}" if not err else err,
        )

    def jr():
        obj, err = journal_load()
        if err:
            return False, f"({storage_mode()}) {err}"
        e2 = journal_save(obj["trades"], obj["position"])
        note = "" if storage_mode() == "gist" else " — مؤقت"
        return (
            (not e2),
            f"{storage_mode()}: {len(obj['trades'])} صفقة | الكتابة "
            + ("نجحت" if not e2 else f"فشلت: {e2}") + note,
        )

    step("1) المفاتيح في Secrets", secrets)
    step(f"2) Yahoo يومي — {t}", yd)
    step(f"3) Yahoo ساعة (لإطار 4H) — {t}", yh)
    step("4) حساب المؤشرات", ind)
    step("5) قائمة نماذج Gemini", gl)
    step("6) نداء Gemini عادي", gc)
    step("7) أخبار Google News (RSS)", rss)
    step("8) قائمة نماذج Groq", ql)
    step("9) نداء Groq", qc)
    step("10) وكيل الأخبار كاملاً", nag)
    step("11) التخزين الدائم للسجل", jr)
    return out


with st.sidebar:
    st.header("الإعدادات")
    asset_label = st.selectbox("الأصل", list(ASSETS.keys()))
    ticker = ASSETS[asset_label]
    strategy = st.selectbox(
        "الاستراتيجية",
        [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE,
         PULLBACK, RSI2, MEAN_REV],
    )
    timeframe = st.selectbox("الإطار الزمني", [TF_DUAL, TF_D1, TF_H4])
    st.caption("المزدوج: لا شراء إلا إذا كان اليومي فوق EMA200. "
               "الاستراتيجيات الانعكاسية معفاة.")
    use_closed = st.checkbox("استخدم آخر شمعة مكتملة فقط", value=True)
    cost_pct = st.number_input(
        "تكلفة الصفقة % (سبريد + عمولة)",
        min_value=0.0, max_value=2.0,
        value=DEFAULT_COST.get(ticker.strip().upper(), 0.05),
        step=0.01, key=f"cost_{ticker.strip().upper()}",
        help="تُستخدم في حساب الأدلة التاريخية والباكتست.",
    )
    balance = st.number_input("رأس المال (للحجم المقترح)",
                              min_value=100.0, value=10000.0, step=100.0)
    risk_pct = st.slider("المخاطرة لكل صفقة %", 0.25, 3.0, 1.0, 0.25)
    alert_threshold = st.slider("حد التنبيه (نقاط الجودة)", 60, 95, 85, 5)
    contract_size = st.number_input(
        "وحدات لكل 1 لوت", min_value=0.0001,
        value=guess_contract_size(ticker),
        key=f"cs_{ticker.strip().upper()}",
    )
    volume_step = st.number_input("خطوة اللوت", min_value=0.0001,
                                   value=0.01, step=0.01,
                                   key=f"vs_{ticker.strip().upper()}")
    min_lot = st.number_input("أقل لوت", min_value=0.0001,
                              value=0.01, step=0.01,
                              key=f"ml_{ticker.strip().upper()}")
    max_daily_loss_pct = st.slider(
        "🛑 أقصى خسارة يومية مسموحة % (قاطع دائرة)",
        0.5, 20.0, 5.0, 0.5,
        help="إذا بلغت خسائر الصفقات المُغلقة اليوم هذا الحد، "
             "يُمنع فتح صفقات جديدة حتى اليوم التالي.",
    )
    analyze_btn = st.button("ابدأ التحليل", type="primary",
                             use_container_width=True)
    st.markdown("---")
    if st.button("📨 اختبر تيليجرام", use_container_width=True):
        res = send_telegram("✅ اختبار: التطبيق متصل بتيليجرام.")
        if res is None:
            st.warning("لم تُضبط مفاتيح تيليجرام.")
        elif res == "ok":
            st.success("وصلت رسالة الاختبار ✅")
        else:
            st.error(res)
    st.caption("لأغراض تعليمية فقط")

    with st.expander("🩺 تشخيص شامل"):
        if st.button("شغّل التشخيص", use_container_width=True):
            with st.spinner("جاري الفحص..."):
                st.session_state["diag"] = run_diagnostics(
                    ticker.strip() or "GC=F")
        for ok_, title_, detail_ in st.session_state.get("diag", []):
            (st.success if ok_ else st.error)(f"{title_}")
            st.code(str(detail_)[:400])


if analyze_btn:
    pos = st.session_state.current_position
    t = ticker.strip()
    if not t:
        st.error("الرجاء إدخال رمز الأصل")
    elif pos is not None and pos["ticker"] != t:
        st.warning(f"⚠️ لديك صفقة مفتوحة على {pos['ticker']}.")
    else:
        with st.spinner(f"جاري تحليل {t}..."):
            try:
                df, bias_s, err = load_prepared(t, timeframe, use_closed)
                if err:
                    st.error(err)
                else:
                    bias = bias_s.iloc[-1] if bias_s is not None else None
                    latest = df.iloc[-1]
                    if len(df) < 60 or latest[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any():
                        st.error("البيانات غير كافية لحساب المؤشرات.")
                    else:
                        regime, adx, checks, chosen = select_strategy(
                            df, strategy, bias)
                        if chosen:
                            used, signal, reason = chosen
                        else:
                            used = checks[0][0]
                            signal = "No Signal"
                            reason = "لا توجد استراتيجية مناسبة تعطي إشارة الآن"

                        price, atr = float(latest["Close"]), float(latest["ATR14"])
                        mean_rev_used = used in (MEAN_REV, RSI2)

                        levels = None
                        if signal in ("BUY", "SELL"):
                            levels = smart_levels(
                                signal, price, df, mean_rev=mean_rev_used)
                            if not levels["valid"]:
                                signal = "No Signal"
                                reason = reason + " — ❌ مرفوضة: العائد/المخاطرة أقل من 1"
                                levels = None

                        ctx = build_context(t, df, timeframe, bias, regime, adx,
                                            used, signal, reason, checks)
                        agents = run_agents(ctx, t, signal, price, atr, df,
                                            mean_rev_used, levels=levels)
                        arb = arbitrate(signal, agents["tech"],
                                        agents["news"], agents["risk"])

                        evidence, score, parts = None, 0, []
                        if signal in ("BUY", "SELL"):
                            evidence = history_evidence(
                                t, timeframe, df, bias_s, used,
                                use_closed, cost=cost_pct)
                            score, parts = compute_quality(
                                signal, df, adx, used, bias,
                                agents["tech"], agents["news"],
                                agents["risk"], evidence)
                        dec = final_decision(signal, arb, score, alert_threshold)
                        plan_obj = build_plan(
                            signal, price, atr, dec, balance, risk_pct,
                            contract_size, df, mean_rev_used,
                            volume_step, min_lot, levels=levels)
                        tg = None
                        if dec.get("enter"):
                            st.toast("🔔 إشارة دخول جاهزة!", icon="🔔")
                            tg = send_telegram(format_alert(
                                t, timeframe, used, plan_obj, score))

                        news = agents.get("news")
                        if news and should_alert_event(
                                news, t, st.session_state.get("event_alerts")):
                            evs = news.get("key_events") or []
                            msg = (
                                f"⚠️ تنبيه حدث اقتصادي — {t}\n"
                                f"مخاطر الأحداث: مرتفعة\n"
                                f"الأحداث: {' | '.join(str(e) for e in evs[:3])}\n"
                                f"راجع خطتك قبل أي قرار جديد."
                            )
                            if send_telegram(msg) == "ok":
                                st.session_state.setdefault("event_alerts", {})[
                                    f"event_{t.strip().upper()}"
                                ] = datetime.now().strftime("%Y-%m-%d %H:%M")
                                st.toast("⚠️ أُرسل تنبيه حدث اقتصادي", icon="⚠️")

                        st.session_state.analysis = {
                            "ticker": t, "mode": strategy, "used": used,
                            "df": df, "tf": timeframe, "bias": bias,
                            "regime": regime, "adx": adx, "checks": checks,
                            "signal": signal, "reason": reason,
                            "price": price, "atr": atr,
                            "rsi": float(latest["RSI14"]),
                            "ema200": float(latest["EMA200"]),
                            "agents": agents, "arb": dec, "score": score,
                            "parts": parts, "threshold": alert_threshold,
                            "evidence": evidence, "tg": tg,
                            "plan": plan_obj,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.session_state.live_price = None
                        st.session_state.live_price_is_live = False
            except Exception as e:
                st.error(f"خطأ: {type(e).__name__}: {e}")
                st.code(traceback.format_exc()[-1800:])


a = st.session_state.analysis
if a is None:
    render_tv_chart(None)
if a is not None:
    plan = a["plan"]

    render_trust_strip(a)
    st.markdown("## 📋 خطة الصفقة")
    getattr(st, plan["level"])(f"**{plan['action']}**")
    if plan["direction"]:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("الاتجاه",
                  "شراء 🟢" if plan["direction"] == "BUY" else "بيع 🔴")
        p2.metric("الدخول", f"{P(plan['entry'])}")
        p3.metric("وقف الخسارة", f"{P(plan['sl'])}")
        p4.metric("الهدف 1", f"{P(plan['tp1'])}")
        p5.metric("الهدف 2", f"{P(plan['tp2'])}")
        if plan.get("size"):
            p6.metric("الحجم",
                      f"{plan['lots']:.2f} لوت" if plan.get("lots")
                      else "أقل من 0.01 لوت")
            st.caption(
                f"≈ {plan['size']:,.2f} وحدة | "
                f"المخاطرة {plan['risk_amount']:,.2f}"
            )
        else:
            p6.metric("الحجم", "—")

    if a.get("score") is not None and a["signal"] in ("BUY", "SELL"):
        st.markdown("## 🎯 درجة التوافق (Heuristic)")
        q1, q2 = st.columns([1, 2])
        q1.metric("الجودة", f"{a['score']}/100")
        q2.progress(min(max(a["score"], 0), 100) / 100)
        q2.caption(f"درجة توافق إرشادية وليست احتمال ربح. "
                   f"الحد المطلوب: {a['threshold']}")
        with st.expander("تفاصيل النقاط"):
            for label, pts_, mx, note in a["parts"]:
                st.markdown(f"- **{label}:** {pts_:g}/{mx} — {note}")

    render_tv_chart(a)

    st.markdown(
        f"**الرمز:** {a['ticker']} | **الاستراتيجية:** {a['used']} | "
        f"**الإطار:** {a.get('tf', TF_D1)} | **الوقت:** {a['time']}"
    )

    st.markdown("## 🧭 حالة السوق")
    r1, r2, r3 = st.columns(3)
    r1.metric("الحالة", a["regime"])
    r2.metric("ADX14", f"{a['adx']:.1f}")
    r3.metric(
        "اتجاه اليومي",
        {"UP": "صاعد 🟢", "DOWN": "هابط 🔴", "NEUTRAL": "محايد ⚪"}.get(
            a.get("bias"), "غير مستخدم"),
    )
    for name, sig, reason in a["checks"]:
        icon = {"BUY": "🟢", "SELL": "🔴"}.get(sig, "⚪")
        st.markdown(f"- {icon} **{name}**: {reason}")

    if a["signal"] in ("BUY", "SELL"):
        corr_note = CORRELATION_NOTES.get(a["ticker"].strip().upper())
        if corr_note:
            st.info(f"🔗 **ارتباط بين الأصول:** {corr_note}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سعر الإغلاق", f"{P(a['price'])}")
    c2.metric("RSI14", f"{a['rsi']:.1f}")
    c3.metric("ATR14", f"{P(a['atr'])}")
    c4.metric("EMA200", f"{P(a['ema200'])}")

    st.markdown("## 📈 شارت الخطة")
    st.plotly_chart(
        create_chart(a["df"], a["ticker"], plan["direction"],
                     plan["sl"], plan["tp1"], plan["tp2"], a.get("tf", TF_D1)),
        use_container_width=True,
    )

    ev = a.get("evidence")
    if ev and ev.get("walk_forward"):
        with st.expander(f"🚶 Walk-Forward — {ev.get('folds', 0)} نوافذ",
                          expanded=True):
            st.caption("في كل نافذة، اختار النظام استراتيجيته على بيانات تدريب "
                       "فقط، ثم اختُبر على نافذة تالية لم يرَها.")
            pf = pd.DataFrame(ev.get("per_fold", []))
            if not pf.empty:
                pf = pf.rename(columns={
                    "fold": "نافذة", "pick": "الاستراتيجية المختارة",
                    "train_r": "R تدريب", "train_n": "صفقات تدريب",
                    "test_r": "R اختبار", "test_n": "صفقات اختبار",
                    "test_total": "إجمالي اختبار %",
                })
                st.dataframe(pf, use_container_width=True, hide_index=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("إجمالي صفقات OOS", ev.get("trades", 0))
            m2.metric("متوسط R (OOS)", f"{ev.get('avg_r', 0):.2f}")
            m3.metric("PF (OOS)",
                      f"{ev.get('pf', 0):.2f}"
                      if np.isfinite(ev.get('pf', 0)) else "∞")
            m4.metric("إجمالي % (OOS)", f"{ev.get('total', 0):+.1f}%")

            if ev.get("picks"):
                uniq = len(set(ev["picks"]))
                if uniq > 1:
                    st.success(
                        f"✅ النظام اختار {uniq} استراتيجية مختلفة عبر "
                        f"{len(ev['picks'])} نوافذ — دليل على التكيّف."
                    )
                else:
                    st.info(
                        f"النظام اختار نفس الاستراتيجية "
                        f"({ev['picks'][0].split(' (')[0]}) في كل النوافذ."
                    )

    st.markdown("## 🧠 فريق الوكلاء")
    ag = a.get("agents")
    if not ag:
        st.info("أعد تشغيل التحليل.")
    else:
        tech, news, risk = ag["tech"], ag["news"], ag["risk"]
        vmap = {"BUY": "شراء 🟢", "SELL": "بيع 🔴", "WAIT": "انتظار ⚪"}

        with st.expander("📈 الوكيل الفني", expanded=True):
            if ag.get("tech_model"):
                st.caption(f"النموذج: {ag['tech_model']}")
            if tech is None:
                st.error(f"غير متاح: {ag['tech_err']}")
                if ag["tech_raw"]:
                    st.caption(ag["tech_raw"][:500])
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("الرأي",
                          vmap.get(str(tech.get("verdict", "")).upper(), "—"))
                m2.metric("الثقة", f"{tech.get('confidence', '—')}%")
                m3.metric("يؤيد الإشارة",
                          "نعم ✅" if truthy(tech.get("supports_signal")) else "لا ❌")
                if tech.get("market_context"):
                    st.markdown(str(tech["market_context"]))
                kl = tech.get("key_levels") if isinstance(tech.get("key_levels"), dict) else {}
                supports = clean_price_levels(kl.get("support"))
                resistances = clean_price_levels(kl.get("resistance"))
                st.markdown(
                    f"**دعم:** {[P(v) for v in supports]} | "
                    f"**مقاومة:** {[P(v) for v in resistances]}"
                )
                for rsn in as_list(tech.get("reasons")):
                    st.markdown(f"- {rsn}")
                if tech.get("invalidation"):
                    st.caption(f"يُلغى التحليل إذا: {tech['invalidation']}")

        with st.expander("📰 وكيل الأخبار", expanded=True):
            if ag.get("news_model"):
                st.caption(f"النموذج: {ag['news_model']}")
            if news is None:
                st.error(f"غير متاح: {ag['news_err']}")
                if ag["news_raw"]:
                    st.caption(ag["news_raw"][:500])
            else:
                sent = {"bullish": "صاعد 🟢", "bearish": "هابط 🔴",
                        "neutral": "محايد ⚪"}
                evr = {"high": "مرتفعة 🔴", "medium": "متوسطة 🟠",
                       "low": "منخفضة 🟢", "unknown": "غير معروفة ⚪"}
                imp = {"supports": "يدعم الإشارة", "conflicts": "يعاكس الإشارة",
                       "neutral": "محايد"}
                m1, m2, m3 = st.columns(3)
                m1.metric("المزاج",
                          sent.get(str(news.get("sentiment", "")).lower(), "—"))
                m2.metric("مخاطر الأحداث",
                          evr.get(str(news.get("event_risk", "")).lower(), "—"))
                m3.metric("الأثر على الإشارة",
                          imp.get(str(news.get("impact_on_signal", "")).lower(), "—"))
                if news.get("summary"):
                    st.markdown(str(news["summary"]))
                for ev in as_list(news.get("key_events")):
                    st.markdown(f"- {ev}")
                if ag["news_sources"]:
                    st.markdown("**المصادر:**")
                    for src_ in ag["news_sources"]:
                        st.markdown(f"- [{src_['title']}]({src_['uri']})")

        with st.expander("🛡️ وكيل المخاطر", expanded=True):
            if ag.get("risk_model"):
                st.caption(f"النموذج: {ag['risk_model']}")
            if ag["risk_skipped"]:
                st.info("لم يُشغَّل: لا توجد إشارة للتقييم.")
            elif risk is None:
                st.error(f"غير متاح: {ag['risk_err']}")
                if ag["risk_raw"]:
                    st.caption(ag["risk_raw"][:500])
            else:
                dmap = {"APPROVE": "موافقة ✅", "REDUCE": "تخفيض ⚠️",
                        "VETO": "اعتراض 🛑"}
                m1, m2 = st.columns(2)
                m1.metric("القرار",
                          dmap.get(str(risk.get("decision", "")).upper(), "—"))
                m2.metric("درجة المخاطرة", f"{risk.get('risk_score', '—')}/10")
                for c_ in as_list(risk.get("concerns")):
                    st.markdown(f"- {c_}")
                if risk.get("comment"):
                    st.caption(str(risk["comment"]))

    arb = a.get("arb")
    if arb:
        st.markdown("## القرار النهائي")
        getattr(st, arb["level"])(f"**{arb['action']}**")
        for n_ in arb["notes"]:
            st.markdown(f"- {n_}")
        if a.get("tg") == "ok":
            st.success("📨 أُرسل تنبيه تيليجرام.")

        if a.get("plan", {}).get("direction"):
            exposed, reason = correlation_exposure_check(
                a["ticker"], st.session_state.trades,
                st.session_state.current_position)
            if exposed:
                st.warning(f"🔗 **حارس الارتباط:** {reason}")
                st.caption("توصية: اعتبر هذه الصفقة جزءًا من تعرّضك الكلي "
                           "للدولار، لا كصفقة منفصلة.")


pos = st.session_state.current_position
if st.session_state.notice:
    kind, msg = st.session_state.notice
    getattr(st, kind)(msg)
    st.session_state.notice = None

if a is not None or pos is not None:
    sim_ticker = pos["ticker"] if pos else a["ticker"]
    atr = pos["atr"] if pos else a["atr"]
    last_close = a["price"] if (a and a["ticker"] == sim_ticker) else pos["entry_price"]
    price = st.session_state.live_price or last_close

    st.markdown("---")
    st.header("🎯 محاكي التداول (Paper Trading)")

    breaker_tripped, pnl_today = daily_breaker_status(
        st.session_state.trades, max_daily_loss_pct)
    if breaker_tripped:
        st.error(
            f"🛑 قاطع الدائرة اليومي مُفعَّل: خسائر اليوم الفعلية "
            f"من رأس المال بلغت {pnl_today:+.2f}% "
            f"(الحد المسموح -{max_daily_loss_pct:.1f}%). "
            f"مُنع فتح صفقات جديدة حتى اليوم التالي."
        )

    if st.session_state.live_price is None:
        src = "آخر إغلاق" if a else "سعر الدخول"
    elif st.session_state.get("live_price_is_live"):
        src = "لحظي ✅"
    else:
        src = "احتياطي (فشل الجلب اللحظي) ⚠️"
    st.markdown(f"**الأصل:** `{sim_ticker}` | **السعر ({src}):** `{P(price)}` | "
                f"**ATR:** `{P(atr)}`")
    if (st.session_state.live_price is not None
            and not st.session_state.get("live_price_is_live")):
        st.caption("⚠️ هذا سعر إغلاق قديم — فحص الوقف/الهدف قد لا يعكس السعر الفعلي.")

    if st.button("🔄 تحديث السعر", use_container_width=True):
        lp, is_live = get_live_price(sim_ticker, last_close)
        st.session_state.live_price = lp
        st.session_state.live_price_is_live = is_live
        if is_live:
            check_exit(lp)
        else:
            st.warning("تعذّر جلب سعر لحظي.")
        st.rerun()

    plan_dir = (a["plan"]["direction"]
                if (a and a["plan"]["level"] != "error") else None)
    if st.button("📌 افتح الصفقة حسب الخطة", use_container_width=True,
                 type="primary",
                 disabled=(pos is not None or plan_dir is None or breaker_tripped)):
        execution_price, exec_is_live = get_live_price(sim_ticker, price)
        plan_entry = a["plan"]["entry"]
        worse_buy = plan_dir == "BUY" and execution_price > plan_entry * 1.02
        worse_sell = plan_dir == "SELL" and execution_price < plan_entry * 0.98
        if worse_buy or worse_sell:
            st.warning(
                f"السعر الحالي ({P(execution_price)}) تحرك ضدك أكثر من 2% "
                f"عن سعر الخطة ({P(plan_entry)})؛ لم تُفتح الصفقة."
            )
        elif not exec_is_live:
            st.warning("تعذّر جلب سعر لحظي للتنفيذ؛ لم تُفتح الصفقة.")
        else:
            mean_rev_exec = a.get("used") in (MEAN_REV, RSI2)
            exec_sl, exec_tp1, exec_tp2 = trade_levels(
                plan_dir, execution_price, atr, a.get("df"), mean_rev_exec)
            exec_risk_pct = risk_pct * (a.get("arb") or {}).get("mult", 1.0)
            open_position(
                plan_dir, execution_price, sim_ticker, atr,
                meta_from(a, "plan"),
                exec_sl, exec_tp2, tp1=exec_tp1,
                account_risk_pct=exec_risk_pct,
            )
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 شراء يدوي", use_container_width=True,
                     disabled=(pos is not None or a is None or breaker_tripped)):
            manual_price, _ = get_live_price(sim_ticker, price)
            mean_rev_manual = a.get("used") in (MEAN_REV, RSI2)
            manual_sl, manual_tp1, manual_tp2 = trade_levels(
                "BUY", manual_price, atr, a.get("df"), mean_rev_manual)
            open_position(
                "BUY", manual_price, sim_ticker, atr,
                meta_from(a, "manual"),
                manual_sl, manual_tp2, tp1=manual_tp1,
                account_risk_pct=risk_pct,
            )
            st.rerun()
    with col2:
        if st.button("🔴 بيع يدوي", use_container_width=True,
                     disabled=(pos is not None or a is None or breaker_tripped)):
            manual_price, _ = get_live_price(sim_ticker, price)
            mean_rev_manual = a.get("used") in (MEAN_REV, RSI2)
            manual_sl, manual_tp1, manual_tp2 = trade_levels(
                "SELL", manual_price, atr, a.get("df"), mean_rev_manual)
            open_position(
                "SELL", manual_price, sim_ticker, atr,
                meta_from(a, "manual"),
                manual_sl, manual_tp2, tp1=manual_tp1,
                account_risk_pct=risk_pct,
            )
            st.rerun()
    with col3:
        if st.button("⏹️ إغلاق الصفقة", use_container_width=True,
                     disabled=pos is None):
            close_trade(price, "Manual")
            st.rerun()

    if pos is not None:
        cur = calc_pnl(pos, price)
        st.markdown("### الصفقة الحالية")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("الاتجاه", pos["direction"])
        c2.metric("سعر الدخول", f"{P(pos['entry_price'])}")
        sl_label = f"{P(pos['sl'])}" + (" (BE)" if pos.get("stage") == 1 else "")
        c3.metric("وقف الخسارة", sl_label)
        c4.metric("هدف 1", f"{P(pos.get('tp1', pos['tp']))}")
        c5.metric("هدف 2", f"{P(pos['tp'])}")
        c6.metric("الربح الحالي", f"{cur:+.2f}%")

        if pos.get("stage") == 1:
            rp = float(pos.get("realized_partial", 0.0))
            pct = int(float(pos.get("partial", 0.5)) * 100)
            st.info(f"✅ تحقق الهدف الأول — أُغلق {pct}% ونُقل الوقف إلى الدخول. "
                    f"محقق جزئيًا: {rp:+.2f}%")

        try:
            opened = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M")
            age_h = (datetime.now() - opened).total_seconds() / 3600.0
            if age_h >= 24:
                st.warning(f"⏰ هذه الصفقة مفتوحة منذ {age_h:.0f} ساعة. "
                           f"اضغط «🔄 تحديث السعر» لفحص الوقف/الهدف.")
            elif age_h >= 8 and not st.session_state.get("live_price_is_live"):
                st.info(f"ℹ️ الصفقة مفتوحة منذ {age_h:.0f} ساعة دون تحديث سعر لحظي.")
        except Exception:
            pass

        if st.session_state.get("live_price_is_live"):
            should, why = should_alert_position(
                pos, price, st.session_state.get("position_alert_last"))
            if should:
                msg = (
                    f"🔔 تنبيه صفقة — {pos['ticker']}\n"
                    f"{why}\n"
                    f"الاتجاه: "
                    f"{'شراء 🟢' if pos['direction'] == 'BUY' else 'بيع 🔴'}\n"
                    f"دخول: {P(pos['entry_price'])} | حاليًا: {P(price)}\n"
                    f"وقف: {P(pos['sl'])} | هدف: {P(pos['tp'])}\n"
                    f"P&L: {cur:+.2f}%\n"
                    f"⚠️ تعليمي فقط."
                )
                res = send_telegram(msg)
                st.session_state["position_alert_last"] = \
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                if res == "ok":
                    st.toast("📨 أُرسل تنبيه الصفقة", icon="📨")
                elif res is not None:
                    st.caption(f"تعذّر إرسال التنبيه: {res}")


st.markdown("---")
st.header("📒 سجل الصفقات الدائم")
if storage_mode() == "gist":
    st.caption("💾 التخزين: GitHub Gist (دائم) ✅")
else:
    st.warning("💾 التخزين مؤقت. أضف GITHUB_TOKEN و GIST_ID في Secrets.")
if st.session_state.get("journal_load_err"):
    st.error(f"تعذّر تحميل السجل: {st.session_state.journal_load_err}")
if st.session_state.get("journal_save_err"):
    st.error(f"تعذّر حفظ السجل: {st.session_state.journal_save_err}")

trades = st.session_state.trades
if not trades:
    st.info("لا صفقات بعد.")
else:
    tdf = pd.DataFrame(trades)
    tdf["pnl_percent"] = pd.to_numeric(tdf["pnl_percent"], errors="coerce")
    names = {
        "entry_time": "وقت الدخول", "ticker": "الأصل", "direction": "الاتجاه",
        "entry_price": "سعر الدخول", "exit_price": "سعر الخروج",
        "exit_reason": "سبب الخروج", "pnl_percent": "حركة السعر %",
        "account_pnl_percent": "أثر رأس المال %", "r_multiple": "R",
        "score": "النقاط", "used": "الاستراتيجية", "tf": "الإطار",
        "source": "المصدر",
    }
    cols = [c for c in names if c in tdf.columns]
    disp = tdf[cols].copy()
    for c in ("used", "tf"):
        if c in disp:
            disp[c] = disp[c].map(
                lambda x: x.split(" (")[0] if isinstance(x, str) else x)
    for c in ("pnl_percent", "account_pnl_percent", "r_multiple"):
        if c in disp:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)
    st.dataframe(disp.rename(columns=names), use_container_width=True,
                 hide_index=True)

    wins = tdf["pnl_percent"] > 0
    gw, gl = tdf.loc[wins, "pnl_percent"].sum(), abs(tdf.loc[~wins, "pnl_percent"].sum())
    rr = (pd.to_numeric(tdf["r_multiple"], errors="coerce").mean()
          if "r_multiple" in tdf else float("nan"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("عدد الصفقات", len(tdf))
    m2.metric("نسبة الربح", f"{wins.mean() * 100:.1f}%")
    m3.metric("متوسط R", f"{rr:.2f}" if rr == rr else "—")
    m4.metric("PF", f"{gw / gl:.2f}" if gl > 0 else "∞")

    with st.expander("📚 إحصاءات التعلم"):
        if len(tdf) < 20:
            st.warning("أقل من 20 صفقة: الإحصاءات للاطلاع فقط.")
        groups = journal_groups(tdf)
        for title, g in groups:
            st.markdown(f"**حسب {title}**")
            st.dataframe(g, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ تنزيل السجل CSV",
        tdf.to_csv(index=False).encode("utf-8-sig"),
        "trades.csv", "text/csv", use_container_width=True,
    )
    with st.expander("⚠️ مسح السجل"):
        sure = st.checkbox("أؤكد حذف كل الصفقات نهائياً", key="confirm_clear")
        if st.button("🗑️ مسح السجل", disabled=not sure):
            st.session_state.trades = []
            persist(force=True)
            st.rerun()


if st.session_state.trades and len(st.session_state.trades) >= 5:
    st.markdown("---")
    st.header("📈 لوحة الأداء (من سجلك الفعلي)")
    st.caption("محسوبة من سجل صفقاتك الحقيقي (المحاكي الورقي). "
               "ليست توقعات مستقبلية.")

    eq_df = compute_equity_curve(st.session_state.trades, initial=100.0)
    if not eq_df.empty:
        final_eq = float(eq_df["equity"].iloc[-1])
        max_dd = float(eq_df["drawdown"].min())
        total_ret = (final_eq - 100.0)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("رأس المال (بداية 100)", f"{final_eq:.2f}",
                  f"{total_ret:+.2f}%")
        k2.metric("أقصى تراجع", f"{max_dd:.2f}%")
        k3.metric("عدد الصفقات", len(eq_df))
        wins = (eq_df["pnl_percent"] > 0).mean() * 100
        k4.metric("نسبة الربح", f"{wins:.1f}%")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=list(range(len(eq_df))), y=eq_df["equity"],
            mode="lines", name="رأس المال",
            line=dict(color="#2ecc71", width=2)))
        fig_eq.add_hline(y=100.0, line_dash="dash", line_color="grey",
                         annotation_text="خط البداية")
        fig_eq.update_layout(title="منحنى رأس المال (بداية = 100)",
                             template="plotly_white", height=350,
                             xaxis_title="رقم الصفقة", yaxis_title="القيمة")
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=list(range(len(eq_df))), y=eq_df["drawdown"],
            mode="lines", fill="tozeroy", name="التراجع %",
            line=dict(color="#e74c3c", width=1.5)))
        fig_dd.update_layout(title="التراجع من القمة",
                             template="plotly_white", height=250,
                             xaxis_title="رقم الصفقة", yaxis_title="%")
        st.plotly_chart(fig_dd, use_container_width=True)

    with st.expander("🎲 محاكاة مونت كارلو", expanded=False):
        mc1, mc2 = st.columns(2)
        n_sims = mc1.number_input("عدد المحاكاات", 100, 10000, 1000, 100,
                                   key="mc_n")
        mc2.caption("يُعاد ترتيب/سحب صفقاتك عشوائيًا، لقياس حساسية النتيجة "
                    "للترتيب.")
        if st.button("🎲 شغّل مونت كارلو", use_container_width=True):
            mc = monte_carlo(st.session_state.trades, n_sims=int(n_sims))
            if mc is None:
                st.warning("لا صفقات كافية.")
            else:
                st.session_state["mc_result"] = mc

        mc = st.session_state.get("mc_result")
        if mc:
            a1, a2, a3 = st.columns(3)
            a1.metric("السيناريو المتوسط", f"{mc['p50']:.2f}",
                      f"{mc['p50'] - 100:+.2f}%")
            a2.metric("السيناريو السيئ (5%)", f"{mc['p05']:.2f}",
                      f"{mc['p05'] - 100:+.2f}%")
            a3.metric("السيناريو الجيد (95%)", f"{mc['p95']:.2f}",
                      f"{mc['p95'] - 100:+.2f}%")

            b1, b2 = st.columns(2)
            b1.metric("متوسط أقصى تراجع", f"{mc['dd_p50']:.2f}%")
            b2.metric("أقصى تراجع سيئ (5%)", f"{mc['dd_p05']:.2f}%")

            st.caption(f"استنادًا إلى {mc['n_trades']} صفقة × "
                       f"{mc['n_sims']} محاكاة.")


st.markdown("---")
st.header("📊 اختبار الاستراتيجيات (Backtest)")
st.caption("يقارن الاستراتيجيات والأطر الزمنية. لا يشمل رأي الذكاء الاصطناعي.")

bc1, bc2, bc3 = st.columns(3)
bt_period = bc1.selectbox("مدة البيانات اليومية", ["3y", "5y", "10y"],
                           index=1)
bt_cost = bc2.number_input("تكلفة الصفقة % (سبريد+عمولة)",
                            0.0, 2.0, cost_pct, 0.01, key=f"btc_{ticker}")
bt_hold = bc3.number_input("أقصى مدة (شموع)", 5, 200, 30)
bt_tfs = st.multiselect("الأطر المطلوب مقارنتها",
                         [TF_D1, TF_H4, TF_DUAL],
                         default=[TF_D1, TF_H4, TF_DUAL])


def short(label):
    return label.split(" (")[0]


if st.button("▶️ شغّل الاختبار", use_container_width=True):
    bt_ticker = ticker.strip()
    if not bt_ticker:
        st.error("الرجاء إدخال رمز الأصل")
    elif not bt_tfs:
        st.error("اختر إطاراً واحداً على الأقل")
    else:
        with st.spinner("جاري الاختبار..."):
            try:
                results, bh, bars, notes, prices = {}, {}, {}, [], {}
                for tf in bt_tfs:
                    dfb, bias_b, err = load_prepared(bt_ticker, tf, True, bt_period)
                    if err:
                        notes.append(f"{short(tf)}: {err}")
                        continue
                    if len(dfb) < 300:
                        notes.append(f"{short(tf)}: البيانات غير كافية")
                        continue
                    if tf in (TF_H4, TF_DUAL) and bt_period in ("5y", "10y"):
                        notes.append(
                            f"{short(tf)}: Yahoo لا يوفر 1H لسنوات 5/10 بشكل موثوق.")
                    sigs = strategy_signals(dfb)
                    raw_sigs = {k: v.copy() for k, v in sigs.items()}
                    sigs[AUTO] = auto_signals(dfb, raw_sigs, bias_b)
                    if bias_b is not None:
                        trend_part = {k: v for k, v in raw_sigs.items()
                                      if k not in CONTRARIAN}
                        contra_part = {k: v for k, v in raw_sigs.items()
                                       if k in CONTRARIAN}
                        trend_part = apply_bias_filter(trend_part, bias_b)
                        sigs.update({**trend_part, **contra_part})
                    for name in [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE,
                                  TURTLE, PULLBACK, RSI2, MEAN_REV]:
                        tr = run_backtest_v2(
                            dfb, sigs[name], bt_cost, int(bt_hold),
                            mean_rev=(name in (MEAN_REV, RSI2)))
                        results[(tf, name)] = {"trades": tr,
                                                "stats": backtest_stats(tr)}
                    bh[tf] = float(
                        (dfb["Close"].iloc[-1] / dfb["Close"].iloc[200] - 1) * 100)
                    bars[tf] = len(dfb) - 200
                    prices[tf] = dfb[["Open", "High", "Low", "Close"]]
                st.session_state.backtest = {
                    "ticker": bt_ticker, "period": bt_period, "cost": bt_cost,
                    "results": results, "buy_hold": bh, "bars": bars,
                    "notes": notes, "prices": prices,
                }
            except Exception as e:
                st.error(f"خطأ في الاختبار: {e}")

bt = st.session_state.get("backtest")
if bt and "results" in bt and isinstance(bt.get("buy_hold"), dict):
    for n in bt["notes"]:
        st.warning(n)
    if not bt["results"]:
        st.error("لا توجد نتائج.")
    else:
        st.markdown(
            f"**{bt['ticker']}** | تكلفة {bt['cost']}% | " +
            " | ".join(f"{short(tf)}: {n} شمعة"
                       for tf, n in bt["bars"].items())
        )

        rows = []
        for (tf, name), res in bt["results"].items():
            s_ = res["stats"]
            rows.append({
                "الإطار": short(tf),
                "الاستراتيجية": short(name),
                "الصفقات": s_["trades"],
                "نسبة الربح %": round(s_["win_rate"], 1),
                "الإجمالي %": round(s_["total"], 1),
                "متوسط R": round(s_["avg_r"], 2),
                "PF": round(s_["pf"], 2) if np.isfinite(s_["pf"]) else "∞",
                "أقصى تراجع %": round(s_["max_dd"], 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption("الشراء والاحتفاظ: " +
                   " | ".join(f"{short(tf)}: {v:+.1f}%"
                              for tf, v in bt["buy_hold"].items()))

        valid = {k: r["stats"] for k, r in bt["results"].items()
                 if r["stats"]["trades"] >= 30}
        if valid:
            best = max(valid, key=lambda k: valid[k]["avg_r"])
            if valid[best]["avg_r"] > 0:
                st.success(
                    f"الأفضل: **{short(best[1])}** على إطار "
                    f"**{short(best[0])}** بمتوسط {valid[best]['avg_r']:.2f}R.")
            else:
                st.warning("لا توجد تركيبة رابحة بعد التكلفة.")
        else:
            st.warning("كل التركيبات أقل من 30 صفقة — العينة صغيرة.")

        tf_keys = list(bt["bars"].keys())
        tf_pick = st.selectbox("منحنى الأرباح للإطار:", tf_keys,
                                format_func=short, key="bt_tf_pick")
        fig_bt = go.Figure()
        for (tf, name), res in bt["results"].items():
            if tf == tf_pick and not res["trades"].empty:
                tr = res["trades"]
                fig_bt.add_trace(go.Scatter(
                    x=tr["exit_date"], y=tr["pnl"].cumsum(),
                    mode="lines", name=short(name)))
        fig_bt.update_layout(title="الأرباح التراكمية %",
                             template="plotly_white", height=400)
        st.plotly_chart(fig_bt, use_container_width=True)

        pick = st.selectbox("عرض صفقات:", list(bt["results"].keys()),
                            key="bt_pick",
                            format_func=lambda k: f"{short(k[0])} | {short(k[1])}")
        tr_pick = bt["results"][pick]["trades"]
        if tr_pick.empty:
            st.info("لا صفقات لهذه التركيبة.")
        else:
            show = tr_pick.copy()
            for col in ("entry_date", "exit_date"):
                show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M")
            show[["entry", "exit", "pnl", "r"]] = show[["entry", "exit", "pnl", "r"]].round(2)
            st.dataframe(show, use_container_width=True, hide_index=True)
            px_ = bt.get("prices", {}).get(pick[0])
            if px_ is not None:
                st.markdown("#### 📍 الدخول والخروج على الشارت")
                st.plotly_chart(
                    create_trades_chart(px_, tr_pick,
                                        f"{short(pick[0])} | {short(pick[1])}"),
                    use_container_width=True,
                )
                st.caption("▲ شراء ▼ بيع ✕ خروج. الأخضر رابحة، الأحمر خاسرة.")


st.markdown("---")
st.header("🌍 مقارنة متعددة الأصول")
st.caption("يشغّل نفس الاستراتيجية والإطار على كل الأصول المدعومة.")

mc1, mc2, mc3, mc4 = st.columns(4)
multi_tf = mc1.selectbox("الإطار", [TF_D1, TF_H4], key="multi_tf")
multi_period = mc2.selectbox("مدة البيانات", ["3y", "5y", "10y"],
                              index=1, key="multi_period")
multi_cost = mc3.number_input("تكلفة الصفقة %", 0.0, 2.0,
                               float(cost_pct), 0.01, key="multi_cost")
multi_strategy = mc4.selectbox(
    "الاستراتيجية",
    [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE, TURTLE, PULLBACK, RSI2, MEAN_REV],
    key="multi_strategy",
)

if st.button("🌍 شغّل المقارنة", use_container_width=True, key="run_multi"):
    with st.spinner("جاري مقارنة الأصول..."):
        try:
            multi_rows = []
            for label, tk in ASSETS.items():
                tk_clean = tk.strip()
                try:
                    dfb, bias_b, err = load_prepared(
                        tk_clean, multi_tf, True, multi_period)
                    if err:
                        multi_rows.append({"الأصل": label, "الرمز": tk_clean,
                                            "الحالة": f"⚠️ {str(err)[:60]}"})
                        continue
                    if dfb is None or len(dfb) < 300:
                        multi_rows.append({"الأصل": label, "الرمز": tk_clean,
                                            "الحالة": "بيانات غير كافية"})
                        continue
                    sigs = strategy_signals(dfb)
                    raw = {k: v.copy() for k, v in sigs.items()}
                    sigs[AUTO] = auto_signals(dfb, raw, bias_b)
                    if bias_b is not None:
                        tp_ = {k: v for k, v in raw.items()
                               if k not in CONTRARIAN}
                        cp_ = {k: v for k, v in raw.items()
                               if k in CONTRARIAN}
                        tp_ = apply_bias_filter(tp_, bias_b)
                        sigs.update({**tp_, **cp_})
                    sig_pick = sigs.get(multi_strategy)
                    if sig_pick is None:
                        multi_rows.append({"الأصل": label, "الرمز": tk_clean,
                                            "الحالة": "استراتيجية غير متاحة"})
                        continue
                    tr = run_backtest_v2(
                        dfb, sig_pick, multi_cost, 30,
                        mean_rev=(multi_strategy in (MEAN_REV, RSI2)))
                    s_ = backtest_stats(tr)
                    bh = float(
                        (dfb["Close"].iloc[-1] / dfb["Close"].iloc[200] - 1) * 100)
                    status = (
                        "✅" if s_["trades"] >= 30
                        else "⚠️ عينة صغيرة" if s_["trades"] > 0
                        else "لا صفقات"
                    )
                    multi_rows.append({
                        "الأصل": label, "الرمز": tk_clean,
                        "الصفقات": s_["trades"],
                        "نسبة الربح %": round(s_["win_rate"], 1),
                        "الإجمالي %": round(s_["total"], 1),
                        "متوسط R": round(s_["avg_r"], 2),
                        "PF": round(s_["pf"], 2) if np.isfinite(s_["pf"]) else "∞",
                        "أقصى تراجع %": round(s_["max_dd"], 1),
                        "Buy&Hold %": round(bh, 1),
                        "الحالة": status,
                    })
                except Exception as e:
                    multi_rows.append({"الأصل": label, "الرمز": tk_clean,
                                        "الحالة": f"استثناء: {type(e).__name__}"})
            st.session_state["multi_backtest"] = multi_rows
        except Exception as e:
            st.error(f"خطأ في المقارنة: {e}")

if st.session_state.get("multi_backtest"):
    st.dataframe(pd.DataFrame(st.session_state["multi_backtest"]),
                 use_container_width=True, hide_index=True)
    st.caption("قارن الإجمالي % مع Buy&Hold %.")