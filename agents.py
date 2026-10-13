"""
agents.py — نسخة خفيفة: بدون news_engine لتسريع التحليل.
الوكلاء الفني والمخاطر يعملان. الأخبار معطّلة مؤقتًا.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import streamlit as st
from google import genai
from google.genai import types
from groq import Groq

from constants import GEMINI_FALLBACK, GROQ_FALLBACK, TF_D1
from risk_engine import P, as_list, clean_price_levels, trade_levels, truthy

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GEMINI_API_KEY = GROQ_API_KEY = ""


def extract_json(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", str(text), flags=re.I).replace("```", "").strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def validate_tech_json(obj, signal):
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).upper()
    if verdict not in {"BUY", "SELL", "WAIT"}:
        return None
    try:
        conf = float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        return None
    if not 0 <= conf <= 100:
        return None
    supports = truthy(obj.get("supports_signal"))
    if signal in ("BUY", "SELL") and supports and verdict != signal:
        supports = False
    obj["confidence"] = conf
    obj["supports_signal"] = supports
    obj["verdict"] = verdict
    obj["reasons"] = [str(x) for x in as_list(obj.get("reasons"))][:3]
    kl = obj.get("key_levels") if isinstance(obj.get("key_levels"), dict) else {}
    obj["key_levels"] = {
        "support": clean_price_levels(kl.get("support")),
        "resistance": clean_price_levels(kl.get("resistance")),
    }
    return obj


def validate_risk_json(obj):
    if not isinstance(obj, dict):
        return None
    decision = str(obj.get("decision", "")).upper()
    if decision not in {"APPROVE", "REDUCE", "VETO"}:
        return None
    try:
        score = float(obj.get("risk_score", 0))
    except (TypeError, ValueError):
        return None
    if not 1 <= score <= 10:
        return None
    obj["decision"] = decision
    obj["risk_score"] = score
    obj["concerns"] = [str(x) for x in as_list(obj.get("concerns"))][:3]
    return obj


def rank_gemini_models(names):
    skip = ("image", "live", "audio", "tts", "embedding", "aqa", "vision",
            "robotics", "computer", "native", "imagen", "veo", "lite")
    def ver(n):
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
    cands = [n for n in names if n.startswith("gemini") and ("flash" in n or "pro" in n)
             and not any(x in n for x in skip) and ver(n)[0] >= 3]
    key = lambda n: (0 if "flash" in n else 1, -ver(n)[0], -ver(n)[1])
    stable = sorted([n for n in cands if not any(x in n for x in ("preview", "exp"))], key=key)
    preview = sorted([n for n in cands if n not in stable], key=key)
    ranked = stable + preview
    out = ranked[:4] + [f for f in GEMINI_FALLBACK if f not in ranked[:4]]
    return out[:5]


def rank_groq_models(ids):
    skip = ("whisper", "tts", "guard", "embedding", "vision", "orpheus",
            "playai", "compound", "distil")
    chat = [i for i in ids if not any(x in i.lower() for x in skip)]
    pri = sorted([i for i in chat if "gpt-oss" in i.lower()],
                 key=lambda i: 0 if "120b" in i else 1)
    rest = [i for i in chat if i not in pri and any(k in i.lower() for k in
                                                    ("llama", "qwen", "kimi", "deepseek"))]
    out = pri + rest
    out += [f for f in GROQ_FALLBACK if f not in out]
    return out[:3]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_gemini():
    client = genai.Client(api_key=GEMINI_API_KEY)
    return [m.name.replace("models/", "") for m in client.models.list()]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_groq():
    client = Groq(api_key=GROQ_API_KEY)
    return [m.id for m in client.models.list().data]


def get_gemini_models():
    try:
        return rank_gemini_models(_list_gemini())
    except Exception:
        return list(GEMINI_FALLBACK)


def get_groq_models():
    try:
        return rank_groq_models(_list_groq())
    except Exception:
        return list(GROQ_FALLBACK)


def _gemini_client():
    try:
        return genai.Client(api_key=GEMINI_API_KEY,
                            http_options=types.HttpOptions(timeout=30000))
    except Exception:
        return genai.Client(api_key=GEMINI_API_KEY)


def _groq_client():
    try:
        return Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=1)
    except TypeError:
        return Groq(api_key=GROQ_API_KEY)


def call_groq(system, user, temperature=0.2, models=None, budget=45):
    models = models or GROQ_FALLBACK
    t0, last_err = time.time(), ""
    for model_name in models:
        if time.time() - t0 > budget:
            last_err = last_err or "انتهت المهلة"
            break
        try:
            c = _groq_client().chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return c.choices[0].message.content or "", "", model_name
        except Exception as e:
            low = str(e).lower()
            last_err = f"{model_name}: {str(e)[:140]}"
            if "401" in low or "invalid_api_key" in low or "invalid api key" in low:
                return "", "Groq: مفتاح API غير صالح.", ""
    return "", f"Groq: {last_err}", ""


def call_gemini(prompt, search=False, models=None, budget=45):
    models = models or GEMINI_FALLBACK
    t0, err = time.time(), ""
    try:
        client = _gemini_client()
        cfg = (types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]) if search else None)
        for model_name in models:
            if time.time() - t0 > budget:
                err = err or "انتهت المهلة"
                break
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=cfg)
                    return (resp.text or ""), [], "", model_name
                except Exception as e:
                    low = str(e).lower()
                    err = f"{model_name}: {str(e)[:140]}"
                    if any(k in low for k in ("api key not valid", "api_key_invalid", "unauthenticated")):
                        return "", [], "Gemini: مفتاح API غير صالح.", ""
                    if attempt == 0 and any(k in low for k in ("503", "unavailable", "overloaded")):
                        time.sleep(2)
                        continue
                    break
    except Exception as e:
        err = f"Gemini: {str(e)[:140]}"
    return "", [], err, ""


def swing_levels(df, window=5, n=3):
    d = df.tail(150)
    if len(d) < 2 * window + 1:
        return [], []
    hi, lo = d["High"], d["Low"]
    k = 2 * window + 1
    roll_max = hi.rolling(k, center=True, min_periods=k).max()
    roll_min = lo.rolling(k, center=True, min_periods=k).min()
    highs = hi[hi == roll_max].tolist()
    lows = lo[lo == roll_min].tolist()
    uniq = lambda xs: list(dict.fromkeys(xs))[-n:]
    return uniq(highs), uniq(lows)


def _safe_float(x):
    try:
        v = float(x)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def build_context(t, df, tf, bias, regime, adx, used, signal, reason, checks):
    latest = df.iloc[-1]
    price, atr = float(latest["Close"]), float(latest["ATR14"])
    dec = 2 if price >= 10 else 5
    def r(x):
        v = _safe_float(x)
        return "n/a" if v is None else round(v, dec)
    def pct(x):
        v = _safe_float(x)
        return "n/a" if v is None else f"{v:+.2f}%"
    def chg(n):
        if len(df) <= n:
            return None
        base = _safe_float(df["Close"].iloc[-1 - n])
        return None if base in (None, 0) else (price / base - 1) * 100
    highs, lows = swing_levels(df)
    candles = df.tail(8)[["Open", "High", "Low", "Close"]].round(dec).to_string()
    candle_kind = "daily" if tf == TF_D1 else "4-hour"
    atr_pct = pct(None if price == 0 else atr / price * 100)
    ema200_v = _safe_float(latest["EMA200"])
    from_ema200 = ("n/a" if (ema200_v is None or ema200_v == 0)
                   else pct((price / ema200_v - 1) * 100))
    rsi_v = _safe_float(latest["RSI14"])
    rsi_s = "n/a" if rsi_v is None else f"{rsi_v:.1f}"
    chg5, chg20 = pct(chg(5)), pct(chg(20))
    return "\n".join([
        f"Asset (Yahoo Finance symbol): {t}",
        f"Timeframe: {tf} (each candle = {candle_kind})",
        f"Daily trend filter (daily close vs daily EMA200): {bias or 'not used'}",
        f"Market regime: {regime}, ADX14 = {adx:.1f}",
        f"Last close: {r(price)} | ATR14: {r(atr)} ({atr_pct} of price)",
        f"EMA20 {r(latest['EMA20'])} | EMA50 {r(latest['EMA50'])} | "
        f"EMA200 {r(latest['EMA200'])} (price is {from_ema200} from EMA200)",
        f"RSI14: {rsi_s}",
        f"20-bar high / low: {r(latest['Donchian_Upper'])} / {r(latest['Donchian_Lower'])}",
        f"Recent swing highs (resistance): {[r(x) for x in highs]}",
        f"Recent swing lows (support): {[r(x) for x in lows]}",
        f"Change: 5 bars {chg5} | 20 bars {chg20}",
        "Last 8 candles:",
        candles,
        f"Strategy used: {used}",
        f"Technical signal: {signal} — {reason}",
        "All strategies checked: " + "; ".join(
            f"{n.split(' (')[0]}={sg}" for n, sg, _ in checks),
    ])


NEWS_NAMES = {
    "GC=F": "gold price", "SI=F": "silver price", "CL=F": "crude oil price",
    "NG=F": "natural gas price", "BTC-USD": "bitcoin", "ETH-USD": "ethereum",
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "DX-Y.NYB": "US dollar index DXY",
}


def news_queries(t):
    u = t.strip().upper()
    if u in NEWS_NAMES:
        return NEWS_NAMES[u], "Fed OR FOMC OR CPI OR payrolls OR OPEC OR ECB OR tariffs"
    if u.endswith("=X") and len(u) == 8:
        return f"{u[:3]}/{u[3:6]} forex", "Fed OR ECB OR CPI OR payrolls OR central bank"
    return f"{u} stock", f"{u} earnings OR guidance OR downgrade OR upgrade"


def news_query_list(t):
    q1, q2 = news_queries(t)
    trusted_sites = "(site:reuters.com OR site:cnbc.com OR site:bloomberg.com OR site:wsj.com OR site:ft.com)"
    return [(q1, 3, 10), (q2, 2, 6), (f"{q1} {trusted_sites}", 3, 8)]


# ============================================================
# الأخبار معطّلة مؤقتًا للسرعة
# ============================================================
def news_agent_full(t, signal, g_models, q_models):
    return {
        "raw": "", "news": None, "sources": [],
        "err": "طبقة الأخبار معطّلة مؤقتًا للسرعة",
        "model": "", "bundle": None,
    }


def news_agent(t, signal, g_models, q_models):
    r = news_agent_full(t, signal, g_models, q_models)
    return r["raw"], r["sources"], r["err"], r["model"]


def fetch_news(t):
    return [], "معطّلة مؤقتًا"


TECH_SYSTEM = (
    "You are a senior technical analyst. Judge ONLY from the data provided. "
    "Be sceptical: if the evidence is mixed, say WAIT. Reply with ONE JSON object only "
    "— no markdown. Text values must be in Arabic."
)

RISK_SYSTEM = (
    "You are a strict risk manager. Look for reasons NOT to take the trade, "
    "but be fair. Judge ONLY from the data provided. Reply with ONE JSON object only "
    "— no markdown. Text values must be in Arabic."
)


def run_agents(ctx, t, signal, price, atr, df=None, mean_rev=False, levels=None):
    has_signal = signal in ("BUY", "SELL")
    tech_user = ctx + """

Return this JSON:
{"verdict":"BUY"|"SELL"|"WAIT","confidence":0-100,"supports_signal":true|false,
 "market_context":"<2 sentences>","key_levels":{"support":[numbers],"resistance":[numbers]},
 "reasons":["<max 3 short reasons>"],"invalidation":"<price condition that cancels the idea>"}
"supports_signal" is true only if you agree with the direction of the technical signal right now."""

    g_models, q_models = get_gemini_models(), get_groq_models()
    with ThreadPoolExecutor(max_workers=1) as ex:
        f_tech = ex.submit(call_groq, TECH_SYSTEM, tech_user, 0.2, q_models)
        try:
            tech_raw, tech_err, tech_model = f_tech.result(timeout=60)
        except Exception as e:
            tech_raw, tech_err, tech_model = "", f"انتهت المهلة: {str(e)[:80]}", ""

    tech = validate_tech_json(extract_json(tech_raw), signal)
    tech_err = tech_err or ("" if tech else "تعذّر قراءة/التحقق من رد الوكيل الفني")

    news = None
    news_raw, news_sources, news_err, news_model = "", [], "معطّل مؤقتًا", ""

    risk, risk_raw, risk_err, risk_model, skipped = None, "", "", "", not has_signal
    if has_signal:
        if levels is not None:
            sl, tp1, tp2 = levels["sl"], levels["tp1"], levels["tp2"]
        else:
            sl, tp1, tp2 = trade_levels(signal, price, atr, df, mean_rev)
        risk_user = f"""{ctx}

Proposed trade: {signal} at {P(price)} | SL {P(sl)} | TP1 {P(tp1)} | TP2 {P(tp2)}.
Technical: {json.dumps(tech, ensure_ascii=False) if tech else "unavailable"}
News: unavailable

Return this JSON:
{{"decision":"APPROVE"|"REDUCE"|"VETO","risk_score":1-10,"concerns":["<max 3 items>"],
 "comment":"<1-2 sentences>"}}"""
        risk_raw, risk_err, risk_model = call_groq(
            RISK_SYSTEM, risk_user, 0.2, q_models)
        risk = validate_risk_json(extract_json(risk_raw))
        risk_err = risk_err or ("" if risk else "تعذّر قراءة/التحقق من رد مدير المخاطر")

    return {
        "tech": tech, "tech_raw": tech_raw, "tech_err": tech_err,
        "news": news, "news_raw": news_raw, "news_err": news_err,
        "news_sources": news_sources,
        "risk": risk, "risk_raw": risk_raw, "risk_err": risk_err,
        "risk_skipped": skipped,
        "tech_model": tech_model, "news_model": news_model, "risk_model": risk_model,
        "news_bundle": None,
    }


def send_telegram(text):
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return None
    if not token or not chat:
        return None
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return "ok" if r.status == 200 else f"فشل الإرسال ({r.status})"
    except Exception as e:
        return f"خطأ تيليجرام: {str(e).replace(str(token), '***')[:120]}"