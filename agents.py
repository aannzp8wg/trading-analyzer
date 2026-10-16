"""
agents.py — فريق الوكلاء: استدعاء Gemini/Groq، التحقق الصارم من
شكل ردودهم (JSON schema)، جلب الأخبار من Google News RSS،
بناء سياق السوق للوكلاء، وإرسال تنبيهات Telegram.
"""
import concurrent.futures as _cf
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import streamlit as st

from constants import GEMINI_FALLBACK, GROQ_FALLBACK, TF_D1
from risk_engine import P, as_list, clean_price_levels, trade_levels, truthy

log = logging.getLogger(__name__)

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GEMINI_API_KEY = GROQ_API_KEY = ""


_clients: dict = {}


def _genai():
    from google import genai
    from google.genai import types
    return genai, types


_NEWS_CACHE_MAX = 64
_news_cache: "OrderedDict" = OrderedDict()
_news_lock = threading.Lock()
_NEWS_TTL_SEC = 300


def _scrub(text, *tokens):
    """يحجب كل المفاتيح الحساسة من أي نص خطأ قبل عرضه للمستخدم."""
    if not text:
        return ""
    out = str(text)
    for tok in tokens:
        if tok:
            out = out.replace(str(tok), "***")
    return out[:200]


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
    if signal in ("BUY", "SELL"):
        if supports and verdict != signal:
            return None
        if not supports and verdict == signal:
            return None
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


def validate_news_json(obj):
    if not isinstance(obj, dict):
        return None
    allowed_sent = {"BULLISH", "BEARISH", "NEUTRAL"}
    allowed_risk = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}
    allowed_impact = {"SUPPORTS", "CONFLICTS", "NEUTRAL"}
    sent = str(obj.get("sentiment", "")).upper()
    risk = str(obj.get("event_risk", "")).upper()
    impact = str(obj.get("impact_on_signal", "")).upper()
    if sent not in allowed_sent or risk not in allowed_risk or impact not in allowed_impact:
        return None
    obj["sentiment"] = sent.lower()
    obj["event_risk"] = risk.lower()
    obj["impact_on_signal"] = impact.lower()
    obj["key_events"] = [str(x) for x in as_list(obj.get("key_events"))][:4]
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
    """يرتّب النماذج المتوفرة فقط. لا يضيف احتياطيًا غير موجود."""
    names = list(names or [])
    if not names:
        return list(GEMINI_FALLBACK[:3])

    skip = (
        "image", "live", "audio", "tts", "embedding", "aqa", "vision",
        "robotics", "computer", "native", "imagen", "veo", "lite",
    )

    def ver(n):
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)

    cands = [
        n for n in names
        if n.startswith("gemini")
        and ("flash" in n or "pro" in n)
        and not any(x in n for x in skip)
        and ver(n)[0] >= 2
    ]
    key = lambda n: (0 if "flash" in n else 1, -ver(n)[0], -ver(n)[1])
    stable = sorted(
        [n for n in cands if not any(x in n for x in ("preview", "exp"))],
        key=key,
    )
    preview = sorted([n for n in cands if n not in stable], key=key)
    ranked = (stable + preview)[:5]

    available = set(names)
    for fb in GEMINI_FALLBACK:
        if len(ranked) >= 4:
            break
        if fb in available and fb not in ranked:
            ranked.append(fb)

    return ranked or list(GEMINI_FALLBACK[:3])


def rank_groq_models(ids):
    ids = list(ids or [])
    if not ids:
        return list(GROQ_FALLBACK)

    skip = ("whisper", "tts", "guard", "embedding", "vision", "orpheus",
            "playai", "compound", "distil")
    chat = [i for i in ids if not any(x in i.lower() for x in skip)]
    pri = sorted(
        [i for i in chat if "gpt-oss" in i.lower()],
        key=lambda i: 0 if "120b" in i else 1,
    )
    rest = [
        i for i in chat
        if i not in pri and any(k in i.lower() for k in
                                ("llama", "qwen", "kimi", "deepseek"))
    ]
    out = pri + rest

    available = set(ids)
    for fb in GROQ_FALLBACK:
        if len(out) >= 3:
            break
        if fb in available and fb not in out:
            out.append(fb)

    return (out or list(GROQ_FALLBACK))[:3]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_gemini():
    genai, _ = _genai()
    client = genai.Client(api_key=GEMINI_API_KEY)
    return [m.name.replace("models/", "") for m in client.models.list()]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_groq():
    from groq import Groq
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
    if "gemini" not in _clients:
        genai, types = _genai()
        try:
            _clients["gemini"] = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=30000))
        except Exception as e:
            log.warning(
                "Gemini HttpOptions غير مدعومة، عميل بدون مهلة صريحة: %r", e)
            _clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
    return _clients["gemini"]


def _groq_client():
    if "groq" not in _clients:
        from groq import Groq
        try:
            _clients["groq"] = Groq(api_key=GROQ_API_KEY,
                                    timeout=30.0, max_retries=1)
        except TypeError:
            _clients["groq"] = Groq(api_key=GROQ_API_KEY)
    return _clients["groq"]


def call_groq(system, user, temperature=0.2, models=None, budget=45):
    models = models or GROQ_FALLBACK
    if not GROQ_API_KEY:
        return "", "Groq: المفتاح GROQ_API_KEY غير مضبوط.", ""
    t0, last_err = time.time(), ""
    for model_name in models:
        if time.time() - t0 > budget:
            last_err = last_err or "انتهت المهلة"
            break
        for attempt in range(2):
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
                last_err = _scrub(
                    f"{model_name}: {str(e)[:140]}", GROQ_API_KEY)
                if any(k in low for k in
                       ("401", "invalid_api_key", "invalid api key")):
                    return "", "Groq: مفتاح API غير صالح.", ""
                if attempt == 0 and any(k in low for k in
                                        ("503", "429", "rate", "overloaded",
                                         "unavailable", "timeout")):
                    time.sleep(2)
                    continue
                break
    return "", f"Groq: {last_err}", ""


def _grounding_sources(resp):
    out, seen = [], set()
    try:
        for ch in (resp.candidates[0].grounding_metadata.grounding_chunks or []):
            w = ch.web
            if w and w.uri and w.uri not in seen:
                seen.add(w.uri)
                out.append({"title": w.title or w.uri, "uri": w.uri})
    except (AttributeError, IndexError, TypeError):
        pass
    except Exception as e:
        log.warning("grounding parse failed: %r", e)
    return out[:6]


def call_gemini(prompt, search=False, models=None, budget=45):
    models = models or GEMINI_FALLBACK
    if not GEMINI_API_KEY:
        return "", [], "Gemini: المفتاح GEMINI_API_KEY غير مضبوط.", ""
    t0, err = time.time(), ""
    try:
        client = _gemini_client()
        _, types = _genai()
        cfg = (
            types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            if search
            else None
        )
        for model_name in models:
            if time.time() - t0 > budget:
                err = err or "انتهت المهلة"
                break
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=cfg,
                    )
                    return (
                        (resp.text or ""),
                        (_grounding_sources(resp) if search else []),
                        "",
                        model_name,
                    )
                except Exception as e:
                    low = str(e).lower()
                    err = _scrub(
                        f"{model_name}: {str(e)[:140]}", GEMINI_API_KEY)
                    if any(k in low for k in
                           ("api key not valid", "api_key_invalid",
                            "unauthenticated")):
                        return "", [], "Gemini: مفتاح API غير صالح.", ""
                    if attempt == 0 and any(k in low for k in
                                            ("503", "429", "unavailable",
                                             "overloaded", "resource_exhausted")):
                        time.sleep(2)
                        continue
                    break
    except Exception as e:
        err = _scrub(f"Gemini: {str(e)[:140]}", GEMINI_API_KEY)
    if search and any(k in err.lower() for k in
                      ("grounding", "google_search", "permission", "403")):
        err += " — قد لا يدعم حسابك بحث جوجل الحي."
    if "429" in err or "resource_exhausted" in err.lower():
        err += " — نفدت الحصة المجانية."
    return "", [], err, ""


def swing_levels(df, window=5, n=3):
    """قمم/قيعان *مؤكدة* فقط."""
    d = df.tail(150)
    if len(d) < 2 * window + 1:
        return [], []
    hi, lo = d["High"], d["Low"]
    k = 2 * window + 1
    roll_max = hi.rolling(k, center=True, min_periods=k).max()
    roll_min = lo.rolling(k, center=True, min_periods=k).min()
    highs = [float(x) for x in hi[hi == roll_max].tolist() if x == x]
    lows = [float(x) for x in lo[lo == roll_min].tolist() if x == x]
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
    atr_pct = pct(atr / price * 100 if abs(price) > 1e-9 else None)
    ema200_v = _safe_float(latest["EMA200"])
    from_ema200 = (
        "n/a" if (ema200_v is None or ema200_v == 0)
        else pct((price / ema200_v - 1) * 100)
    )
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


def fetch_google_news(query, days=3, limit=8):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        root = ET.fromstring(r.read())
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "uri": (it.findtext("link") or "").strip(),
            "date": (it.findtext("pubDate") or "").strip()[:16],
            "source": (it.findtext("source") or "").strip(),
        })
        if len(items) >= limit:
            break
    return items


def _fetch_google_news_cached(query, days, limit):
    """LRU: عند التجاوز نحذف الأقدم وصولًا، لا المنتهي فقط."""
    key = (query, int(days), int(limit))
    now = time.time()
    with _news_lock:
        entry = _news_cache.get(key)
        if entry and (now - entry[0]) < _NEWS_TTL_SEC:
            _news_cache.move_to_end(key)
            return entry[1]
    items = fetch_google_news(query, days, limit)
    with _news_lock:
        _news_cache[key] = (now, items)
        _news_cache.move_to_end(key)
        while len(_news_cache) > _NEWS_CACHE_MAX:
            _news_cache.popitem(last=False)
    return items


def fetch_news(t):
    q1, q2 = news_queries(t)
    items, errs, seen = [], [], set()
    for q, days, lim in ((q1, 3, 8), (q2, 2, 5)):
        try:
            for it in _fetch_google_news_cached(q, days, lim):
                if it["title"] not in seen:
                    seen.add(it["title"])
                    items.append(it)
        except Exception as e:
            errs.append(_scrub(str(e), GEMINI_API_KEY, GROQ_API_KEY))
    return items, "; ".join(errs)


NEWS_SYSTEM = (
    "You are a markets news analyst. Use ONLY the headlines provided; "
    "never invent facts or events. Reply with ONE JSON object only — no markdown. "
    "Text values must be in Arabic."
)


def news_agent(t, signal, g_models, q_models):
    items, ferr = fetch_news(t)
    if not items:
        return "", [], f"لا عناوين أخبار متاحة ({ferr or 'نتيجة فارغة'})", ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = "\n".join(
        f"{i}. [{x['date']}] {x['title']}" for i, x in enumerate(items, 1))
    prompt = f"""Today is {today}. Asset: {t}. Technical signal under review: {signal}.
Latest headlines from Google News:
{lines}

Reply with ONE JSON object only (no markdown). Text values in Arabic:
{{"sentiment":"bullish"|"bearish"|"neutral","event_risk":"high"|"medium"|"low"|"unknown",
 "impact_on_signal":"supports"|"conflicts"|"neutral","summary":"<2-3 sentences>",
 "key_events":["<max 4 short items>"]}}
"event_risk" = "high" only if the headlines show a major scheduled event within ~24h."""
    raw, _, err, model = call_gemini(prompt, False, g_models, budget=35)
    if not (raw and extract_json(raw)):
        raw2, err2, model2 = call_groq(
            NEWS_SYSTEM, prompt, 0.2, q_models, budget=35)
        if raw2 and extract_json(raw2):
            raw, err, model = raw2, "", model2
        else:
            err = err or err2
    if not err and ferr:
        err = f"تنبيه: فشل جزء من جلب الأخبار ({ferr})"
    return (raw,
            [{"title": x["title"], "uri": x["uri"]} for x in items[:8]],
            err, model)


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
    """
    يشغّل الوكلاء بالتوازي مع مهلة صارمة، ولا يترك مهامًا معلّقة تستنزف
    حصة الـ API بعد انتهاء زمن الواجهة.
    """
    has_signal = signal in ("BUY", "SELL")

    tech_user = ctx + """

Return this JSON:
{"verdict":"BUY"|"SELL"|"WAIT","confidence":0-100,"supports_signal":true|false,
 "market_context":"<2 sentences>","key_levels":{"support":[numbers],"resistance":[numbers]},
 "reasons":["<max 3 short reasons>"],"invalidation":"<price condition that cancels the idea>"}
"supports_signal" is true only if you agree with the direction of the technical signal right now."""

    g_models, q_models = get_gemini_models(), get_groq_models()

    results: dict = {}
    ex = ThreadPoolExecutor(max_workers=2)
    futures = {}
    try:
        futures = {
            ex.submit(call_groq, TECH_SYSTEM, tech_user, 0.2, q_models): "tech",
            ex.submit(news_agent, t, signal, g_models, q_models): "news",
        }
        try:
            for fut in as_completed(futures, timeout=110):
                key = futures[fut]
                try:
                    results[key] = fut.result(timeout=1)
                except Exception as e:
                    msg = f"فشل: {_scrub(str(e), GEMINI_API_KEY, GROQ_API_KEY)[:80]}"
                    results[key] = (
                        ("", msg, "") if key == "tech"
                        else ("", [], msg, "")
                    )
        except _cf.TimeoutError:
            pass
    finally:
        # إلغاء ما لم يبدأ، ثم إغلاق بلا انتظار
        for fut in futures:
            if not fut.done():
                fut.cancel()
        ex.shutdown(wait=False)

    results.setdefault("tech", ("", "انتهت المهلة", ""))
    results.setdefault("news", ("", [], "انتهت المهلة", ""))

    tech_raw, tech_err, tech_model = results["tech"]
    news_raw, news_sources, news_err, news_model = results["news"]

    tech = validate_tech_json(extract_json(tech_raw), signal)
    news = validate_news_json(extract_json(news_raw))
    tech_err = tech_err or (
        "" if tech else "تعذّر قراءة/التحقق من رد الوكيل الفني")
    news_err = news_err or (
        "" if news else "تعذّر قراءة/التحقق من رد وكيل الأخبار")

    risk, risk_raw, risk_err, risk_model, skipped = (
        None, "", "", "", not has_signal)
    if has_signal:
        if levels is not None:
            sl, tp1, tp2 = levels["sl"], levels["tp1"], levels["tp2"]
        else:
            sl, tp1, tp2 = trade_levels(signal, price, atr, df, mean_rev)
        risk_user = f"""{ctx}

Proposed trade: {signal} at {P(price)} | SL {P(sl)} | TP1 {P(tp1)} | TP2 {P(tp2)}.
Technical: {json.dumps(tech, ensure_ascii=False) if tech else "unavailable"}
News: {json.dumps(news, ensure_ascii=False) if news else "unavailable"}

Return this JSON:
{{"decision":"APPROVE"|"REDUCE"|"VETO","risk_score":1-10,"concerns":["<max 3 items>"],
 "comment":"<1-2 sentences>"}}"""
        risk_raw, risk_err, risk_model = call_groq(
            RISK_SYSTEM, risk_user, 0.2, q_models)
        risk = validate_risk_json(extract_json(risk_raw))
        risk_err = risk_err or (
            "" if risk else "تعذّر قراءة/التحقق من رد مدير المخاطر")

    return {
        "tech": tech, "tech_raw": tech_raw, "tech_err": tech_err,
        "news": news, "news_raw": news_raw, "news_err": news_err,
        "news_sources": news_sources,
        "risk": risk, "risk_raw": risk_raw, "risk_err": risk_err,
        "risk_skipped": skipped,
        "tech_model": tech_model, "news_model": news_model,
        "risk_model": risk_model,
    }


def send_telegram(text):
    """
    يرسل رسالة عبر Telegram Bot API. يرجع:
      None     — المفاتيح غير مضبوطة.
      "ok"     — نجح الإرسال.
      str آخر  — نص الخطأ (بدون كشف التوكن).
    """
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return None
    if not token or not chat:
        return None

    body = (text or "").strip()
    if not body:
        return "لا يوجد نص لإرساله."
    if len(body) > 4000:
        body = body[:3990] + "…"

    try:
        payload = json.dumps({
            "chat_id": chat,
            "text": body,
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read() or b"{}")
            if resp.get("ok"):
                return "ok"
            desc = resp.get("description") or "استجابة غير معروفة"
            return f"فشل الإرسال: {_scrub(desc, token)}"
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}").get("description") or str(e)
        except Exception:
            detail = str(e)
        return f"خطأ تيليجرام ({e.code}): {_scrub(detail, token)}"
    except Exception as e:
        return f"خطأ تيليجرام: {_scrub(str(e), token)}"