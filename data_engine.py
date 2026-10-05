"""
data_engine.py — جلب بيانات Yahoo Finance (يومي و4 ساعات)،
حساب المؤشرات الفنية، واتجاه اليومي (daily bias) مع المحاذاة
الآمنة من تسريب المستقبل (look-ahead bias).
"""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from constants import DATA_PERIOD, TF_D1, TF_H4
from gold_engine import add_extra_indicators


class DataError(Exception):
    """فشل جلب البيانات."""


# أخطاء لا فائدة من إعادة المحاولة عليها
_NON_RETRIABLE = (
    "no data", "delisted", "not found", "invalid",
    "404", "400", "401", "403",
)


def _fetch_yf(ticker, period, interval):
    """جلب Yahoo مع إعادة محاولة ذكية."""
    last = ""
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=True,
            )
            if df is not None and not df.empty:
                return df
            last = "empty"
        except Exception as e:
            last = str(e)

        low = last.lower()
        if last == "empty" or any(k in low for k in _NON_RETRIABLE):
            break
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    if last == "empty":
        try:
            yf.Ticker(ticker).history(
                period=period, interval=interval, raise_errors=True,
            )
        except TypeError:
            pass
        except Exception as e:
            last = str(e)

    low = last.lower()
    if any(k in low for k in ("rate", "too many", "429")):
        raise DataError(
            "Yahoo Finance حجب الطلبات مؤقتاً. "
            "انتظر 5–10 دقائق ثم أعد المحاولة."
        )
    if last == "empty" or any(k in low for k in ("no data", "delisted", "not found")):
        raise DataError(
            f"لا توجد بيانات لـ {ticker} ({interval}). "
            f"تحقق من الرمز أو جرّب إطارًا آخر."
        )
    if any(k in low for k in ("401", "403", "400", "404", "invalid")):
        raise DataError(
            f"Yahoo رفض الطلب لـ {ticker} ({interval}). "
            f"التفاصيل: {last[:120]}"
        )
    raise DataError(f"فشل جلب البيانات ({interval}): {last[:150]}")


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, period=DATA_PERIOD):
    df = _fetch_yf(ticker, period, "1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_data_h4(ticker, period="730d"):
    fallback = {
        "730d": ["730d", "365d", "180d"],
        "365d": ["365d", "180d", "60d"],
        "180d": ["180d", "60d", "30d"],
        "60d": ["60d", "30d"],
        "30d": ["30d"],
    }
    per = period if period in fallback else "730d"
    last_err = None
    df = None
    for candidate in fallback[per]:
        try:
            df = _fetch_yf(ticker, candidate, "1h")
            if df is not None and not df.empty:
                break
        except DataError as e:
            last_err = e
            df = None
    if df is None:
        raise last_err or DataError(f"لا توجد بيانات 1H لـ {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx
    agg = {
        c: f
        for c, f in {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }.items()
        if c in df.columns
    }
    return df.resample("4h", origin="epoch", offset="0h").agg(agg).dropna(
        subset=["Open", "High", "Low", "Close"]
    )


def drop_incomplete_candle(df, hours=None):
    if len(df) <= 1:
        return df
    if hours:
        tz = df.index.tz
        now = (
            pd.Timestamp.now(tz=tz)
            if tz is not None
            else pd.Timestamp.now(tz="UTC").tz_localize(None)
        )
        if df.index[-1] + pd.Timedelta(hours=hours) > now:
            return df.iloc[:-1]
    elif df.index[-1].date() >= datetime.now(timezone.utc).date():
        return df.iloc[:-1]
    return df


def calculate_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Donchian_Upper"] = df["High"].rolling(20).max()
    df["Donchian_Lower"] = df["Low"].rolling(20).min()

    prev_close = df["Close"].shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        / df["ATR14"]
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        / df["ATR14"]
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    return add_extra_indicators(df)


def daily_bias_series(daily_df):
    c, e = daily_df["Close"], daily_df["EMA200"]
    return pd.Series(
        np.select([c > e, c < e], ["UP", "DOWN"], default="NEUTRAL"),
        index=daily_df.index,
    )


def align_bias(df_lower, bias_daily):
    """تُلصق آخر اتجاه يومي *مكتمل* قبل كل شمعة من الإطار الأقل."""
    d = pd.Series(bias_daily.values, index=pd.DatetimeIndex(bias_daily.index))
    if d.index.tz is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    d = d[~d.index.duplicated(keep="last")].sort_index()

    l_idx = pd.DatetimeIndex(df_lower.index)
    l_naive = (
        l_idx.tz_convert("UTC").tz_localize(None)
        if l_idx.tz is not None
        else l_idx
    )

    pos = d.index.searchsorted(l_naive, side="left") - 1
    vals = np.where(pos >= 0, d.iloc[np.maximum(pos, 0)].to_numpy(), "NEUTRAL")
    vals[pos < 0] = "NEUTRAL"
    return pd.Series(vals, index=df_lower.index)


def apply_bias_filter(sigs, bias):
    b = bias.values
    out = {}
    for k, sr in sigs.items():
        v = sr.values.copy()
        v[(v == 1) & (b != "UP")] = 0
        v[(v == -1) & (b != "DOWN")] = 0
        out[k] = pd.Series(v, index=sr.index)
    return out


def load_prepared(ticker, tf, use_closed=True, daily_period=DATA_PERIOD):
    try:
        return _load_prepared(ticker, tf, use_closed, daily_period)
    except DataError as e:
        return None, None, str(e)


def _load_prepared(ticker, tf, use_closed=True, daily_period=DATA_PERIOD):
    if tf == TF_D1:
        raw = load_data(ticker, daily_period)
        if raw.empty:
            return None, None, f"لم يتم العثور على بيانات يومية لـ {ticker}"
        df = drop_incomplete_candle(raw) if use_closed else raw
        return calculate_indicators(df), None, None

    h4_period = {
        "1y": "365d", "2y": "730d", "3y": "730d",
        "5y": "730d", "10y": "730d",
    }.get(daily_period, "730d")

    raw4 = load_data_h4(ticker, h4_period)
    if raw4.empty:
        return None, None, f"لا تتوفر بيانات 4 ساعات لـ {ticker}"
    df4 = drop_incomplete_candle(raw4, 4) if use_closed else raw4
    df4 = calculate_indicators(df4)

    if tf == TF_H4:
        return df4, None, None

    rawd = load_data(ticker, daily_period)
    if rawd.empty:
        return None, None, f"لم يتم العثور على بيانات يومية لـ {ticker}"
    dfd = calculate_indicators(rawd)
    if use_closed:
        dfd = drop_incomplete_candle(dfd)
    return df4, align_bias(df4, daily_bias_series(dfd)), None