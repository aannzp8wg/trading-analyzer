"""
walkforward.py — اختبار خارج العيّنة (Walk-Forward).

الباكتست العادي يختار "أفضل استراتيجية" ويقيّمها على البيانات نفسها، فتكون النتيجة متفائلة.
هنا في كل طيّة (fold): تُختار الاستراتيجية على نافذة تدريب فقط، ثم تُشغَّل وحدها على نافذة
اختبار تالية لم تُستخدم في الاختيار. تُجمع نتائج نوافذ الاختبار وحدها.

لا تسريب: الإشارات سببية (كل شمعة تعتمد على ما قبلها)، والتدريب والاختبار يُقصّان بالشرائح
فلا تستعمل صفقة التدريب أي سعر من نافذة الاختبار (تُغلَق قسراً عند نهاية الشريحة).
"""
import numpy as np
import pandas as pd

from gold_engine import run_backtest_v2
from risk_engine import backtest_stats

MIN_TRAIN_TRADES = 10     # أقل صفقات تدريب ليُؤخذ تقييم استراتيجية على محمل الجد
MIN_OOS_TRADES = 30       # أقل صفقات خارج العيّنة ليكون الحكم ذا معنى


def _mr_slice(mr, name, lo, hi):
    v = mr.get(name, False) if isinstance(mr, dict) else mr
    arr = np.asarray(v, dtype=bool)
    return bool(arr) if arr.ndim == 0 else arr[lo:hi]


def bootstrap_ci(r, n_boot=2000, seed=7, alpha=0.05):
    """فاصل ثقة (بدون افتراض توزيع) لمتوسط R. حتمي بسبب seed ثابت."""
    r = np.asarray([x for x in r if np.isfinite(x)], dtype=float)
    if len(r) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, len(r), size=(n_boot, len(r)))].mean(axis=1)
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def verdict_for(n_trades, avg_r, ci_low, min_oos=MIN_OOS_TRADES):
    if n_trades < min_oos:
        return "insufficient"            # عيّنة صغيرة: لا حكم
    if np.isfinite(ci_low) and ci_low > 0:
        return "supported"               # حتى الحد الأدنى لفاصل الثقة موجب
    if avg_r > 0:
        return "unproven"                # موجب لكن قد يكون حظاً
    return "failed"


def make_folds(n, start, train_bars, test_bars):
    """نوافذ متدحرجة غير متداخلة في الاختبار: [(tr_lo,tr_hi,te_lo,te_hi), ...]."""
    folds, a = [], start
    while a + train_bars + test_bars <= n:
        folds.append((a, a + train_bars, a + train_bars, a + train_bars + test_bars))
        a += test_bars
    return folds


def walk_forward(df, signals, mean_rev, cost_pct=0.05, max_hold=30, start=200,
                 train_bars=None, test_bars=None, candidates=None,
                 min_train_trades=MIN_TRAIN_TRADES, min_oos_trades=MIN_OOS_TRADES):
    """
    df: بيانات مجهّزة بالمؤشرات | signals: {اسم: Series} | mean_rev: {اسم: bool/مصفوفة}
    يرجع dict: ok، folds، oos_trades، oos_stats، ci، verdict، hindsight، params، error.
    """
    n = len(df)
    usable = n - start
    train_bars = int(train_bars or usable * 0.5)
    test_bars = int(test_bars or usable * 0.125)
    names = [c for c in (candidates or list(signals)) if c in signals]
    if test_bars < 40 or train_bars < 100 or not names:
        return {"ok": False, "error": "البيانات غير كافية لنوافذ تدريب/اختبار مفيدة."}
    folds_idx = make_folds(n, start, train_bars, test_bars)
    if len(folds_idx) < 2:
        return {"ok": False, "error": "أقل من طيّتين ممكنتين؛ زد مدة البيانات."}

    def run(name, lo, hi):
        return run_backtest_v2(df.iloc[lo:hi], signals[name].iloc[lo:hi], cost_pct, max_hold, start=1,
                               mean_rev=_mr_slice(mean_rev, name, lo, hi))

    folds, oos_parts = [], []
    for k, (tl, th, el, eh) in enumerate(folds_idx, 1):
        scored = []
        for name in names:
            st_ = backtest_stats(run(name, tl, th))
            if st_["trades"] >= min_train_trades and st_["avg_r"] > 0:
                scored.append((st_["avg_r"], name, st_))
        row = {"fold": k, "train": f"{df.index[tl]:%Y-%m-%d} → {df.index[th - 1]:%Y-%m-%d}",
               "test": f"{df.index[el]:%Y-%m-%d} → {df.index[eh - 1]:%Y-%m-%d}",
               "test_range": (el, eh), "train_range": (tl, th)}
        if not scored:
            row.update(chosen=None, is_trades=0, is_avg_r=0.0, oos_trades=0, oos_avg_r=0.0)
        else:
            is_r, best, is_st = max(scored, key=lambda x: x[0])
            tr = run(best, el, eh)
            o_st = backtest_stats(tr)
            if not tr.empty:
                tr = tr.assign(fold=k, strategy=best)
                oos_parts.append(tr)
            row.update(chosen=best, is_trades=is_st["trades"], is_avg_r=is_r,
                       oos_trades=o_st["trades"], oos_avg_r=o_st["avg_r"])
        folds.append(row)

    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    stats = backtest_stats(oos)
    ci = bootstrap_ci(oos["r"].to_numpy()) if not oos.empty else (float("nan"), float("nan"))

    chosen_is = [f for f in folds if f["chosen"]]
    w = sum(f["is_trades"] for f in chosen_is)
    is_weighted = (sum(f["is_avg_r"] * f["is_trades"] for f in chosen_is) / w) if w else 0.0

    # "أفضل بأثر رجعي": أفضل استراتيجية على كامل الفترة (هذا ما يعرضه الباكتست العادي = متفائل)
    hind = []
    for name in names:
        st_ = backtest_stats(run(name, start, n))
        if st_["trades"] >= min_train_trades:
            hind.append((st_["avg_r"], name, st_["trades"]))
    hb = max(hind) if hind else None

    return {"ok": True, "folds": folds, "oos_trades": oos, "oos_stats": stats, "ci": ci,
            "verdict": verdict_for(stats["trades"], stats["avg_r"], ci[0], min_oos_trades),
            "is_avg_r": is_weighted,
            "hindsight": {"strategy": hb[1], "avg_r": hb[0], "trades": hb[2]} if hb else None,
            "params": {"train_bars": train_bars, "test_bars": test_bars, "folds": len(folds_idx),
                       "cost": cost_pct, "max_hold": max_hold}}