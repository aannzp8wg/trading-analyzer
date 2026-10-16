def arbitrate(signal, tech, news, risk):
    if signal not in ("BUY", "SELL"):
        return {"action": "انتظر — لا توجد إشارة مناسبة الآن",
                "level": "info", "mult": 0.0, "notes": []}

    notes = []
    if tech is None:
        return {"action": "لا تدخل — الوكيل الفني غير متاح/غير صالح",
                "level": "error", "mult": 0.0,
                "notes": ["لا يتم فتح صفقة عند فشل التحقق من رد الوكيل الفني."]}
    if risk is None:
        return {"action": "لا تدخل — مدير المخاطر غير متاح/غير صالح",
                "level": "error", "mult": 0.0,
                "notes": ["النظام يعمل بمبدأ fail-closed عند غياب تقييم المخاطر."]}

    tech_ok = (
        truthy(tech.get("supports_signal"))
        and str(tech.get("verdict", "")).upper() == signal
    )
    risk_dec = str(risk.get("decision", "")).upper()
    npol = news_policy(news)
    news_conflict = npol["impact"] == "conflicts"
    event_high = npol["event_risk"] == "high"

    if not tech_ok:
        return {"action": "لا تدخل — التأكيد الفني لا يدعم الإشارة",
                "level": "error", "mult": 0.0,
                "notes": [str(tech.get("market_context", "لا يوجد تأكيد فني."))]}
    if risk_dec == "VETO":
        return {"action": "لا تدخل — مدير المخاطر اعترض",
                "level": "error", "mult": 0.0,
                "notes": [str(c) for c in as_list(risk.get("concerns"))]}

    if risk_dec == "REDUCE":
        notes += [f"مدير المخاطر: {c}" for c in as_list(risk.get("concerns"))] \
                 or ["مدير المخاطر يطلب تخفيض الحجم"]
    if news_conflict:
        notes.append("الأخبار تعاكس اتجاه الإشارة")
    if event_high:
        notes.append("يوجد خطر أحداث مرتفع بحسب العناوين المتاحة")
    if news is None:
        notes.append("الأخبار غير متاحة؛ لن تُستخدم كدليل تأكيد")

    # تدرّج الحجم: 1.0 (نظيف) → 0.5 (تحفّظ واحد) → 0.25 (تحفّظان أو أكثر)
    blockers = int(news_conflict) + int(event_high) + int(news is None)
    reduce_penalty = 1 if risk_dec == "REDUCE" else 0
    score = blockers + reduce_penalty

    if score == 0:
        return {"action": "تأكيد — الفني والمخاطر متوافقان",
                "level": "success", "mult": 1.0, "notes": notes}
    if score == 1:
        return {"action": "دخول بحذر — نصف الحجم",
                "level": "warning", "mult": 0.5, "notes": notes}
    return {"action": "دخول حذر جدًا — ربع الحجم",
            "level": "warning", "mult": 0.25, "notes": notes}