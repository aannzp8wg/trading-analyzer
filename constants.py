"""
constants.py — ثوابت التطبيق المشتركة. لا يحتوي أي منطق.
"""

# نماذج LLM الاحتياطية
GEMINI_FALLBACK = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]
GROQ_FALLBACK = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# الإعدادات العامة
DATA_PERIOD = "1y"

# الأصول المدعومة
ASSETS = {
    "الذهب 🥇 (XAU/USD)": "GC=F",
    "اليورو/دولار 💶 (EUR/USD)": "EURUSD=X",
    "مؤشر الدولار 💵 (DXY)": "DX-Y.NYB",
}

# التكلفة الافتراضية لكل أصل (سبريد + عمولة %)
DEFAULT_COST = {
    "GC=F": 0.05,
    "EURUSD=X": 0.02,
    "DX-Y.NYB": 0.03,
}

# ملاحظات الارتباط بين الأصول
CORRELATION_NOTES = {
    "GC=F": (
        "الذهب يرتبط عكسيًا بمؤشر الدولار (DXY) غالبًا، "
        "وطرديًا جزئيًا مع اليورو/دولار. إذا كان لديك صفقة "
        "حقيقية مفتوحة على DX-Y.NYB أو EUR/USD، فكّر في هذه "
        "الصفقة كجزء من التعرض الكلي للدولار، لا كصفقة مستقلة."
    ),
    "EURUSD=X": (
        "اليورو/دولار يرتبط عكسيًا بشكل قوي بمؤشر الدولار "
        "(اليورو ~57% من تكوين DXY). صفقة على اليورو/دولار "
        "وصفقة معاكسة على DX-Y.NYB قد تكونان الرهان نفسه مكرَّرًا."
    ),
    "DX-Y.NYB": (
        "مؤشر الدولار يرتبط عكسيًا بالذهب واليورو/دولار غالبًا. "
        "تحقق من أي صفقات حقيقية مفتوحة على هذين الأصلين قبل "
        "إضافة هذه الصفقة."
    ),
}

# أسماء الاستراتيجيات (تُعرَّف هنا لكسر الاستيراد الدائري)
AUTO = "Auto (اختيار تلقائي حسب السوق)"
CONFLUENCE = "Confluence (تلاقي الاستراتيجيات)"
SQUEEZE = "Squeeze Breakout (انفجار بعد الانضغاط)"
TURTLE = "Turtle 55 (اختراق السلاحف)"
RSI2 = "RSI2 Reversion (ارتداد كونورز)"
TREND = "Trend Follow (تتبع الاتجاه)"
BREAKOUT = "Breakout (اختراق)"
PULLBACK = "Trend + Pullback (اتجاه+ارتداد)"
MEAN_REV = "Mean Reversion (عودة للمتوسط)"

CONTRARIAN = (MEAN_REV, RSI2)

NEW_STRATEGIES = [CONFLUENCE, SQUEEZE, TURTLE, RSI2]

# عائلات الاستراتيجيات — لمنع التلاقي الزائف في Confluence
# الاستراتيجيات داخل العائلة نفسها مترابطة، فتأييدها المزدوج دليل
# واحد مضاعف لا دليلان مستقلان. Confluence يعطي كل عائلة صوتًا واحدًا.
STRATEGY_FAMILIES = {
    TREND: "momentum",
    BREAKOUT: "momentum",
    TURTLE: "momentum",
    SQUEEZE: "momentum",
    PULLBACK: "pullback",
    MEAN_REV: "reversion",
    RSI2: "reversion",
    CONFLUENCE: "ensemble",
    AUTO: "auto",
}

# الأطر الزمنية
TF_D1 = "يومي (1D)"
TF_H4 = "4 ساعات (4H)"
TF_DUAL = "مزدوج (اتجاه يومي + دخول 4H)"

# مضاعفات ATR الافتراضية
SL_ATR = 1.5
TP1_ATR = 1.5
TP2_ATR = 3.0