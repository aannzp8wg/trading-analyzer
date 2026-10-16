"""
constants.py — ثوابت التطبيق المشتركة. لا يحتوي أي منطق.
"""

# نماذج LLM الاحتياطية — يجب أن تكون موجودة فعليًا في كل مزوّد.
GEMINI_FALLBACK = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]
GROQ_FALLBACK = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

DATA_PERIOD = "1y"

ASSETS = {
    "الذهب 🥇 (XAU/USD)": "GC=F",
    "اليورو/دولار 💶 (EUR/USD)": "EURUSD=X",
    "مؤشر الدولار 💵 (DXY)": "DX-Y.NYB",
}

DEFAULT_COST = {
    "GC=F": 0.05,
    "EURUSD=X": 0.02,
    "DX-Y.NYB": 0.03,
}

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

AUTO = "Auto (اختيار تلقائي حسب السوق)"
CONFLUENCE = "Confluence (تلاقي الاستراتيجيات)"
SQUEEZE = "Squeeze Breakout (انفجار بعد الانضغاط)"
TURTLE = "Turtle 55 (اختراق السلاحف)"
RSI2 = "RSI2 Reversion (ارتداد كونورز)"
TREND = "Trend Follow (تتبع الاتجاه)"
BREAKOUT = "Breakout (اختراق)"
PULLBACK = "Trend + Pullback (اتجاه+ارتداد)"
MEAN_REV = "Mean Reversion (عودة للمتوسط)"
GANN = "Gann HiLo (متوسطات القمم والقيعان)"

CONTRARIAN = (MEAN_REV, RSI2)

NEW_STRATEGIES = [CONFLUENCE, SQUEEZE, TURTLE, RSI2, GANN]

STRATEGY_FAMILIES = {
    TREND: "momentum",
    BREAKOUT: "momentum",
    TURTLE: "momentum",
    SQUEEZE: "momentum",
    GANN: "momentum",
    PULLBACK: "pullback",
    MEAN_REV: "reversion",
    RSI2: "reversion",
    CONFLUENCE: "ensemble",
    AUTO: "auto",
}

TF_D1 = "يومي (1D)"
TF_H4 = "4 ساعات (4H)"
TF_DUAL = "مزدوج (اتجاه يومي + دخول 4H)"

SL_ATR = 1.5
TP1_ATR = 1.5
TP2_ATR = 3.0

ALL_STRATEGIES = [AUTO, CONFLUENCE, TREND, BREAKOUT, SQUEEZE,
                  TURTLE, GANN, PULLBACK, RSI2, MEAN_REV]