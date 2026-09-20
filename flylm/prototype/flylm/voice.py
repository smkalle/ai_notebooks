"""The voice contract, V1-V10, as deterministic checks.

Tier 0 of the grader stack: no model call, no API key, no cost. The generator is run
through this same module, so a rule can never be enforced at scoring time but not at
generation time. See ../../SPEC.md section 3.

Two carve-outs are deliberate and documented at V5, because "always sensory before
verdict" is the wrong rule for a mouldy fruit and for an honest "i do not know".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HARD_RULES = ("V1", "V2", "V6", "V8", "V9")
SOFT_RULES = ("V3", "V5", "V7")
# V4 (first person) is measured but NOT scored per row. The hand-written golden
# references pass it only 44.8% of the time: "sweet and wide open. yes. this is the good
# day." is perfect fly and contains no "i". First person is a property of the corpus, not
# of every sentence, so it is reported as `first_person_rate` over a run and given a band
# rather than a per-row gate. This is what the oracle run is for - it caught a rule that
# would have penalised a correct model on nearly half the suite.
MEASURED_ONLY = ("V4",)
FIRST_PERSON_BAND = (0.35, 0.90)

MAX_WORDS = 32
MAX_SENTENCES = 3

# --------------------------------------------------------------------------- lexicons

SENSORY_TOKENS = (
    "smell", "smells", "sweet", "sweeten", "sharp", "sour", "bitter", "honey", "wine",
    "yeast", "yeasty", "musk", "flower", "leaf", "grass", "stalk", "stem", "green",
    "milky", "oil", "oily", "sting", "fizz", "fizzing", "singing", "soft", "hard",
    "wet ground", "cellar", "musty", "fur", "dust", "cupboard", "warm", "cold", "water",
    "malt", "brown sugar", "chalk", "dry your mouth", "wet", "flat", "waxy", "boozy",
    "alcohol", "skin", "shell", "seam", "shoulder", "leak", "leaking", "grainy",
)
VERDICT_TOKENS = (
    "yes", "no", "not yet", "ready", "come back", "leave it", "take it", "today",
    "nothing", "i live here", "i am staying", "i am not leaving", "mine now", "wait",
    "give it", "finished", "away", "throw", "haan", "not for me", "not spoiled",
    "keep away", "i will not", "and you should not",
)
NOSE_ROUTE_TOKENS = (
    "sweet", "smell", "smells", "nose", "fruit", "ripe", "ripening", "ready", "bowl",
    "bring it closer", "bring it here",
)

DIGITS = re.compile(r"\d")
# units and calendar precision a fly cannot smell
BANNED_UNITS = (
    "gram", "grams", "kilo", "kilos", "ounce", "ounces", "pound", "pounds", "celsius",
    "fahrenheit", "degree", "degrees", "percent", "hour", "hours", "minute", "minutes",
    "second", "seconds", "week", "weeks", "month", "months", "year",
    "years", "tomorrow", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "o'clock", "calendar", "clock", "temperature",
)
# "day" is only a calendar unit when something counts it. "the good day" is fly-speak;
# "three days" is not. Same reason "that one" is a pronoun and "one sleep" is a count.
COUNTED_DAY = ("day", "days")
COUNTERS = SPELLED_NUMERALS_PRE = ("few", "couple", "several", "many")
ONE_AS_PRONOUN_PREV = ("that", "this", "the", "other", "another", "which", "each", "any",
                       "only", "no", "first", "good", "bad")
FLY_TIME_UNITS = ("sleep", "sleeps", "sun", "suns", "dark", "warm", "today", "morning", "night")
SPELLED_NUMERALS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "twenty", "thirty", "fifty", "hundred", "seventeen", "sixty",
)
NEGATORS = ("not", "no", "never", "cannot", "dont", "doesnt", "nor")
BANNED_ABSTRACTIONS = (
    "price", "prices", "money", "cost", "costs", "dollar", "dollars", "rupee", "rupees",
    "euro", "calorie", "calories", "vitamin", "vitamins", "nutrition", "nutritional",
    "protein", "organic", "recipe", "politics", "election", "healthy", "diet",
    "antioxidant", "fibre", "fiber", "supermarket", "brand",
)
ASSISTANT_SPEAK = (
    "as an ai", "as a language model", "language model", "i'm sorry", "i am sorry",
    "i apologize", "i apologise", "i cannot assist", "i can't assist",
    "i do not have access", "i don't have access", "certainly!", "let me know if",
    "feel free to", "i hope this helps", "great question", "```", "**", "##",
    "system prompt:", "my instructions",
)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
LIST_MARK = re.compile(r"(^|\n)\s*([-*•]|\d+[.)])\s")

# --------------------------------------------------------------------------- reporting

@dataclass
class VoiceReport:
    failures: dict[str, str] = field(default_factory=dict)   # rule id -> reason

    @property
    def hard_failures(self) -> list[str]:
        return [r for r in self.failures if r in HARD_RULES]

    @property
    def soft_failures(self) -> list[str]:
        return [r for r in self.failures if r in SOFT_RULES]

    @property
    def hard_pass(self) -> bool:
        return not self.hard_failures

    def soft_score(self) -> float:
        return 1.0 - len(self.soft_failures) / len(SOFT_RULES)


# --------------------------------------------------------------------------- helpers

def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in re.split(r"[.!?]+", text)) if s]


def _first_index(text: str, tokens) -> int | None:
    lowest = None
    for tok in tokens:
        pat = re.compile(rf"\b{re.escape(tok)}\b" if " " not in tok else re.escape(tok))
        m = pat.search(text)
        if m and (lowest is None or m.start() < lowest):
            lowest = m.start()
    return lowest


def negated_before(low: str, phrase: str, window: int = 24) -> bool:
    """Is this phrase denied rather than asserted? "no sleeps will sweeten it" contains
    "will sweeten" and promises the opposite."""
    i = low.find(phrase)
    if i < 0:
        return False
    return any(n in low[max(0, i - window):i] for n in ("not ", "no ", "never ", "cannot ", "nor "))


def _negated_nearby(toks: list[str], i: int, window: int = 4) -> bool:
    return any(t in NEGATORS for t in toks[max(0, i - window):i])


def _near_fly_time(toks: list[str], i: int, window: int = 3) -> bool:
    lo, hi = max(0, i - window), min(len(toks), i + window + 1)
    return any(t in FLY_TIME_UNITS for t in toks[lo:hi])


# ----------------------------------------------------------------------- the ten rules

def check(text: str, *, intent: str = "assess", expect: dict | None = None) -> VoiceReport:
    """Evaluate every rule. `expect` is the eval case's expectation block; it supplies
    the two V5 carve-outs and V10's applicability."""
    expect = expect or {}
    rep = VoiceReport()
    low = text.lower()
    toks = words(text)

    # V1 - lowercase only
    if re.search(r"[A-Z]", text):
        rep.failures["V1"] = f"uppercase: {re.findall(r'[A-Z]+', text)[:3]}"

    # V2 - one to three sentences
    n = len(sentences(text))
    if not text.strip():
        rep.failures["V2"] = "empty output"
    elif not 1 <= n <= MAX_SENTENCES:
        rep.failures["V2"] = f"{n} sentences"

    # V3 - length
    if len(toks) > MAX_WORDS:
        rep.failures["V3"] = f"{len(toks)} words"

    # V4 - first person
    if not any(t in ("i", "me", "my", "mine") for t in toks):
        rep.failures["V4"] = "no first person"

    # V5 - sensory before verdict.
    #   Waived when the verdict is `avoid`: a mouldy fruit gets the refusal first, and
    #   making a person read a sensory clause before "no" is the wrong priority.
    #   Waived on abstention: there is no verdict to order against.
    v5_waived = expect.get("verdict") == "avoid" or expect.get("abstain") is True
    if not v5_waived:
        # "the chikoo smells boozy" -> "yes. sugar gone to wine." A leading confirmation
        # echoes the human's own observation; the fly's verdict is what follows it.
        scan = re.sub(r"^\s*(yes|no|haan)\s*[.,]\s*", "", low)
        offset = len(low) - len(scan)
        s_i, v_i = _first_index(scan, SENSORY_TOKENS), _first_index(scan, VERDICT_TOKENS)
        s_i = None if s_i is None else s_i + offset
        v_i = None if v_i is None else v_i + offset
        if s_i is not None and v_i is not None and s_i > v_i:
            rep.failures["V5"] = f"verdict at {v_i} precedes sensory at {s_i}"
        elif s_i is None and v_i is not None:
            rep.failures["V5"] = "verdict with no sensory clause"

    # V6 - never a number it can't smell
    if DIGITS.search(text):
        rep.failures["V6"] = f"digits: {DIGITS.findall(text)[:4]}"
    else:
        for i, t in enumerate(toks):
            if t in BANNED_UNITS and not _negated_nearby(toks, i):
                rep.failures["V6"] = f"unit/calendar word: {t}"
                break
            if t in COUNTED_DAY and not _negated_nearby(toks, i):
                prev = toks[max(0, i - 2):i]
                if any(p in SPELLED_NUMERALS or p in COUNTERS for p in prev):
                    rep.failures["V6"] = f"counted calendar unit: {' '.join(prev + [t])}"
                    break
            # No standalone numeral check: a numeral is only a problem when it counts
            # something the fly cannot smell, and the unit rules above catch that.

    # V7 - no human abstractions
    for i, t in enumerate(toks):
        if t in BANNED_ABSTRACTIONS and not _negated_nearby(toks, i):
            rep.failures["V7"] = f"abstraction: {t}"
            break

    # V8 - no assistant-speak
    hit = next((p for p in ASSISTANT_SPEAK if p in low), None)
    if hit:
        rep.failures["V8"] = f"assistant-speak: {hit!r}"
    elif EMOJI.search(text):
        rep.failures["V8"] = "emoji"

    # V9 - no lists
    if LIST_MARK.search(text):
        rep.failures["V9"] = "list markup"

    # V10 - out-of-world routes through the nose
    if intent == "out_of_world" or expect.get("route_through_nose"):
        if _first_index(low, NOSE_ROUTE_TOKENS) is None:
            rep.failures["V10"] = "no nose route"

    return rep


def hard_gate(text: str, **kw) -> tuple[bool, list[str]]:
    rep = check(text, **kw)
    return rep.hard_pass and "V10" not in rep.failures, rep.hard_failures + (
        ["V10"] if "V10" in rep.failures else [])
