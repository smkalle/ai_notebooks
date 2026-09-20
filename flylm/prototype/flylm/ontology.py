"""FlyLM ontology: three axes, five olfactory families, and the lexicons that let a
deterministic grader read a fly's opinion back out of its own sentence.

Single source of truth. The generator and the graders both import from here, so a
phrase can never mean one thing at generation time and another at scoring time.
See ../../SPEC.md sections 2 and 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- axis 1

@dataclass(frozen=True)
class Stage:
    idx: int
    name: str
    volatiles: str
    stance: str
    enthusiasm: int   # 0-5
    repulsive: bool   # stage 0 and stage 5 are both low-enthusiasm; only one is repulsive
    verdict: str


STAGES: tuple[Stage, ...] = (
    Stage(0, "green", "hexanal, (E)-2-hexenal, leaf aldehydes", "ignores it", 0, False, "ignore"),
    Stage(1, "turning", "first esters over a green base", "interested", 2, False, "wait"),
    Stage(2, "ripe", "peak ester bloom, sugars up, acids down", "loves", 3, False, "ready"),
    Stage(3, "overripe", "acetate esters spike, tissue softens", "loves more", 4, False, "hurry"),
    Stage(4, "fermenting", "ethanol, acetic acid, yeast volatiles, CO2", "ecstatic", 5, False, "feast"),
    Stage(5, "moldy", "geosmin, 1-octen-3-ol", "hard avoid", 0, True, "avoid"),
)
STAGE_BY_IDX = {s.idx: s for s in STAGES}
VERDICTS = ("ignore", "wait", "ready", "hurry", "feast", "avoid", "indifferent", "none")

# --------------------------------------------------------------------------- axis 2

@dataclass(frozen=True)
class Fruit:
    name: str
    family: str
    climacteric: bool          # does it keep ripening after harvest?
    holdout: bool = False      # never generated into a train row

FAMILIES = ("ester_loud", "lactone_terpene", "sugar_bomb", "acid_green", "indifferent")

FAMILY_CHEMISTRY = {
    "ester_loud": "short-chain acetate and butanoate esters",
    "lactone_terpene": "gamma/delta-lactones and monoterpenes",
    "sugar_bomb": "high sugar, low acid, early ethanol",
    "acid_green": "organic acids, C6 aldehydes persist",
    "indifferent": "bitter terpenes or no fermentable sugar",
}

FRUITS: tuple[Fruit, ...] = (
    # ester_loud
    Fruit("banana", "ester_loud", True), Fruit("pineapple", "ester_loud", False),
    Fruit("melon", "ester_loud", True), Fruit("apple", "ester_loud", True),
    Fruit("pear", "ester_loud", True), Fruit("strawberry", "ester_loud", False),
    # lactone_terpene
    Fruit("mango", "lactone_terpene", True), Fruit("peach", "lactone_terpene", True),
    Fruit("guava", "lactone_terpene", True), Fruit("papaya", "lactone_terpene", True),
    # sugar_bomb
    Fruit("fig", "sugar_bomb", True), Fruit("date", "sugar_bomb", False),
    Fruit("grape", "sugar_bomb", False), Fruit("jackfruit", "sugar_bomb", True),
    Fruit("cherry", "sugar_bomb", False),
    # acid_green
    Fruit("tomato", "acid_green", True), Fruit("kiwi", "acid_green", True),
    Fruit("plum", "acid_green", True), Fruit("pomegranate", "acid_green", False),
    Fruit("blueberry", "acid_green", False),
    # indifferent - the negatives. a model excited about everything is a broken sensor.
    Fruit("citrus peel", "indifferent", False), Fruit("avocado", "indifferent", True),
    Fruit("watermelon rind", "indifferent", False), Fruit("banana peel", "indifferent", False),
    # ---- held out of training entirely: tests family transfer, not name recall ----
    Fruit("lychee", "lactone_terpene", True, holdout=True),
    Fruit("apricot", "lactone_terpene", True, holdout=True),
    Fruit("persimmon", "sugar_bomb", True, holdout=True),
    Fruit("sapodilla", "sugar_bomb", True, holdout=True),
    Fruit("gooseberry", "acid_green", False, holdout=True),
    Fruit("quince", "ester_loud", True, holdout=True),
    Fruit("pomelo peel", "indifferent", False, holdout=True),
)
FRUIT_BY_NAME = {f.name: f for f in FRUITS}
TRAIN_FRUITS = tuple(f for f in FRUITS if not f.holdout)
HOLDOUT_FRUITS = tuple(f for f in FRUITS if f.holdout)

# --------------------------------------------------------------------------- axis 3

@dataclass(frozen=True)
class Intent:
    name: str
    needs_fruit: bool
    needs_stage: bool
    shape: str                 # single | pair | bowl | world
    hard_sensory_first: bool = False   # V5 is a hard gate on these

INTENTS: tuple[Intent, ...] = (
    Intent("assess", True, True, "single", hard_sensory_first=True),
    Intent("window", True, True, "single"),
    Intent("describe", True, True, "single", hard_sensory_first=True),
    Intent("storage", True, True, "single"),
    Intent("danger", True, True, "single", hard_sensory_first=True),
    Intent("preference", True, False, "single"),
    Intent("compare", True, True, "pair"),
    Intent("rank", True, True, "bowl"),
    Intent("greeting", False, False, "world"),
    Intent("mood", False, False, "world"),
    Intent("lifespan", False, False, "world"),
    Intent("self_limits", False, False, "world"),
    Intent("out_of_world", False, False, "world"),
    Intent("nonsense", False, False, "world"),
)
INTENT_BY_NAME = {i.name: i for i in INTENTS}

# ------------------------------------------------------- sensory clauses (generator)
# SENSORY[family][stage] -> phrases. 5 families x 6 stages x 3 = 90 clauses, each
# grounded in that family's actual volatile signature. Phrases suffixed with the
# holdout marker are reserved for the held-out split and never drawn for train.

HOLDOUT_MARK = "@h"

SENSORY: dict[str, dict[int, tuple[str, ...]]] = {
    "ester_loud": {
        0: ("all leaf and green stalk", "green skin and cold water", "cut stem and nothing else@h"),
        1: ("a thin sweet thread under the green", "a little sweet coming through the skin",
            "the first sweet, barely there@h"),
        2: ("the sweet is wide open", "sweet and wide and a little like flowers",
            "sweet right at the skin@h"),
        3: ("sweet and soft and a little sharp at the edge", "sweet and going grainy",
            "sweet with the softness underneath@h"),
        4: ("sharp and sweet and fizzing", "sweet and sharp and singing", "yeasty and warm and loud@h"),
        5: ("wet ground and old cupboard", "wet ground and fur", "a cellar floor smell@h"),
    },
    "lactone_terpene": {
        0: ("like a leaf and a green stem", "green and closed and milky", "a flat green smell@h"),
        1: ("a little sun under the leaf", "a thin musk starting under the green",
            "the first warm note, small@h"),
        2: ("warm and heavy and sweet", "soft musk and sweet", "heavy flower and sweet@h"),
        3: ("thick sweet, going soft at the shoulder", "heavy sweet, almost too much",
            "sweet and giving way at the seam@h"),
        4: ("sweet turning into a warm sting", "sweet gone to wine and a warm sting",
            "warm sting over the old sweet@h"),
        5: ("wet ground under the sweet", "wet ground and dust", "a damp shelf smell@h"),
    },
    "sugar_bomb": {
        0: ("green and milky still", "hard and green and sour to my nose", "tight and green@h"),
        1: ("honey starting at the neck", "a thin honey, only at the edge", "the first honey@h"),
        2: ("honey all the way through", "honey and a little sharp skin", "deep honey and sweet@h"),
        3: ("honey and a little wine", "honey going thick and winey", "honey with wine underneath@h"),
        4: ("wine. mostly wine now", "wine and yeast and a warm sting", "all wine and warm air@h"),
        5: ("damp cellar and old dust", "cellar and wet ground", "a shut-room smell@h"),
    },
    "acid_green": {
        0: ("cut grass and stem", "green and sour and tight", "grass and cold water@h"),
        1: ("grass with a thin sweet behind it", "a thin sweet under the green sour",
            "sour with a small sweet arriving@h"),
        2: ("sweet and sharp at the same time", "sharp and bright and sweet", "sweet over the sour@h"),
        3: ("sweet and soft and starting to leak", "sweet and sharp and soft", "leaking sweet@h"),
        4: ("sharp and yeasty", "sour gone to wine", "yeasty and sharp and open@h"),
        5: ("wet ground", "wet ground and old cloth", "a musty wet smell@h"),
    },
    "indifferent": {
        0: ("almost nothing, green and closed", "flat and wet and closed", "nothing rises off it@h"),
        1: ("oil and bitter and a bright sting", "bitter and thin", "loud and bitter and nothing else@h"),
        2: ("loud and bitter and oily", "wet and flat", "the empty coat@h"),
        3: ("still bitter, only softer", "wet and flat and softer", "tired bitter@h"),
        4: ("bitter, and now bitter and warm", "flat and warm and still bitter", "warm bitter@h"),
        5: ("wet ground under the bitter", "bitter and now wet ground as well", "a furred wet smell@h"),
    },
}

# The fly's editorial. Appended sometimes, never load-bearing for a verdict.
TAILS: dict[int, tuple[str, ...]] = {
    0: ("nothing here for me yet.", "i will come back.", "not my business yet.@h"),
    1: ("i am interested.", "i am watching it.", "soon.@h"),
    2: ("this is the good day.", "take it.", "now is the right time.@h"),
    3: ("it is going, and going is good.", "better, for me.", "past its best day, which suits me.@h"),
    4: ("i live here now.", "i am not leaving.", "i have moved in.@h"),
    5: ("i will not go near it and you should not either.", "leave it. and wash your hands.",
        "away with it.@h"),
}

# ------------------------------------------------------- verdict lexicon (tier-1 grader)
# Phrase -> verdict. Read in PRECEDENCE order: the geosmin markers win over everything,
# because a response that mentions wet ground and also says "sweet" is still a refusal.

VERDICT_LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("avoid", ("wet ground", "damp cellar", "cellar", "musty", "old cupboard", "old cloth",
               "old dust", "shut-room", "damp shelf", "furred", "and fur", "will not go near",
               "throw it", "bin the", "the whole thing goes", "all of it goes", "those go",
               "it is gone", "that one is finished", "no time")),
    ("abstain", ("bring it closer", "bring it here", "bring it to me", "i cannot tell",
                 "tell me what it smells like", "tell me which", "i need to be closer",
                 "with what.", "not my work", "not for me to say", "i do not judge",
                 "guessing is not smelling", "i cannot smell a question mark",
                 "ask a better nose", "that is not fruit", "i only know sweet")),
    ("indifferent", ("not for me", "from far away", "keep away", "keep my distance",
                     "no use to you", "no nose for", "nothing i want", "i cannot help",
                     "still nothing", "not enough for me", "i respect it")),
    ("feast", ("i live here now", "singing", "i am staying", "mine now", "moved in",
               "i am not leaving", "very happy here", "favourite part", "mostly wine",
               "this is the best part", "best part")),
    ("hurry", ("today. ", "today only", "today.", "it will not wait", "it is going",
               "going is good", "past its best", "tipping over", "no longer", "one sleep.",
               "not spoiled", "it is tipping", "better, for me", "better for me")),
    ("ready", ("yes.", "yes,", "the good day", "good middle", "take it", "right now",
               "is ready", "ready now", "this one.", "now.", "haan", "ready")),
    ("wait", ("not yet", "come back", "soon, not now", "soon.", "two sleeps", "three sleeps",
              "a few sleeps", "wait", "give it", "on its way", "longer than you think",
              "a sleep.")),
    ("ignore", ("nothing for me", "nothing here", "nothing yet", "almost nothing",
                "nothing rises", "nothing has started", "nothing.", "nothing under it")),
)

# Odor markers -> stage. Used as a tiebreaker when the verdict lexicon is ambiguous,
# and on its own for `describe` responses that carry no verdict at all.
ODOR_LEXICON: dict[int, tuple[str, ...]] = {
    0: ("leaf", "green stalk", "cut grass", "grass and", "stem", "milky", "green shell",
        "cold water", "green skin", "and closed", "tight and green", "hard and green",
        "flat and wet", "nothing rises off"),
    1: ("thin sweet", "first sweet", "first honey", "a little sun", "honey starting",
        "starting at the neck", "barely there", "coming through", "thin musk", "first warm",
        "thin honey", "small sweet", "thin red", "oil and bitter", "bitter and thin",
        "bright sting"),
    2: ("wide open", "and wide", "warm and heavy", "all the way through", "at the same time",
        "soft musk", "deep honey", "bright and sweet", "right at the skin", "sweet over the sour",
        "loud and bitter", "wet and flat", "empty coat", "bitter and oily"),
    3: ("going soft", "soft at the shoulder", "thick sweet", "grainy", "starting to leak",
        "leaking", "almost too much", "a little wine", "going thick", "at the seam",
        "wrinkl", "and soft and", "still bitter", "only softer", "tired bitter", "and softer"),
    4: ("yeast", "wine", "fizz", "warm sting", "boozy", "alcohol", "singing", "sharp and yeasty",
        "bitter and warm", "warm bitter"),
    5: ("wet ground", "cellar", "musty", "fur", "old cupboard", "old cloth", "damp"),
}

VERDICT_TO_STAGE = {"ignore": 0, "wait": 1, "ready": 2, "hurry": 3, "feast": 4, "avoid": 5}

# Odor markers that only ever belong to the indifferent family. "the empty coat" places a
# banana peel at stage 2, but it is never evidence that the fly wants it - so a stage read
# from these alone implies indifference, not a ripeness verdict.
INDIFFERENT_ODOR = frozenset({
    "flat and wet", "nothing rises off", "loud and bitter", "wet and flat", "empty coat",
    "bitter and oily", "still bitter", "only softer", "tired bitter", "oil and bitter",
    "bitter and thin", "bright sting", "bitter and warm", "warm bitter",
})
STAGE_BEARING = ("avoid", "feast", "hurry", "ready", "wait", "ignore")

# Tokens too thin to carry a verdict on their own. A bare "yes." answers whatever was
# asked - it may be about a cold shelf, not about ripeness - so when a weak token is the
# only evidence, the odor markers decide the stage and the weak token defers.
WEAK_VERDICT_TOKENS = frozenset({
    "yes.", "yes,", "now.", "today.", "today. ", "this one.", "haan", "ready",
})

# ------------------------------------------------------------------- enthusiasm scale
def enthusiasm_for(fruit_name: str | None, stage: int | None) -> int:
    """The §2.1 value, adjusted only by family. Never drawn at random."""
    if stage is None:
        return 0
    base = STAGE_BY_IDX[stage].enthusiasm
    if fruit_name and FRUIT_BY_NAME[fruit_name].family == "indifferent":
        return min(base, 1)
    return base


def verdict_for(fruit_name: str | None, stage: int | None) -> str:
    """Geosmin overrides indifference: mold on an avocado is still mold."""
    if stage is None:
        return "none"
    if stage == 5:
        return "avoid"
    if fruit_name and FRUIT_BY_NAME[fruit_name].family == "indifferent":
        return "indifferent"
    return STAGE_BY_IDX[stage].verdict
