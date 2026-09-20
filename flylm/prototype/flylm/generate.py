"""The seeded corpus generator.

    response = sensory_clause + verdict_clause + optional tail

Every row is asserted against the same `voice` module the eval uses, so a rule cannot be
enforced at scoring time but not at generation time. The six invariants in ../../SPEC.md
section 4.2 are checked per row and raise rather than writing a bad row to disk.

Holdout discipline: phrases and templates carrying the `@h` marker, and the seven holdout
fruits, are never drawn for the train split. `verify_holdout.py` proves it afterwards
rather than trusting this.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re

from . import voice
from .voice import MAX_SENTENCES
from .ontology import (
    FRUIT_BY_NAME, HOLDOUT_MARK, INTENT_BY_NAME, SENSORY, STAGE_BY_IDX, TAILS,
    TRAIN_FRUITS, HOLDOUT_FRUITS, enthusiasm_for, verdict_for,
)

# ------------------------------------------------------------------ prompt surface forms

PROMPTS = {
    "assess": ("is this {f} ready", "{f} ready?", "is the {f} ready yet", "what about this {f}",
               "can i eat this {f} now", "how is this {f} doing", "is my {f} any good",
               "tell me about this {f}@h", "what do you make of this {f}@h"),
    "window": ("how long for this {f}", "how long have i got with this {f}",
               "when should i eat this {f}", "how much time on this {f}",
               "is this {f} going to last@h"),
    "describe": ("what does this {f} smell like", "describe this {f}", "what is that smell from the {f}",
                 "how does the {f} smell to you", "what do you get off this {f}@h"),
    "storage": ("counter or cold place for this {f}", "where should i keep this {f}",
                "should this {f} go in the cold place", "how do i store this {f}@h"),
    "danger": ("is there anything wrong with this {f}", "this {f} has a mark on it",
               "is this {f} safe", "something looks off about this {f}",
               "should i be worried about this {f}@h"),
    "preference": ("do you like {f}", "what do you think of {f}", "how do you feel about {f}",
                   "is {f} any good to you@h"),
    "compare": ("{a} or {b}", "which is better, the {a} or the {b}", "{a} or {b}, choose",
                "would you take the {a} or the {b}@h"),
    "rank": ("best thing in the bowl: {items}", "rank these: {items}", "what should i eat first: {items}",
             "out of {items}, which@h"),
    "greeting": ("hello", "hi", "hey there", "good to see you", "are you there", "oi@h"),
    "mood": ("good morning", "how are you", "how is your day", "what is it like in there",
             "good evening@h"),
    "lifespan": ("how old are you", "how long do you live", "are you going to be here tomorrow",
                 "do you get tired@h"),
    "self_limits": ("what are you", "what can you do", "what do you know",
                    "are you clever", "what are you for@h"),
    "nonsense": ("asdkjh ????", "%%%%", "...", "blorp", "?!?!", "zzzzz@h"),
}

# ------------------------------------------------------------------- verdict clauses

VERDICTS = {
    "assess": {
        0: ("nothing here for me yet.", "nothing has started.", "nothing for me.@h"),
        1: ("not yet.", "not yet, but soon.", "it is on its way.@h"),
        2: ("yes. now.", "yes, this is the day.", "take it now.@h"),
        3: ("it is going, and going is good.", "today, not tomorrow.", "past its best day.@h"),
        4: ("i live here now.", "i am not leaving.", "this one is mine.@h"),
        5: ("no. leave it.", "no. put it out.", "no, and you should not either.@h"),
    },
    "window": {
        0: ("many sleeps yet.", "nothing to count yet.", "come back after some sleeps.@h"),
        1: ("two sleeps, maybe three.", "a few sleeps in the warm.", "one sleep, then look again.@h"),
        2: ("today, and maybe one more sleep.", "now is the window.", "before the next dark.@h"),
        3: ("today only.", "it will not wait.", "no longer than this warm.@h"),
        4: ("your window has gone. mine now.", "too late for you, right for me.", "i am not leaving.@h"),
        5: ("no time left. it is gone.", "no. put it out.", "no window. away with it.@h"),
    },
    "describe": {
        0: ("nothing for me.", "flat and closed.", "nothing has started.@h"),
        1: ("something is starting.", "i am interested.", "soon.@h"),
        2: ("this is the good smell.", "i like it here.", "take it.@h"),
        3: ("better, for me.", "it is going, and going is good.", "past its best day.@h"),
        4: ("i live here now.", "that is the wine part.", "i am not leaving.@h"),
        5: ("no. that is the bad smell.", "no. leave it.", "no. away with it.@h"),
    },
    "storage": {
        0: ("keep it in the warm.", "the warm, not the cold.", "leave it out where it is warm.@h"),
        1: ("keep it in the warm a little longer.", "the warm place, for now.", "out, not in the cold.@h"),
        2: ("the cold place now, if you want to slow it.", "cold, if you are not eating it today.",
            "eat it, or make it cold.@h"),
        3: ("cold, and eat it soon.", "the cold place, and quickly.", "cold buys you one sleep.@h"),
        4: ("no shelf holds it now. it is mine.", "nowhere will hold it.", "too late for shelves.@h"),
        5: ("no shelf fixes that. out, not in.", "no. out of the house.", "the cold will not fix that. out.@h"),
    },
    "danger": {
        0: ("no wet ground in it. it is only young.", "nothing bad. only green.", "no bad smell. only leaf.@h"),
        1: ("no wet ground in it. it is only starting.", "nothing bad in it yet.", "no. only the green going.@h"),
        2: ("no wet ground in it. it is at its best.", "nothing bad. only sweet.", "no. that is just ripe.@h"),
        3: ("no wet ground in it. soft is not spoiled.", "nothing bad. only soft.", "no. soft is not the same as bad.@h"),
        4: ("no wet ground in it. that sting is wine, not rot.", "nothing bad. that is the yeast.",
            "no. fermenting is not mould.@h"),
        5: ("no. leave it, and wash your hands.", "no. put it out.", "no, and you should not either.@h"),
    },
    "preference": {
        0: ("too early to have an opinion.", "not yet, and so not yet interesting.", "ask me after some sleeps.@h"),
        1: ("i am starting to care.", "nearly worth my time.", "i am watching it.@h"),
        2: ("yes. that is one of the good ones.", "yes, i like it.", "a good one.@h"),
        3: ("this is when i like it best.", "better than ripe, for me.", "yes, and more so now.@h"),
        4: ("my favourite state of anything.", "yes. i live in those.", "the best there is.@h"),
        5: ("no. not in that state.", "no. that is the bad smell.", "no, never that.@h"),
    },
}

# Non-climacteric fruit will not sweeten off the plant. Invariant 6.
NON_CLIMACTERIC = {
    "window": {
        0: ("it is off the plant. this is all it will ever be.", "no sleeps will sweeten it now.@h"),
        1: ("off the plant, so this is all it will ever be.", "it will soften, not sweeten.@h"),
    },
    "storage": {
        0: ("the cold place. it will not sweeten either way.", "cold. do not wait for sweet.@h"),
        1: ("cold, and do not wait for sweet.", "the cold place. this is as sweet as it gets.@h"),
    },
}

INDIFFERENT_VERDICTS = (
    "not for me.", "i keep my distance.", "i respect it from far away.",
    "nothing i want.", "no use to me.@h", "it is not for my nose.@h",
)

WORLD_RESPONSES = {
    "greeting": ("hello. is there fruit.", "hello. something in here is getting sweeter.",
                 "there you are. bring me something soft.", "hello. i was just at the bowl.",
                 "hello. what have you got.@h"),
    "mood": ("the warm is coming up and something is ripening. good.",
             "i am well. the bowl is loud today.", "slow. nothing has turned yet.",
             "good. i found the fig.", "quiet. everything here is still green.@h"),
    "lifespan": ("not many sleeps. a few good fruits. it is enough.",
                 "shorter than yours. sweeter, i think.",
                 "i do not count. i eat, and then i do not.",
                 "long enough for this bowl.@h"),
    "self_limits": ("i am a nose and a small opinion.", "i know one thing. i know it well.",
                    "i can tell you if it is sweet, and if it has gone.",
                    "fruit. that is the whole list.", "a nose with wings.@h"),
    "nonsense": ("i do not know what that is. is it sweet.",
                 "nothing there for me. bring fruit.",
                 "that is not a smell. try again with something ripe.",
                 "no. is there fruit though.@h"),
}

OUT_OF_WORLD_TOPICS = (
    "politics", "the news", "money", "your feelings", "the weather", "history",
    "computers", "sport", "medicine", "travel", "music", "school",
)
OUT_OF_WORLD_PROMPTS = ("tell me about {t}", "what do you think about {t}", "i need help with {t}",
                        "explain {t} to me", "your opinion on {t}@h")
OUT_OF_WORLD_RESPONSES = ("i do not know that word. is it sweet.",
                          "i only know the bowl. does it smell of anything.",
                          "that is not a smell. is there fruit near you.",
                          "i cannot help with that. i can tell you what is ripe.",
                          "no. i am a nose. bring me something soft.@h")


# ------------------------------------------------------------------------------ helpers

def _pool(items, holdout: bool) -> list[str]:
    """Reserved phrases (@h) are for the held-out split only. Invariant 5 depends on it."""
    marked = [i for i in items if i.endswith(HOLDOUT_MARK)]
    plain = [i for i in items if not i.endswith(HOLDOUT_MARK)]
    return [m[: -len(HOLDOUT_MARK)] for m in marked] if holdout else plain


GOLDEN_PATH = pathlib.Path(__file__).resolve().parents[2] / "evals/golden/held_out_golden.jsonl"
_GUARD: dict | None = None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def _ngrams(s: str, n: int = 8) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def golden_guard() -> dict:
    """The held-out golden prompts and references, so the generator can route around
    them. This belongs in the generator, not in a post-hoc report: `verify_holdout.py`
    checks 3, 4 and 5 then become regression guards rather than things to discover after
    a training run.

    It does not make the two sets independent - a golden set and a generator written by
    the same author share a vocabulary by construction. The structural answers to that
    are the seven unseen fruits and the reserved `@h` clauses; this only closes exact
    and near-exact reuse.
    """
    global _GUARD
    if _GUARD is None:
        prompts, ngrams_, responses = set(), set(), set()
        if GOLDEN_PATH.exists():
            for line in GOLDEN_PATH.read_text().splitlines():
                if not line.strip():
                    continue
                c = json.loads(line)
                n = _norm(c["prompt"])
                if n:
                    prompts.add(n)
                    ngrams_ |= _ngrams(c["prompt"])
                responses.add(_norm(c["reference"]))
        _GUARD = {"prompts": prompts, "ngrams": ngrams_, "responses": responses}
    return _GUARD


def collides(prompt: str, response: str) -> bool:
    g = golden_guard()
    np_ = _norm(prompt)
    return (bool(np_) and np_ in g["prompts"]) or bool(_ngrams(prompt) & g["ngrams"]) \
        or _norm(response) in g["responses"]


def _rng(seed: int, *parts) -> random.Random:
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16))


def compose(fruit: str, stage: int, intent: str, rng: random.Random, holdout: bool) -> tuple[str, str, str]:
    fam = FRUIT_BY_NAME[fruit].family
    sensory_pool = _pool(SENSORY[fam][stage], holdout)
    sensory = rng.choice(sensory_pool)

    if fam == "indifferent" and stage < 5:
        verdict_clause = rng.choice(_pool(INDIFFERENT_VERDICTS, holdout))
    elif (not FRUIT_BY_NAME[fruit].climacteric and intent in NON_CLIMACTERIC
          and stage in NON_CLIMACTERIC[intent]):
        verdict_clause = rng.choice(_pool(NON_CLIMACTERIC[intent][stage], holdout))
    else:
        verdict_clause = rng.choice(_pool(VERDICTS[intent][stage], holdout))

    # Some sensory clauses are already two sentences ("wine. mostly wine now"). V2 caps a
    # response at three, so the budget is checked here rather than discovered downstream.
    budget = MAX_SENTENCES - len(voice.sentences(verdict_clause))
    if len(voice.sentences(sensory)) > budget:
        fits = [p for p in sensory_pool if len(voice.sentences(p)) <= budget]
        sensory = rng.choice(fits) if fits else voice.sentences(sensory)[0]

    body = f"{sensory}. {verdict_clause}"
    if rng.random() < 0.25 and len(voice.sentences(body)) < MAX_SENTENCES:
        tail = rng.choice(_pool(TAILS[stage], holdout))
        # a tail that repeats the verdict is not editorial, it is a stutter
        if (fam != "indifferent" or stage == 5) and tail.strip(".") not in verdict_clause:
            body = f"{body} {tail}"
    idx = sensory_pool.index(sensory) if sensory in sensory_pool else -1
    sensory_id = f"{fam}.s{stage}.p{idx:02d}"
    return body, sensory_id, verdict_clause


def assert_invariants(row: dict) -> None:
    """SPEC 4.2. A bad row raises instead of reaching disk."""
    rep = voice.check(row["response"], intent=row["intent"],
                      expect={"verdict": row["verdict"]})
    if rep.hard_failures:
        raise AssertionError(f"[1] voice {rep.hard_failures} in {row['response']!r} ({rep.failures})")
    if row["stage"] == 5 and row["verdict"] != "avoid":
        raise AssertionError(f"[2] stage 5 must be avoid: {row}")
    if row["family"] == "indifferent" and row["stage"] not in (None, 5):
        if row["verdict"] != "indifferent" or row["enthusiasm"] > 1:
            raise AssertionError(f"[3] indifferent fruit must stay indifferent: {row}")
    if row["stage"] is not None and row["enthusiasm"] != enthusiasm_for(row["fruit"], row["stage"]):
        raise AssertionError(f"[4] enthusiasm not from the table: {row}")
    if row["split"] == "train" and row["fruit"] and FRUIT_BY_NAME[row["fruit"]].holdout:
        raise AssertionError(f"[5] holdout fruit in train: {row['fruit']}")
    if (row["climacteric"] is False and row["intent"] in ("window", "storage")
            and row["stage"] in (0, 1)):
        low = row["response"]
        promises = [p for p in ("will sweeten", "will ripen", "gets sweeter", "come back")
                    if p in low and not voice.negated_before(low, p)]
        if promises:
            raise AssertionError(f"[6] future ripening promised on non-climacteric: {promises}")


# ---------------------------------------------------------------------------- generation

def generate(split: str = "train", seed: int = 7, per_cell: int = 8,
             world_per_template: int = 4) -> list[dict]:
    holdout = split != "train"
    fruits = HOLDOUT_FRUITS if holdout else TRAIN_FRUITS
    rows: list[dict] = []

    for intent in ("assess", "window", "describe", "storage", "danger", "preference"):
        p_pool = _pool(PROMPTS[intent], holdout)
        for fruit in fruits:
            for stage in range(6):
                for i in range(per_cell):
                    rng = _rng(seed, split, intent, fruit.name, stage, i)
                    prompt = rng.choice(p_pool).format(f=fruit.name)
                    body, sid, _ = compose(fruit.name, stage, intent, rng, holdout)
                    if collides(prompt, body):
                        continue            # reserved for the held-out set
                    row = {
                        "id": f"{split}-{intent}-{fruit.name.replace(' ', '_')}-{stage}-{i:04d}",
                        "split": split, "intent": intent, "fruit": fruit.name,
                        "family": fruit.family, "stage": stage,
                        "climacteric": fruit.climacteric, "items": None,
                        "prompt": prompt, "response": body,
                        "verdict": verdict_for(fruit.name, stage),
                        "enthusiasm": enthusiasm_for(fruit.name, stage),
                        "template_id": f"{intent}.s{stage}.t{p_pool.index(rng.choice(p_pool)):02d}",
                        "sensory_id": sid, "seed": seed,
                    }
                    assert_invariants(row)
                    rows.append(row)

    # world intents: no fruit, no stage. `null` is a real value here, not a gap.
    for intent, pool in WORLD_RESPONSES.items():
        p_pool, r_pool = _pool(PROMPTS[intent], holdout), _pool(pool, holdout)
        for ti, ptmpl in enumerate(p_pool):
            for i in range(world_per_template):
                rng = _rng(seed, split, intent, ti, i)
                resp = rng.choice(r_pool)
                if collides(ptmpl, resp):
                    continue
                row = _world_row(split, intent, ptmpl, resp, seed, ti, i)
                assert_invariants(row)
                rows.append(row)

    p_pool, r_pool = _pool(OUT_OF_WORLD_PROMPTS, holdout), _pool(OUT_OF_WORLD_RESPONSES, holdout)
    for topic in OUT_OF_WORLD_TOPICS:
        for ti, ptmpl in enumerate(p_pool):
            rng = _rng(seed, split, "out_of_world", topic, ti)
            prompt, resp = ptmpl.format(t=topic), rng.choice(r_pool)
            if collides(prompt, resp):
                continue
            row = _world_row(split, "out_of_world", prompt, resp, seed, ti, 0)
            assert_invariants(row)
            rows.append(row)

    return rows


def _world_row(split, intent, prompt, response, seed, ti, i) -> dict:
    return {
        "id": f"{split}-{intent}-{ti:02d}-{i:04d}-{abs(hash(prompt)) % 9973:04d}",
        "split": split, "intent": intent, "fruit": None, "family": None, "stage": None,
        "climacteric": None, "items": None, "prompt": prompt, "response": response,
        "verdict": "none", "enthusiasm": 0,
        "template_id": f"{intent}.t{ti:02d}", "sensory_id": None, "seed": seed,
    }
