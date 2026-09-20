"""Deep-trace instrumentation.

The notebook's job is to show WHY a grader reached a verdict, not just that it did. So
every interesting decision is traceable at four levels, and the level is a single knob:

    trace.configure(level="deep")     # every lexicon hit, every rule, every timing
    trace.configure(level="normal")   # section banners, decisions, results  (default)
    trace.configure(level="quiet")    # results only
    trace.configure(level="off")      # nothing

Nothing here is required by the library - `graders`, `harness` and `generate` all work
with tracing off. It is an observation layer, deliberately kept out of their logic so a
print statement can never change a score.

Plain ASCII markers, no colour codes: a notebook is read in a browser, a terminal, a
diff, and a PDF, and escape sequences survive only the first.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

LEVELS = {"off": 0, "quiet": 1, "normal": 2, "deep": 3}

MARK = {
    "step": "->",
    "ok": "[ok]",
    "fail": "[FAIL]",
    "warn": "[warn]",
    "info": " . ",
    "hit": " * ",
    "time": " ~ ",
}


@dataclass
class _State:
    level: int = LEVELS["normal"]
    indent: int = 0
    width: int = 88
    stream: object = field(default_factory=lambda: sys.stdout)
    counters: dict[str, int] = field(default_factory=dict)


_S = _State()


def configure(level: str = "normal", width: int = 88, stream=None) -> None:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    _S.level = LEVELS[level]
    _S.width = width
    if stream is not None:
        _S.stream = stream


def level_name() -> str:
    return next(k for k, v in LEVELS.items() if v == _S.level)


def _emit(text: str, min_level: int) -> None:
    if _S.level < min_level:
        return
    pad = "  " * _S.indent
    print(f"{pad}{text}", file=_S.stream)


# ------------------------------------------------------------------------------ banners

def section(n: str | int, title: str, subtitle: str = "") -> None:
    """A numbered notebook section. Printed at `quiet` and above: it is the map."""
    if _S.level < LEVELS["quiet"]:
        return
    bar = "=" * _S.width
    print(f"\n{bar}", file=_S.stream)
    print(f"SECTION {n}  {title}", file=_S.stream)
    if subtitle:
        print(f"          {subtitle}", file=_S.stream)
    print(bar, file=_S.stream)
    _S.indent = 0


def step(text: str) -> None:
    _emit(f"{MARK['step']} {text}", LEVELS["normal"])


def info(text: str) -> None:
    _emit(f"{MARK['info']} {text}", LEVELS["normal"])


def detail(text: str) -> None:
    """Deep level only: the per-hit, per-rule firehose."""
    _emit(f"{MARK['hit']} {text}", LEVELS["deep"])


def result(text: str, ok: bool | None = None) -> None:
    m = MARK["ok"] if ok else (MARK["fail"] if ok is False else MARK["info"])
    _emit(f"{m} {text}", LEVELS["quiet"])


def warn(text: str) -> None:
    _emit(f"{MARK['warn']} {text}", LEVELS["quiet"])


def note(text: str) -> None:
    """A wrapped prose aside - the 'why', which is the part worth reading."""
    if _S.level < LEVELS["normal"]:
        return
    import textwrap
    pad = "  " * _S.indent + "   "
    for line in textwrap.wrap(text, width=_S.width - len(pad)):
        print(f"{pad}{line}", file=_S.stream)


@contextmanager
def block(title: str, min_level: str = "normal"):
    """Indented, timed sub-block."""
    if _S.level < LEVELS[min_level]:
        yield
        return
    _emit(f"{MARK['step']} {title}", LEVELS[min_level])
    _S.indent += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        _emit(f"{MARK['time']} {dt:.1f} ms", LEVELS["deep"])
        _S.indent -= 1


def count(key: str, n: int = 1) -> None:
    _S.counters[key] = _S.counters.get(key, 0) + n


def counters() -> dict[str, int]:
    return dict(_S.counters)


def reset_counters() -> None:
    _S.counters.clear()


# ------------------------------------------------------------------------------- tables

def table(headers: list[str], rows: list[list], *, align: str | None = None,
          min_level: str = "quiet", max_rows: int | None = None) -> None:
    """A plain-text table. Every chart in this notebook ships one of these beside it.

    That is not decoration: the validated chart palette has a light-mode slot below 3:1
    contrast, and the documented relief for that is visible labels or a table view. A
    static PNG also has no hover layer, so the table IS the data-inspection path.
    """
    if _S.level < LEVELS[min_level]:
        return
    body = [[("" if c is None else str(c)) for c in r] for r in rows]
    shown = body if max_rows is None else body[:max_rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in shown)) if shown else len(headers[i])
              for i in range(len(headers))]
    align = align or ("l" + "r" * (len(headers) - 1))

    def fmt(cells):
        return "  ".join(c.ljust(w) if align[i] == "l" else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))

    pad = "  " * _S.indent
    print(f"{pad}{fmt(headers)}", file=_S.stream)
    print(f"{pad}{'  '.join('-' * w for w in widths)}", file=_S.stream)
    for r in shown:
        print(f"{pad}{fmt(r)}", file=_S.stream)
    if max_rows is not None and len(body) > max_rows:
        print(f"{pad}... {len(body) - max_rows} more rows", file=_S.stream)


def kv(pairs: dict, min_level: str = "normal") -> None:
    if _S.level < LEVELS[min_level]:
        return
    w = max((len(k) for k in pairs), default=0)
    for k, v in pairs.items():
        _emit(f"{MARK['info']} {k.ljust(w)} : {v}", LEVELS[min_level])


# -------------------------------------------------------- grader-specific trace helpers

def trace_reading(text: str, reading) -> None:
    """Unpack one classifier decision. This is the single most useful trace in the
    notebook: it shows which phrase produced which verdict, so a wrong score is
    attributable to a lexicon entry rather than to a mystery."""
    if _S.level < LEVELS["deep"]:
        return
    detail(f"input   : {text!r}")
    detail(f"verdict : {reading.verdict}   stage={reading.stage}   "
           f"enthusiasm={reading.enthusiasm}   abstained={reading.abstained}")
    detail(f"conf    : {reading.confidence}  (source={reading.source})")
    for k, v in sorted(reading.hits.items()):
        detail(f"hit     : {k:22} {v}")


def trace_case(case: dict, output: str, res) -> None:
    """One graded case, end to end."""
    if _S.level < LEVELS["deep"]:
        return
    detail(f"case    : {case['id']}  [{case['slice']}]")
    detail(f"prompt  : {case['prompt']!r}")
    detail(f"output  : {output!r}")
    trace_reading(output, res.reading)
    detail(f"gates   : failed={res.gates_failed or 'none'}  soft={res.soft_score:.3f}")
    detail(f"metrics : {res.metrics}")
    detail(f"verdict : slice_pass={res.slice_pass}  ({res.reason})")
