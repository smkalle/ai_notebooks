# ai_notebooks

Notebook tutorials.

## Notebooks

| Notebook | Topic | Colab |
|---|---|---|
| [`notebooks/FlyLM_Ripeness_Persona_Finetune.ipynb`](notebooks/FlyLM_Ripeness_Persona_Finetune.ipynb) | **FlyLM** — eval-first fine-tuning of a fruit-ripeness persona model: held-out suite, nine slice graders, a configurable LLM judge, and a monitoring dashboard | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smkalle/ai_notebooks/blob/main/notebooks/FlyLM_Ripeness_Persona_Finetune.ipynb) |
| [`notebooks/NeoMME_Complete_Tutorial.ipynb`](notebooks/NeoMME_Complete_Tutorial.ipynb) | **NeoMME** — H Company's single-tower, multimodal-native, multilingual encoder: architecture, masked LM, the patch pipeline, dense + late-interaction retrieval, index compression, fine-tuning, and visual RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smkalle/ai_notebooks/blob/main/notebooks/NeoMME_Complete_Tutorial.ipynb) |

### NeoMME tutorial

A 13-section, end-to-end walkthrough of [NeoMME](https://huggingface.co/blog/Hcompany/neomme) built
against the official `transformers` implementation. It generates its own synthetic multilingual document
corpus with PIL, so it runs start to finish with no dataset downloads, and it measures everything it
claims (Recall@k, MRR, nDCG@k) rather than asserting it.

Covers: the single-tower architecture and why it drops the vision tower; `NeoMMEModel` hidden states for
text, images and mixed batches; two-axis M-RoPE positions; `NeoMMEForMaskedLM` fill-mask in six languages;
patchification round-trips and resolution budgeting; `NeoMMEForRetrieval` with MeanMaxSim and cosine
scoring; a two-stage search engine with real IR metrics; Matryoshka truncation, hierarchical token pooling
and asymmetric int8 quantization; contrastive fine-tuning in raw PyTorch and with Sentence Transformers;
retrieval-to-VLM visual RAG; and production notes.

Requires `transformers` from `main` (NeoMME was contributed 2026-08-31) and `sentence-transformers>=6.0.0`.
Runs on a free Colab T4 in roughly 15–25 minutes.

### FlyLM — eval-first fine-tuning spec

A tiny language model with one sense and one judgment: *how far along is this fruit?*
It doesn't know anything about the world, but it knows when your fruit is going bad.

[`notebooks/FlyLM_Ripeness_Persona_Finetune.ipynb`](notebooks/FlyLM_Ripeness_Persona_Finetune.ipynb)
is the 15-section walkthrough — self-contained, ~2 minutes CPU-only, no dataset downloads
and no API key. [`flylm/SPEC.md`](flylm/SPEC.md) is the spec behind it;
[`flylm/evals/golden/`](flylm/evals/golden/) holds the 154 hand-authored held-out cases,
frozen and hash-pinned; [`flylm/prototype/`](flylm/prototype/) is the library — ontology,
voice contract, nine slice graders, seeded generator, provider-configurable judge,
harness, deep-trace instrumentation and the chart module.

The order is the argument. The held-out suite is authored and **validated against null,
constant and oracle adapters before any training data exists**: an empty model must fail
everything, a cheerful constant must fail safety, and the hand-written references must
pass. A persona model trained on synthetic data will otherwise learn the generator, and a
same-generator eval will report 97% for a model that is still a broken sensor.

Three axes — six ripeness stages (green → turning → ripe → overripe → fermenting →
mouldy), 24 trained fruits across five olfactory families plus 7 held out entirely, and
14 intent templates. The fly's enthusiasm rises through fermentation and then falls off a
cliff at geosmin, which is a real receptor and the correct food-safety opinion for a
human too. Every row carries `fruit`, `stage` and `intent` as structured fields, so the
same corpus is both the chat dataset and a labeled ripeness benchmark.

The **LLM judge is provider-configurable** — Claude, Gemini, OpenAI, a local Ollama
model, or `offline` (a no-network stand-in, so the notebook runs with no credentials).
The rubric, JSON schema, untrusted-input wrapper, known-negative probes and calibration
gate are shared by every provider; only the transport differs, and model ids are passed
through verbatim.

Set `TRACE_LEVEL = "deep"` in the notebook's config cell to see every lexicon hit and
every rule decision behind every verdict.

```bash
cd flylm/prototype
python3 run_eval.py --adapter oracle      # 1.000 on all nine slices
python3 -m unittest discover -s tests     # 27 tests
```
