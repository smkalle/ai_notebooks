# ai_notebooks

Notebook tutorials.

## Notebooks

| Notebook | Topic | Colab |
|---|---|---|
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
