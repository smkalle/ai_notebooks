# NeoMME Tutorial — Feature Handout

**Notebook:** `notebooks/NeoMME_Complete_Tutorial.ipynb`
**Subject:** [NeoMME](https://huggingface.co/blog/Hcompany/neomme) — H Company's single-tower, multimodal-native, multilingual encoder (260M / 800M)
**Runtime:** ~15–25 min on a free Colab T4 · 122 cells · 13 sections
**Requires:** `transformers` from `main` (NeoMME contributed 2026-08-31), `sentence-transformers >= 6.0.0`
**Licence of all checkpoints:** Apache 2.0

> **Self-contained.** The notebook renders its own document corpus with PIL and ships its own labelled
> query set, so it runs start to finish with **no dataset downloads** — and it *measures* every claim
> (Recall@k, MRR, nDCG@k) rather than asserting it.

---

## At a glance

| # | Section | Headline capability |
|---|---------|---------------------|
| 1 | Why single-tower | Architecture contrasted with dual-tower and VLM-adapter designs |
| 2 | Setup | Install, environment check, model IDs |
| 3 | The backbone | `NeoMMEModel` — text, images, and mixed batches through one encoder |
| 4 | Masked language modelling | `NeoMMEForMaskedLM` — multilingual fill-mask, PLL scoring |
| 5 | The image pipeline | Patchification, resolution budgets, two-axis positions |
| 6 | Sample data | A synthetic 6-language document corpus + labelled queries |
| 7 | Retrieval | `NeoMMEForRetrieval` — dense *and* late interaction in one pass |
| 8 | A real search engine | Two-stage pipeline with genuine IR metrics |
| 9 | Compression | Matryoshka, token pooling, asymmetric int8 |
| 10 | Fine-tuning | Raw PyTorch and Sentence Transformers routes |
| 11 | Visual RAG | Retrieved page images → VLM → grounded answer |
| 12 | Production notes | Throughput, batching, index storage, ship checklist |
| 13 | Troubleshooting & FAQ | Every error you are likely to hit, with its fix |

---

## 1. Architecture and concepts

- Single-tower design set against **dual-tower (CLIP)** and **VLM-adapter (ColPali)** shapes, with a side-by-side ASCII diagram
- **ALBERT-style factorized text embedding** (`vocab_size → embedding_rank → hidden_size`), with the parameter saving computed live from the config
- The **32×32 patch stem** exposed as a 2-layer MLP — 4 tensors, ~2% of parameters, and that is the *entire* "vision encoder"
- **Masked discrete-diffusion** pretraining objective, and why it produces bidirectional representations
- Interleaved `sliding_attention` / `full_attention` schedule rendered as a glyph strip, with **per-layer-type RoPE spectra** (θ = 10k full-rotary for local layers vs. θ = 1M quarter-rotary for global layers) and alternating 256 / 1024 windows
- The **16,384-token context** sized against real page resolutions

## 2. `NeoMMEModel` — the backbone

- `AutoModel` / `AutoProcessor` loading, full config introspection, correct per-layer `sliding_window` access
- **Special-token table** (`<doc>`, `<img>`, `<row>`, `<query>`, `<mask>`, `<pad>`) cross-checked against config IDs
- Text encoding in **five languages**, with token-by-token inspection
- Image-only encoding, with the `<doc><img>` + patch-grid + `<row>` layout decoded and verified against hand-computed grid maths
- **Two-axis M-RoPE `position_ids` plotted** — a staircase for rows, a sawtooth for columns
- A sentence **and** a page image in one batch, one forward pass, the same weights
- `get_image_features` called directly to show raw patches → hidden space

## 3. `NeoMMEForMaskedLM`

- Fill-mask with top-k probability bars
- The **same fact masked across six languages** — no language ID, no translation layer
- **Bidirectional evidence demo**: the disambiguating word appears *after* the mask, which a causal LM could not use
- Multiple masks resolved **jointly**, not left-to-right
- **Pseudo-log-likelihood** text scoring built from the MLM head
- Tied-weight inspection (`lm_head` ↔ factorized embedding)

## 4. The image pipeline

- A full **patchification round-trip** — `pixel_values` inverted back into the original page, proving the format is lossless
- **Token-cost table** from thumbnail to 4K UHD, showing % of context consumed and flagging over-context sizes
- All three **resize knobs** compared side by side (`max_side`, `min_pixels`, `max_pixels`) and how they compose
- `get_number_of_image_patches` — cost a page *before* you pay for it
- **Mixed-size batching** explained: `pixel_values` concatenated vs. `input_ids` padded

## 5. Sample data (no downloads)

- A declarative **PIL document renderer** with four primitives: heading, prose, bar chart, table
- **12 synthetic document pages** across 6 languages (EN/FR/ES/DE/IT/PT) and 3 document types, each carrying one findable fact
- **20 labelled queries** tagged `mono` / `cross` / `visual`, with a gold-label integrity assertion

## 6. `NeoMMEForRetrieval`

- `apply_chat_template(task="query" | "document")` with the resulting token streams printed for both
- The **10 `<mask>` query-expansion tokens** explained and shown in situ
- **Both heads from one forward pass**: `embeddings` (multi-vector) + `dense_embeddings`
- Verified invariants: L2-normalized token vectors, exactly-zero padding rows
- **Five guard rails probed live** — image + `task="query"`, text + image together, two images, no task, truncation eating the expansion masks — each with its real exception message
- **MeanMaxSim vs. plain MaxSim vs. cosine** on the same pairs, including the `MaxSim / MeanMaxSim == query token count` identity, printed
- **Per-patch MaxSim heatmap** overlaid on the page — see exactly which patches the query tokens latched onto

## 7. Search engine and measurement

- An `Index` dataclass holding **variable-length multi-vector lists**, as a real store would
- Batched document / query encoders with timing
- **Recall@k, MRR and nDCG@k implemented from scratch** — nothing to take on trust
- Three strategies compared: **dense only**, **late interaction only**, and **two-stage dense → rerank** at four depths
- **Per-query rank table** with dense-vs-late win/loss flags
- Breakdowns **by query kind and by language**, quantifying cross-lingual retrieval with no translation step anywhere
- An end-to-end `search()` returning ranked page images

## 8. Index compression

- **Matryoshka `dense_dim`** sweep (64 → 1024) with quality-retention bars
- **`HierarchicalTokenPooling`** sweep (`pool_factor` 2 → 32) with vector counts and index size
- Hand-rolled **asymmetric int8 quantization** — percentile calibration, quantize/dequantize, error statistics; documents quantized, queries left in fp32
- `quantize_embeddings` for float32 / int8 / binary on the dense side
- **Stacked pooling × quantization** table plus a size-vs-quality curve with a 95% threshold line
- Extrapolation to a **10M-page corpus** in real bytes
- An explicit, honest caveat that 12 pages cannot reproduce a published benchmark number

## 9. Fine-tuning

- Train/eval split with a **page-leakage assertion**
- **Raw-PyTorch contrastive loop**: joint late-interaction + dense InfoNCE, temperature scaling, OneCycleLR, gradient clipping, gradient checkpointing
- **Before/after evaluation on held-out pages**, plus a loss / in-batch-accuracy plot
- **Sentence Transformers route**: `MultiVectorEncoder`, `encode_query` / `encode_document`, cached MNRL loss (GradCache), trainer arguments, IR evaluator — with graceful fallback if the ST layout does not load
- **Domain-adaptive continued MLM pretraining** recipe, including why NeoMME wants a higher and randomized `mlm_probability` rather than BERT's fixed 15%

## 10. Visual RAG

- Retrieval → page images → VLM → **grounded answer**, with no OCR anywhere in the pipeline
- Pluggable generator that **skips cleanly** if unavailable
- A **cross-lingual end-to-end run**: English question → Spanish page → English answer
- Failure-mode table, including the *index at one resolution, generate at another* rule

## 11. Production notes

- **Live throughput benchmark** across resize settings, with OOM handling
- Knob table: resolution, dtype, FlashAttention, batching, head flags, `dense_dim`
- Loading patterns, **length-sorted batching**, memory-bounded scoring via `chunk_elements`
- **Index storage sketch** — pgvector + object storage, two-stage layout
- A **pre-ship checklist**

## 12. Troubleshooting and FAQ

- **10 real exception messages** taken from the implementation, each with its cause and its fix
- FAQ: is it generative? do I need OCR? how many images per pass? dense or late interaction? which languages? what are the `predecay` checkpoints? licensing?
- Links to the blog post, the model collection, the Transformers docs, the Sentence Transformers multi-vector quickstart, and ViDoRe

---

## Notes on provenance and verification

- Built against the **official NeoMME implementation** in `huggingface/transformers@main` — `configuration_neomme.py`, `modeling_neomme.py`, `processing_neomme.py`, `image_processing_neomme.py`, the model-doc page and the processor tests — plus `sentence-transformers` 6.0.1 source.
- **All 59 executable code cells were run** against a locally constructed tiny NeoMME model (same classes, same processor contract) to verify API paths, tensor shapes and error messages. Zero failures.
- The notebook has **not** been executed against the real checkpoints in this environment. Shapes and API calls are verified; the *numbers* it prints only become meaningful when you run it with real weights.
- The 800M checkpoint IDs are inferred from the naming pattern — only the 260M IDs appear verbatim in the official docs. The notebook flags this at the point where it would matter.

## Links

- [NeoMME blog post](https://huggingface.co/blog/Hcompany/neomme)
- [NeoMME model collection](https://huggingface.co/collections/Hcompany/neomme)
- [Transformers `NeoMME` documentation](https://huggingface.co/docs/transformers/main/model_doc/neomme)
- [Sentence Transformers multi-vector quickstart](https://sbert.net/docs/quickstart.html#multi-vector-encoder)
- [ViDoRe benchmark](https://huggingface.co/vidore)
