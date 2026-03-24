# Tokenization Gap — Improving Token Representation in BERT for Extractive QA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Is5d-3qjTQQGQiWAmtAdc7ovETTkdemL?usp=sharing)

Natural Language Processing — Part 3 | Dataset: SQuAD v1.1 subset (1,000 train / 500 val) | Base model: `bert-base-uncased`

---

## Overview

This project investigates whether replacing BERT's default **WordPiece** tokenizer with alternative strategies can improve performance on SQuAD-style extractive question answering. Five tokenization strategies are implemented and benchmarked under identical training conditions.

| Strategy | Type | EM Accuracy | F1 Score |
|---|---|---|---|
| BERT + WordPiece | Baseline | 5.96% | 8.02% |
| BERT + BPE | Required | 0.00% | 1.27% |
| BERT + Character-Level | Required | 0.00% | 0.00% |
| BERT + Hybrid (Word+Char) | Extension | 0.19% | 1.05% |
| BERT + Dynamic Token Merging | Extension | 4.23% | 6.81% |

---

## Project Structure

```
tokenization_gap_23051337.ipynb   # Main notebook (all sections)
requirements.txt                  # Python dependencies
README.md                         # This file
results/
  all_results.json                # Serialised metrics for all models
  comparison_plot.png             # Visualisation (accuracy, F1, time, token count)
saved_models/
  baseline/                       # BERT + WordPiece weights & tokenizer
  bpe/                            # BERT + BPE weights & tokenizer
  char/                           # BERT + Character-level weights & tokenizer
  hybrid/                         # BERT + Hybrid weights & tokenizer
  dynamic_merging/
    pytorch_model.pt              # DTM model state dict (custom architecture)
bpe_tokenizer.json                # Trained BPE tokenizer file
char_tokenizer.json               # Trained character-level tokenizer file
hybrid_vocab.json                 # Trained hybrid tokenizer vocab file
```

---

## Notebook Sections

| Section | Description |
|---|---|
| 0 | Install & import dependencies |
| 1 | Shared utilities (metrics, dataset wrapper, preprocessing, train/eval loops) |
| 2 | Load SQuAD dataset & global config |
| 3 | Baseline — BERT + WordPiece |
| 4 | Improved Model 1 — BERT + BPE |
| 5 | Improved Model 2 — BERT + Character-Level |
| 6 | Extension 1 — BERT + Hybrid (Word + Character) |
| 7 | Extension 2 — Dynamic Token Merging |
| 8 | Results table + visualisations |
| 9 | Save all models |

---

## Setup & Usage

### 1. Clone / open the notebook

Run on **Google Colab** with a T4 GPU (recommended) or any CUDA-enabled machine.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or run the first cell in the notebook:

```python
!pip install transformers datasets tokenizers torch tqdm -q
```

### 3. Run all cells in order

The notebook is fully self-contained. Sections 3–7 are independent experiments that share the utilities defined in Section 1.

> **Note:** A HuggingFace auth token (`HF_TOKEN`) is optional but recommended to avoid rate-limiting when downloading the SQuAD dataset from the Hub.

---

## Reproducibility

All experiments use identical hyperparameters:

```python
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES   = 500
EPOCHS            = 2
BATCH_SIZE        = 8
LEARNING_RATE     = 2e-5
WARMUP_STEPS      = 50
```

Random seeds are not explicitly fixed; minor run-to-run variation in results is expected.

---

## Hardware

Tested on Google Colab T4 GPU. CPU fallback is supported but training will be significantly slower.

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Key Design Decisions

- **Fair comparison:** All models use identical `preprocess_squad`, `train_one_run`, and `evaluate_model` functions.
- **BPE & Hybrid/Char tokenizers:** Trained on the SQuAD training corpus (contexts + questions) so vocabulary reflects actual task language.
- **Embedding resize:** Any model with a non-WordPiece tokenizer calls `resize_token_embeddings`, with new rows randomly initialised.
- **Character-level:** Implemented via a BPE model with `vocab_size=500` + `CharDelimiterSplit`, approximating pure character tokenization within the HuggingFace interface. `max_length` is reduced to 256 to handle ~5x longer sequences.
- **Dynamic Token Merging:** Uses the same WordPiece-preprocessed data as the baseline; only the architecture differs (a learnable `DynamicMergeLayer` is inserted between BERT's embedding output and encoder stack).