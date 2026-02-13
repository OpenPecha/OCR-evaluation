# OCR Evaluation Pipeline — Data & Process Flow

This document describes the end-to-end flow of data and processes in the OCR
evaluation pipeline, from raw benchmark images to final CER scores.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Directory Layout](#2-directory-layout)
3. [Stage 1 — Benchmark Preparation](#3-stage-1--benchmark-preparation)
4. [Stage 2 — OCR Inference Pipeline](#4-stage-2--ocr-inference-pipeline)
5. [Stage 3 — CER Evaluation](#5-stage-3--cer-evaluation)
6. [Configuration Reference](#6-configuration-reference)
7. [Data Formats](#7-data-formats)
8. [Execution Examples](#8-execution-examples)

---

## 1. High-Level Overview

The project follows a three-stage pipeline:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Stage 1            │     │  Stage 2            │     │  Stage 3            │
│  Benchmark Prep     │────▶│  OCR Inference      │────▶│  CER Evaluation     │
│                     │     │                     │     │                     │
│  Images + Transcripts    │  Run models on       │     │  Compare inference  │
│  → benchmark.csv    │     │  benchmark images   │     │  against ground     │
│                     │     │  → {model}.csv      │     │  truth → CER scores │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

**Data flows as:**

```
Raw images + transcripts
    │
    ▼
data/benchmark/benchmark.csv          (ground-truth: image_name, batch_id, transcript)
data/benchmark/images/{batch_id}/     (image files organised by batch)
    │
    ▼  [pipeline.py — OCR inference]
    │
data/models/{model_name}.csv          (inference: image_name, batch_id, inference)
    │
    ▼  [CER_evaluation.py — CER computation]
    │
data/evaluation/{model}_cer.csv       (per-image CER scores)
data/evaluation/summary.csv           (cross-model CER comparison)
```

---

## 2. Directory Layout

```
OCR-evaluation/
├── src/
│   ├── OCR-evaluation/
│   │   ├── pipeline.py           # Stage 2: OCR inference runner
│   │   ├── CER_evaluation.py     # Stage 3: CER computation
│   │   ├── eval_config.py        # All paths, model configs, constants
│   │   ├── data_pipeline.py      # Image collection + CSV writing helpers
│   │   ├── utils.py              # General-purpose I/O utilities
│   │   ├── logging_config.py     # Centralised logging setup
│   │   └── OCR/
│   │       ├── BDRC_models.py    # BDRC OCRPipeline wrapper
│   │       └── google_cloud_vision.py  # Google Cloud Vision wrapper
│   └── data-preparations/
│       └── benchmark_data.py     # Stage 1: benchmark dataset builder
│
├── data/
│   ├── benchmark/                # INPUT: ground-truth data
│   │   ├── benchmark.csv
│   │   ├── images/{batch-1..7}/
│   │   └── transcripts/{batch-1..7}/
│   ├── models/                   # OUTPUT of Stage 2: inference CSVs
│   │   ├── Ume_Druma.csv
│   │   ├── Ume_Petsuk.csv
│   │   └── Woodblock.csv
│   └── evaluation/               # OUTPUT of Stage 3: CER results
│       ├── summary.csv
│       └── {model}_cer.csv
│
├── vendor/
│   └── tibetan-ocr-app/          # External: BDRC OCR library (Git submodule)
│
├── logs/
│   └── ocr_evaluation.log        # Debug-level log file
│
└── tests/                        # Unit tests
```

---

## 3. Stage 1 — Benchmark Preparation

**Script:** `src/data-preparations/benchmark_data.py`

**Purpose:** Take raw image and transcript data and organise it into the
standardised benchmark format the pipeline expects.

### Process Flow

```
Source data (images + annotations)
    │
    ▼
copy_images_to_destination()
    │  Copies images → data/benchmark/images/{batch_id}/
    │  Writes transcripts → data/benchmark/transcripts/{batch_id}/{stem}.txt
    │
    ▼
make_benchmark_csv()
    │  Walks images/ and transcripts/ directories
    │  Generates benchmark.csv with columns:
    │      image_name, batch_id, transcript
    │  Uses csv.writer for proper quoting (handles Tibetan text with commas/quotes)
    │
    ▼
make_stats()
    │  Counts images per batch
    │  Writes stats.json
    │
    ▼
data/benchmark/benchmark.csv    ← The single source of ground truth
```

### Key Details

- **Batch ID normalisation:** Handles legacy naming (`b1` → `batch-1`,
  `batch-05` → `batch-5`).
- **CSV quoting:** Uses `csv.writer` (not raw string formatting) so that
  Tibetan text containing commas, quotes, or newlines is properly escaped.
- This stage only needs to be run **once** when setting up the benchmark
  dataset. After that, the benchmark.csv and images are static inputs.

---

## 4. Stage 2 — OCR Inference Pipeline

**Script:** `src/OCR-evaluation/pipeline.py`

**Purpose:** Run one or more OCR models over all benchmark images and save
the raw inference text to per-model CSV files.

### Process Flow

```
                         eval_config.py
                              │
               ┌──────────────┼──────────────────┐
               │              │                  │
               ▼              ▼                  ▼
         MODELS_CONFIG    BENCHMARK_DIR      OUTPUT_DIR
         (with provider)  (image paths)      (where to save)
               │              │                  │
               └──────────────┼──────────────────┘
                              │
                              ▼
                       pipeline.py main()
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
         parse_args()   get_all_image_   Resolve which
         (--model,      paths()          models to run
          --batch)      from data_       from config
                        pipeline.py
                              │
                              ▼
                     For each model:
                              │
                     Read settings["provider"]
                              │
                     Look up _PROVIDERS[provider]
                              │
               ┌──────────────┼──────────────────────┐
               │              │                      │
               ▼              ▼                      ▼
          "bdrc"     "google_cloud_vision"    "gemini" / "deepseek"
               │              │                      │
               ▼              ▼                      ▼
         _run_bdrc()   _run_google_cloud_     _run_gemini() /
               │        vision()              _run_deepseek()
               │              │                      │
               ▼              │                      │
       initialize_pipeline()  │                      │
       (BDRC_models.py)       │                      │
       Load local ONNX        │  No local init       │  Configure API
       model weights          │  needed — uses       │  client with
               │              │  GOOGLE_APPLICATION_  │  API key from
               │              │  CREDENTIALS env var  │  config
               │              │                      │
               └──────────────┼──────────────────────┘
                              │
                              ▼
                    ThreadPoolExecutor
                    (concurrent inference)
                              │
                              ▼
                    write_model_output()
                    (data_pipeline.py)
                              │
                              ▼
                data/models/{model_name}.csv
```

### Provider Dispatch Table

The pipeline uses a dispatch table (`_PROVIDERS`) to route each model to
the correct backend based on its `"provider"` key:

| Provider | Runner Function | Initialisation | OCR Call |
|----------|----------------|----------------|----------|
| `"bdrc"` | `_run_bdrc()` | Loads local ONNX model + line detection | `run_ocr_model()` via BDRC OCRPipeline |
| `"google_cloud_vision"` | `_run_google_cloud_vision()` | None (uses `GOOGLE_APPLICATION_CREDENTIALS` env var) | `run_google_vision_ocr()` |
| `"gemini"` | `_run_gemini()` | `configure_gemini(api_key)` | `run_gemini_ocr()` |
| `"deepseek"` | `_run_deepseek()` | None (API key passed per call) | `run_deepseek_ocr()` |

Only `"bdrc"` models call `initialize_pipeline()` to load local model
weights and the line-detection ONNX file. API-based providers skip this
entirely and go straight to making API calls.

### Detailed Step-by-Step

1. **Argument parsing** — Optional `--model` (run one model) and `--batch`
   (filter to one batch). Without arguments, all models run on all batches.

2. **Image collection** — `get_all_image_paths()` walks
   `data/benchmark/images/`, listing every batch subdirectory, then every
   image file inside it. Returns a sorted list of `(image_name, batch_id,
   image_path)` tuples.

3. **Provider dispatch** — For each model in `MODELS_CONFIG`:
   - Read `settings["provider"]` (defaults to `"bdrc"` if missing).
   - Look up the matching runner function from `_PROVIDERS`.
   - If the provider is unknown, log an error and skip the model.

4. **Model initialisation (provider-specific)**:
   - **BDRC:** `initialize_pipeline()` loads the OCR model weights from
     the model directory and the line-detection ONNX model. Builds a BDRC
     `OCRPipeline` object ready for inference.
   - **Google Cloud Vision:** No initialisation — the client is lazily
     created on first API call. Requires `GOOGLE_APPLICATION_CREDENTIALS`.
   - **Gemini:** Calls `configure_gemini()` with the API key from config.
   - **DeepSeek:** No initialisation — API key is passed per request.

5. **Concurrent inference** — Images are processed in parallel using
   `ThreadPoolExecutor` (default `NUM_WORKERS=4` threads). Each provider's
   runner function manages its own thread pool with the same pattern:
   - Submit one `_ocr_task` per image.
   - Collect results in a pre-allocated indexed list.
   - `tqdm` progress bar tracks completion.

6. **CSV output** — `write_model_output()` writes results to
   `data/models/{model_name}.csv` using `csv.writer` for proper escaping.

### Concurrency Model

```
Main Thread
    │
    ├── ThreadPoolExecutor (NUM_WORKERS=4)
    │       ├── Thread 1: _ocr_task(image_0)
    │       ├── Thread 2: _ocr_task(image_1)
    │       ├── Thread 3: _ocr_task(image_2)
    │       └── Thread 4: _ocr_task(image_3)
    │       ... (processes all images with up to 4 in flight)
    │
    ▼ Results collected in original order (indexed array)
```

Results are stored in a pre-allocated list (`results[idx]`) so that the
output CSV preserves the original image ordering regardless of which thread
finishes first.

---

## 5. Stage 3 — CER Evaluation

**Script:** `src/OCR-evaluation/CER_evaluation.py`

**Purpose:** Compare model inference against ground-truth and compute
Character Error Rate (CER) at per-image, per-batch, and overall levels.

### Process Flow

```
data/benchmark/benchmark.csv          data/models/{model}.csv
        │                                      │
        ▼                                      ▼
  load_benchmark()                      load_model_csv()
  → Dict[(image_name, batch_id)]        → List[(image_name, batch_id,
    → transcript                             inference_text)]
        │                                      │
        └──────────────┬───────────────────────┘
                       │
                       ▼
              evaluate_model()
                       │
           For each (image_name, batch_id, inference):
                       │
                       ▼
              Look up ground-truth reference
              by (image_name, batch_id) key
                       │
                       ▼
              _normalise() both strings
              (optional — see CLI flags below)
                       │
                       ▼
              compute_cer(inference, reference)
              = _edit_distance(inf, ref) / len(ref)
                       │
                       ▼
              Accumulate per-image and per-batch CER
                       │
                       ▼
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
      save_evaluation()   save_summary()
              │                 │
              ▼                 ▼
    {model}_cer.csv       summary.csv
                          (all models compared)
```

### Output Files

| File | Contents |
|------|----------|
| `{model}_cer.csv` | Per-image CER scores (image_name, batch_id, cer) |
| `summary.csv` | One row per model with overall CER and per-batch CER columns |

(See [CER_EVALUATION.md](CER_EVALUATION.md) for a detailed analysis of
the CER algorithm and its correctness for Tibetan script.)

---

## 6. Configuration Reference

All configuration lives in `src/OCR-evaluation/eval_config.py`.

| Constant | Description |
|----------|-------------|
| `PROJECT_ROOT` | Auto-detected project root (2 levels up from `eval_config.py`) |
| `TIBETAN_OCR_APP_DIR` | Path to `vendor/tibetan-ocr-app/` |
| `OCR_MODELS_DIR` | Path to downloaded OCR model weights |
| `LINE_DETECTION_MODEL` | Path to the line-detection ONNX file |
| `LINE_DETECTION_PATCH_SIZE` | Patch size for line detection (default: 512) |
| `NUM_WORKERS` | Thread count for concurrent OCR inference (default: 4) |
| `BENCHMARK_DIR` | Path to `data/benchmark/` |
| `BENCHMARK_CSV` | Path to `data/benchmark/benchmark.csv` |
| `OUTPUT_DIR` | Path to `data/models/` (inference CSV output) |
| `MODELS_CONFIG` | Dict mapping model names to their OCR settings |

### MODELS_CONFIG Structure

Every model entry **must** include a `"provider"` key that tells the
pipeline which OCR backend to use. The remaining keys depend on the provider.

**BDRC models** (provider: `"bdrc"`):

```python
"Ume_Druma": {
    "provider": "bdrc",          # ← required: selects the BDRC backend
    "model_dir": str,            # Path to model weights directory
    "k_factor": float,           # Line-detection sensitivity
    "bbox_tolerance": float,     # Bounding box merging tolerance
    "encoding": str,             # "unicode" or "wylie"
    "merge_lines": bool,         # Whether to merge detected lines
    "dewarp": bool,              # Whether to apply TPS dewarping
}
```

**Google Cloud Vision** (provider: `"google_cloud_vision"`):

```python
"Google_Vision": {
    "provider": "google_cloud_vision",
    "max_workers": 4,            # Concurrent API calls
    # Requires: GOOGLE_APPLICATION_CREDENTIALS env var
}
```

**Gemini** (provider: `"gemini"`):

```python
"Gemini": {
    "provider": "gemini",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash",
}
```

**DeepSeek** (provider: `"deepseek"`):

```python
"DeepSeek": {
    "provider": "deepseek",
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "base_url": "https://api.deepseek.com",
    "model_name": "deepseek-chat",
}
```

---

## 7. Data Formats

### benchmark.csv (Ground Truth)

```csv
image_name,batch_id,transcript
08430009.png,batch-1,"བོད་སྐད་ཀྱི་ཡི་གེ..."
```

- **image_name:** Filename of the image (e.g., `08430009.png`)
- **batch_id:** Which batch the image belongs to (e.g., `batch-1`)
- **transcript:** Human-verified Tibetan text (ground truth)

### {model}.csv (Inference Output)

```csv
image_name,batch_id,inference
08430009.png,batch-1,"ཏུ་+ཡ་བྱ།ནམ་མཁའི..."
```

Same structure as benchmark.csv but with `inference` instead of `transcript`.

### {model}_cer.csv (Per-Image CER)

```csv
image_name,batch_id,cer
08430009.png,batch-1,0.057297
37080031.tif,batch-1,0.555894
```

### summary.csv (Cross-Model Comparison)

```csv
model,overall_cer,cer_batch-1,cer_batch-2,...
Ume_Druma,0.572113,0.634295,0.646037,...
Woodblock,0.31588,0.207804,0.275043,...
```

---

## 8. Execution Examples

### Run all models on all batches

```bash
cd src/OCR-evaluation
python pipeline.py
```

### Run a single model

```bash
python pipeline.py --model Woodblock
```

### Run one model on one batch

```bash
python pipeline.py --model Ume_Druma --batch batch-1
```

### Evaluate all models (raw comparison — no normalisation)

```bash
python CER_evaluation.py
```

### Evaluate a single model

```bash
python CER_evaluation.py --model Woodblock
```

### Evaluate with whitespace normalisation

Strip leading/trailing whitespace and collapse internal runs to a single
space before computing CER:

```bash
python CER_evaluation.py --normalize-whitespace
```

### Evaluate with Unicode normalisation

Apply a Unicode normalisation form (NFC, NFD, NFKC, or NFKD) to both
inference and reference before computing CER.  Useful when different OCR
engines return the same visual characters in different decomposition forms:

```bash
python CER_evaluation.py --normalize-unicode NFC
```

### Combine both normalisation options

```bash
python CER_evaluation.py --normalize-whitespace --normalize-unicode NFC
```

> **Note:** If neither `--normalize-whitespace` nor `--normalize-unicode`
> is given, CER is computed on the raw text with **no normalisation at all**.

### Reformat existing CSVs (fix quoting issues)

```bash
python data_pipeline.py
```
