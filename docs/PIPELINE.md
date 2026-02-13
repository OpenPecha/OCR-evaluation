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
data/evaluation/{model}_cer.json      (full results: overall, per-batch, per-image)
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
│       ├── {model}_cer.csv
│       └── {model}_cer.json
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
              ┌─────────────┼──────────────────┐
              │             │                  │
              ▼             ▼                  ▼
        MODELS_CONFIG   BENCHMARK_DIR      OUTPUT_DIR
        (model settings)  (image paths)     (where to save)
              │             │                  │
              └─────────────┼──────────────────┘
                            │
                            ▼
                      pipeline.py main()
                            │
              ┌─────────────┼───────────────┐
              │             │               │
              ▼             ▼               ▼
        parse_args()    get_all_image_   Resolve which
        (--model,       paths()          models to run
         --batch)       from data_       from config
                        pipeline.py
                            │
                            ▼
                   For each model:
                            │
              ┌─────────────┤
              │             │
              ▼             ▼
        initialize_     Run OCR on each image
        pipeline()      using ThreadPoolExecutor
        (BDRC_models.py)  (NUM_WORKERS threads)
              │             │
              │             ▼
              │        run_ocr_model()
              │          │ Read image with OpenCV
              │          │ Run pipeline.run_ocr()
              │          │ Join OCR lines into text
              │          │ Return (image_name, batch_id, text)
              │             │
              └─────────────┘
                            │
                            ▼
                   write_model_output()
                   (data_pipeline.py)
                            │
                            ▼
               data/models/{model_name}.csv
```

### Detailed Step-by-Step

1. **Argument parsing** — Optional `--model` (run one model) and `--batch`
   (filter to one batch). Without arguments, all models run on all batches.

2. **Image collection** — `get_all_image_paths()` walks
   `data/benchmark/images/`, listing every batch subdirectory, then every
   image file inside it. Returns a sorted list of `(image_name, batch_id,
   image_path)` tuples.

3. **Model initialisation** — For each model in `MODELS_CONFIG`:
   - `initialize_pipeline()` loads the OCR model weights from the model
     directory and the line-detection ONNX model.
   - Builds a BDRC `OCRPipeline` object ready for inference.

4. **Concurrent inference** — Images are processed in parallel using
   `ThreadPoolExecutor` (default 4 workers). For each image:
   - `cv2.imread()` loads the image.
   - `pipeline.run_ocr()` performs line detection, text recognition, and
     optional dewarping.
   - Recognised lines are joined with `\n` into a single text string.
   - A `tqdm` progress bar tracks completion.

5. **CSV output** — `write_model_output()` writes results to
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
              (strip + collapse whitespace)
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
    {model}_cer.json      summary.csv
    {model}_cer.csv       (all models compared)
```

### Output Files

| File | Contents |
|------|----------|
| `{model}_cer.json` | Full results: overall CER, per-batch CER (with counts), per-image CER |
| `{model}_cer.csv` | Flat CSV of per-image CER for easy inspection / plotting |
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

Each model entry contains:

```python
"Model_Name": {
    "model_dir": str,        # Path to model weights directory
    "k_factor": float,       # Line-detection sensitivity
    "bbox_tolerance": float,  # Bounding box merging tolerance
    "encoding": str,          # "unicode" or "wylie"
    "merge_lines": bool,      # Whether to merge detected lines
    "dewarp": bool,           # Whether to apply TPS dewarping
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
```

### {model}_cer.json (Full Results)

```json
{
  "model": "Woodblock",
  "overall_cer": 0.31588,
  "per_batch": {
    "batch-1": { "cer": 0.207804, "count": 44 }
  },
  "per_image": [
    { "image_name": "08430009.png", "batch_id": "batch-1", "cer": 0.057297 }
  ]
}
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

### Evaluate all models

```bash
python CER_evaluation.py
```

### Evaluate a single model

```bash
python CER_evaluation.py --model Woodblock
```

### Reformat existing CSVs (fix quoting issues)

```bash
python data_pipeline.py
```
