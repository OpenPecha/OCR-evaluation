# CER Evaluation — Algorithm, Tibetan Script Review & Adding New Models

This document provides a detailed analysis of the Character Error Rate (CER)
evaluation logic, reviews its correctness for Tibetan script (བོད་ཡིག),
and explains how to plug in new OCR models (Gemini, Google Cloud Vision,
DeepSeek, etc.) without modifying the evaluation code.

---

## Table of Contents

1. [What is CER?](#1-what-is-cer)
2. [Algorithm Walkthrough](#2-algorithm-walkthrough)
3. [Tibetan Script Review — Is This CER Correct?](#3-tibetan-script-review--is-this-cer-correct)
4. [Normalisation Analysis](#4-normalisation-analysis)
5. [Aggregation Strategy](#5-aggregation-strategy)
6. [Edge Cases](#6-edge-cases)
7. [Recommendations for Improvement](#7-recommendations-for-improvement)
8. [Adding New OCR Models](#8-adding-new-ocr-models)

---

## 1. What is CER?

**Character Error Rate (CER)** measures how many character-level edits are
needed to transform the OCR output (inference) into the correct text
(reference), normalised by the length of the reference:

```
CER = edit_distance(inference, reference) / len(reference)
```

- **CER = 0.0** — Perfect recognition; inference matches reference exactly.
- **CER = 1.0** — The number of errors equals the reference length.
- **CER > 1.0** — Possible when inference is much longer than reference
  (many insertions).

CER is the standard metric for OCR evaluation across scripts, including
Tibetan. It is preferred over Word Error Rate (WER) for Tibetan because
Tibetan does not use spaces between words (it uses tsheg `་` between
syllables), making "word" boundaries ambiguous.

---

## 2. Algorithm Walkthrough

### 2.1 Levenshtein Edit Distance

The core of CER is the `_edit_distance()` function in `CER_evaluation.py`.

**What it computes:** The minimum number of single-character operations
(insertion, deletion, substitution) to transform string `s1` into string `s2`.

**Implementation:**

```python
def _edit_distance(s1: str, s2: str) -> int:
    # Optimisation: always iterate over the longer string in the outer loop
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    # Single-row DP (O(min(n,m)) space instead of O(n*m))
    prev = list(range(len(s2) + 1))    # Base case: row 0
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr
    return prev[-1]
```

**How this works, step by step:**

1. **Swap for space efficiency.** If `s1` is shorter than `s2`, swap them.
   This ensures the outer loop runs over the longer string and the inner
   loop (which determines memory usage) runs over the shorter one.

2. **Initialise the base row.** `prev = [0, 1, 2, ..., len(s2)]` represents
   the cost of converting an empty prefix of `s1` into each prefix of `s2`
   (all insertions).

3. **Fill the DP table row by row.** For each character `c1` in `s1`:
   - Start the row with `i` (cost of deleting all characters so far).
   - For each character `c2` in `s2`, pick the minimum of:
     - **Insertion:** `curr[j-1] + 1` (insert `c2` after position `j-1`)
     - **Deletion:** `prev[j] + 1` (delete `c1` at position `i`)
     - **Substitution:** `prev[j-1] + cost` (replace `c1` with `c2`;
       cost is 0 if they match, 1 otherwise)

4. **Return `prev[-1]`** — the bottom-right cell of the conceptual matrix.

**Correctness:** This is the textbook Levenshtein algorithm. The single-row
optimisation is mathematically equivalent to the full matrix version. The
swap optimisation reduces space from O(max(n,m)) to O(min(n,m)) without
affecting the result because edit distance is symmetric: `ed(s1, s2) = ed(s2, s1)`.

**Verdict: Correct.**

### 2.2 CER Computation

```python
def compute_cer(inference: str, reference: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(inference) == 0 else 1.0
    return _edit_distance(inference, reference) / len(reference)
```

- Both-empty → 0.0 (no error when there's nothing to recognise)
- Reference empty, inference non-empty → 1.0 (capped by convention)
- Otherwise → standard CER formula

**Verdict: Correct and follows standard conventions.**

---

## 3. Tibetan Script Review — Is This CER Correct?

### 3.1 How Tibetan Unicode Works

Tibetan (བོད་ཡིག) is an Indic abugida written left-to-right. Key features
relevant to CER:

| Feature | Example | Unicode Representation |
|---------|---------|----------------------|
| Base consonants | ཀ ཁ ག ང | Single code points (U+0F40–U+0F6C) |
| Vowel signs | ཀི ཀུ ཀེ ཀོ | Base + combining mark (e.g., U+0F40 + U+0F72) |
| Subjoined consonants | རྒ (ra + subjoined ga) | Base + subjoined mark (e.g., U+0F62 + U+0F92) |
| Stacked clusters | རྒྱ | Multiple code points: ར + ྒ + ྱ (U+0F62 + U+0F92 + U+0FB1) |
| Tsheg (syllable separator) | ་ | U+0F0B |
| Shad (punctuation) | ། | U+0F0D |
| Nyis-shad (double shad) | ༎ | U+0F0E |

**Critical observation:** A single visual Tibetan "character" (a syllable
cluster) can be composed of **multiple Unicode code points**. For example:

- **བསྒྲུབས** (a single Tibetan syllable) = བ + ས + ྒ + ྲ + ུ + བ + ས
  = 7 Unicode code points

### 3.2 What Level Does This CER Operate On?

This implementation computes CER at the **Unicode code point level** — each
Python `str` character (which is one Unicode code point) is treated as one
unit.

This means:
- Base consonant `ཀ` = 1 unit
- Vowel sign `ི` = 1 unit
- Subjoined consonant `ྒ` = 1 unit
- Tsheg `་` = 1 unit

So `རྒྱལ` (4 code points: ར + ྒ + ྱ + ལ) counts as 4 characters.

### 3.3 Is Code-Point-Level CER Appropriate for Tibetan?

**Yes — this is the standard and accepted approach.**

Here is the reasoning:

1. **Industry standard.** Most published Tibetan OCR research (including work
   from BDRC, Esukhia, and academic papers on Tibetan OCR) uses Unicode
   code-point-level CER. This makes results directly comparable with the
   existing literature.

2. **Granularity matches OCR errors.** Common OCR mistakes in Tibetan
   correspond naturally to code-point-level operations:
   - Misrecognising a vowel sign (one substitution at code-point level)
   - Missing a subjoined consonant (one deletion at code-point level)
   - Confusing similar-looking base consonants like ཤ/ཥ (one substitution)
   - Extra or missing tsheg marks (one insertion/deletion)

3. **Alternative: grapheme-cluster-level CER** would treat each visual stack
   as one unit. This would mask important distinctions — for example,
   confusing `རྒ` with `རྒྱ` (missing one subjoined consonant) would be
   1 substitution at cluster level but 1 deletion at code-point level. The
   code-point approach gives a more precise error breakdown.

4. **Alternative: syllable-level or tsheg-unit CER** would be closer to
   Word Error Rate and would lose character-level granularity entirely.

**Verdict: The code-point-level CER is correct and appropriate for Tibetan.**

### 3.4 One Caveat: Unicode Normalisation

The current implementation does **not** apply Unicode normalisation (NFC/NFD).
In practice, Tibetan Unicode text is almost always in NFC (precomposed) form,
but if one source (e.g., a particular OCR engine) produces NFD
(decomposed) output, the same visual text could have different byte
representations, causing inflated CER.

**Risk level:** Low for the current BDRC models (they output consistent
Unicode), but **important when adding new models** like Gemini or Google
Vision that may produce differently-normalised output.

**Recommendation:** Add `unicodedata.normalize("NFC", text)` to the
`_normalise()` function. See [Section 7](#7-recommendations-for-improvement).

---

## 4. Normalisation Analysis

The `_normalise()` function applies minimal preprocessing before CER
computation:

```python
def _normalise(text: str) -> str:
    return " ".join(text.split())
```

**What this does:**
1. `text.split()` — splits on any whitespace (spaces, tabs, newlines,
   `\r\n`, etc.) and discards empty strings from leading/trailing/consecutive
   whitespace.
2. `" ".join(...)` — rejoins with a single space.

**Effect:** Strips leading/trailing whitespace and collapses internal runs
of whitespace to a single space.

**Why this matters for Tibetan:**
- Tibetan text uses tsheg `་` (U+0F0B) as a syllable delimiter, **not**
  spaces. Spaces in Tibetan OCR output are generally artifacts of line
  breaks or formatting, not linguistic content.
- The normalisation ensures that differences in line-break conventions
  (`\n` vs `\r\n`) or trailing whitespace don't inflate CER.
- Tsheg marks are **not** affected by this normalisation (they are not
  whitespace), which is correct — tsheg presence/absence is a real OCR
  error.

**Verdict: The normalisation is appropriate.** It handles the most common
source of trivial differences (whitespace) without touching
linguistically meaningful Tibetan characters.

---

## 5. Aggregation Strategy

### Per-Image CER

Each image gets its own CER score:

```python
cer = compute_cer(normalise(inference), normalise(reference))
```

The per-image CER is stored with 6 decimal places of precision.

### Per-Batch CER

**Method: Macro-average** (mean of per-image CERs within the batch):

```python
batch_cer = sum(image_cers) / len(image_cers)
```

This gives equal weight to every image regardless of text length. An image
with 10 characters and an image with 1000 characters contribute equally.

**Alternative not used:** Micro-average (total edits / total reference length)
would weight longer texts more heavily.

### Overall CER

**Method: Macro-average** across all images:

```python
overall_cer = sum(all_image_cers) / len(all_image_cers)
```

Same approach — every image has equal weight.

**Verdict:** Macro-averaging is a reasonable default. It prevents a few
very long texts from dominating the score. The choice between macro and
micro averaging is a design decision, not a correctness issue — both are
valid. The current approach is clearly documented and consistently applied.

---

## 6. Edge Cases

| Scenario | Behaviour | Correct? |
|----------|-----------|----------|
| Both reference and inference are empty | CER = 0.0 | Yes |
| Reference is empty, inference is not | CER = 1.0 | Yes (convention) |
| Inference is empty, reference is not | CER = len(ref)/len(ref) = 1.0 | Yes |
| Inference is NaN (missing) | Treated as empty string `""` | Yes |
| Reference not found in benchmark | Treated as empty string `""` | Reasonable |
| Image in benchmark but not in model CSV | Not evaluated (silently skipped) | Acceptable |

The code handles `NaN` values from pandas explicitly:

```python
_normalise(inference_text if pd.notna(inference_text) else "")
_normalise(reference if pd.notna(reference) else "")
```

**Verdict: Edge cases are handled correctly.**

---

## 7. Recommendations for Improvement

### 7.1 Add Unicode NFC Normalisation (Recommended)

Add NFC normalisation to prevent inflated CER when different OCR engines
produce differently-normalised Unicode:

```python
import unicodedata

def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())
```

**Why:** Google Cloud Vision, Gemini, and DeepSeek may output Tibetan text
in NFD form. A single visual vowel like `ི` could be represented differently
in NFC vs NFD, causing spurious edit distance.

### 7.2 Consider Micro-Average as an Additional Metric

Report both macro-average CER (current) and micro-average CER (total edits
divided by total reference characters) in the JSON output. This gives a
fuller picture — macro-average tells you "how well does the model do on a
typical image" while micro-average tells you "what fraction of all
characters are wrong."

### 7.3 Consider Tibetan-Specific Normalisation (Optional)

For stricter evaluation, you could normalise Tibetan-specific whitespace
and punctuation variations:

- Multiple tsheg marks `་་` → single tsheg `་`
- Trailing tsheg before shad `་།` → `།` (stylistic variation)
- Zero-width spaces (U+200B) → removed

This would be optional and should be clearly documented if applied, as it
changes what the CER measures.

### 7.4 Use `jiwer` or `editdistance` Libraries (Optional)

The custom Levenshtein implementation is correct, but using a C-extension
library like `python-Levenshtein` or `editdistance` would be faster for
large-scale evaluation. For the current benchmark size (~630 images), the
custom implementation is perfectly adequate.

---

## 8. Adding New OCR Models

The project is designed so that the CER evaluation (`CER_evaluation.py`) is
**completely decoupled** from the OCR inference. The evaluation only reads
CSV files. This means:

> **To add a new model, you only need to produce a CSV file in the right
> format. You do not need to modify the evaluation code.**

### 8.1 The Contract

Any new model must produce a CSV file at `data/models/{model_name}.csv` with
exactly these columns:

```csv
image_name,batch_id,inference
08430009.png,batch-1,"ཏུ་+ཡ་བྱ།..."
```

- **image_name** must match the filenames in `data/benchmark/images/`
- **batch_id** must match the batch directory names
- **inference** is the OCR output text (properly CSV-quoted)

Once this CSV exists, `python CER_evaluation.py` will automatically discover
it and compute CER against the benchmark.

### 8.2 Step-by-Step: Adding Google Cloud Vision

Google Cloud Vision is already partially integrated
(`src/OCR-evaluation/OCR/google_cloud_vision.py`). Here's how to complete it:

**1. Install the dependency:**

```bash
pip install google-cloud-vision
```

**2. Set up credentials:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**3. Add the model to `eval_config.py`:**

```python
# ── External / API-based models ──────────────────────────────────────────
EXTERNAL_MODELS_CONFIG = {
    "Google_Vision": {
        "provider": "google_cloud_vision",
        "max_workers": 4,      # concurrent API calls
    },
}
```

**4. Create a runner script** (or extend `pipeline.py`):

```python
"""run_google_vision.py — Run Google Cloud Vision on benchmark images."""
import csv
from pathlib import Path
from data_pipeline import get_all_image_paths
from eval_config import BENCHMARK_DIR, OUTPUT_DIR
from OCR.google_cloud_vision import run_google_vision_ocr

images = get_all_image_paths(BENCHMARK_DIR)
output_csv = OUTPUT_DIR / "Google_Vision.csv"

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "batch_id", "inference"])
    for image_name, batch_id, image_path in images:
        text = run_google_vision_ocr(str(image_path))
        writer.writerow([image_name, batch_id, text])
```

**5. Evaluate:**

```bash
python CER_evaluation.py --model Google_Vision
```

### 8.3 Step-by-Step: Adding Gemini

**1. Install the dependency:**

```bash
pip install google-generativeai
```

**2. Create the OCR wrapper** at `src/OCR-evaluation/OCR/gemini.py`:

```python
"""Gemini OCR integration."""
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed.")


def configure_gemini(api_key: str):
    """Configure Gemini with your API key."""
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("pip install google-generativeai")
    genai.configure(api_key=api_key)


def run_gemini_ocr(image_path: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Run Gemini vision model on a single image for Tibetan OCR.

    Returns the extracted Tibetan text, or "" on failure.
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("pip install google-generativeai")

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found: %s", image_path)
        return ""

    try:
        model = genai.GenerativeModel(model_name)
        image_data = path.read_bytes()

        response = model.generate_content([
            "Extract all Tibetan text from this image. "
            "Return only the raw Tibetan text, no translation or explanation.",
            {
                "mime_type": f"image/{path.suffix.lstrip('.')}",
                "data": image_data,
            },
        ])
        return response.text.strip()
    except Exception:
        logger.exception("Gemini OCR failed for %s", image_path)
        return ""
```

**3. Add config in `eval_config.py`:**

```python
# Gemini config
GEMINI_CONFIG = {
    "api_key": "YOUR_API_KEY",  # or use: os.environ.get("GEMINI_API_KEY")
    "model_name": "gemini-2.0-flash",
}
```

**4. Create a runner script** similar to the Google Vision one above, calling
`run_gemini_ocr()` for each image and writing to
`data/models/Gemini.csv`.

**5. Evaluate:**

```bash
python CER_evaluation.py --model Gemini
```

### 8.4 Step-by-Step: Adding DeepSeek

**1. Install the dependency:**

```bash
pip install openai  # DeepSeek uses OpenAI-compatible API
```

**2. Create the OCR wrapper** at `src/OCR-evaluation/OCR/deepseek.py`:

```python
"""DeepSeek vision OCR integration."""
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _DEEPSEEK_AVAILABLE = True
except ImportError:
    _DEEPSEEK_AVAILABLE = False
    logger.warning("openai package not installed.")


def run_deepseek_ocr(
    image_path: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
) -> str:
    """
    Run DeepSeek vision model on a single image for Tibetan OCR.

    Returns the extracted Tibetan text, or "" on failure.
    """
    if not _DEEPSEEK_AVAILABLE:
        raise RuntimeError("pip install openai")

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found: %s", image_path)
        return ""

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime_type = f"image/{path.suffix.lstrip('.')}"

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all Tibetan text from this image. "
                                "Return only the raw Tibetan text.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                        },
                    },
                ],
            }],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("DeepSeek OCR failed for %s", image_path)
        return ""
```

**3. Add config in `eval_config.py`:**

```python
# DeepSeek config
DEEPSEEK_CONFIG = {
    "api_key": "YOUR_API_KEY",  # or use: os.environ.get("DEEPSEEK_API_KEY")
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
}
```

### 8.5 Generic Pattern for Any New Model

The pattern is always the same:

```
1. Create  src/OCR-evaluation/OCR/{provider}.py
             └── def run_{provider}_ocr(image_path, **kwargs) -> str

2. Add credentials/settings to eval_config.py

3. Write a runner script that:
     a. Calls get_all_image_paths() to get benchmark images
     b. Calls your OCR function for each image
     c. Writes results to data/models/{Model_Name}.csv

4. Run:  python CER_evaluation.py --model {Model_Name}
```

The evaluation pipeline auto-discovers any `.csv` in `data/models/` and
computes CER against the benchmark — no changes needed to
`CER_evaluation.py`.

### 8.6 Config Summary for All Models

Here is what `eval_config.py` would look like with all models configured:

```python
# ── BDRC local models ────────────────────────────────────────────────────
MODELS_CONFIG = {
    "Ume_Druma":  { "model_dir": ..., "k_factor": 2.5, ... },
    "Ume_Petsuk": { "model_dir": ..., "k_factor": 2.5, ... },
    "Woodblock":  { "model_dir": ..., "k_factor": 2.5, ... },
}

# ── Google Cloud Vision ──────────────────────────────────────────────────
# Requires: GOOGLE_APPLICATION_CREDENTIALS env var
GOOGLE_VISION_CONFIG = {
    "max_workers": 4,
}

# ── Gemini ───────────────────────────────────────────────────────────────
GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash",
}

# ── DeepSeek ─────────────────────────────────────────────────────────────
DEEPSEEK_CONFIG = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
}
```

**Best practice:** Never hardcode API keys. Use environment variables:

```bash
export GEMINI_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Levenshtein edit distance | **Correct** | Standard DP algorithm with space optimisation |
| CER formula | **Correct** | Standard: edit_distance / reference_length |
| Operates at Unicode code-point level | **Correct for Tibetan** | Matches published Tibetan OCR research |
| Whitespace normalisation | **Appropriate** | Handles line-break artifacts without touching tsheg |
| Unicode NFC normalisation | **Missing but recommended** | Important when mixing OCR engines |
| Macro-average aggregation | **Valid** | Equal weight per image; consider adding micro-average |
| Edge case handling | **Correct** | Empty strings, NaN values handled properly |
| Plugging in new models | **Supported** | Just produce a CSV in the right format |
