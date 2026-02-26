# OCR Evaluation

Evaluate multiple OCR models on Tibetan benchmark images. Results are saved as CSVs in `data/models/`.

## Setup

```bash
pip install -r requirements.txt
```

## Setting API keys (current terminal session only)

```bash
# Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# Google Cloud Vision (uses a service account JSON)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# DeepSeek (if enabled)
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

## Running the pipeline

All commands are run from `src/OCR-evaluation/`:

```bash
cd src/OCR-evaluation
```

### Run a single model on a single batch

```bash
python pipeline.py --model Gemini_Prompt_3_Flash --batch batch-1
```

### Run a single model on all batches

```bash
python pipeline.py --model Gemini_Prompt_3_Flash
```

### Run all configured models

```bash
python pipeline.py
```

## Available models

| `--model` value | Provider | Description |
|---|---|---|
| `Ume_Druma` | BDRC (local) | Ume Druma OCR model |
| `Ume_Petsuk` | BDRC (local) | Ume Petsuk OCR model |
| `Woodblock` | BDRC (local) | Woodblock OCR model |
| `Google_Vision` | Google Cloud Vision | Cloud Vision TEXT_DETECTION API |
| `Gemini_2_Flash` | Gemini | gemini-2.0-flash |
| `Gemini_2.5_Flash` | Gemini | gemini-2.5-flash |
| `Gemini_3_Flash` | Gemini | gemini-3-flash-preview (plain prompt) |
| `Gemini_3_Pro` | Gemini | gemini-3-pro-preview (plain prompt) |
| `Gemini_Prompt_3_Flash` | Gemini | gemini-3-flash-preview (structured prompt with JSON output) |

## Output

- **CSV results**: `data/models/{model_name}.csv` with columns `image_name, batch_id, inference`
- **Gemini raw text**: `data/gemini_raw/{model_name}/{batch_id}/{image_stem}.txt`
- **Gemini structured JSON** (Gemini_Prompt_3_Flash only): `data/gemini_raw/Gemini_Prompt_3_Flash/{batch_id}/{image_stem}.json`
- **Google Vision raw responses**: `data/google_vision_raw/{batch_id}/{image_stem}.json.gz`

Cached raw files are reused on subsequent runs; delete them to force re-OCR.
