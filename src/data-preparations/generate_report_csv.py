"""
Report generation: build consolidated metadata from benchmark catalog CSVs
and script directory CSVs, merge with benchmark data, attach CER evaluation
scores, and produce clustered report CSVs.

Catalog technologies
--------------------
blockprints, digital_fonts, manuscripts, metal_types, modern, typewriters.

Pipeline
--------
Stage 1 – Consolidated metadata
  1. Read all per-technology catalog CSVs (keeping **all** columns).
  2. Join with Script and Multi-script CSVs on ``script_id`` to fill in
     script-level metadata (name, 3_types, 8_categories, period, …).
  3. Write ``data/metadata/consolidated_metadata.csv``.

Stage 2 – Merge with benchmark
  Using ``data/benchmark/benchmark.csv`` and consolidated metadata, produce
  three files:
  - ``merged_report.csv``        : images present in BOTH benchmark and
                                   consolidated, with all metadata columns.
  - ``only_in_consolidated.csv`` : images in consolidated but NOT in benchmark.
  - ``only_in_benchmark.csv``    : images in benchmark but NOT in consolidated.

Stage 3 – Attach CER scores & produce clusters
  Load ``*_cer.csv`` evaluation files and join CER scores onto the merged
  report.  Then generate per-cluster CSVs grouped by technology, main style,
  and rarity—each containing **all** metadata columns plus per-model CER
  values (same column set as the merged report).  Also produce a
  ``final_summary.csv`` with average CER per model across groupings.

Output layout
-------------
data/metadata/
└── consolidated_metadata.csv

data/reports/
├── merged_report.csv
├── only_in_consolidated.csv
├── only_in_benchmark.csv
├── final_summary.csv
├── by_technology/
│   └── <technology>.csv
├── by_main_styles/
│   └── <style>.csv
└── by_rarity/
    └── <rarity>.csv
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CATALOG_DIR = DATA_DIR / "metadata" / "catalog"
SCRIPT_DIR = DATA_DIR / "metadata" / "script"
BENCHMARK_CSV = DATA_DIR / "benchmark" / "benchmark.csv"
EVALUATION_DIR = DATA_DIR / "evaluation"

METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = DATA_DIR / "reports"

# ── Catalog file stems → technology labels ────────────────────────────────────
CATALOG_FILES: dict[str, str] = {
    "Benchmark catalog - blockprints": "blockprints",
    "Benchmark catalog - digital_fonts": "digital_fonts",
    "Benchmark catalog - manuscripts": "manuscripts",
    "Benchmark catalog - metal_types": "metal_types",
    "Benchmark catalog - modern": "modern",
    "Benchmark catalog - typewriters": "typewriters",
}

# ── Column normalization (applied after lowercasing + stripping) ──────────────
CATALOG_COLUMN_RENAMES: dict[str, str] = {
    "file name": "file_name",
    "script id": "script_id",
    "exact font": "exact_font",
    "font ps_name": "font_ps_name",
    "font other_names": "font_other_names",
    "note": "notes",
}

SCRIPTS_COLUMN_RENAMES: dict[str, str] = {
    "id": "script_id",
    "name (phonetics, wylie in parentheses, and english)": "script_name",
    "3 types": "3_types",
    "8 categories": "8_categories",
    "descenders length ratio": "descenders_length_ratio",
    "gigu angle for cursives": "gigu_angle_for_cursives",
    "popularity on bdrc": "popularity_on_bdrc",
}

MULTI_SCRIPTS_COLUMN_RENAMES: dict[str, str] = {
    "ids": "script_id",
    "names (phonetics)": "script_name",
    "3 type combination": "3_type_combination",
    "8 cat. combination": "8_cat_combination",
    "number of scripts": "number_of_scripts",
    "popularity on bdrc": "popularity_on_bdrc",
}

# ── Preferred column ordering for cluster CSVs ───────────────────────────────
#    All columns from the merged report are included in each cluster CSV.
#    Columns listed here appear first (in this order); any remaining metadata
#    columns follow alphabetically, then CER columns at the end.
CLUSTER_LEADING_COLUMNS: list[str] = [
    "image_name",
    "batch_id",
    "technology",
    "script_id",
    "script_name",
    "format",
    "legibility",
    "color",
    "interlinear",
    "source",
    "notes",
    "exact_font",
    "font_ps_name",
    "font_other_names",
    "font_path",
    "ttc_face_index",
    "font_size_pt",
    "dpi",
    "skt_ok",
    "3_types",
    "8_categories",
    "3_type_combination",
    "8_cat_combination",
    "number_of_scripts",
    "descenders_length_ratio",
    "gigu_angle_for_cursives",
    "popularity_on_bdrc",
    "period",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Lowercase, strip, drop unnamed/empty columns, apply *renames*."""
    df.columns = df.columns.str.lower().str.strip()
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    df = df.drop(columns=[""], errors="ignore")
    df = df.rename(columns=renames)
    return df


def _safe_filename(value: str) -> str:
    """Convert a cluster value into a filesystem-safe filename (no extension)."""
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _merge_with_fill(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Left-join *left* with *right* on *on*.

    For columns present in both DataFrames, values already in *left* take
    precedence; only NaN cells are filled from *right*.
    """
    right_cols = [c for c in right.columns if c != on]

    merged = left.merge(right, on=on, how="left", suffixes=("", suffix))

    for col in right_cols:
        suffixed = f"{col}{suffix}"
        if suffixed in merged.columns:
            merged[col] = merged[col].combine_first(merged[suffixed])
            merged.drop(columns=[suffixed], inplace=True)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Build consolidated metadata (catalogs + scripts, all columns)
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogs(catalog_dir: Path = CATALOG_DIR) -> pd.DataFrame:
    """
    Read every per-technology catalog CSV, keeping **all** columns.
    A ``technology`` column is prepended to each.
    """
    frames: list[pd.DataFrame] = []

    for stem, technology in CATALOG_FILES.items():
        path = catalog_dir / f"{stem}.csv"
        if not path.exists():
            logger.warning("Missing catalog file: %s", path)
            continue

        df = pd.read_csv(path, dtype=str)
        df = _normalize_columns(df, CATALOG_COLUMN_RENAMES)
        df.insert(0, "technology", technology)
        frames.append(df)
        logger.info(
            "Loaded catalog  %-30s  %d rows, %d cols",
            technology, len(df), len(df.columns),
        )

    if not frames:
        raise FileNotFoundError(f"No catalog CSVs found in {catalog_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows without a file name (empty padding rows in some sheets)
    combined = combined.dropna(subset=["file_name"])
    combined = combined[combined["file_name"].str.strip() != ""]

    logger.info(
        "All catalogs combined: %d rows, %d columns",
        len(combined), len(combined.columns),
    )
    return combined


def load_script_csvs(
    script_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Scripts and Multi-scripts CSVs, normalized for joining."""
    scripts_path = script_dir / "Script lists - Scripts.csv"
    scripts = pd.read_csv(scripts_path, dtype=str)
    scripts = _normalize_columns(scripts, SCRIPTS_COLUMN_RENAMES)
    logger.info("Loaded scripts CSV: %d rows, cols %s", len(scripts), list(scripts.columns))

    multi_path = script_dir / "Script lists - Multi-scripts.csv"
    multi = pd.read_csv(multi_path, dtype=str)
    multi = _normalize_columns(multi, MULTI_SCRIPTS_COLUMN_RENAMES)
    logger.info("Loaded multi-scripts CSV: %d rows, cols %s", len(multi), list(multi.columns))

    return scripts, multi


def join_script_metadata(
    catalog: pd.DataFrame,
    scripts: pd.DataFrame,
    multi_scripts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join catalog with both script CSVs on ``script_id``.

    Catalog values always take precedence; script values only fill NaN cells.
    """
    # Clean script_id for matching
    for df in (catalog, scripts, multi_scripts):
        if "script_id" in df.columns:
            df["script_id"] = df["script_id"].astype(str).str.strip()

    result = _merge_with_fill(catalog, scripts, on="script_id", suffix="_script")
    result = _merge_with_fill(result, multi_scripts, on="script_id", suffix="_multi")

    matched_scripts = result["script_name"].notna().sum() if "script_name" in result.columns else 0
    logger.info(
        "Script join: %d / %d rows matched a script entry",
        matched_scripts, len(result),
    )
    return result


def build_consolidated_metadata(
    catalog_dir: Path = CATALOG_DIR,
    script_dir: Path = SCRIPT_DIR,
) -> pd.DataFrame:
    """
    Build consolidated metadata from all catalog CSVs joined with script
    metadata.  **All** columns from both sources are preserved—no filtering
    by evaluation or benchmark data at this stage.

    Returns
    -------
    consolidated : pd.DataFrame
        Every catalog image with full catalog + script metadata.
    """
    # 1. Load all catalog CSVs (all columns preserved)
    catalog = load_catalogs(catalog_dir)

    # 2. Load script CSVs and join on script_id
    scripts, multi_scripts = load_script_csvs(script_dir)
    consolidated = join_script_metadata(catalog, scripts, multi_scripts)

    logger.info(
        "Consolidated metadata: %d rows, %d columns",
        len(consolidated), len(consolidated.columns),
    )
    return consolidated


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – Merge consolidated metadata with benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def merge_with_benchmark(
    consolidated: pd.DataFrame,
    benchmark_csv: Path = BENCHMARK_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge consolidated metadata with the benchmark CSV.

    Returns
    -------
    merged : pd.DataFrame
        Images present in BOTH benchmark and consolidated (inner join),
        with all metadata columns from consolidated plus ``batch_id``
        from the benchmark (transcript is excluded).
    only_in_consolidated : pd.DataFrame
        Images in consolidated that are NOT in the benchmark.
    only_in_benchmark : pd.DataFrame
        Images in benchmark that are NOT in consolidated.
    """
    benchmark = pd.read_csv(benchmark_csv, dtype=str)
    benchmark.columns = benchmark.columns.str.strip()
    logger.info("Loaded benchmark: %d rows", len(benchmark))

    # Normalize join keys
    consolidated = consolidated.copy()
    consolidated["file_name"] = consolidated["file_name"].str.strip()
    benchmark["image_name"] = benchmark["image_name"].str.strip()

    benchmark_names = set(benchmark["image_name"])
    consolidated_names = set(consolidated["file_name"])

    # 1. Merged: inner join (images in BOTH)
    merged = benchmark.merge(
        consolidated,
        left_on="image_name",
        right_on="file_name",
        how="inner",
    )
    # Drop file_name (duplicate of image_name) and transcript
    merged.drop(columns=["file_name", "transcript"], errors="ignore", inplace=True)
    logger.info("Merged (in both): %d images", len(merged))

    # 2. Only in consolidated (not in benchmark)
    only_consol_mask = ~consolidated["file_name"].isin(benchmark_names)
    only_in_consolidated = consolidated[only_consol_mask].copy()
    logger.info(
        "Only in consolidated (not in benchmark): %d images",
        len(only_in_consolidated),
    )

    # 3. Only in benchmark (not in consolidated)
    only_bench_mask = ~benchmark["image_name"].isin(consolidated_names)
    only_in_benchmark = benchmark[only_bench_mask].copy()
    logger.info(
        "Only in benchmark (not in consolidated): %d images",
        len(only_in_benchmark),
    )

    return merged, only_in_consolidated, only_in_benchmark


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – Attach CER scores & produce clusters
# ═══════════════════════════════════════════════════════════════════════════════

def load_evaluation_csvs(evaluation_dir: Path = EVALUATION_DIR) -> pd.DataFrame:
    """
    Load all ``*_cer.csv`` files and pivot each model's CER into its own
    column.  Returns columns: ``image_name``, ``batch_id``, ``<Model>_cer …``
    """
    cer_files = sorted(evaluation_dir.glob("*_cer.csv"))
    if not cer_files:
        raise FileNotFoundError(f"No *_cer.csv files in {evaluation_dir}")

    frames: list[pd.DataFrame] = []
    for path in cer_files:
        model_name = path.stem  # e.g. "Ume_Druma_cer"
        df = pd.read_csv(path, dtype={"image_name": str, "batch_id": str})
        df = df.rename(columns={"cer": model_name})
        frames.append(df)
        logger.info("Loaded evaluation: %-25s  %d rows", model_name, len(df))

    evaluation = frames[0]
    for df in frames[1:]:
        evaluation = evaluation.merge(df, on=["image_name", "batch_id"], how="outer")

    logger.info(
        "Combined evaluation: %d images × %d model(s)",
        len(evaluation), len(frames),
    )
    return evaluation


def attach_cer_scores(
    merged: pd.DataFrame,
    evaluation_dir: Path = EVALUATION_DIR,
) -> pd.DataFrame:
    """
    Join CER evaluation scores onto the merged report.

    Joins on ``["image_name", "batch_id"]`` so that CER values line up
    precisely with the benchmark images.
    """
    evaluation = load_evaluation_csvs(evaluation_dir)

    result = merged.merge(
        evaluation,
        on=["image_name", "batch_id"],
        how="left",
    )

    cer_cols = [c for c in result.columns if c.endswith("_cer")]
    matched = result[cer_cols[0]].notna().sum() if cer_cols else 0
    logger.info(
        "Attached CER scores: %d / %d images have CER values (%d models)",
        matched, len(result), len(cer_cols),
    )
    return result


# ── Report writers ────────────────────────────────────────────────────────────

def _ordered_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all columns of *df* in a stable, readable order:
    leading metadata columns first (from ``CLUSTER_LEADING_COLUMNS``),
    then any remaining non-CER columns alphabetically, then CER columns
    alphabetically at the end.
    """
    all_cols = set(df.columns)
    cer_cols = sorted(c for c in all_cols if c.endswith("_cer"))
    non_cer = all_cols - set(cer_cols)

    ordered: list[str] = []
    for c in CLUSTER_LEADING_COLUMNS:
        if c in non_cer:
            ordered.append(c)
            non_cer.discard(c)

    ordered += sorted(non_cer)
    ordered += cer_cols
    return ordered


def generate_cluster_csvs(
    merged: pd.DataFrame,
    group_column: str,
    output_dir: Path,
) -> list[Path]:
    """
    Split *merged* by unique values in *group_column* and write one
    CSV per group, keeping **all** columns in a consistent order.

    Rows where *group_column* is NaN / empty go to ``_unspecified.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    cols = _ordered_columns(merged)

    mask_na = merged[group_column].isna() | (
        merged[group_column].astype(str).str.strip() == ""
    )

    for value, group in merged[~mask_na].groupby(group_column, sort=True):
        path = output_dir / f"{_safe_filename(value)}.csv"
        group[cols].to_csv(path, index=False)
        written.append(path)
        logger.info("  %-40s %4d rows → %s", value, len(group), path.name)

    if mask_na.any():
        path = output_dir / "_unspecified.csv"
        merged[mask_na][cols].to_csv(path, index=False)
        written.append(path)
        logger.info(
            "  %-40s %4d rows → %s",
            "(unspecified / NA)", mask_na.sum(), path.name,
        )

    return written


def generate_final_summary(
    merged: pd.DataFrame,
    output_path: Path,
    full_evaluation: pd.DataFrame | None = None,
    only_in_benchmark_names: set[str] | None = None,
) -> Path:
    """
    Produce a summary CSV with average CER per model, grouped by
    technology, main style, and rarity.

    If *full_evaluation* is provided, two extra rows are included:
    - ``overall / whole_benchmark`` – CER across **all** benchmark images
    - ``overall / only_in_benchmark`` – CER for images that are in the
      benchmark but have no catalog metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cer_cols = [c for c in merged.columns if c.endswith("_cer")]
    rows: list[dict] = []

    # ── Whole benchmark (all images evaluated, whether cataloged or not) ──
    if full_evaluation is not None:
        eval_cer_cols = [c for c in full_evaluation.columns if c.endswith("_cer")]
        row: dict = {
            "category": "overall",
            "group": "whole_benchmark",
            "image_count": len(full_evaluation),
        }
        for col in eval_cer_cols:
            model = col.removesuffix("_cer")
            row[f"{model}_avg_cer"] = round(full_evaluation[col].mean(), 6)
        rows.append(row)

    # ── Merged (images in both benchmark and catalog) ─────────────────────
    row = {"category": "overall", "group": "in_both", "image_count": len(merged)}
    for col in cer_cols:
        model = col.removesuffix("_cer")
        row[f"{model}_avg_cer"] = round(merged[col].mean(), 6)
    rows.append(row)

    # ── Only in benchmark (no catalog metadata) ───────────────────────────
    if full_evaluation is not None and only_in_benchmark_names:
        eval_cer_cols = [c for c in full_evaluation.columns if c.endswith("_cer")]
        bench_only = full_evaluation[
            full_evaluation["image_name"].isin(only_in_benchmark_names)
        ]
        if not bench_only.empty:
            row = {
                "category": "overall",
                "group": "only_in_benchmark",
                "image_count": len(bench_only),
            }
            for col in eval_cer_cols:
                model = col.removesuffix("_cer")
                row[f"{model}_avg_cer"] = round(bench_only[col].mean(), 6)
            rows.append(row)

    # ── Grouped summaries ─────────────────────────────────────────────────
    group_configs = {
        "technology": "technology",
        "main_style": "3_types",
        "rarity": "popularity_on_bdrc",
    }
    for category_label, column in group_configs.items():
        if column not in merged.columns:
            continue
        mask = merged[column].notna() & (merged[column].astype(str).str.strip() != "")
        for value, group in merged[mask].groupby(column, sort=True):
            row = {
                "category": category_label,
                "group": value,
                "image_count": len(group),
            }
            for col in cer_cols:
                model = col.removesuffix("_cer")
                row[f"{model}_avg_cer"] = round(group[col].mean(), 6)
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    avg_cols = sorted(c for c in df_summary.columns if c.endswith("_avg_cer"))
    df_summary = df_summary[["category", "group", "image_count"] + avg_cols]

    df_summary.to_csv(output_path, index=False)
    logger.info("Wrote final summary → %s  (%d rows)", output_path, len(df_summary))
    return output_path


def generate_clusters_from_merged(
    merged_csv: Path,
    reports_dir: Path = REPORTS_DIR,
    evaluation_dir: Path = EVALUATION_DIR,
    only_in_benchmark_csv: Path | None = None,
) -> None:
    """
    Read the single ``merged_report.csv`` (which already contains all
    metadata columns **and** CER scores) and produce:

    - ``final_summary.csv``   – average CER per model across groupings,
      including whole-benchmark and only-in-benchmark rows
    - ``by_technology/*.csv``  – one CSV per technology
    - ``by_main_styles/*.csv`` – one CSV per 3_types value
    - ``by_rarity/*.csv``      – one CSV per popularity_on_bdrc value

    The merged CSV is the single source for cluster CSVs.  The evaluation
    directory and only-in-benchmark list are used only for the summary's
    whole-benchmark and only-in-benchmark aggregate rows.
    """
    merged = pd.read_csv(merged_csv, dtype=str)
    logger.info(
        "Loaded merged report for clustering: %d rows, %d cols",
        len(merged), len(merged.columns),
    )

    # Convert CER columns back to float for aggregation
    cer_cols = [c for c in merged.columns if c.endswith("_cer")]
    for col in cer_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Load full evaluation data for whole-benchmark summary rows
    full_evaluation = load_evaluation_csvs(evaluation_dir)
    eval_cer_cols = [c for c in full_evaluation.columns if c.endswith("_cer")]
    for col in eval_cer_cols:
        full_evaluation[col] = pd.to_numeric(full_evaluation[col], errors="coerce")

    # Load only-in-benchmark image names
    only_in_benchmark_names: set[str] = set()
    if only_in_benchmark_csv is None:
        only_in_benchmark_csv = reports_dir / "only_in_benchmark.csv"
    if only_in_benchmark_csv.exists():
        oib = pd.read_csv(only_in_benchmark_csv, dtype=str)
        only_in_benchmark_names = set(oib["image_name"].str.strip())

    # ── Final summary ─────────────────────────────────────────────────────
    generate_final_summary(
        merged,
        reports_dir / "final_summary.csv",
        full_evaluation=full_evaluation,
        only_in_benchmark_names=only_in_benchmark_names,
    )

    # ── Cluster CSVs (all columns included in each) ─────────────────────
    cluster_configs = {
        "technology": reports_dir / "by_technology",
        "3_types": reports_dir / "by_main_styles",
        "popularity_on_bdrc": reports_dir / "by_rarity",
    }
    for column, out_dir in cluster_configs.items():
        if column in merged.columns:
            logger.info("Clustering by '%s' → %s/", column, out_dir)
            generate_cluster_csvs(merged, column, out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_reports(
    catalog_dir: Path = CATALOG_DIR,
    script_dir: Path = SCRIPT_DIR,
    benchmark_csv: Path = BENCHMARK_CSV,
    evaluation_dir: Path = EVALUATION_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """
    End-to-end report generation:

    1. Build consolidated metadata (catalogs + scripts, all columns)
    2. Save consolidated metadata CSV
    3. Merge with benchmark → three files (merged, only_consolidated, only_benchmark)
    4. Attach CER scores to merged data
    5. Save merged report with CER and final summary
    6. Generate cluster CSVs from merged data (with CER values)
    """
    # ── 1. Build consolidated metadata ────────────────────────────────────
    consolidated = build_consolidated_metadata(catalog_dir, script_dir)

    # ── 2. Save consolidated metadata ─────────────────────────────────────
    consolidated_csv = METADATA_DIR / "consolidated_metadata.csv"
    consolidated_csv.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(consolidated_csv, index=False)
    logger.info("Wrote consolidated metadata → %s", consolidated_csv)

    # ── 3. Merge with benchmark → three files ─────────────────────────────
    merged, only_in_consolidated, only_in_benchmark = merge_with_benchmark(
        consolidated, benchmark_csv,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)

    only_consol_path = reports_dir / "only_in_consolidated.csv"
    only_in_consolidated[["file_name"]].rename(columns={"file_name": "image_name"}).to_csv(
        only_consol_path, index=False,
    )
    logger.info(
        "Wrote only-in-consolidated → %s  (%d images)",
        only_consol_path, len(only_in_consolidated),
    )

    only_bench_path = reports_dir / "only_in_benchmark.csv"
    only_in_benchmark[["image_name"]].to_csv(only_bench_path, index=False)
    logger.info(
        "Wrote only-in-benchmark → %s  (%d images)",
        only_bench_path, len(only_in_benchmark),
    )

    # ── 4. Attach CER scores to merged data ───────────────────────────────
    merged = attach_cer_scores(merged, evaluation_dir)

    # ── 5. Save merged report (with CER included) ────────────────────────
    #    This single CSV is the source of truth for all downstream outputs:
    #    clusters, summaries, etc.
    merged_csv = reports_dir / "merged_report.csv"
    merged.to_csv(merged_csv, index=False)
    logger.info("Wrote merged report → %s  (%d rows)", merged_csv, len(merged))

    # ── 6. Generate clusters & summary from the saved merged CSV ─────────
    generate_clusters_from_merged(
        merged_csv, reports_dir,
        evaluation_dir=evaluation_dir,
        only_in_benchmark_csv=only_bench_path,
    )

    return merged_csv


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )

    merged_path = generate_reports()

    # ── Quick summary ─────────────────────────────────────────────────────
    merged = pd.read_csv(merged_path)
    cer_cols = [c for c in merged.columns if c.endswith("_cer")]

    print(f"\n{'=' * 70}")
    print(f"Merged report:  {merged_path}")
    print(f"  Total rows:   {len(merged)}")
    print(f"  Columns:      {list(merged.columns)}")
    print(f"  CER columns:  {cer_cols}")

    if "technology" in merged.columns:
        print("\n  Rows per technology:")
        for tech, count in (
            merged["technology"].value_counts(dropna=False).sort_index().items()
        ):
            print(f"    {str(tech):30s} {count:>5d}")

    print("\n  Clustered outputs:")
    for label, d in [
        ("by_technology", REPORTS_DIR / "by_technology"),
        ("by_main_styles", REPORTS_DIR / "by_main_styles"),
        ("by_rarity", REPORTS_DIR / "by_rarity"),
    ]:
        if d.exists():
            files = sorted(d.glob("*.csv"))
            print(f"    {label}/  ({len(files)} files)")
            for f in files:
                df_f = pd.read_csv(f)
                print(f"      {f.name:40s} {len(df_f):>5d} rows, cols: {list(df_f.columns)}")

    # Report split counts
    for label, fname in [
        ("Only in consolidated (not in benchmark)", "only_in_consolidated.csv"),
        ("Only in benchmark (not in consolidated)", "only_in_benchmark.csv"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            df_f = pd.read_csv(p)
            print(f"\n  {label}: {len(df_f)} images")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
