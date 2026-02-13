"""
Report generation: merge per-model CER evaluation CSVs with consolidated
metadata and produce clustered report CSVs by technology, main style,
and rarity on BDRC.

Output directory layout
-----------------------
data/reports/
├── final_summary.csv                   # average CER per model across all groupings
├── merged_report.csv                   # all models × all images with metadata
├── by_model/
│   ├── Ume_Druma_report.csv            # single-model CER + metadata
│   ├── Ume_Petsuk_report.csv
│   └── Woodblock_report.csv
├── by_technology/
│   ├── blockprints.csv
│   ├── digital_fonts.csv
│   ├── manuscripts.csv
│   └── …
├── by_main_styles/
│   ├── Uchen.csv
│   └── …
└── by_rarity/
    ├── 4 - very common.csv
    ├── 3 - common.csv
    └── …

- by_model CSVs each contain: image_name, batch_id, cer, + all metadata columns.
- Clustered CSVs contain the same columns as the merged report, filtered
  to the rows that belong to that cluster value.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "consolidated_metadata.csv"

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
MERGED_CSV = REPORTS_DIR / "merged_report.csv"

CLUSTER_DIRS = {
    "technology": REPORTS_DIR / "by_technology",
    "3_types": REPORTS_DIR / "by_main_styles",
    "popularity_on_bdrc": REPORTS_DIR / "by_rarity",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_filename(value: str) -> str:
    """Convert a cluster value into a filesystem-safe filename (without extension)."""
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


# ── Core functions ────────────────────────────────────────────────────────────
def load_evaluation_csvs(evaluation_dir: Path) -> pd.DataFrame:
    """
    Load all ``*_cer.csv`` files in *evaluation_dir* and pivot model CER
    values into separate columns.

    Returns a DataFrame with columns:
        ``image_name``, ``batch_id``, ``<Model1>_cer``, ``<Model2>_cer``, …

    Each row is one image; each model's CER appears in its own column.
    """
    cer_files = sorted(evaluation_dir.glob("*_cer.csv"))
    if not cer_files:
        raise FileNotFoundError(
            f"No *_cer.csv files found in {evaluation_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in cer_files:
        model_name = path.stem  # e.g. "Ume_Druma_cer"
        df = pd.read_csv(path, dtype={"image_name": str, "batch_id": str})
        df = df.rename(columns={"cer": model_name})
        frames.append(df)
        logger.info("Loaded evaluation: %-25s  %d rows", model_name, len(df))

    # Merge all model frames on (image_name, batch_id)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["image_name", "batch_id"], how="outer")

    logger.info(
        "Combined evaluation: %d images × %d model(s)",
        len(merged),
        len(frames),
    )
    return merged


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Load the consolidated metadata CSV."""
    df = pd.read_csv(metadata_csv, dtype=str)
    logger.info("Loaded metadata: %d rows, %d columns", len(df), len(df.columns))
    return df


def merge_evaluation_with_metadata(
    evaluation: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join evaluation data with metadata on
    ``evaluation.image_name == metadata.file_name``.

    Only images present in the evaluation CSVs are kept; metadata columns
    are added where a match exists.
    """
    merged = evaluation.merge(
        metadata,
        left_on="image_name",
        right_on="file_name",
        how="left",
    )

    # Drop the duplicate join key (file_name) since image_name already carries it
    merged = merged.drop(columns=["file_name"], errors="ignore")

    matched = merged["technology"].notna().sum()
    total = len(merged)
    logger.info(
        "Merged: %d / %d images matched metadata (%.1f%%)",
        matched,
        total,
        matched / total * 100 if total else 0,
    )
    return merged


def save_merged_report(df: pd.DataFrame, output_path: Path) -> Path:
    """Write the full merged report CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote merged report → %s  (%d rows)", output_path, len(df))
    return output_path


def save_per_model_reports(
    merged: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """
    Write one CSV per model into *output_dir*.

    Each CSV has columns:
        ``image_name``, ``batch_id``, ``cer``, + all metadata columns.

    Returns a list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cer_cols = [c for c in merged.columns if c.endswith("_cer")]
    metadata_cols = [
        c for c in merged.columns
        if c not in {"image_name", "batch_id"} and not c.endswith("_cer")
    ]

    written: list[Path] = []
    for cer_col in cer_cols:
        # Derive a clean model name:  "Ume_Druma_cer" → "Ume_Druma"
        model_name = cer_col.removesuffix("_cer")

        df_model = merged[["image_name", "batch_id", cer_col] + metadata_cols].copy()
        df_model = df_model.rename(columns={cer_col: "cer"})

        path = output_dir / f"{model_name}_report.csv"
        df_model.to_csv(path, index=False)
        written.append(path)
        logger.info("Wrote per-model report: %-25s  %d rows → %s", model_name, len(df_model), path.name)

    return written


def generate_clustered_csvs(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
) -> list[Path]:
    """
    Split *df* by unique values in *column* and write one CSV per group
    into *output_dir*.

    Rows where *column* is NaN/empty are collected into a special
    ``_unspecified.csv`` file (if any exist).

    Returns a list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Separate rows with missing values
    mask_na = df[column].isna() | (df[column].astype(str).str.strip() == "")
    df_specified = df[~mask_na]
    df_unspecified = df[mask_na]

    for value, group in df_specified.groupby(column, sort=True):
        fname = _safe_filename(value) + ".csv"
        path = output_dir / fname
        group.to_csv(path, index=False)
        written.append(path)
        logger.info("  %-40s %4d rows → %s", value, len(group), path.name)

    if len(df_unspecified) > 0:
        path = output_dir / "_unspecified.csv"
        df_unspecified.to_csv(path, index=False)
        written.append(path)
        logger.info(
            "  %-40s %4d rows → %s",
            "(unspecified / NA)",
            len(df_unspecified),
            path.name,
        )

    return written


def generate_final_summary(
    merged: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Produce a single summary CSV with average CER per model, grouped by
    every clustering dimension.

    Columns:
        ``category``, ``group``, ``image_count``,
        ``<Model1>_avg_cer``, ``<Model2>_avg_cer``, …

    Rows include:
        - overall average
        - one row per technology
        - one row per main style (3_types)
        - one row per rarity level (popularity_on_bdrc)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cer_cols = [c for c in merged.columns if c.endswith("_cer")]
    rows: list[dict] = []

    # ── Overall ───────────────────────────────────────────────────────────
    row: dict = {
        "category": "overall",
        "group": "all",
        "image_count": len(merged),
    }
    for col in cer_cols:
        model = col.removesuffix("_cer")
        row[f"{model}_avg_cer"] = round(merged[col].mean(), 6)
    rows.append(row)

    # ── Grouped summaries ─────────────────────────────────────────────────
    group_configs = {
        "technology": "technology",
        "main_style": "3_types",
        "rarity": "popularity_on_bdrc",
    }

    for category_label, column in group_configs.items():
        # rows with actual values
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

    # Ensure column order: category, group, image_count, then model avg_cer cols
    avg_cols = [c for c in df_summary.columns if c.endswith("_avg_cer")]
    df_summary = df_summary[["category", "group", "image_count"] + sorted(avg_cols)]

    df_summary.to_csv(output_path, index=False)
    logger.info("Wrote final summary → %s  (%d rows)", output_path, len(df_summary))
    return output_path


# ── Main entry point ─────────────────────────────────────────────────────────
def generate_reports(
    evaluation_dir: Path = EVALUATION_DIR,
    metadata_csv: Path = METADATA_CSV,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """
    End-to-end report generation:
    1. Load & pivot evaluation CSVs
    2. Load consolidated metadata
    3. Merge them
    4. Save the merged report
    5. Generate clustered CSVs by technology, main style, and rarity

    Returns the path to the merged report CSV.
    """
    # 1. Load evaluation data
    evaluation = load_evaluation_csvs(evaluation_dir)

    # 2. Load metadata
    metadata = load_metadata(metadata_csv)

    # 3. Merge
    merged = merge_evaluation_with_metadata(evaluation, metadata)

    # 4. Save merged report
    merged_csv = reports_dir / "merged_report.csv"
    save_merged_report(merged, merged_csv)

    # 5. Save per-model reports (one CSV per model with cer + metadata)
    save_per_model_reports(merged, reports_dir / "by_model")

    # 6. Generate final summary (average CER across all groupings)
    generate_final_summary(merged, reports_dir / "final_summary.csv")

    # 7. Generate clustered CSVs
    cluster_configs = {
        "technology": reports_dir / "by_technology",
        "3_types": reports_dir / "by_main_styles",
        "popularity_on_bdrc": reports_dir / "by_rarity",
    }

    for column, out_dir in cluster_configs.items():
        logger.info("Clustering by '%s' → %s/", column, out_dir)
        generate_clustered_csvs(merged, column, out_dir)

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

    print(f"\n{'='*70}")
    print(f"Merged report:  {merged_path}")
    print(f"  Total rows:   {len(merged)}")
    print(f"  Columns:      {list(merged.columns)}")
    print(f"  CER columns:  {cer_cols}")

    if "technology" in merged.columns:
        print(f"\n  Rows per technology:")
        for tech, count in (
            merged["technology"].value_counts(dropna=False).sort_index().items()
        ):
            print(f"    {str(tech):30s} {count:>5d}")

    print(f"\n  Clustered outputs:")
    for label, d in [
        ("by_model", REPORTS_DIR / "by_model"),
        ("by_technology", REPORTS_DIR / "by_technology"),
        ("by_main_styles", REPORTS_DIR / "by_main_styles"),
        ("by_rarity", REPORTS_DIR / "by_rarity"),
    ]:
        if d.exists():
            files = sorted(d.glob("*.csv"))
            print(f"    {label}/  ({len(files)} files)")
            for f in files:
                row_count = sum(1 for _ in open(f)) - 1  # subtract header
                print(f"      {f.name:40s} {row_count:>5d} rows")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
