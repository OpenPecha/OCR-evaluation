"""
Metadata preparation: consolidate all per-technology benchmark catalog CSVs
into a single unified CSV with only the columns needed for the CER evaluation
report.

Kept columns (report-relevant only):
  - technology        : clustering axis (derived from source sheet)
  - file_name         : image identifier
  - script_id         : pivot key into the script catalog
  - format            : image characteristic (pecha, book, scroll …)
  - legibility        : image characteristic
  - color             : image characteristic
  - interlinear       : image characteristic
  - 3_types           : script catalog – large style categories
  - 8_categories      : script catalog – more specific categories
  - descenders_length_ratio : script catalog – descender length
  - popularity_on_bdrc      : script catalog – rarity on BDRC
  - period            : script catalog – time period

Columns that exist in the source sheets but are NOT needed for the report
(font paths, font names, dpi, source, notes, woodblock_style, etc.) are
dropped during consolidation.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
METADATA_DIR = Path("./data/metadata")
OUTPUT_PATH = METADATA_DIR / "consolidated_metadata.csv"

# Maps source CSV filename (stem, without .csv) → technology label
CATALOG_FILES = {
    "Benchmark catalog - blockprints": "blockprints",
    "Benchmark catalog - digital_fonts": "digital_fonts",
    "Benchmark catalog - manuscripts": "manuscripts",
    "Benchmark catalog - metal_types": "metal_types",
    "Benchmark catalog - modern": "modern",
    "Benchmark catalog - printeries_woodblocks": "printeries_woodblocks",
    "Benchmark catalog - typewriters": "typewriters",
}

# Canonical column rename map (applied *after* lowercasing + stripping)
COLUMN_RENAMES = {
    "file name": "file_name",
    "script id": "script_id",
    "descenders length ratio": "descenders_length_ratio",
    "3 types": "3_types",
    "8 categories": "8_categories",
    "popularity on bdrc": "popularity_on_bdrc",
}

# Only these columns are kept in the final output (in this order).
# Any column not listed here is dropped.
KEEP_COLUMNS = [
    "technology",
    "file_name",
    "script_id",
    # image characteristics
    "format",
    "legibility",
    "color",
    "interlinear",
    # script-catalog fields (populated for printeries_woodblocks; NA for others)
    "3_types",
    "8_categories",
    "descenders_length_ratio",
    "popularity_on_bdrc",
    "period",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and apply canonical renames to column names."""
    # lowercase + strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    # drop unnamed / empty columns (trailing-comma artifacts)
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    df = df.drop(columns=[""], errors="ignore")
    # apply canonical renames
    df = df.rename(columns=COLUMN_RENAMES)
    return df


def parse_metadata() -> pd.DataFrame:
    """
    Read all per-technology catalog CSVs, normalize columns,
    tag each row with its technology, keep only the report-relevant
    columns, and concatenate into one DataFrame.
    """
    frames: list[pd.DataFrame] = []

    for stem, technology in CATALOG_FILES.items():
        path = METADATA_DIR / f"{stem}.csv"
        if not path.exists():
            logger.warning("Missing catalog file: %s", path)
            continue

        df = pd.read_csv(path)
        df = _normalize_columns(df)
        df.insert(0, "technology", technology)
        frames.append(df)
        logger.info("Loaded %-30s  %d rows, %d cols", technology, len(df), len(df.columns))

    if not frames:
        raise FileNotFoundError(f"No catalog CSVs found in {METADATA_DIR}")

    combined = pd.concat(frames, ignore_index=True)

    # drop rows with no file_name (empty padding rows in some sheets)
    combined = combined.dropna(subset=["file_name"], how="any")
    combined = combined[combined["file_name"].str.strip() != ""]

    # keep only report-relevant columns (in order); missing ones become NA
    keep = [c for c in KEEP_COLUMNS if c in combined.columns]
    combined = combined[keep]

    # add any KEEP_COLUMNS that were missing across all sheets as NA columns
    for col in KEEP_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    # ensure final column order matches KEEP_COLUMNS
    combined = combined[KEEP_COLUMNS]

    logger.info(
        "Consolidated: %d rows, %d columns, %d technologies",
        len(combined),
        len(combined.columns),
        combined["technology"].nunique(),
    )
    return combined


def main():
    """Main function to run the metadata preparation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )

    consolidated = parse_metadata()
    consolidated.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote consolidated CSV → %s", OUTPUT_PATH)

    # quick summary
    print(f"\n{'='*60}")
    print(f"Consolidated metadata written to: {OUTPUT_PATH}")
    print(f"  Total rows:    {len(consolidated)}")
    print(f"  Total columns: {len(consolidated.columns)}")
    print(f"  Columns:       {list(consolidated.columns)}")
    print(f"\n  Rows per technology:")
    for tech, count in consolidated["technology"].value_counts().sort_index().items():
        print(f"    {tech:30s} {count:>5d}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()